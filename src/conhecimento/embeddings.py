"""Backends de embeddings compatíveis com o índice acadêmico do projeto.

O ambiente local usa SentenceTransformer para indexação e treinamento. No
deploy de consulta, o mesmo encoder é executado pelo ONNX Runtime em versão
quantizada. Isso evita carregar PyTorch no processo web sem trocar o
espaço vetorial usado pelo snapshot da literatura.
"""

from __future__ import annotations

import gc
import os
import threading
from typing import Iterable


REPOSITORIO_MODELO = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
REVISAO_MODELO = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
ARQUIVO_MODELO_ONNX = "onnx/model_quint8_avx2.onnx"
ARQUIVO_TOKENIZER = "tokenizer.json"
MAX_TOKENS = 128


def backend_embeddings(*, modo_consulta: bool) -> str:
    """Resolve o backend, permitindo override explícito por ambiente."""
    configurado = os.getenv("AL_IADO_EMBEDDINGS_BACKEND", "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "onnx": "onnx",
        "baixo_consumo": "onnx",
        "sentence-transformers": "sentence-transformers",
        "sentence_transformers": "sentence-transformers",
        "torch": "sentence-transformers",
    }
    if configurado not in aliases:
        opcoes = "auto, onnx ou sentence-transformers"
        raise ValueError(
            f"AL_IADO_EMBEDDINGS_BACKEND inválido: {configurado!r}; use {opcoes}."
        )
    resolvido = aliases[configurado]
    if resolvido == "auto":
        return "onnx" if modo_consulta else "sentence-transformers"
    return resolvido


def criar_modelo_embeddings(*, modo_consulta: bool):
    """Cria o encoder adequado sem importar PyTorch no modo de consulta."""
    backend = backend_embeddings(modo_consulta=modo_consulta)
    if backend == "onnx":
        return ModeloEmbeddingsONNX()

    from sentence_transformers import SentenceTransformer
    from src.core.config import MODELO_EMBEDDINGS

    return SentenceTransformer(MODELO_EMBEDDINGS)


class ModeloEmbeddingsONNX:
    """Adaptador leve com a interface ``SentenceTransformer.encode``.

    A sessão ONNX é carregada apenas na primeira consulta. O tokenizer
    multilíngue ocupa memória relevante quando materializado; por isso ele é
    aberto somente durante a tokenização e liberado antes da inferência. Um
    lock serializa consultas concorrentes para impedir picos multiplicados por
    sessões web simultâneas.
    """

    def __init__(
        self,
        *,
        repositorio: str = REPOSITORIO_MODELO,
        revisao: str = REVISAO_MODELO,
        arquivo_modelo: str = ARQUIVO_MODELO_ONNX,
        arquivo_tokenizer: str = ARQUIVO_TOKENIZER,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.repositorio = repositorio
        self.revisao = revisao
        self.arquivo_modelo = arquivo_modelo
        self.arquivo_tokenizer = arquivo_tokenizer
        self.max_tokens = int(max_tokens)
        self._sessao = None
        self._caminho_tokenizer: str | None = None
        self._lock = threading.RLock()

    def _baixar(self, arquivo: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self.repositorio,
            filename=arquivo,
            revision=self.revisao,
        )

    def _obter_sessao(self):
        if self._sessao is not None:
            return self._sessao

        import onnxruntime as ort

        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = max(
            1, int(os.getenv("AL_IADO_ONNX_THREADS", "1"))
        )
        opcoes.inter_op_num_threads = 1
        opcoes.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opcoes.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opcoes.enable_cpu_mem_arena = False
        opcoes.enable_mem_pattern = False
        self._sessao = ort.InferenceSession(
            self._baixar(self.arquivo_modelo),
            sess_options=opcoes,
            providers=["CPUExecutionProvider"],
        )
        return self._sessao

    def _tokenizar(self, textos: list[str]):
        import numpy as np
        from tokenizers import Tokenizer

        if self._caminho_tokenizer is None:
            self._caminho_tokenizer = self._baixar(self.arquivo_tokenizer)

        tokenizer = Tokenizer.from_file(self._caminho_tokenizer)
        tokenizer.enable_truncation(max_length=self.max_tokens)
        tokenizer.enable_padding()
        codificados = tokenizer.encode_batch(textos)
        entradas = {
            "input_ids": np.asarray(
                [item.ids for item in codificados], dtype=np.int64
            ),
            "attention_mask": np.asarray(
                [item.attention_mask for item in codificados], dtype=np.int64
            ),
            "token_type_ids": np.asarray(
                [item.type_ids for item in codificados], dtype=np.int64
            ),
        }

        del codificados, tokenizer
        gc.collect()
        return entradas

    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        import numpy as np

        mascara = attention_mask[..., None].astype(np.float32, copy=False)
        soma = (token_embeddings * mascara).sum(axis=1)
        divisor = np.clip(mascara.sum(axis=1), 1e-9, None)
        return soma / divisor

    def encode(
        self,
        sentences: str | Iterable[str],
        *,
        batch_size: int = 16,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        show_progress_bar: bool | None = None,
        **kwargs,
    ):
        """Codifica texto(s), preservando o contrato usado no projeto."""
        del show_progress_bar, kwargs
        import numpy as np

        entrada_unica = isinstance(sentences, str)
        textos = [sentences] if entrada_unica else [str(item) for item in sentences]
        if not textos:
            vazio = np.empty((0, 384), dtype=np.float32)
            return vazio if convert_to_numpy else vazio.tolist()

        vetores = []
        tamanho_lote = max(1, min(int(batch_size), 32))
        with self._lock:
            sessao = self._obter_sessao()
            for inicio in range(0, len(textos), tamanho_lote):
                entradas = self._tokenizar(textos[inicio:inicio + tamanho_lote])
                token_embeddings = sessao.run(None, entradas)[0]
                lote = self._mean_pooling(
                    token_embeddings, entradas["attention_mask"]
                ).astype(np.float32, copy=False)
                if normalize_embeddings:
                    normas = np.linalg.norm(lote, axis=1, keepdims=True)
                    lote = lote / np.clip(normas, 1e-12, None)
                vetores.append(lote)

        resultado = np.concatenate(vetores, axis=0)
        if entrada_unica:
            resultado = resultado[0]
        return resultado if convert_to_numpy else resultado.tolist()
