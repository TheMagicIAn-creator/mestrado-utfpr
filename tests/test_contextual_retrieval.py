"""Contracts for deterministic Contextual Retrieval R3."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.conhecimento.contextual_retrieval import (
    ESTRATEGIA_CONTEXTO_R3,
    ContextualizacaoInvalida,
    construir_prefixo_contextual,
    contextualizar_snapshot,
)
from src.conhecimento.indice_portatil import (
    exportar_colecao,
    importar_colecao,
    ler_manifesto,
)

HASH = "a" * 64


class ColecaoFalsa:
    name = "literatura_pv"

    def __init__(self, itens=None):
        self.itens = {item["id"]: item for item in (itens or [])}

    def count(self):
        return len(self.itens)

    def get(self, limit=100, offset=0, include=None):
        del include
        lote = list(self.itens.values())[offset:offset + limit]
        return {
            "ids": [item["id"] for item in lote],
            "documents": [item["documento"] for item in lote],
            "metadatas": [item["metadata"] for item in lote],
            "embeddings": [item["embedding"] for item in lote],
        }

    def upsert(self, ids, documents, metadatas, embeddings):
        for chunk_id, documento, metadata, embedding in zip(
            ids, documents, metadatas, embeddings
        ):
            self.itens[chunk_id] = {
                "id": chunk_id,
                "documento": documento,
                "metadata": metadata,
                "embedding": embedding,
            }


class EncoderFalso:
    def __init__(self):
        self.textos = []

    def encode(self, textos, **kwargs):
        assert kwargs["normalize_embeddings"] is False
        self.textos.extend(textos)
        return [[float(len(texto)), float(indice)] for indice, texto in enumerate(textos)]


@pytest.fixture
def item():
    return {
        "id": f"{HASH}__chunk_00000",
        "documento": "The observed failure rate decreases after maintenance.",
        "metadata": {
            "arquivo": "autor_estudo_2022.pdf",
            "arquivo_hash": HASH,
            "pagina_inicio": 12,
            "pagina_fim": 12,
            "chunk_index": 0,
            "total_chunks": 1,
            "idioma": "en",
            "autor": "Autor A",
            "titulo": "Reliability of photovoltaic inverters",
            "ano": "2022",
            "pasta": "confiabilidade",
        },
        "embedding": [0.1, 0.2],
    }


def test_contextualizacao_preserva_raw_ids_e_recalcula_embedding(tmp_path, item):
    origem = tmp_path / "r2.jsonl.gz"
    destino = tmp_path / "r3.jsonl.gz"
    exportar_colecao(
        ColecaoFalsa([item]),
        origem,
        modelo_embeddings="modelo-multilingue",
        hash_corpus="corpus",
        n_documentos=1,
    )
    encoder = EncoderFalso()

    resultado = contextualizar_snapshot(origem, destino, encoder, tamanho_lote=1)

    assert resultado["ja_estava_pronto"] is False
    assert resultado["retrieval_text_strategy"] == ESTRATEGIA_CONTEXTO_R3
    assert resultado["contextual_retrieval"]["llm_used"] is False
    assert resultado["contextual_retrieval"]["raw_text_unchanged"] is True
    assert ler_manifesto(origem)["retrieval_text_strategy"] == "identity_raw_text"
    with gzip.open(destino, "rt", encoding="utf-8") as arquivo:
        manifesto = json.loads(next(arquivo))
        registro = json.loads(next(arquivo))
    assert manifesto["hash_corpus_sha256"] == "corpus"
    assert registro["chunk_id"] == item["id"]
    assert registro["raw_text"] == item["documento"]
    assert "Documento: Reliability of photovoltaic inverters" in registro[
        "retrieval_text"
    ]
    assert registro["retrieval_text"].endswith(item["documento"])
    assert registro["embedding"] != item["embedding"]
    assert encoder.textos == [registro["retrieval_text"]]

    restaurada = ColecaoFalsa()
    importar_colecao(restaurada, destino)
    assert restaurada.itens[item["id"]]["documento"] == registro["retrieval_text"]
    assert contextualizar_snapshot(origem, destino, encoder)["ja_estava_pronto"] is True


def test_prefixo_usa_apenas_metadados_disponiveis():
    prefixo = construir_prefixo_contextual(
        {
            "arquivo": "paper.pdf",
            "titulo": "Study",
            "pagina_inicio": 3,
            "pagina_fim": 4,
        }
    )

    assert prefixo == (
        "Contexto documental deterministico:\n"
        "Documento: Study\n"
        "Pagina fisica: 3-4"
    )
    assert "Autores" not in prefixo
    assert "Secao" not in prefixo


def test_contextualizacao_recusa_sobrescrever_snapshot_base(tmp_path, item):
    origem = tmp_path / "r2.jsonl.gz"
    exportar_colecao(
        ColecaoFalsa([item]),
        origem,
        modelo_embeddings="modelo",
        hash_corpus="corpus",
        n_documentos=1,
    )

    with pytest.raises(ContextualizacaoInvalida, match="separado"):
        contextualizar_snapshot(origem, origem, EncoderFalso())


def test_cli_contextualiza_snapshot_r3_sem_promover_indice(
    monkeypatch, tmp_path, capsys
):
    from scripts import manter_base

    destino = tmp_path / "candidato.jsonl.gz"
    chamadas = {}

    def contextualizar(origem, caminho_destino, modelo, *, tamanho_lote):
        chamadas.update(
            origem=origem,
            destino=caminho_destino,
            modelo=modelo,
            tamanho_lote=tamanho_lote,
        )
        return {"stage": "R3", "promovido": False}

    monkeypatch.setitem(
        sys.modules,
        "src.conhecimento.benchmark_retrieval",
        SimpleNamespace(resolver_caminho_no_projeto=lambda caminho: Path(caminho)),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.conhecimento.contextual_retrieval",
        SimpleNamespace(contextualizar_snapshot=contextualizar),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.conhecimento.embeddings",
        SimpleNamespace(criar_modelo_embeddings=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.core.config",
        SimpleNamespace(ARQUIVO_INDICE_LITERATURA=Path("origem-r2.jsonl.gz")),
    )

    assert manter_base.contextualizar_literatura_r3(destino, tamanho_lote=7) == 0
    assert chamadas == {
        "origem": Path("origem-r2.jsonl.gz"),
        "destino": destino,
        "modelo": {"modo_consulta": False},
        "tamanho_lote": 7,
    }
    assert json.loads(capsys.readouterr().out) == {
        "stage": "R3",
        "promovido": False,
    }


def test_cli_expõe_comando_contextual_r3(monkeypatch, tmp_path):
    from scripts import manter_base

    destino = tmp_path / "candidato.jsonl.gz"
    chamadas = []
    monkeypatch.setattr(
        manter_base,
        "contextualizar_literatura_r3",
        lambda caminho, *, tamanho_lote: chamadas.append(
            (caminho, tamanho_lote)
        )
        or 0,
    )

    retorno = manter_base.main(
        [
            "contextualizar-literatura-r3",
            "--destino",
            str(destino),
            "--tamanho-lote",
            "9",
        ]
    )

    assert retorno == 0
    assert chamadas == [(destino, 9)]
