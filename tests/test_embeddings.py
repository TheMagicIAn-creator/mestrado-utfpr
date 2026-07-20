from __future__ import annotations

import numpy as np
import pytest

from src.conhecimento.embeddings import (
    ModeloEmbeddingsONNX,
    backend_embeddings,
)


def test_backend_auto_separa_calculo_de_consulta(monkeypatch):
    monkeypatch.delenv("AL_IADO_EMBEDDINGS_BACKEND", raising=False)

    assert backend_embeddings(modo_consulta=True) == "onnx"
    assert backend_embeddings(modo_consulta=False) == "sentence-transformers"


@pytest.mark.parametrize(
    ("configurado", "esperado"),
    [
        ("onnx", "onnx"),
        ("baixo_consumo", "onnx"),
        ("torch", "sentence-transformers"),
        ("sentence_transformers", "sentence-transformers"),
    ],
)
def test_backend_respeita_override(monkeypatch, configurado, esperado):
    monkeypatch.setenv("AL_IADO_EMBEDDINGS_BACKEND", configurado)
    assert backend_embeddings(modo_consulta=False) == esperado


def test_backend_rejeita_valor_desconhecido(monkeypatch):
    monkeypatch.setenv("AL_IADO_EMBEDDINGS_BACKEND", "modelo-magico")
    with pytest.raises(ValueError, match="AL_IADO_EMBEDDINGS_BACKEND"):
        backend_embeddings(modo_consulta=True)


def test_mean_pooling_ignora_padding():
    tokens = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[2.0, 6.0], [4.0, 8.0], [6.0, 10.0]],
        ],
        dtype=np.float32,
    )
    mascara = np.asarray([[1, 1, 0], [1, 1, 1]], dtype=np.int64)

    resultado = ModeloEmbeddingsONNX._mean_pooling(tokens, mascara)

    np.testing.assert_allclose(resultado[0], [2.0, 3.0])
    np.testing.assert_allclose(resultado[1], [4.0, 8.0])


def test_encode_preserva_contrato_sem_carregar_dependencias_pesadas(monkeypatch):
    modelo = ModeloEmbeddingsONNX()

    class SessaoFake:
        def run(self, _saidas, entradas):
            ids = entradas["input_ids"].astype(np.float32)
            return [np.stack((ids, ids * 2), axis=-1)]

    def tokenizar(textos):
        ids = np.asarray([[len(texto), 2] for texto in textos], dtype=np.int64)
        return {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),
        }

    monkeypatch.setattr(modelo, "_obter_sessao", lambda: SessaoFake())
    monkeypatch.setattr(modelo, "_tokenizar", tokenizar)

    lote = modelo.encode(["abc", "abcdef"])
    unico = modelo.encode("abc", normalize_embeddings=True)

    assert lote.shape == (2, 2)
    np.testing.assert_allclose(lote[0], [2.5, 5.0])
    assert unico.shape == (2,)
    np.testing.assert_allclose(np.linalg.norm(unico), 1.0)
