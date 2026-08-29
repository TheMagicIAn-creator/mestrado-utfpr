"""
Sprint 4 — métricas de recuperação (7.2): Recall@k, MRR, nDCG@k, cobertura.
"""

import pytest

from src.conhecimento.retrieval_metrics import (
    agregimar,
    cobertura_fontes,
    hit_rate_at_k,
    metricas_por_evidencias,
    metricas_por_grupos,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    rec = ["a", "b", "c", "d"]
    rel = {"c", "x"}
    assert recall_at_k(rec, rel, 3) == 0.5   # c no top-3, x não → 1/2
    assert recall_at_k(rec, rel, 2) == 0.0   # nenhum no top-2


def test_mrr():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5  # rank 2
    assert mrr(["a", "b"], {"z"}) == 0.0


def test_precision_e_hit_rate_at_k():
    recuperados = ["a", "b", "c"]
    relevantes = {"a", "c"}
    assert precision_at_k(recuperados, relevantes, 2) == 0.5
    assert hit_rate_at_k(recuperados, relevantes, 2) == 1.0
    assert hit_rate_at_k(["x", "y"], relevantes, 2) == 0.0


def test_ndcg_topo_vs_fim():
    rel = {"a"}
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == 1.0       # relevante no topo
    assert 0.0 < ndcg_at_k(["b", "c", "a"], rel, 3) < 1.0  # relevante no fim


def test_cobertura_e_agrega():
    assert cobertura_fontes(["a", "a", "b"]) == 2
    linhas = [
        {"recuperados": ["a", "b"], "relevantes": {"a"}},
        {"recuperados": ["x", "y"], "relevantes": {"y"}},
    ]
    ag = agregimar(linhas, k=2)
    assert ag["n"] == 2 and ag["taxa_fonte_correta"] == 1.0
    assert ag["precision@2"] == 0.5


def test_metricas_por_grupos_nao_premiam_chunks_vizinhos_repetidos():
    grupos = [
        {"chunk_ids": ["a1", "a2"], "relevance": 3},
        {"chunk_ids": ["b1"], "relevance": 2},
    ]
    metricas = metricas_por_grupos(["a2", "a1", "x", "b1"], grupos, 4)

    assert metricas["grupos_recuperados"] == 2
    assert metricas["recall@4"] == 1.0
    assert metricas["precision@4"] == 0.5
    assert metricas["mrr"] == 1.0
    assert 0.0 < metricas["ndcg@4"] < 1.0


def test_metricas_por_grupos_validam_o_contrato():
    try:
        metricas_por_grupos(["a"], [{"chunk_ids": [], "relevance": 2}], 5)
    except ValueError as exc:
        assert "chunk_id" in str(exc)
    else:
        raise AssertionError("Grupo sem chunk deveria ser rejeitado.")


def test_metricas_por_evidencias_separam_chunk_pagina_e_documento():
    grupos = [
        {
            "document_id": "doc-a",
            "pages": [7],
            "chunk_ids": ["chunk-esperado"],
            "relevance": 3,
        }
    ]
    recuperados = [
        {
            "document_id": "doc-a",
            "pages": [7],
            "chunk_id": "chunk-vizinho",
        }
    ]

    assert metricas_por_evidencias(recuperados, grupos, 1, nivel="chunk")["recall@1"] == 0.0
    assert metricas_por_evidencias(recuperados, grupos, 1, nivel="page")["recall@1"] == 1.0
    assert metricas_por_evidencias(recuperados, grupos, 1, nivel="document")["recall@1"] == 1.0

    with pytest.raises(ValueError, match="Nível"):
        metricas_por_evidencias(recuperados, grupos, 1, nivel="seção")
