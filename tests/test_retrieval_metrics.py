"""
Sprint 4 — métricas de recuperação (7.2): Recall@k, MRR, nDCG@k, cobertura.
"""

from src.ml.retrieval_metrics import (
    agregimar,
    cobertura_fontes,
    mrr,
    ndcg_at_k,
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
