"""
retrieval_metrics.py — Al IAdo PV / Sprint 4 (7.2)

Métricas de RECUPERAÇÃO (separadas da qualidade da geração): Recall@k, MRR,
nDCG@k e cobertura/diversidade de fontes. Funções puras e determinísticas
(testáveis sem ChromaDB). `recuperados` é a lista ORDENADA de fontes retornadas;
`relevantes` é o conjunto de fontes esperadas.
"""

from __future__ import annotations

import math


def recall_at_k(recuperados: list, relevantes, k: int) -> float:
    rel = set(relevantes)
    if not rel:
        return 0.0
    topk = set(recuperados[:k])
    return len(rel & topk) / len(rel)


def mrr(recuperados: list, relevantes) -> float:
    """Reciprocal rank da PRIMEIRA fonte relevante (0 se nenhuma)."""
    rel = set(relevantes)
    for i, item in enumerate(recuperados, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(recuperados: list, relevantes, k: int) -> float:
    """nDCG@k com relevância binária."""
    rel = set(relevantes)
    dcg = 0.0
    for i, item in enumerate(recuperados[:k], start=1):
        if item in rel:
            dcg += 1.0 / math.log2(i + 1)
    n_ideal = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def cobertura_fontes(recuperados: list) -> int:
    """Número de fontes DISTINTAS recuperadas (diversidade)."""
    return len(set(recuperados))


def agregimar(linhas: list[dict], k: int = 5) -> dict:
    """Agrega métricas por pergunta em médias (linhas: [{recuperados, relevantes}])."""
    if not linhas:
        return {"n": 0}
    n = len(linhas)
    rec = sum(recall_at_k(x["recuperados"], x["relevantes"], k) for x in linhas) / n
    rr = sum(mrr(x["recuperados"], x["relevantes"]) for x in linhas) / n
    nd = sum(ndcg_at_k(x["recuperados"], x["relevantes"], k) for x in linhas) / n
    acerto = sum(1 for x in linhas if mrr(x["recuperados"], x["relevantes"]) > 0) / n
    return {
        "n": n, "k": k,
        f"recall@{k}": round(rec, 4),
        "mrr": round(rr, 4),
        f"ndcg@{k}": round(nd, 4),
        "taxa_fonte_correta": round(acerto, 4),
    }
