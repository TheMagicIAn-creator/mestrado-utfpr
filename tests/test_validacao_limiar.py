"""
Sprint 1 — validação formal com limiar CONGELADO (item 3.3).

Garante que metricas_no_limiar() aplica o limiar RECEBIDO (carregado de
limiar.json, congelado) e NÃO escolhe o limiar ótimo no próprio conjunto:
- mudar o limiar muda as predições (logo, o limiar é respeitado como entrada);
- AUC-ROC é invariante ao limiar (métrica independente de limiar);
- AUC-PR é reportada.
"""

import numpy as np

from src.ml.validacao import metricas_no_limiar


def test_metricas_respeitam_limiar_passado():
    rng = np.random.default_rng(0)
    erros_neg = rng.normal(0.20, 0.05, 200)  # saudável → erro baixo
    erros_pos = rng.normal(1.00, 0.20, 200)  # falha → erro alto

    m_baixo = metricas_no_limiar(erros_neg, erros_pos, limiar=0.30)
    m_alto = metricas_no_limiar(erros_neg, erros_pos, limiar=0.90)

    # as predições mudam com o limiar → a função usa o limiar DADO,
    # não otimiza internamente sobre o conjunto avaliado
    assert (
        m_baixo["recall"] != m_alto["recall"]
        or m_baixo["precision"] != m_alto["precision"]
    )
    assert "auc_pr" in m_baixo and "auc_roc" in m_baixo


def test_auc_roc_invariante_ao_limiar():
    rng = np.random.default_rng(1)
    neg = rng.normal(0.20, 0.05, 150)
    pos = rng.normal(1.00, 0.20, 150)
    a = metricas_no_limiar(neg, pos, 0.50)["auc_roc"]
    b = metricas_no_limiar(neg, pos, 0.80)["auc_roc"]
    assert abs(a - b) < 1e-9  # AUC não depende do limiar (independente do corte)
