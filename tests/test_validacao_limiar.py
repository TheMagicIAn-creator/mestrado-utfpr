"""
Validação sintética interna E2 com limiar CONGELADO.

Garante que metricas_no_limiar() aplica o limiar RECEBIDO (carregado de
limiar.json, congelado) e NÃO escolhe o limiar ótimo no próprio conjunto:
- mudar o limiar muda as predições (logo, o limiar é respeitado como entrada);
- AUC-ROC é invariante ao limiar (métrica independente de limiar);
- AUC-PR é reportada.
"""

import numpy as np
import pandas as pd

from src.ml.validacao import (
    _normalizar_matriz_por_linha,
    _severidade_alvo,
    _severidade_transicao,
    _severidades_matrizes,
    metricas_no_limiar,
)


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
    assert m_baixo["auc_pr"] == m_baixo["average_precision"]


def test_auc_roc_invariante_ao_limiar():
    rng = np.random.default_rng(1)
    neg = rng.normal(0.20, 0.05, 150)
    pos = rng.normal(1.00, 0.20, 150)
    a = metricas_no_limiar(neg, pos, 0.50)["auc_roc"]
    b = metricas_no_limiar(neg, pos, 0.80)["auc_roc"]
    assert abs(a - b) < 1e-9  # AUC não depende do limiar (independente do corte)


def test_matriz_visual_e_normalizada_por_classe_real():
    cm = np.array([[90, 10], [2, 8]])
    proporcoes = _normalizar_matriz_por_linha(cm)

    np.testing.assert_allclose(proporcoes.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(proporcoes, [[0.9, 0.1], [0.2, 0.8]])


def test_selecao_visual_prioriza_transicao_e_primeiro_nivel_no_alvo():
    resultados = {}
    recalls = {0.05: 0.02, 0.1: 0.12, 0.2: 0.48, 0.3: 0.91, 0.5: 0.97,
               0.7: 1.0, 1.0: 1.0}
    for sev, recall in recalls.items():
        resultados[f"fusivel_ac_sev{sev}"] = {"recall": recall}

    assert _severidade_alvo(resultados, "fusivel_ac") == 0.5
    assert _severidade_transicao(resultados, "fusivel_ac") == 0.2
    assert _severidades_matrizes(resultados, "fusivel_ac") == [0.05, 0.2, 0.5]


def test_fusivel_reduz_corrente_sem_fabricar_queda_de_tensao():
    from src.ml.injecao_falhas import falha_perda_fase_fusivel
    from src.ml.gpvs_principal import COLUNAS_CORRENTE, COLUNAS_TENSAO

    dados = {
        coluna: np.linspace(1.0, 2.0, 16)
        for coluna in COLUNAS_CORRENTE + COLUNAS_TENSAO
    }
    janela = pd.DataFrame(dados)
    falha = falha_perda_fase_fusivel(janela, severidade=1.0)

    np.testing.assert_allclose(
        falha[COLUNAS_CORRENTE[0]], janela[COLUNAS_CORRENTE[0]] * 0.88
    )
    for coluna in COLUNAS_TENSAO:
        np.testing.assert_allclose(falha[coluna], janela[coluna])
