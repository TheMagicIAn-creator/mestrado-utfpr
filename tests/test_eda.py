"""Contratos leves da análise exploratória."""

import numpy as np
import pandas as pd

from src.ml import eda


def test_analise_fases_calcula_desequilibrio_nulo_para_fases_iguais():
    sinal = np.sin(np.linspace(0, 8 * np.pi, 140))
    df = pd.DataFrame({"i_a_k": sinal, "i_b_k": sinal, "i_c_k": sinal})

    resultado = eda.analise_fases(df.copy()).dropna(subset=["desequilibrio"])

    assert not resultado.empty
    assert np.allclose(resultado["desequilibrio"], 0.0)


def test_executar_eda_orquestra_todas_as_etapas(monkeypatch):
    chamadas = []
    df = pd.DataFrame({"n_k": [1.0]})

    monkeypatch.setattr(eda, "carregar_dados", lambda: df)
    monkeypatch.setattr(eda, "analise_basica", lambda dados: chamadas.append("basica"))
    monkeypatch.setattr(
        eda,
        "analise_fases",
        lambda dados: chamadas.append("fases") or dados.assign(desequilibrio=0.0),
    )
    for nome in (
        "plotar_series_temporais",
        "plotar_distribuicoes",
        "plotar_correlacoes",
        "plotar_desequilibrio",
    ):
        monkeypatch.setattr(eda, nome, lambda dados, nome=nome: chamadas.append(nome))

    assert eda.executar_eda() is True
    assert chamadas == [
        "basica",
        "fases",
        "plotar_series_temporais",
        "plotar_distribuicoes",
        "plotar_correlacoes",
        "plotar_desequilibrio",
    ]
