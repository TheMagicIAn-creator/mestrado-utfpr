from __future__ import annotations

import math

import numpy as np
import pytest

from src.ml.confiabilidade_fisica_v2 import (
    CENARIOS,
    HOURS_PER_YEAR,
    auditoria_dimensional,
    confiabilidade_exponencial,
    curvas_cenarios,
    densidade_falha_exponencial,
    marcos_cenarios,
    probabilidade_falha_exponencial,
    quantil_falha_exponencial,
    tabela_cenarios,
    taxa_risco_exponencial,
)


def _cenario(scenario_id: str):
    return next(cenario for cenario in CENARIOS if cenario.scenario_id == scenario_id)


def test_cenarios_preservam_cinco_contextos_bibliograficos_distintos():
    assert len(CENARIOS) == 5
    assert len({cenario.scenario_id for cenario in CENARIOS}) == 5
    assert {cenario.source_type for cenario in CENARIOS} == {
        "secondary_bibliographic_rate",
        "literature_assumption",
        "mil_hdbk_217f_prediction",
        "illustrative_markov_parameter",
    }


def test_conversoes_dimensionais_das_fontes():
    torres = _cenario("torres_colli_rate")
    cristaldi = _cenario("cristaldi_inverter_rate")
    obeidat_alta = _cenario("obeidat_high_quality")
    obeidat_baixa = _cenario("obeidat_low_quality")
    dhople = _cenario("dhople_markov_example")

    assert torres.lambda_per_hour == pytest.approx(1.75e-4)
    assert torres.lambda_per_year == pytest.approx(1.75e-4 * HOURS_PER_YEAR)
    assert torres.reciprocal_time_years == pytest.approx(1 / (1.75e-4 * 8760))
    assert cristaldi.lambda_per_year == pytest.approx(0.125)
    assert obeidat_alta.lambda_per_hour == pytest.approx(8.069e-6)
    assert obeidat_baixa.lambda_per_hour == pytest.approx(50.76e-6)
    assert dhople.lambda_per_year == pytest.approx(0.1)


def test_mtbf_predito_reconcilia_com_arredondamento_da_fonte():
    for scenario_id in ("obeidat_high_quality", "obeidat_low_quality"):
        cenario = _cenario(scenario_id)
        diferenca = abs(cenario.reciprocal_time_hours - cenario.reported_mean_hours)
        assert diferenca / cenario.reported_mean_hours < 1e-4


def test_identidades_da_distribuicao_exponencial():
    t = np.linspace(0, 20, 101)
    for cenario in CENARIOS:
        taxa = cenario.lambda_per_year
        r = confiabilidade_exponencial(t, taxa)
        f_acumulada = probabilidade_falha_exponencial(t, taxa)
        densidade = densidade_falha_exponencial(t, taxa)
        risco = taxa_risco_exponencial(t, taxa)
        np.testing.assert_allclose(r + f_acumulada, 1.0, rtol=1e-13)
        np.testing.assert_allclose(densidade, risco * r, rtol=1e-13)
        np.testing.assert_allclose(risco, taxa, rtol=0, atol=0)
        assert np.all(np.diff(r) <= 0)
        assert np.all(np.diff(f_acumulada) >= 0)


def test_b1_b10_mediana_e_media_tem_ordem_correta():
    for cenario in CENARIOS:
        taxa = cenario.lambda_per_year
        b1 = quantil_falha_exponencial(0.01, taxa)
        b10 = quantil_falha_exponencial(0.10, taxa)
        mediana = quantil_falha_exponencial(0.50, taxa)
        assert b1 < b10 < mediana < 1.0 / taxa
        assert float(probabilidade_falha_exponencial(b10, taxa)) == pytest.approx(0.10)


def test_beta_eta_fisicos_permanecem_nao_estimados():
    tabela = tabela_cenarios()
    assert "beta" not in tabela.columns
    assert "eta" not in tabela.columns
    assert not tabela["caveat"].str.contains("estimado pelo GPVS", case=False).any()


def test_curvas_e_marcos_tem_contrato_completo():
    curvas = curvas_cenarios(horizonte_anos=12, n_points=25)
    marcos = marcos_cenarios()
    assert len(curvas) == 5 * 25
    assert curvas.groupby("scenario_id")["time_years"].agg(["min", "max"]).eq(
        [0.0, 12.0]
    ).all().all()
    assert len(marcos) == 5
    assert {
        "b1_years",
        "b10_years",
        "median_years",
        "reciprocal_time_years",
    }.issubset(marcos.columns)


def test_auditoria_registra_erros_de_unidade_sem_corrigir_a_fonte_em_silencio():
    auditoria = auditoria_dimensional()
    por_id = {item["audit_id"]: item for item in auditoria}
    taxa = por_id["torres_rate_reciprocal_unit"]
    reparo = por_id["torres_repair_rate_reciprocal_unit"]
    assert taxa["status"] == "source_unit_inconsistency"
    assert "horas" in taxa["dimensional_result"]
    assert "12 horas" in reparo["dimensional_result"]
    assert math.isclose(1 / 1.8e-4, 5555.555555555556)


@pytest.mark.parametrize(
    "func,args",
    (
        (confiabilidade_exponencial, (-1.0, 0.1)),
        (confiabilidade_exponencial, (1.0, 0.0)),
        (quantil_falha_exponencial, (0.0, 0.1)),
        (quantil_falha_exponencial, (1.0, 0.1)),
    ),
)
def test_entradas_dimensionais_invalidas_sao_recusadas(func, args):
    with pytest.raises(ValueError):
        func(*args)

