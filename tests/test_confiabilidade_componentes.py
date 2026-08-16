from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.config import RAIZ_PROJETO
from src.ml.confiabilidade_componentes import (
    HOURS_PER_YEAR,
    INVERTER_RATE_PER_HOUR,
    SCENARIOS,
    component_curves,
    cumulative_failure,
    failure_density,
    hazard_rate,
    methodology,
    reliability,
    scenario_table,
)
from src.ml.proveniencia import funcao_de_hash_para


ROOT = Path(RAIZ_PROJETO)


def _scenario(scenario_id: str):
    return next(item for item in SCENARIOS if item.scenario_id == scenario_id)


def test_rates_are_traceable_and_component_scenarios_are_explicit():
    assert _scenario("contator_ac_derived").lambda_per_hour == pytest.approx(
        INVERTER_RATE_PER_HOUR * 0.12
    )
    assert _scenario("igbt_derived").lambda_per_hour == pytest.approx(
        INVERTER_RATE_PER_HOUR * 0.06
    )
    assert _scenario("fusivel_ac_derived").lambda_per_hour == pytest.approx(
        INVERTER_RATE_PER_HOUR * 0.04
    )
    assert _scenario("fusivel_ac_direct").lambda_per_hour == pytest.approx(2.17e-6)
    assert {item.pdf_page for item in SCENARIOS} == {35}
    assert {item.printed_page for item in SCENARIOS} == {34}


def test_exponential_functions_are_dimensionally_consistent():
    time_hours = np.linspace(0, 10 * HOURS_PER_YEAR, 101)
    for scenario in SCENARIOS:
        rate = scenario.lambda_per_hour
        r = reliability(time_hours, rate)
        f_cumulative = cumulative_failure(time_hours, rate)
        density = failure_density(time_hours, rate)
        hazard = hazard_rate(time_hours, rate)
        np.testing.assert_allclose(r + f_cumulative, 1.0, rtol=1e-13)
        np.testing.assert_allclose(density, hazard * r, rtol=1e-13)
        np.testing.assert_allclose(hazard, rate, rtol=0, atol=0)


def test_curves_publish_hours_years_density_and_hazard():
    curves = component_curves(horizon_years=2, n_points=9)
    assert len(curves) == len(SCENARIOS) * 9
    assert curves["time_hours"].max() == pytest.approx(2 * HOURS_PER_YEAR)
    assert curves["time_years"].max() == pytest.approx(2)
    assert {
        "reliability",
        "cumulative_failure_probability",
        "failure_density_per_hour",
        "failure_density_per_year",
        "hazard_per_hour",
        "hazard_per_year",
    }.issubset(curves.columns)


def test_no_physical_weibull_is_fabricated():
    contract = methodology()
    assert contract["physical_weibull"] == {
        "status": "not_estimable_from_current_dataset",
        "beta": None,
        "eta": None,
        "reason": "Ausência de tempos de vida, exposição e censura por ativo no GPVS-Faults",
    }
    assert "beta" not in scenario_table().columns
    assert "eta" not in scenario_table().columns


def test_published_reliability_manifest_reconciles_all_outputs():
    manifest_path = ROOT / "resultados" / "manifestos" / "confiabilidade_componentes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 2
    assert manifest["evidence_level"] == "bibliographic_sensitivity"
    assert len(manifest["outputs"]) == 10
    assert set(manifest["outputs"]) == set(manifest["output_artifacts"])
    for relative_path, expected_hash in manifest["output_artifacts"].items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert funcao_de_hash_para(path)(path) == expected_hash, relative_path


def test_published_vector_and_raster_figures_are_valid_files():
    output = ROOT / "resultados" / "confiabilidade"
    stems = {
        "confiabilidade_probabilidade_falha",
        "densidade_taxa_falha",
        "taxas_componentes",
    }
    for stem in stems:
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        assert (output / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("bad_time", [-1.0, np.nan, np.inf])
def test_invalid_time_is_rejected(bad_time):
    with pytest.raises(ValueError):
        reliability(bad_time, 1e-6)
