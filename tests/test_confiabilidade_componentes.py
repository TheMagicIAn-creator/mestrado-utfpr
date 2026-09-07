from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.config import RAIZ_PROJETO
from src.ml.confiabilidade_componentes import (
    FMECA_COMPONENTS,
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
    distribution_model_contracts,
)
from src.ml.graficos_confiabilidade import generate_all
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
    # Cenários históricos do TCC citam página do PDF; os bibliográficos externos
    # citam fonte própria e não têm página no PDF.
    tcc = [item for item in SCENARIOS if item.source_pdf is not None]
    external = [item for item in SCENARIOS if item.source_citation is not None]
    assert {item.pdf_page for item in tcc} == {35}
    assert {item.printed_page for item in tcc} == {34}
    assert all(item.pdf_page is None for item in external)
    assert len(external) == 7
    # λ externos com faixa entram como par (lower/upper), nunca ponto fabricado.
    assert _scenario("igbt_field_lower").lambda_per_hour == pytest.approx(6.85e-7)
    assert _scenario("igbt_field_upper").lambda_per_hour == pytest.approx(2.74e-6)
    assert _scenario("contactors_norm").lambda_per_hour == pytest.approx(1.0e-7)
    assert {item.bound for item in external} == {"lower", "upper", "point"}


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
    assert contract["schema_version"] == 8
    assert contract["fmeca"]["formula"] == "NPR = S * O * D"
    assert contract["physical_weibull"] == {
        "status": "blocked_no_traceable_igbt_parameters_in_current_corpus",
        "beta": None,
        "eta": None,
        "reason": (
            "O corpus não fornece beta/eta de IGBT nem tempos individuais de "
            "falha, exposição e censura por ativo."
        ),
    }
    assert contract["evidence_scope"] == "bibliographic_reliability_only"
    assert "experimental_dataset" not in contract
    assert "beta" not in scenario_table().columns
    assert "eta" not in scenario_table().columns


def test_current_fmeca_scope_has_six_items_with_mixed_provenance():
    contract = methodology()["fmeca"]

    assert contract["status"] == "validated"
    assert contract["calculation_enabled"] is True
    assert contract["scope_item_count"] == 6
    assert {item.component_id for item in FMECA_COMPONENTS} == {
        "igbt",
        "sensor_feedback_system",
        "inverter_control_system",
        "pcb",
        "ac_dc_contactors",
        "cooling_fans",
    }
    # Só os itens com ensaio no GPVS entram na injeção E2.
    assert contract["injection_covered_items"] == [
        "igbt",
        "sensor_feedback_system",
        "inverter_control_system",
    ]
    # Valores bibliográficos de Cristaldi (2017); o sensor mantém o do pesquisador.
    expected = {
        "igbt": (3, 3, 7, 63),
        "sensor_feedback_system": (5, 8, 7, 280),
        "inverter_control_system": (6, 7, 3, 126),
        "pcb": (7, 4, 6, 168),
        "ac_dc_contactors": (6, 5, 5, 150),
        "cooling_fans": (4, 3, 4, 48),
    }
    expected_provenance = {
        "igbt": "bibliographic_cristaldi_2017",
        "sensor_feedback_system": "researcher_defined",
        "inverter_control_system": "bibliographic_cristaldi_2017",
        "pcb": "bibliographic_cristaldi_2017",
        "ac_dc_contactors": "bibliographic_cristaldi_2017",
        "cooling_fans": "bibliographic_cristaldi_2017",
    }
    for item in contract["components"]:
        cid = item["component_id"]
        scores = (
            item["severity"],
            item["occurrence"],
            item["detectability"],
            item["npr"],
        )
        assert item["status"] == "validated"
        assert item["calculation_enabled"] is True
        assert scores == expected[cid]
        assert item["provenance"] == expected_provenance[cid]
        assert item["npr"] == (
            item["severity"] * item["occurrence"] * item["detectability"]
        )
    # Itens sem contrapartida no GPVS não têm ensaio nativo.
    non_native = {"pcb", "ac_dc_contactors", "cooling_fans"}
    for item in contract["components"]:
        if item["component_id"] in non_native:
            assert item["has_native_experiment"] is False
            assert item["native_experiments"] == []


def test_lambda_not_found_is_recorded_not_estimated():
    contract = methodology()
    absent = {item["component_id"] for item in contract["lambda_not_found"]}
    assert absent == {"pcb", "inverter_control_system"}
    # Nenhum cenário de λ para os itens sem taxa rastreável.
    scenario_components = {item["component_id"] for item in contract["scenarios"]}
    assert "pcb" not in scenario_components


def test_distribution_contract_lists_missing_parameters_without_curves():
    contracts = distribution_model_contracts()

    assert contracts["exponential"]["status"] == "published_bibliographic_sensitivity"
    assert contracts["exponential"]["outputs"] == ["R(t)", "F(t)", "f(t)", "h(t)"]
    assert contracts["weibull_2p"]["status"] == (
        "blocked_no_traceable_igbt_parameters_in_current_corpus"
    )
    audit = contracts["weibull_2p"]["corpus_audit"]
    assert audit["igbt_chunks_reviewed"] == 22
    assert audit["weibull_chunks_found"] == 74
    assert contracts["weibull_2p"]["outputs"] == []
    assert all(
        value is None
        for value in contracts["weibull_2p"]["parameters"].values()
    )
    for model in ("normal", "lognormal", "lifetime_histogram"):
        assert contracts[model]["status"].startswith("blocked_missing_lifetime")
        assert contracts[model]["outputs"] == []
        assert all(value is None for value in contracts[model]["parameters"].values())


def test_published_reliability_manifest_reconciles_all_outputs():
    manifest_path = ROOT / "resultados" / "manifestos" / "confiabilidade_componentes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 2
    assert manifest["evidence_level"] == "bibliographic_sensitivity"
    assert len(manifest["outputs"]) == 14
    assert set(manifest["outputs"]) == set(manifest["output_artifacts"])
    for relative_path, expected_hash in manifest["output_artifacts"].items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert funcao_de_hash_para(path)(path) == expected_hash, relative_path


def test_published_vector_and_raster_figures_are_valid_files():
    output = ROOT / "resultados" / "confiabilidade"
    stems = {
        "curva_confiabilidade",
        "curva_probabilidade_falha",
        "curva_densidade_falha",
        "curva_taxa_falha",
        "taxas_componentes",
    }
    for stem in stems:
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        assert (output / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_all_reliability_figures_are_generated_from_canonical_tables(tmp_path):
    paths = generate_all(
        tmp_path,
        curves=component_curves(horizon_years=2, n_points=9),
        scenarios=scenario_table(),
    )

    assert len(paths) == 10
    assert {path.suffix for path in paths} == {".pdf", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


@pytest.mark.parametrize("bad_time", [-1.0, np.nan, np.inf])
def test_invalid_time_is_rejected(bad_time):
    with pytest.raises(ValueError):
        reliability(bad_time, 1e-6)
