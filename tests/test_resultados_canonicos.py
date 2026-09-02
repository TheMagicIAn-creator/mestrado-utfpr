from __future__ import annotations

import json
from pathlib import Path

from src.ml import resultados


def test_default_summary_reconciles_comparison_and_physical_reliability():
    response = resultados.resumir_resultados("resuma os resultados", incluir_imagens=False)
    text = response["mensagem"]
    assert "Autoencoder Denso" in text
    assert "AE-LSTM" in text
    assert "14 ensaios experimentais" in text
    assert "SMD95" not in text
    assert "R(t)=exp(-λt)" in text
    assert "não são medições" in text


def test_e3_focus_returns_only_comparison_figures():
    response = resultados.resumir_resultados("mostre os gráficos ROC e AUC da E3")
    assert response["imagens"]
    assert all(("resultados", "comparacao") == Path(item["path"]).parts[-3:-1] for item in response["imagens"])
    assert all(item["inline"] for item in response["imagens"])
    assert all("e3_" in item["path"] for item in response["imagens"])


def test_reliability_focus_does_not_mix_detectability_figures():
    response = resultados.resumir_resultados("mostre a confiabilidade física e h(t)")
    assert len(response["imagens"]) == 5
    assert all(("resultados", "confiabilidade") == Path(item["path"]).parts[-3:-1] for item in response["imagens"])
    assert "a_det" not in response["mensagem"]


def test_published_json_contracts_are_strict_json():
    comparison = json.loads(resultados.COMPARISON_JSON.read_text(encoding="utf-8"))
    reliability = json.loads(resultados.RELIABILITY_JSON.read_text(encoding="utf-8"))
    assert comparison["schema_version"] == 2
    assert reliability["schema_version"] == 7
    assert set(comparison["models"]) == {"ae_denso", "ae_lstm"}
    assert reliability["evidence_scope"] == "bibliographic_reliability_only"
    fmeca = reliability["fmeca"]
    assert fmeca["status"] == "validated"
    assert fmeca["calculation_enabled"] is True
    assert fmeca["traceability_status"] == "pending_source_documentation"
    expected = {
        "igbt": (5, 6, 5, 150),
        "sensor_feedback_system": (5, 8, 7, 280),
        "inverter_control_system": (5, 6, 8, 240),
    }
    assert {
        item["component_id"]: (
            item["severity"],
            item["occurrence"],
            item["detectability"],
            item["npr"],
        )
        for item in fmeca["components"]
    } == expected
    assert all(
        item["npr"]
        == item["severity"] * item["occurrence"] * item["detectability"]
        for item in fmeca["components"]
    )
    serialized = json.dumps(reliability, ensure_ascii=False).lower()
    for revoked in ("pod_mon", "d_mon", "d_proj", "npr_proj"):
        assert revoked not in serialized
    assert reliability["distribution_models"]["exponential"]["status"].startswith(
        "published"
    )
    for model in ("weibull_2p", "normal", "lognormal"):
        assert reliability["distribution_models"][model]["status"].startswith(
            "blocked"
        )
