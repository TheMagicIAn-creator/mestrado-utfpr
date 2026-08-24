from __future__ import annotations

import csv
from pathlib import Path

import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.webapp.chart_data import (
    e3_discrimination_series,
    reliability_curve_series,
)


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "resultados" / "comparacao"
RELIABILITY = ROOT / "resultados" / "confiabilidade"


def test_curvas_e3_compactas_preservam_areas_publicadas():
    path = COMPARISON / "e3_escores_referencia.csv"
    series = e3_discrimination_series(path, maximum_points=201)

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for model, payload in series["models"].items():
        block = [row for row in rows if row["model"] == model]
        labels = [int(row["y_true"]) for row in block]
        scores = [float(row["anomaly_index"]) for row in block]
        assert payload["pooled_auc_roc"] == pytest.approx(
            roc_auc_score(labels, scores), abs=1e-12
        )
        assert payload["pooled_average_precision"] == pytest.approx(
            average_precision_score(labels, scores), abs=1e-12
        )
        assert len(payload["roc"]) <= 201
        assert len(payload["precision_recall"]) <= 201
        assert payload["roc"][0] == [0.0, 0.0]
        assert payload["roc"][-1] == [1.0, 1.0]


def test_curvas_fisicas_sao_limitadas_sem_perder_extremos():
    scenarios = {
        "contator_ac_derived": "Contator AC",
        "igbt_derived": "IGBT",
        "fusivel_ac_derived": "Fusível derivado",
        "fusivel_ac_direct": "Fusível direto",
    }
    series = reliability_curve_series(
        RELIABILITY / "curvas.csv", scenarios, maximum_points=121
    )

    assert len(series) == 4
    assert all(len(item["points"]) == 121 for item in series)
    assert all(item["points"][0]["time_years"] == 0 for item in series)
    assert all(item["points"][-1]["time_years"] == 20 for item in series)
    assert all(
        point["hazard_per_year"] > 0
        for item in series
        for point in item["points"]
    )
