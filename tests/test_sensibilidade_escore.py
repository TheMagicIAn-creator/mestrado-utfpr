from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.ml.sensibilidade_escore as sensitivity


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


def _errors(rows: int) -> np.ndarray:
    base = np.arange(1, 25, dtype=np.float32)
    return np.vstack([base + index for index in range(rows)])


@pytest.mark.leve
def test_sensitivity_uses_healthy_calibration_and_never_selects_on_faults(monkeypatch):
    monkeypatch.setattr(sensitivity, "FAULT_EXPERIMENTS", ("F1L",))
    monkeypatch.setattr(
        sensitivity,
        "normalize_commissioning",
        lambda _features, _normalization: (
            np.zeros((3, 24), dtype=np.float32),
            np.asarray([0]),
            np.asarray([1, 2]),
            {},
        ),
    )
    monkeypatch.setattr(
        sensitivity,
        "_healthy_feature_errors",
        lambda _model_id, _run, _prepared, role: _errors(
            4 if role == "calibration" else 3
        ),
    )
    monkeypatch.setattr(
        sensitivity,
        "dense_feature_squared_errors",
        lambda _model, values: _errors(len(values)),
    )
    monkeypatch.setattr(
        sensitivity,
        "lstm_feature_squared_errors",
        lambda _model, values: _errors(len(values)),
    )
    prepared = SimpleNamespace(
        scaler=_IdentityScaler(),
        baseline_normalization={},
    )
    runs = {
        model_id: [SimpleNamespace(model=object(), seed=42)]
        for model_id in ("ae_denso", "ae_lstm")
    }
    faults = pd.DataFrame(
        {"experiment": ["F1L"] * 3, "window_index": [0, 1, 2]}
    )

    result = sensitivity.evaluate_score_threshold_sensitivity(
        prepared,
        runs,
        faults,
        top_k_values=(1, 5),
        percentiles=(50.0, 99.9),
    )

    assert len(result) == 8
    assert result["uses_fault_data_for_selection"].eq(False).all()  # noqa: E712
    assert set(result["calibration_n"]) == {4}
    assert set(result["threshold_percentile_resolution"]) == {25.0}
    canonical = result[result["is_canonical_configuration"]]
    assert set(canonical["score_top_k"]) == {5}
    assert set(canonical["threshold_requested_percentile"]) == {99.9}
