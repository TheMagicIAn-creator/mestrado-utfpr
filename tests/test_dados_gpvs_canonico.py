from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.dados_gpvs import (
    ALL_EXPERIMENTS,
    DATASET_DIR,
    FEATURE_COLUMNS,
    SOURCE_COLUMNS,
    WINDOW_SAMPLES,
    dataset_files,
    extract_experiment_features,
    feature_vector,
    split_healthy_features,
)


def _synthetic_window() -> pd.DataFrame:
    time = np.arange(WINDOW_SAMPLES) / 10_000.0
    angle = 2.0 * np.pi * 50.0 * time
    return pd.DataFrame(
        {
            "Ipv": np.full(WINDOW_SAMPLES, 8.0),
            "Vpv": np.full(WINDOW_SAMPLES, 320.0),
            "Vdc": np.full(WINDOW_SAMPLES, 400.0),
            "ia": 5.0 * np.sin(angle),
            "ib": 5.0 * np.sin(angle - 2.0 * np.pi / 3.0),
            "ic": 5.0 * np.sin(angle + 2.0 * np.pi / 3.0),
            "va": 230.0 * np.sqrt(2.0) * np.sin(angle),
            "vb": 230.0 * np.sqrt(2.0) * np.sin(angle - 2.0 * np.pi / 3.0),
            "vc": 230.0 * np.sqrt(2.0) * np.sin(angle + 2.0 * np.pi / 3.0),
        }
    )


@pytest.mark.leve
def test_feature_vector_has_exactly_24_finite_electrical_features():
    vector = feature_vector(_synthetic_window())
    assert vector.shape == (24,)
    assert len(FEATURE_COLUMNS) == 24
    assert np.isfinite(vector).all()


@pytest.mark.leve
def test_temporal_roles_are_disjoint_and_purged_per_healthy_experiment():
    rows = []
    for experiment in ("F0L", "F0M"):
        rows.extend({"experiment": experiment} for _ in range(100))
    split = split_healthy_features(pd.DataFrame(rows))
    roles = ("train", "validation", "calibration", "test")
    combined = np.concatenate([split[role] for role in roles])
    assert len(combined) == len(np.unique(combined))
    assert split["nominal_fractions"] == {
        "train": 0.50,
        "validation": 0.15,
        "calibration": 0.15,
        "test": 0.20,
    }
    assert split["purge_windows"] == 2
    for experiment in ("F0L", "F0M"):
        blocks = split["per_experiment"][experiment]
        for left, right in zip(roles, roles[1:], strict=False):
            assert min(blocks[right]) - max(blocks[left]) >= 3


@pytest.mark.leve
def test_fault_boundary_is_explicitly_nominal_mid_record():
    base = _synthetic_window()
    frame = pd.concat([base] * 6, ignore_index=True)
    frame.insert(0, "Time", np.arange(len(frame)) / 10_000.0)
    for column in set(SOURCE_COLUMNS) - set(frame.columns):
        frame[column] = 0.0
    features, metadata = extract_experiment_features(frame, "F1L")
    assert metadata["fault_boundary_method"] == "nominal_mid_record"
    assert metadata["fault_sample_nominal"] == len(frame) // 2
    assert set(features["phase"]) == {"pre_fault", "post_fault"}


@pytest.mark.pesado
def test_local_gpvs_dataset_has_exactly_the_16_expected_trials():
    files = dataset_files(DATASET_DIR)
    assert tuple(files) == ALL_EXPERIMENTS
    assert all(path.stat().st_size > 1_000_000 for path in files.values())
