from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.assinaturas_fmeca import (
    SIGNATURES,
    inject_contactor,
    inject_fuse,
    inject_igbt,
)


def _window() -> pd.DataFrame:
    time = np.arange(200) / 10_000.0
    angle = 2.0 * np.pi * 50.0 * time
    return pd.DataFrame(
        {
            "ia": np.sin(angle),
            "ib": np.sin(angle - 2.0 * np.pi / 3.0),
            "ic": np.sin(angle + 2.0 * np.pi / 3.0),
            "va": 10.0 * np.sin(angle),
            "vb": 10.0 * np.sin(angle - 2.0 * np.pi / 3.0),
            "vc": 10.0 * np.sin(angle + 2.0 * np.pi / 3.0),
        }
    )


@pytest.mark.leve
def test_fmeca_contract_preserves_components_and_npr():
    assert [(item.component_id, item.npr) for item in SIGNATURES] == [
        ("contator_ac", 315),
        ("igbt", 90),
        ("fusivel_ac", 30),
    ]


@pytest.mark.leve
def test_zero_magnitude_preserves_all_three_signatures():
    window = _window()
    for result in (
        inject_contactor(window, 0.0, seed=10),
        inject_igbt(window, 0.0),
        inject_fuse(window, 0.0),
    ):
        pd.testing.assert_frame_equal(result, window)


@pytest.mark.leve
def test_contactor_realization_is_deterministic_for_shared_seed():
    first = inject_contactor(_window(), 0.5, seed=123)
    second = inject_contactor(_window(), 0.5, seed=123)
    np.testing.assert_array_equal(first["ia"], second["ia"])


@pytest.mark.leve
def test_fuse_changes_current_without_fabricating_voltage_drop():
    window = _window()
    result = inject_fuse(window, 1.0)
    np.testing.assert_allclose(result["ia"], window["ia"] * 0.88)
    for column in ("va", "vb", "vc"):
        np.testing.assert_array_equal(result[column], window[column])


@pytest.mark.leve
def test_igbt_adds_harmonics_to_each_current_phase():
    window = _window()
    result = inject_igbt(window, 0.7)
    for column in ("ia", "ib", "ic"):
        assert not np.array_equal(result[column].to_numpy(), window[column].to_numpy())

