from __future__ import annotations

import numpy as np
import pytest

from src.ml.detectabilidade import (
    empirical_detection_functions,
    first_sustained_crossings,
    fit_weibull_diagnostic,
    probability_positions,
)


@pytest.mark.leve
def test_first_crossing_requires_sustained_detection_width():
    grid = np.linspace(0.0, 1.0, 11)
    detections = np.array(
        [
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    first, observed = first_sustained_crossings(
        detections, grid, persistence_width=0.2
    )
    np.testing.assert_allclose(first, [0.6, 1.0])
    np.testing.assert_array_equal(observed, [True, False])


@pytest.mark.leve
def test_empirical_functions_are_monotonic_on_magnitude_axis():
    grid = np.linspace(0.0, 1.0, 6)
    frame = empirical_detection_functions(
        np.array([0.2, 0.4, 1.0]),
        np.array([True, True, False]),
        grid,
    )
    assert np.all(np.diff(frame["survival"]) <= 1e-12)
    assert np.all(np.diff(frame["cumulative_detection"]) >= -1e-12)
    assert frame["discrete_hazard"].between(0, 1).all()


@pytest.mark.leve
def test_probability_positions_group_tied_grid_events():
    x, probability, method = probability_positions(
        np.array([0.2, 0.2, 0.5, 1.0]),
        np.array([True, True, True, False]),
    )
    np.testing.assert_allclose(x, [0.2, 0.5])
    assert len(probability) == 2
    assert "grouped" in method


@pytest.mark.leve
def test_weibull_diagnostic_rejects_insufficient_events_without_rul_semantics():
    result = fit_weibull_diagnostic(
        np.array([0.2, 0.4, 1.0, 1.0]),
        np.array([True, True, False, False]),
        grid_step=0.1,
        n_bootstrap_ci=0,
        n_bootstrap_gof=0,
    )
    assert result["fit_converged"] is False
    assert result["parametric_recommended"] is False
    assert result["axis"] == "a_det"
    assert result["axis_is_time"] is False
    assert not {"rul", "mttf", "b10"}.intersection(result)

