"""Análise de primeiro cruzamento no eixo de magnitude ``a_det``.

As funções deste módulo descrevem detectabilidade sintética E2. O eixo é
adimensional e não pode ser interpretado como tempo, RUL ou vida física.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import weibull_min


MIN_WEIBULL_EVENTS = 10
MIN_DISTINCT_LEVELS = 8
MAX_CENSORING_PERCENT = 50.0
GOODNESS_OF_FIT_ALPHA = 0.05
PROBABILITY_POSITION_METHOD = "modified_KM_NIST_grouped_ties"


def first_sustained_crossings(
    detections: np.ndarray,
    magnitudes: np.ndarray,
    *,
    persistence_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(detections, dtype=bool)
    grid = np.asarray(magnitudes, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(grid):
        raise ValueError("A matriz de detecção deve estar alinhada à grade a_det")
    if len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("A grade a_det deve ser estritamente crescente")
    if not np.isfinite(grid).all() or float(persistence_width) < 0:
        raise ValueError("Grade e largura de persistência devem ser válidas")
    step = float(np.min(np.diff(grid)))
    ratio = float(persistence_width) / step
    required = max(1, int(np.ceil(ratio - 1e-12)) + 1)
    first = np.full(matrix.shape[0], float(grid[-1]), dtype=float)
    observed = np.zeros(matrix.shape[0], dtype=bool)
    kernel = np.ones(required, dtype=int)
    for row_index, row in enumerate(matrix):
        sustained = np.convolve(row.astype(int), kernel, mode="valid") >= required
        matches = np.flatnonzero(sustained)
        if len(matches):
            crossing_index = int(matches[0] + required - 1)
            first[row_index] = float(grid[crossing_index])
            observed[row_index] = True
    return first, observed


def empirical_detection_functions(
    first_crossing: np.ndarray,
    events: np.ndarray,
    magnitudes: np.ndarray,
) -> pd.DataFrame:
    crossing = np.asarray(first_crossing, dtype=float)
    observed = np.asarray(events, dtype=bool)
    grid = np.asarray(magnitudes, dtype=float)
    if len(crossing) != len(observed):
        raise ValueError("Cruzamentos e eventos devem ter o mesmo comprimento")
    survival = 1.0
    rows = []
    for magnitude in grid:
        at_risk = int(np.sum(crossing >= magnitude - 1e-12))
        event_count = int(
            np.sum(observed & np.isclose(crossing, magnitude, atol=1e-12))
        )
        hazard = event_count / at_risk if at_risk else 0.0
        survival *= 1.0 - hazard
        rows.append(
            {
                "magnitude": float(magnitude),
                "at_risk": at_risk,
                "events": event_count,
                "survival": float(survival),
                "cumulative_detection": float(1.0 - survival),
                "discrete_hazard": float(hazard),
            }
        )
    return pd.DataFrame(rows)


def probability_positions(
    magnitudes: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    values = np.asarray(magnitudes, dtype=float)
    observed = np.asarray(events, dtype=bool)
    if len(values) != len(observed):
        raise ValueError("Magnitudes e eventos devem ter o mesmo comprimento")
    if not len(values):
        return np.asarray([]), np.asarray([]), PROBABILITY_POSITION_METHOD
    order = np.lexsort((~observed, values))
    ordered_values = values[order]
    ordered_events = observed[order]
    n = len(values)
    survival = (n + 0.7) / (n + 0.4)
    grouped: dict[float, list[float]] = {}
    for rank, (value, event) in enumerate(
        zip(ordered_values, ordered_events, strict=True), start=1
    ):
        if not event:
            continue
        survival *= (n - rank + 0.7) / (n - rank + 1.7)
        grouped.setdefault(float(value), []).append(float(1.0 - survival))
    x = np.asarray(list(grouped), dtype=float)
    probability = np.asarray(
        [float(np.mean(items)) for items in grouped.values()], dtype=float
    )
    return x, probability, PROBABILITY_POSITION_METHOD


def _fit_interval_censored_weibull(
    magnitudes: np.ndarray,
    events: np.ndarray,
    *,
    grid_step: float,
) -> tuple[float, float, bool]:
    values = np.asarray(magnitudes, dtype=float)
    observed = np.asarray(events, dtype=bool)
    if len(values) != len(observed):
        raise ValueError("Magnitudes e eventos devem ter o mesmo comprimento")
    if int(observed.sum()) < MIN_WEIBULL_EVENTS:
        return float("nan"), float("nan"), False
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Magnitudes devem ser finitas e não negativas")
    step = float(grid_step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("O passo da grade deve ser positivo e finito")
    clipped = np.clip(values, 1e-12, None)

    def negative_log_likelihood(log_parameters: np.ndarray) -> float:
        beta, eta = np.exp(log_parameters)
        log_likelihood = 0.0
        if observed.any():
            right = clipped[observed]
            left = np.maximum(0.0, right - step)
            z_right = np.power(right / eta, beta)
            z_left = np.power(left / eta, beta)
            difference = np.maximum(z_right - z_left, np.finfo(float).tiny)
            interval_mass_log = -z_left + np.log(-np.expm1(-difference))
            log_likelihood += float(np.sum(interval_mass_log))
        if (~observed).any():
            log_likelihood += float(
                np.sum(-np.power(clipped[~observed] / eta, beta))
            )
        return -log_likelihood if np.isfinite(log_likelihood) else 1e30

    initial_eta = max(float(np.median(clipped[observed])), step)
    result = minimize(
        negative_log_likelihood,
        x0=np.log((2.0, initial_eta)),
        method="L-BFGS-B",
        bounds=((np.log(0.05), np.log(50.0)), (np.log(1e-6), np.log(1e5))),
    )
    beta, eta = np.exp(result.x)
    converged = bool(result.success and np.isfinite(beta) and np.isfinite(eta))
    return float(beta), float(eta), converged


def _probability_plot_diagnostic(
    magnitudes: np.ndarray,
    events: np.ndarray,
    beta: float,
    eta: float,
) -> dict:
    x_values, probabilities, method = probability_positions(magnitudes, events)
    valid = (x_values > 0) & (probabilities > 0) & (probabilities < 1)
    x = np.log(x_values[valid])
    y = np.log(-np.log1p(-probabilities[valid]))
    if len(x) < 3:
        return {
            "n_points": int(len(x)),
            "r2": None,
            "rmse": None,
            "position_method": method,
        }
    fitted = float(beta) * (x - np.log(float(eta)))
    residuals = y - fitted
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / total if total > 0 else None
    return {
        "n_points": int(len(x)),
        "r2": float(r2) if r2 is not None else None,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "position_method": method,
    }


def _quantized_gof(
    magnitudes: np.ndarray,
    events: np.ndarray,
    beta: float,
    eta: float,
    *,
    grid_step: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    values = np.asarray(magnitudes, dtype=float)
    observed = np.asarray(events, dtype=bool)
    empirical_x, empirical_f, method = probability_positions(values, observed)
    if len(empirical_x) < 3 or int(n_bootstrap) <= 0:
        return {
            "method": "quantized_parametric_bootstrap_cdf",
            "statistic": None,
            "p_value": None,
            "requested": int(max(n_bootstrap, 0)),
            "valid": 0,
            "position_method": method,
        }
    fitted_f = weibull_min.cdf(empirical_x, beta, loc=0, scale=eta)
    statistic = float(np.mean((empirical_f - fitted_f) ** 2))
    horizon = float(np.max(values))
    rng = np.random.default_rng(int(seed))
    bootstrap_statistics: list[float] = []
    for _ in range(int(n_bootstrap)):
        continuous = eta * rng.weibull(beta, size=len(values))
        simulated_events = continuous <= horizon
        simulated = np.ceil(continuous / float(grid_step)) * float(grid_step)
        simulated = np.clip(simulated, float(grid_step), horizon)
        beta_b, eta_b, ok = _fit_interval_censored_weibull(
            simulated, simulated_events, grid_step=grid_step
        )
        if not ok:
            continue
        x_b, f_b, _ = probability_positions(simulated, simulated_events)
        if len(x_b) < 3:
            continue
        fitted_b = weibull_min.cdf(x_b, beta_b, loc=0, scale=eta_b)
        bootstrap_statistics.append(float(np.mean((f_b - fitted_b) ** 2)))
    p_value = None
    if bootstrap_statistics:
        p_value = float(
            (1 + np.sum(np.asarray(bootstrap_statistics) >= statistic))
            / (1 + len(bootstrap_statistics))
        )
    return {
        "method": "quantized_parametric_bootstrap_cdf",
        "statistic": statistic,
        "p_value": p_value,
        "requested": int(n_bootstrap),
        "valid": len(bootstrap_statistics),
        "position_method": method,
    }


def fit_weibull_diagnostic(
    magnitudes: np.ndarray,
    events: np.ndarray,
    *,
    grid_step: float,
    n_bootstrap_ci: int = 100,
    n_bootstrap_gof: int = 100,
    seed: int = 42,
) -> dict:
    """Ajuste 2P diagnóstico; nunca converte ``a_det`` em tempo."""

    values = np.asarray(magnitudes, dtype=float)
    observed = np.asarray(events, dtype=bool)
    beta, eta, converged = _fit_interval_censored_weibull(
        values, observed, grid_step=grid_step
    )
    diagnostic = (
        _probability_plot_diagnostic(values, observed, beta, eta)
        if converged
        else {"n_points": 0, "r2": None, "rmse": None, "position_method": None}
    )
    goodness = (
        _quantized_gof(
            values,
            observed,
            beta,
            eta,
            grid_step=grid_step,
            n_bootstrap=n_bootstrap_gof,
            seed=int(seed) + 10_000,
        )
        if converged
        else {
            "method": "quantized_parametric_bootstrap_cdf",
            "statistic": None,
            "p_value": None,
            "requested": int(n_bootstrap_gof),
            "valid": 0,
            "position_method": None,
        }
    )

    bootstrap: list[tuple[float, float]] = []
    if converged and int(n_bootstrap_ci) > 0:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(n_bootstrap_ci)):
            indices = rng.integers(0, len(values), size=len(values))
            beta_b, eta_b, ok = _fit_interval_censored_weibull(
                values[indices], observed[indices], grid_step=grid_step
            )
            if ok:
                bootstrap.append((beta_b, eta_b))
    if bootstrap:
        matrix = np.asarray(bootstrap)
        beta_ci = [
            float(np.percentile(matrix[:, 0], 2.5)),
            float(np.percentile(matrix[:, 0], 97.5)),
        ]
        eta_ci = [
            float(np.percentile(matrix[:, 1], 2.5)),
            float(np.percentile(matrix[:, 1], 97.5)),
        ]
    else:
        beta_ci = eta_ci = [None, None]

    censoring_percent = float((~observed).mean() * 100.0) if len(observed) else 100.0
    distinct_levels = int(np.unique(values[observed]).size)
    p_value = goodness["p_value"]
    accepted = bool(
        converged
        and censoring_percent <= MAX_CENSORING_PERCENT
        and distinct_levels >= MIN_DISTINCT_LEVELS
        and p_value is not None
        and p_value >= GOODNESS_OF_FIT_ALPHA
    )
    rejection_reasons = []
    if not converged:
        rejection_reasons.append("fit_not_converged_or_insufficient_events")
    if censoring_percent > MAX_CENSORING_PERCENT:
        rejection_reasons.append("censoring_above_50_percent")
    if distinct_levels < MIN_DISTINCT_LEVELS:
        rejection_reasons.append("fewer_than_8_distinct_event_levels")
    if p_value is None:
        rejection_reasons.append("formal_goodness_of_fit_unavailable")
    elif p_value < GOODNESS_OF_FIT_ALPHA:
        rejection_reasons.append("formal_goodness_of_fit_rejected")
    return {
        "fit_converged": converged,
        "beta": beta if converged else None,
        "eta": eta if converged else None,
        "beta_ci95": beta_ci,
        "eta_ci95": eta_ci,
        "probability_plot": diagnostic,
        "goodness_of_fit": goodness,
        "goodness_of_fit_alpha": GOODNESS_OF_FIT_ALPHA,
        "parametric_recommended": accepted,
        "rejection_reasons": rejection_reasons,
        "n_trajectories": int(len(values)),
        "n_events": int(observed.sum()),
        "n_censored": int((~observed).sum()),
        "censoring_percent": censoring_percent,
        "distinct_event_levels": distinct_levels,
        "grid_step": float(grid_step),
        "axis": "a_det",
        "axis_is_time": False,
        "fit_method": "2P Weibull interval-MLE with right censoring",
        "bootstrap_ci_valid": len(bootstrap),
    }


__all__ = [
    "GOODNESS_OF_FIT_ALPHA",
    "MAX_CENSORING_PERCENT",
    "MIN_DISTINCT_LEVELS",
    "MIN_WEIBULL_EVENTS",
    "empirical_detection_functions",
    "first_sustained_crossings",
    "fit_weibull_diagnostic",
    "probability_positions",
]
