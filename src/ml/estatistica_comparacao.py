"""Estatística inferencial da comparação entre os dois detectores."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260815
METRIC_NAMES = (
    "recall",
    "f1",
    "precision",
    "auc_roc",
    "auc_pr",
    "false_positive_rate",
    "specificity",
    "sensitivity",
    "balanced_accuracy",
    "mcc",
)


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if int(total) <= 0:
        return float("nan"), float("nan")
    proportion = int(successes) / int(total)
    denominator = 1.0 + z * z / int(total)
    center = (proportion + z * z / (2.0 * int(total))) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / int(total)
            + z * z / (4.0 * int(total) ** 2)
        )
        / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float, int]:
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    if not len(finite):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(finite), size=(int(n_resamples), len(finite)))
    means = finite[indices].mean(axis=1)
    return (
        float(finite.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        int(len(finite)),
    )


def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    target = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if target.shape != values.shape or target.ndim != 1:
        raise ValueError("y_true e scores devem ser vetores alinhados")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("A avaliação binária exige as duas classes")
    if not np.isfinite(values).all() or not np.isfinite(float(threshold)):
        raise ValueError("Escores e limiar devem ser finitos")
    prediction = values > float(threshold)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=(0, 1)).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    predicted_positive = int(tp + fp)
    precision = float(tp / predicted_positive) if predicted_positive else float("nan")
    sensitivity_ci = wilson_interval(int(tp), int(tp + fn))
    specificity_ci = wilson_interval(int(tn), int(tn + fp))
    curve_precision, curve_recall, _ = precision_recall_curve(target, values)
    return {
        "auc_pr": float(auc(curve_recall, curve_precision)),
        "auc_roc": float(roc_auc_score(target, values)),
        "sensitivity": float(sensitivity),
        "sensitivity_ci95_low": sensitivity_ci[0],
        "sensitivity_ci95_high": sensitivity_ci[1],
        "specificity": float(specificity),
        "specificity_ci95_low": specificity_ci[0],
        "specificity_ci95_high": specificity_ci[1],
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "mcc": float(matthews_corrcoef(target, prediction)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "precision": precision,
        "precision_defined": bool(predicted_positive),
        "recall": float(sensitivity),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "METRIC_NAMES",
    "binary_metrics",
    "bootstrap_mean",
    "wilson_interval",
]
