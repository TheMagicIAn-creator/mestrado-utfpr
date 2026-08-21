"""Estatística inferencial da comparação entre os dois detectores."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260815
METRIC_NAMES = (
    "auc_pr",
    "auc_roc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "f1",
    "precision",
    "false_positive_rate",
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
) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if not len(data) or not np.isfinite(data).all():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(data), size=(int(n_resamples), len(data)))
    means = data[indices].mean(axis=1)
    return (
        float(data.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
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
    sensitivity_ci = wilson_interval(int(tp), int(tp + fn))
    specificity_ci = wilson_interval(int(tn), int(tn + fp))
    return {
        "auc_pr": float(average_precision_score(target, values)),
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
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
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
