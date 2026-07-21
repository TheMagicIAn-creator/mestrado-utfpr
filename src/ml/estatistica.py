"""Utilitarios estatisticos pequenos e reproduziveis do pipeline CA."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def intervalo_wilson(
    sucessos: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Intervalo de Wilson bilateral de 95% para uma proporcao binomial."""
    if n <= 0:
        return float("nan"), float("nan")
    p = sucessos / n
    denom = 1.0 + z * z / n
    centro = (p + z * z / (2.0 * n)) / denom
    margem = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
        / denom
    )
    return max(0.0, centro - margem), min(1.0, centro + margem)


def bootstrap_auc_ci(
    erros_neg: np.ndarray,
    erros_pos: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """IC bootstrap estratificado para AUC-ROC e Average Precision."""
    neg = np.asarray(erros_neg, dtype=float)
    pos = np.asarray(erros_pos, dtype=float)
    if len(neg) < 2 or len(pos) < 2:
        return {
            "auc_roc_ci_low": float("nan"),
            "auc_roc_ci_high": float("nan"),
            "auc_pr_ci_low": float("nan"),
            "auc_pr_ci_high": float("nan"),
            "n_boot": 0,
        }

    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    aps: list[float] = []
    for _ in range(n_boot):
        neg_b = rng.choice(neg, size=len(neg), replace=True)
        pos_b = rng.choice(pos, size=len(pos), replace=True)
        y_true = np.concatenate([np.zeros(len(neg_b)), np.ones(len(pos_b))])
        y_score = np.concatenate([neg_b, pos_b])
        aucs.append(float(roc_auc_score(y_true, y_score)))
        aps.append(float(average_precision_score(y_true, y_score)))

    def limite(valores, percentil: float) -> float:
        # Interpolação em ponto flutuante pode gerar 1.0000000000000002.
        # Consumidores dos artefatos devem poder confiar no domínio [0, 1].
        return float(np.clip(np.percentile(valores, percentil), 0.0, 1.0))

    return {
        "auc_roc_ci_low": limite(aucs, 2.5),
        "auc_roc_ci_high": limite(aucs, 97.5),
        "auc_pr_ci_low": limite(aps, 2.5),
        "auc_pr_ci_high": limite(aps, 97.5),
        "n_boot": int(n_boot),
    }
