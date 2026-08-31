"""Sensibilidade descritiva do top-k e do limiar saudável.

Esta análise não seleciona configuração. Cada limiar é derivado somente do
bloco saudável de calibração, e os ensaios de falha entram apenas depois que o
ponto operacional está congelado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.dados_gpvs import (
    FAULT_EXPERIMENTS,
    PreparedData,
    normalize_commissioning,
    role_blocks,
)
from src.ml.estatistica_comparacao import binary_metrics
from src.ml.modelos_autoencoder import (
    SCORE_TOP_K,
    dense_feature_squared_errors,
    lstm_feature_squared_errors,
    sequences_for_blocks,
    sequences_from_flow,
    top_k_scores_from_feature_errors,
)
from src.ml.treino_comparacao import (
    MODEL_IDS,
    MODEL_NAMES,
    REFERENCE_SEED,
    THRESHOLD_PERCENTILE,
    ModelRun,
    calibrate_threshold,
)


SENSITIVITY_TOP_K = (1, 3, 5, 8, 12, 24)
SENSITIVITY_PERCENTILES = (95.0, 97.5, 99.0, 99.5, 99.9)
PRIMARY_METRICS = ("recall", "f1", "precision")


def _healthy_feature_errors(
    model_id: str,
    run: ModelRun,
    prepared: PreparedData,
    role: str,
) -> np.ndarray:
    blocks = role_blocks(prepared.split, role)
    if model_id == "ae_denso":
        values = prepared.scaled_values[np.concatenate(blocks)]
        return dense_feature_squared_errors(run.model, values)
    sequences, _ = sequences_for_blocks(prepared.scaled_values, blocks)
    return lstm_feature_squared_errors(run.model, sequences)


def _fault_blocks(
    prepared: PreparedData,
    fault_features: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    blocks: dict[str, dict[str, np.ndarray]] = {}
    for experiment in FAULT_EXPERIMENTS:
        features = (
            fault_features[fault_features["experiment"].eq(experiment)]
            .sort_values("window_index")
            .reset_index(drop=True)
        )
        normalized, pre_test, post_test, _ = normalize_commissioning(
            features,
            prepared.baseline_normalization,
        )
        scaled = prepared.scaler.transform(normalized).astype(np.float32)
        blocks[experiment] = {
            "scaled": scaled,
            "evaluation_indices": np.r_[pre_test, post_test],
            "y_true": np.r_[
                np.zeros(len(pre_test), dtype=int),
                np.ones(len(post_test), dtype=int),
            ],
        }
    return blocks


def evaluate_score_threshold_sensitivity(
    prepared: PreparedData,
    runs: dict[str, list[ModelRun]],
    fault_features: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...] = SENSITIVITY_TOP_K,
    percentiles: tuple[float, ...] = SENSITIVITY_PERCENTILES,
) -> pd.DataFrame:
    """Avalia uma grade pré-fixada sem usar as falhas para calibrar ou escolher."""

    normalized_k = tuple(int(value) for value in top_k_values)
    normalized_percentiles = tuple(float(value) for value in percentiles)
    if len(set(normalized_k)) != len(normalized_k):
        raise ValueError("A grade top-k não pode conter duplicatas")
    if len(set(normalized_percentiles)) != len(normalized_percentiles):
        raise ValueError("A grade de percentis não pode conter duplicatas")
    blocks = _fault_blocks(prepared, fault_features)
    rows: list[dict] = []
    for model_id in MODEL_IDS:
        for run in runs[model_id]:
            calibration_errors = _healthy_feature_errors(
                model_id, run, prepared, "calibration"
            )
            healthy_test_errors = _healthy_feature_errors(
                model_id, run, prepared, "test"
            )
            fault_errors = {}
            for experiment, block in blocks.items():
                errors = (
                    dense_feature_squared_errors(run.model, block["scaled"])
                    if model_id == "ae_denso"
                    else lstm_feature_squared_errors(
                        run.model,
                        sequences_from_flow(block["scaled"]),
                    )
                )
                fault_errors[experiment] = errors[block["evaluation_indices"]]
            for top_k in normalized_k:
                calibration_scores = top_k_scores_from_feature_errors(
                    calibration_errors,
                    top_k=top_k,
                )
                healthy_test_scores = top_k_scores_from_feature_errors(
                    healthy_test_errors,
                    top_k=top_k,
                )
                fault_scores = {
                    experiment: top_k_scores_from_feature_errors(errors, top_k=top_k)
                    for experiment, errors in fault_errors.items()
                }
                for percentile in normalized_percentiles:
                    threshold = calibrate_threshold(calibration_scores, percentile)
                    per_experiment = [
                        binary_metrics(
                            blocks[experiment]["y_true"],
                            fault_scores[experiment],
                            threshold.value,
                        )
                        for experiment in FAULT_EXPERIMENTS
                    ]
                    row = {
                        "model": model_id,
                        "model_name": MODEL_NAMES[model_id],
                        "seed": run.seed,
                        "is_reference": run.seed == REFERENCE_SEED,
                        "score_top_k": top_k,
                        "threshold_requested_percentile": percentile,
                        **threshold.as_dict(),
                        "healthy_test_false_positive_rate": float(
                            np.mean(healthy_test_scores > threshold.value)
                        ),
                        "healthy_test_n": len(healthy_test_scores),
                        "fault_experiment_count": len(per_experiment),
                        "uses_fault_data_for_selection": False,
                        "is_canonical_configuration": (
                            top_k == SCORE_TOP_K
                            and percentile == THRESHOLD_PERCENTILE
                        ),
                    }
                    for metric in PRIMARY_METRICS:
                        values = np.asarray(
                            [metrics[metric] for metrics in per_experiment],
                            dtype=float,
                        )
                        finite = values[np.isfinite(values)]
                        row[f"macro_{metric}"] = (
                            float(finite.mean()) if len(finite) else float("nan")
                        )
                        row[f"macro_{metric}_n_valid"] = int(len(finite))
                    rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "PRIMARY_METRICS",
    "SENSITIVITY_PERCENTILES",
    "SENSITIVITY_TOP_K",
    "evaluate_score_threshold_sensitivity",
]
