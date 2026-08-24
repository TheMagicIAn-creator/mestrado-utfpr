"""Avaliação experimental E3 dos dois autoencoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.ml.dados_gpvs import (
    FAULT_EXPERIMENTS,
    FAULT_NAMES,
    OPERATING_MODES,
    PreparedData,
    normalize_commissioning,
)
from src.ml.estatistica_comparacao import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    METRIC_NAMES,
    binary_metrics,
    bootstrap_mean,
)
from src.ml.modelos_autoencoder import (
    score_dense,
    score_lstm,
    sequences_from_flow,
)
from src.ml.treino_comparacao import (
    MODEL_IDS,
    MODEL_NAMES,
    REFERENCE_SEED,
    ModelRun,
)


def evaluate_e3(
    prepared: PreparedData,
    runs: dict[str, list[ModelRun]],
    fault_features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    scenario_rows: list[dict] = []
    score_rows: list[dict] = []
    for experiment in FAULT_EXPERIMENTS:
        features = (
            fault_features[fault_features["experiment"].eq(experiment)]
            .sort_values("window_index")
            .reset_index(drop=True)
        )
        normalized, pre_test, post_test, baseline_meta = normalize_commissioning(
            features, prepared.baseline_normalization
        )
        scaled = prepared.scaler.transform(normalized).astype(np.float32)
        sequences = sequences_from_flow(scaled)
        evaluation_indices = np.r_[pre_test, post_test]
        y_true = np.r_[
            np.zeros(len(pre_test), dtype=int),
            np.ones(len(post_test), dtype=int),
        ]
        fault_number = int(experiment[1])
        for model_id in MODEL_IDS:
            for run in runs[model_id]:
                all_scores = (
                    score_dense(run.model, scaled)
                    if model_id == "ae_denso"
                    else score_lstm(run.model, sequences)
                )
                selected_scores = all_scores[evaluation_indices]
                metrics = binary_metrics(y_true, selected_scores, run.threshold)
                scenario_rows.append(
                    {
                        "model": model_id,
                        "model_name": MODEL_NAMES[model_id],
                        "seed": run.seed,
                        "is_reference": run.seed == REFERENCE_SEED,
                        "experiment": experiment,
                        "fault": fault_number,
                        "fault_type": FAULT_NAMES[fault_number],
                        "mode": experiment[-1],
                        "mode_name": OPERATING_MODES[experiment[-1]],
                        "n_commissioning": baseline_meta["n_baseline"],
                        "n_pre_fault_test": len(pre_test),
                        "n_post_fault_test": len(post_test),
                        "fault_boundary_method": "nominal_mid_record",
                        "score_threshold": run.threshold,
                        **metrics,
                    }
                )
                if run.seed == REFERENCE_SEED:
                    for index, target, score in zip(
                        evaluation_indices, y_true, selected_scores, strict=True
                    ):
                        row = features.iloc[int(index)]
                        score_rows.append(
                            {
                                "model": model_id,
                                "experiment": experiment,
                                "window_index": int(row["window_index"]),
                                "time_center_s": float(row["time_center_s"]),
                                "phase": "pre_fault" if target == 0 else "post_fault",
                                "y_true": int(target),
                                "score": float(score),
                                "score_threshold": run.threshold,
                                "anomaly_index": float(score / run.threshold),
                            }
                        )

    scenarios = pd.DataFrame(scenario_rows)
    scores = pd.DataFrame(score_rows)
    reference = scenarios[scenarios["is_reference"]]
    macro_rows = []
    for model_id in MODEL_IDS:
        block = reference[reference["model"].eq(model_id)]
        for metric in METRIC_NAMES:
            estimate, low, high = bootstrap_mean(
                block[metric].to_numpy(dtype=float),
                seed=BOOTSTRAP_SEED + sum(map(ord, model_id + metric)),
            )
            macro_rows.append(
                {
                    "model": model_id,
                    "model_name": MODEL_NAMES[model_id],
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_experiments": len(block),
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_unit": "experiment",
                }
            )
    macro = pd.DataFrame(macro_rows)

    stability_rows = []
    for (model_id, seed), block in scenarios.groupby(["model", "seed"]):
        for metric in METRIC_NAMES:
            stability_rows.append(
                {
                    "model": model_id,
                    "seed": int(seed),
                    "metric": metric,
                    "macro_mean": float(block[metric].mean()),
                }
            )
    stability = pd.DataFrame(stability_rows)

    paired_rows = []
    for metric in METRIC_NAMES:
        pivot = reference.pivot(index="experiment", columns="model", values=metric)
        differences = pivot["ae_denso"].to_numpy() - pivot["ae_lstm"].to_numpy()
        estimate, low, high = bootstrap_mean(
            differences,
            seed=BOOTSTRAP_SEED + 50_000 + sum(map(ord, metric)),
        )
        paired_rows.append(
            {
                "metric": metric,
                "difference_dense_minus_lstm": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "n_paired_experiments": len(pivot),
                "bootstrap_unit": "paired_experiment",
            }
        )
    paired = pd.DataFrame(paired_rows)

    confusion_rows = []
    for model_id in MODEL_IDS:
        block = scores[scores["model"].eq(model_id)]
        prediction = block["score"] > block["score_threshold"]
        tn, fp, fn, tp = confusion_matrix(
            block["y_true"], prediction, labels=(0, 1)
        ).ravel()
        confusion_rows.append(
            {
                "model": model_id,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "unit": "window_descriptive_only",
            }
        )
        for metric in ("auc_roc", "auc_pr"):
            value = macro[
                macro["model"].eq(model_id) & macro["metric"].eq(metric)
            ]["estimate"].iloc[0]
            scores.loc[scores["model"].eq(model_id), f"{metric}_macro"] = value
    return {
        "scenarios": scenarios,
        "scores": scores,
        "macro": macro,
        "stability": stability,
        "paired": paired,
        "confusion": pd.DataFrame(confusion_rows),
    }


__all__ = ["evaluate_e3"]
