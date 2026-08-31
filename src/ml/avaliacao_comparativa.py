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
    SEQUENCE_LENGTH,
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


TEMPORAL_ANALYSES = (
    "current_full",
    "transition",
    "sustained",
    "post_fault_reset",
)


def _temporal_ablation_for_experiment(
    *,
    experiment: str,
    scaled: np.ndarray,
    pre_test: np.ndarray,
    post_test: np.ndarray,
    runs: dict[str, list[ModelRun]],
) -> list[dict]:
    """Avalia o possível ganho de transição sem alterar o resultado E3 principal."""

    transition_size = SEQUENCE_LENGTH - 1
    if len(post_test) <= transition_size:
        raise ValueError(
            f"{experiment} não possui pós-falha suficiente para a coorte sustentada"
        )
    transition = post_test[:transition_size]
    sustained = post_test[transition_size:]
    continuous_sequences = sequences_from_flow(scaled)
    reset_post_sequences = sequences_from_flow(scaled[post_test])
    analysis_indices = {
        "current_full": np.r_[pre_test, post_test],
        "transition": np.r_[pre_test, transition],
        "sustained": np.r_[pre_test, sustained],
    }
    fault_number = int(experiment[1])
    rows: list[dict] = []
    for model_id in MODEL_IDS:
        for run in runs[model_id]:
            continuous_scores = (
                score_dense(run.model, scaled, top_k=run.score_top_k)
                if model_id == "ae_denso"
                else score_lstm(
                    run.model,
                    continuous_sequences,
                    top_k=run.score_top_k,
                )
            )
            selected_by_analysis = {
                analysis: continuous_scores[indices]
                for analysis, indices in analysis_indices.items()
            }
            selected_by_analysis["post_fault_reset"] = np.r_[
                continuous_scores[pre_test],
                (
                    continuous_scores[post_test]
                    if model_id == "ae_denso"
                    else score_lstm(
                        run.model,
                        reset_post_sequences,
                        top_k=run.score_top_k,
                    )
                ),
            ]
            positive_counts = {
                "current_full": len(post_test),
                "transition": len(transition),
                "sustained": len(sustained),
                "post_fault_reset": len(post_test),
            }
            for analysis in TEMPORAL_ANALYSES:
                n_positive = positive_counts[analysis]
                y_true = np.r_[
                    np.zeros(len(pre_test), dtype=int),
                    np.ones(n_positive, dtype=int),
                ]
                metrics = binary_metrics(
                    y_true,
                    selected_by_analysis[analysis],
                    run.threshold,
                )
                rows.append(
                    {
                        "analysis": analysis,
                        "model": model_id,
                        "model_name": MODEL_NAMES[model_id],
                        "seed": run.seed,
                        "is_reference": run.seed == REFERENCE_SEED,
                        "experiment": experiment,
                        "fault": fault_number,
                        "fault_type": FAULT_NAMES[fault_number],
                        "mode": experiment[-1],
                        "mode_name": OPERATING_MODES[experiment[-1]],
                        "sequence_length": SEQUENCE_LENGTH,
                        "transition_post_windows": transition_size,
                        "n_pre_fault_test": len(pre_test),
                        "n_post_fault_evaluated": n_positive,
                        "post_fault_context_reset": analysis == "post_fault_reset",
                        "score_threshold": run.threshold,
                        "score_top_k": run.score_top_k,
                        **metrics,
                    }
                )
    return rows


def _summarize_temporal_ablation(
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    paired_rows = []
    for (seed, analysis), block in scenarios.groupby(["seed", "analysis"]):
        for metric in METRIC_NAMES:
            pivot = block.pivot(index="experiment", columns="model", values=metric)
            differences = pivot["ae_lstm"].to_numpy() - pivot["ae_denso"].to_numpy()
            estimate, low, high, n_valid = bootstrap_mean(
                differences,
                seed=(
                    BOOTSTRAP_SEED
                    + 70_000
                    + int(seed)
                    + sum(map(ord, analysis + metric))
                ),
            )
            paired_rows.append(
                {
                    "seed": int(seed),
                    "is_reference": int(seed) == REFERENCE_SEED,
                    "analysis": analysis,
                    "metric": metric,
                    "difference_lstm_minus_dense": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_paired_experiments": len(pivot),
                    "n_valid_paired_experiments": n_valid,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_unit": "paired_experiment",
                }
            )
    paired = pd.DataFrame(paired_rows)
    reference = paired[
        paired["is_reference"] & paired["analysis"].eq("sustained")
    ].set_index("metric")
    required = reference.reindex(["recall", "f1", "precision"])
    finite = np.isfinite(
        required[["difference_lstm_minus_dense", "ci95_low", "ci95_high"]]
        .to_numpy(dtype=float)
    ).all()
    if not finite:
        status = "inconclusive"
        reason = "Ao menos uma métrica principal não possui diferença pareada finita."
    elif (
        float(required.loc["recall", "ci95_low"]) > 0.0
        and float(required.loc["f1", "ci95_low"]) > 0.0
        and float(required.loc["precision", "difference_lstm_minus_dense"]) >= 0.0
    ):
        status = "survives"
        reason = (
            "Recall e F1 têm IC95% da diferença acima de zero na falha sustentada, "
            "sem diferença pontual negativa de Precision."
        )
    elif (
        float(required.loc["recall", "difference_lstm_minus_dense"]) < 0.0
        or float(required.loc["f1", "difference_lstm_minus_dense"]) < 0.0
    ):
        status = "does_not_survive"
        reason = (
            "Recall ou F1 apresenta diferença pontual negativa na falha sustentada."
        )
    else:
        status = "inconclusive"
        reason = (
            "Os intervalos da falha sustentada não sustentam superioridade inequívoca."
        )
    conclusion = {
        "status": status,
        "analysis": "sustained",
        "reference_seed": REFERENCE_SEED,
        "criterion": (
            "Recall e F1 com limite inferior do IC95% acima de zero e diferença "
            "pontual de Precision não negativa."
        ),
        "reason": reason,
    }
    return paired, conclusion


def evaluate_e3(
    prepared: PreparedData,
    runs: dict[str, list[ModelRun]],
    fault_features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    scenario_rows: list[dict] = []
    score_rows: list[dict] = []
    temporal_rows: list[dict] = []
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
        temporal_rows.extend(
            _temporal_ablation_for_experiment(
                experiment=experiment,
                scaled=scaled,
                pre_test=pre_test,
                post_test=post_test,
                runs=runs,
            )
        )
        for model_id in MODEL_IDS:
            for run in runs[model_id]:
                all_scores = (
                    score_dense(run.model, scaled, top_k=run.score_top_k)
                    if model_id == "ae_denso"
                    else score_lstm(run.model, sequences, top_k=run.score_top_k)
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
                        "score_top_k": run.score_top_k,
                        "threshold_requested_percentile": (
                            run.threshold_calibration.requested_percentile
                        ),
                        "threshold_effective_percentile": (
                            run.threshold_calibration.effective_percentile
                        ),
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
            estimate, low, high, n_valid = bootstrap_mean(
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
                    "n_valid_experiments": n_valid,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_unit": "experiment",
                }
            )
    macro = pd.DataFrame(macro_rows)

    stability_rows = []
    for (model_id, seed), block in scenarios.groupby(["model", "seed"]):
        for metric in METRIC_NAMES:
            finite = block[metric].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            stability_rows.append(
                {
                    "model": model_id,
                    "seed": int(seed),
                    "metric": metric,
                    "macro_mean": float(finite.mean()) if len(finite) else float("nan"),
                    "n_valid_experiments": int(len(finite)),
                }
            )
    stability = pd.DataFrame(stability_rows)

    paired_rows = []
    for metric in METRIC_NAMES:
        pivot = reference.pivot(index="experiment", columns="model", values=metric)
        differences = pivot["ae_denso"].to_numpy() - pivot["ae_lstm"].to_numpy()
        estimate, low, high, n_valid = bootstrap_mean(
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
                "n_valid_paired_experiments": n_valid,
                "bootstrap_unit": "paired_experiment",
            }
        )
    paired = pd.DataFrame(paired_rows)

    confusion_rows = []
    for model_id in MODEL_IDS:
        block = scores[scores["model"].eq(model_id)]
        reference_run = next(
            run for run in runs[model_id] if run.seed == REFERENCE_SEED
        )
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
                "tn_rate_actual_healthy": float(tn / max(tn + fp, 1)),
                "fp_rate_actual_healthy": float(fp / max(tn + fp, 1)),
                "fn_rate_actual_fault": float(fn / max(fn + tp, 1)),
                "tp_rate_actual_fault": float(tp / max(fn + tp, 1)),
                "unit": "window_descriptive_only",
                "normalization": "within_actual_class",
                "threshold_requested_percentile": (
                    reference_run.threshold_calibration.requested_percentile
                ),
                "threshold_effective_percentile": (
                    reference_run.threshold_calibration.effective_percentile
                ),
            }
        )
        for metric in ("auc_roc", "auc_pr"):
            value = macro[
                macro["model"].eq(model_id) & macro["metric"].eq(metric)
            ]["estimate"].iloc[0]
            scores.loc[scores["model"].eq(model_id), f"{metric}_macro"] = value
    temporal_ablation = pd.DataFrame(temporal_rows)
    temporal_paired, temporal_conclusion = _summarize_temporal_ablation(
        temporal_ablation
    )
    return {
        "scenarios": scenarios,
        "scores": scores,
        "macro": macro,
        "stability": stability,
        "paired": paired,
        "confusion": pd.DataFrame(confusion_rows),
        "temporal_ablation": temporal_ablation,
        "temporal_ablation_paired": temporal_paired,
        "temporal_ablation_conclusion": temporal_conclusion,
    }


__all__ = ["TEMPORAL_ANALYSES", "evaluate_e3"]
