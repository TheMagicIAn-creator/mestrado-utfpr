"""Avaliações E3 experimental e E2 sintética dos dois autoencoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.ml.assinaturas_fmeca import INJECTORS, SIGNATURES
from src.ml.dados_gpvs import (
    FAULT_EXPERIMENTS,
    FAULT_NAMES,
    FEATURE_COLUMNS,
    OPERATING_MODES,
    PreparedData,
    feature_vector,
    load_holdout_windows,
    normalize_commissioning,
    normalize_f0_vectors,
)
from src.ml.detectabilidade import (
    empirical_detection_functions,
    first_sustained_crossings,
    fit_weibull_diagnostic,
    probability_positions,
)
from src.ml.estatistica_comparacao import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    METRIC_NAMES,
    binary_metrics,
    bootstrap_mean,
    wilson_interval,
)
from src.ml.modelos_autoencoder import (
    SEQUENCE_LENGTH,
    score_dense,
    score_lstm,
    sequences_from_flow,
    sequences_with_current_values,
)
from src.ml.treino_comparacao import (
    MODEL_IDS,
    MODEL_NAMES,
    REFERENCE_SEED,
    ModelRun,
)


E2_MAGNITUDE_STEPS = 101
E2_PERSISTENCE_MAGNITUDE = 0.02


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


def _score_injected(
    model_id: str,
    run: ModelRun,
    prepared: PreparedData,
    healthy_scaled: np.ndarray,
    injected_features: np.ndarray,
    experiments: np.ndarray,
) -> np.ndarray:
    normalized = normalize_f0_vectors(
        injected_features, experiments, prepared.baseline_normalization
    )
    scaled = prepared.scaler.transform(normalized).astype(np.float32)
    if model_id == "ae_denso":
        return score_dense(run.model, scaled)
    sequences = sequences_with_current_values(
        healthy_scaled, scaled, experiments, SEQUENCE_LENGTH
    )
    return score_lstm(run.model, sequences)


def evaluate_e2(
    prepared: PreparedData,
    runs: dict[str, list[ModelRun]],
    *,
    n_steps: int = E2_MAGNITUDE_STEPS,
) -> dict[str, pd.DataFrame | dict]:
    if int(n_steps) < 3:
        raise ValueError("A grade E2 exige ao menos três magnitudes")
    reference_runs = {
        model_id: next(run for run in runs[model_id] if run.seed == REFERENCE_SEED)
        for model_id in MODEL_IDS
    }
    windows, holdout_metadata = load_holdout_windows(prepared)
    experiments = np.asarray([str(window.attrs["experiment"]) for window in windows])
    test_indices = np.asarray(prepared.split["test"], dtype=int)
    expected = prepared.features.iloc[test_indices]["experiment"].astype(str).to_numpy()
    if len(windows) != len(test_indices) or not np.array_equal(experiments, expected):
        raise ValueError("Holdout de sinal e de features não estão alinhados")
    healthy_scaled = prepared.scaled_values[test_indices]
    magnitudes = np.linspace(0.0, 1.0, int(n_steps))
    grid_step = float(magnitudes[1] - magnitudes[0])
    curve_rows: list[dict] = []
    crossing_rows: list[dict] = []
    empirical_rows: list[dict] = []
    fit_rows: list[dict] = []
    probability_rows: list[dict] = []

    for signature in SIGNATURES:
        injector = INJECTORS[signature.component_id]
        detection_matrices = {
            model_id: np.zeros((len(windows), len(magnitudes)), dtype=bool)
            for model_id in MODEL_IDS
        }
        for magnitude_index, magnitude in enumerate(magnitudes):
            vectors = []
            for trajectory_id, window in enumerate(windows):
                if signature.component_id == "contator_ac":
                    injected = injector(
                        window, float(magnitude), seed=20_000 + trajectory_id
                    )
                else:
                    injected = injector(window, float(magnitude))
                injected.attrs.update(window.attrs)
                vectors.append(feature_vector(injected))
            feature_matrix = np.asarray(vectors, dtype=np.float32)
            for model_id, run in reference_runs.items():
                scores = _score_injected(
                    model_id,
                    run,
                    prepared,
                    healthy_scaled,
                    feature_matrix,
                    experiments,
                )
                detections = scores > run.threshold
                detection_matrices[model_id][:, magnitude_index] = detections
                low, high = wilson_interval(int(detections.sum()), len(detections))
                curve_rows.append(
                    {
                        "model": model_id,
                        "component": signature.component_id,
                        "component_name": signature.component_name,
                        "npr": signature.npr,
                        "magnitude": float(magnitude),
                        "detections": int(detections.sum()),
                        "n_trajectories": len(detections),
                        "detection_probability": float(detections.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

        for model_id, detections in detection_matrices.items():
            first, observed = first_sustained_crossings(
                detections,
                magnitudes,
                persistence_width=E2_PERSISTENCE_MAGNITUDE,
            )
            for trajectory_id, (value, event, experiment) in enumerate(
                zip(first, observed, experiments, strict=True)
            ):
                crossing_rows.append(
                    {
                        "model": model_id,
                        "component": signature.component_id,
                        "trajectory_id": trajectory_id,
                        "experiment": experiment,
                        "a_det": float(value),
                        "event_observed": bool(event),
                    }
                )
            empirical = empirical_detection_functions(first, observed, magnitudes)
            empirical.insert(0, "component", signature.component_id)
            empirical.insert(0, "model", model_id)
            empirical_rows.extend(empirical.to_dict(orient="records"))
            fit = fit_weibull_diagnostic(
                first,
                observed,
                grid_step=grid_step,
                n_bootstrap_ci=100,
                n_bootstrap_gof=100,
                seed=REFERENCE_SEED + sum(map(ord, model_id + signature.component_id)),
            )
            fit_rows.append(
                {
                    "model": model_id,
                    "component": signature.component_id,
                    "fit_converged": fit["fit_converged"],
                    "beta": fit["beta"],
                    "eta": fit["eta"],
                    "beta_ci95_low": fit["beta_ci95"][0],
                    "beta_ci95_high": fit["beta_ci95"][1],
                    "eta_ci95_low": fit["eta_ci95"][0],
                    "eta_ci95_high": fit["eta_ci95"][1],
                    "probability_plot_r2": fit["probability_plot"]["r2"],
                    "adherence_p_value": fit["goodness_of_fit"]["p_value"],
                    "adherence_alpha": fit["goodness_of_fit_alpha"],
                    "parametric_recommended": fit["parametric_recommended"],
                    "rejection_reasons": ";".join(fit["rejection_reasons"]),
                    "n_events": fit["n_events"],
                    "n_trajectories": fit["n_trajectories"],
                    "indetectable_at_max_pct": fit["censoring_percent"],
                    "axis_is_time": False,
                }
            )
            x, probability, method = probability_positions(first, observed)
            valid = (x > 0) & (probability > 0) & (probability < 1)
            for value, probability_value in zip(x[valid], probability[valid], strict=True):
                probability_rows.append(
                    {
                        "model": model_id,
                        "component": signature.component_id,
                        "magnitude": float(value),
                        "empirical_cdf": float(probability_value),
                        "log_magnitude": float(np.log(value)),
                        "weibull_y": float(np.log(-np.log1p(-probability_value))),
                        "position_method": method,
                    }
                )

    curves = pd.DataFrame(curve_rows)
    crossings = pd.DataFrame(crossing_rows)
    empirical = pd.DataFrame(empirical_rows)
    fits = pd.DataFrame(fit_rows)
    points = pd.DataFrame(probability_rows)
    summary_rows = []
    npr_by_component = {item.component_id: item.npr for item in SIGNATURES}
    for (model_id, component), block in curves.groupby(["model", "component"]):
        eligible = block[block["ci95_low"] >= 0.95].sort_values("magnitude")
        crossing_block = crossings[
            crossings["model"].eq(model_id)
            & crossings["component"].eq(component)
        ]
        fit = fits[
            fits["model"].eq(model_id) & fits["component"].eq(component)
        ].iloc[0]
        summary_rows.append(
            {
                "model": model_id,
                "component": component,
                "npr": npr_by_component[component],
                "smd95": None if eligible.empty else float(eligible.iloc[0]["magnitude"]),
                "smd95_status": "not_reached" if eligible.empty else "reached",
                "detection_at_max": float(
                    block.sort_values("magnitude").iloc[-1]["detection_probability"]
                ),
                "n_trajectories": len(crossing_block),
                "n_events": int(crossing_block["event_observed"].sum()),
                "indetectable_at_max_pct": float(
                    (~crossing_block["event_observed"]).mean() * 100.0
                ),
                "weibull_parametric_recommended": bool(
                    fit["parametric_recommended"]
                ),
            }
        )
    return {
        "curves": curves,
        "crossings": crossings,
        "empirical": empirical,
        "fits": fits,
        "probability_points": points,
        "summary": pd.DataFrame(summary_rows),
        "holdout_metadata": holdout_metadata,
    }


__all__ = [
    "E2_MAGNITUDE_STEPS",
    "E2_PERSISTENCE_MAGNITUDE",
    "evaluate_e2",
    "evaluate_e3",
]
