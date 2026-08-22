"""Series compactas e deterministicas para os graficos interativos do ALIAdo."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sample_evenly(values: list, maximum: int) -> list:
    if maximum < 2:
        raise ValueError("maximum deve ser ao menos 2")
    if len(values) <= maximum:
        return values
    indices = {
        round(index * (len(values) - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return [values[index] for index in sorted(indices)]


def _binary_curve(y_true: list[int], scores: list[float]) -> dict:
    if len(y_true) != len(scores) or not y_true:
        raise ValueError("Rotulos e escores devem ter o mesmo tamanho nao nulo")
    if any(label not in {0, 1} for label in y_true):
        raise ValueError("A curva binaria aceita somente rotulos 0 e 1")

    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("A curva exige exemplos positivos e negativos")

    ordered = sorted(zip(scores, y_true, strict=True), reverse=True)
    roc = [[0.0, 0.0]]
    precision_recall = [[0.0, 1.0]]
    true_positives = 0
    false_positives = 0
    cursor = 0
    while cursor < len(ordered):
        score = ordered[cursor][0]
        group_true = 0
        group_false = 0
        while cursor < len(ordered) and ordered[cursor][0] == score:
            if ordered[cursor][1] == 1:
                group_true += 1
            else:
                group_false += 1
            cursor += 1
        true_positives += group_true
        false_positives += group_false
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        roc.append([false_positives / negatives, recall])
        precision_recall.append([recall, precision])

    auc_roc = sum(
        (current[0] - previous[0]) * (current[1] + previous[1]) / 2
        for previous, current in zip(roc, roc[1:], strict=False)
    )
    average_precision = sum(
        (current[0] - previous[0]) * current[1]
        for previous, current in zip(
            precision_recall, precision_recall[1:], strict=False
        )
    )
    return {
        "roc": roc,
        "precision_recall": precision_recall,
        "pooled_auc_roc": auc_roc,
        "pooled_average_precision": average_precision,
        "prevalence": positives / len(y_true),
        "n_windows": len(y_true),
    }


def e3_discrimination_series(path: Path, maximum_points: int = 201) -> dict:
    """Calcula curvas agregadas e limita o payload sem alterar suas areas."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows(path):
        grouped[row["model"]].append(row)

    models = {}
    prevalence_values = []
    for model, values in sorted(grouped.items()):
        curve = _binary_curve(
            [int(item["y_true"]) for item in values],
            [float(item["anomaly_index"]) for item in values],
        )
        prevalence_values.append(curve["prevalence"])
        models[model] = {
            **curve,
            "roc": _sample_evenly(curve["roc"], maximum_points),
            "precision_recall": _sample_evenly(
                curve["precision_recall"], maximum_points
            ),
            "macro_auc_roc": float(values[0]["auc_roc_macro"]),
            "macro_auc_pr": float(values[0]["auc_pr_macro"]),
        }
    return {
        "models": models,
        "prevalence": sum(prevalence_values) / len(prevalence_values),
        "maximum_points_per_curve": maximum_points,
        "aggregation_unit": "window_descriptive_only",
    }


def e2_detection_series(path: Path, model_names: dict[str, str]) -> list[dict]:
    return [
        {
            "model": row["model"],
            "model_name": model_names[row["model"]],
            "component": row["component"],
            "component_name": row["component_name"],
            "npr": int(row["npr"]),
            "magnitude": float(row["magnitude"]),
            "detection_probability": float(row["detection_probability"]),
            "ci95_low": float(row["ci95_low"]),
            "ci95_high": float(row["ci95_high"]),
            "n_trajectories": int(row["n_trajectories"]),
        }
        for row in _rows(path)
    ]


def e2_empirical_series(
    path: Path,
    model_names: dict[str, str],
    component_names: dict[str, str],
) -> list[dict]:
    return [
        {
            "model": row["model"],
            "model_name": model_names[row["model"]],
            "component": row["component"],
            "component_name": component_names[row["component"]],
            "magnitude": float(row["magnitude"]),
            "at_risk": int(row["at_risk"]),
            "events": int(row["events"]),
            "survival": float(row["survival"]),
            "cumulative_detection": float(row["cumulative_detection"]),
            "discrete_hazard": float(row["discrete_hazard"]),
        }
        for row in _rows(path)
    ]


def reliability_curve_series(
    path: Path,
    scenario_names: dict[str, str],
    maximum_points: int = 121,
) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows(path):
        grouped[row["scenario_id"]].append(row)

    output = []
    for scenario_id, values in sorted(grouped.items()):
        sampled = _sample_evenly(values, maximum_points)
        output.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_names[scenario_id],
                "component_id": values[0]["component_id"],
                "evidence_type": values[0]["evidence_type"],
                "points": [
                    {
                        "time_hours": float(row["time_hours"]),
                        "time_years": float(row["time_years"]),
                        "reliability": float(row["reliability"]),
                        "cumulative_failure_probability": float(
                            row["cumulative_failure_probability"]
                        ),
                        "failure_density_per_year": float(
                            row["failure_density_per_year"]
                        ),
                        "hazard_per_year": float(row["hazard_per_year"]),
                    }
                    for row in sampled
                ],
            }
        )
    return output


__all__ = [
    "e2_detection_series",
    "e2_empirical_series",
    "e3_discrimination_series",
    "reliability_curve_series",
]
