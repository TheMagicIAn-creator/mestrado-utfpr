from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

import src.ml.avaliacao_comparativa as evaluation  # noqa: E402
import src.ml.comparacao_autoencoders as orchestrator  # noqa: E402
import src.ml.graficos_comparacao as charts  # noqa: E402


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


def _reference_run(model_id: str, *, threshold: float = 0.5):
    calibration = SimpleNamespace(
        value=threshold,
        requested_percentile=99.9,
        effective_percentile=100.0,
    )
    return SimpleNamespace(
        model_id=model_id,
        model=object(),
        seed=42,
        threshold=threshold,
        threshold_calibration=calibration,
        score_top_k=5,
    )


@pytest.mark.integracao
def test_evaluate_e3_aggregates_models_metrics_and_confusion(monkeypatch):
    monkeypatch.setattr(evaluation, "FAULT_EXPERIMENTS", ("F1L",))
    monkeypatch.setattr(
        evaluation,
        "normalize_commissioning",
        lambda _features, _baseline: (
            np.asarray([[float(index)] * 24 for index in range(10)]),
            np.asarray([0]),
            np.arange(1, 10),
            {"n_baseline": 1},
        ),
    )
    monkeypatch.setattr(evaluation, "sequences_from_flow", lambda values: values)
    monkeypatch.setattr(
        evaluation,
        "score_dense",
        lambda _model, values, *, top_k: np.linspace(0.1, 0.9, len(values)),
    )
    monkeypatch.setattr(
        evaluation,
        "score_lstm",
        lambda _model, values, *, top_k: np.linspace(0.1, 0.9, len(values)),
    )
    prepared = SimpleNamespace(
        scaler=_IdentityScaler(),
        baseline_normalization={},
    )
    runs = {
        model_id: [_reference_run(model_id)]
        for model_id in ("ae_denso", "ae_lstm")
    }
    faults = pd.DataFrame(
        {
            "experiment": ["F1L"] * 10,
            "window_index": range(10),
            "time_center_s": np.arange(10, dtype=float),
        }
    )

    result = evaluation.evaluate_e3(prepared, runs, faults)

    assert len(result["scenarios"]) == 2
    assert set(result["scores"]["model"]) == {"ae_denso", "ae_lstm"}
    assert result["confusion"][["tn", "fp", "fn", "tp"]].to_numpy().tolist() == [
        [1, 0, 4, 5],
        [1, 0, 4, 5],
    ]
    assert result["macro"]["n_valid_experiments"].min() == 1
    assert set(result["temporal_ablation"]["analysis"]) == set(
        evaluation.TEMPORAL_ANALYSES
    )
    sustained = result["temporal_ablation"]
    sustained = sustained[sustained["analysis"].eq("sustained")]
    assert set(sustained["n_post_fault_evaluated"]) == {2}


@pytest.mark.integracao
def test_temporal_ablation_resets_only_the_post_fault_lstm_context(monkeypatch):
    captured_sequences = []
    monkeypatch.setattr(
        evaluation,
        "score_dense",
        lambda _model, values, *, top_k: np.linspace(0.1, 0.9, len(values)),
    )

    def fake_score_lstm(_model, values, *, top_k):
        captured_sequences.append(np.asarray(values).copy())
        return np.linspace(0.1, 0.9, len(values))

    monkeypatch.setattr(evaluation, "score_lstm", fake_score_lstm)
    scaled = np.repeat(np.arange(13, dtype=float)[:, None], 24, axis=1)
    runs = {
        model_id: [_reference_run(model_id)]
        for model_id in ("ae_denso", "ae_lstm")
    }

    rows = evaluation._temporal_ablation_for_experiment(
        experiment="F1L",
        scaled=scaled,
        pre_test=np.asarray([2, 3]),
        post_test=np.arange(4, 13),
        runs=runs,
    )

    reset_sequences = next(values for values in captured_sequences if len(values) == 9)
    assert np.all(reset_sequences[0, :, 0] == 4.0)
    assert reset_sequences.min() >= 4.0
    sustained = [row for row in rows if row["analysis"] == "sustained"]
    assert {row["n_post_fault_evaluated"] for row in sustained} == {2}


def test_orchestrator_forwards_scientific_configuration(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        orchestrator,
        "load_or_extract_features",
        lambda *, force: ("healthy", "faults", {"dataset": "GPVS-Faults"}),
    )
    monkeypatch.setattr(orchestrator, "prepare_healthy_data", lambda data: "prepared")

    def fake_train(prepared, **kwargs):
        captured["training"] = (prepared, kwargs)
        return "runs"

    monkeypatch.setattr(orchestrator, "train_models", fake_train)
    monkeypatch.setattr(
        orchestrator,
        "evaluate_e3",
        lambda prepared, runs, faults: {
            "prepared": prepared,
            "runs": runs,
            "faults": faults,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "evaluate_score_threshold_sensitivity",
        lambda prepared, runs, faults: "sensitivity",
    )

    def fake_save(dataset_manifest, prepared, runs, e3, *, seeds):
        captured["publication"] = (dataset_manifest, prepared, runs, e3, seeds)
        return {
            "manifest": Path("manifest.json"),
            "outputs": [Path("result.json")],
            "payload": {"ok": True},
        }

    monkeypatch.setattr(orchestrator, "save_results", fake_save)

    result = orchestrator.run(
        force_features=True,
        seeds=(42,),
        threshold_percentile=99.0,
        score_top_k=3,
    )

    assert captured["training"] == (
        "prepared",
        {"seeds": (42,), "threshold_percentile": 99.0, "score_top_k": 3},
    )
    assert captured["publication"][-1] == (42,)
    assert captured["publication"][3]["score_threshold_sensitivity"] == "sensitivity"
    assert result["output_count"] == 1


def test_comparison_cli_accepts_threshold_and_top_k(monkeypatch, capsys):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "payload": {"omitted": True}}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comparacao_autoencoders",
            "--seeds",
            "42",
            "--threshold-percentile",
            "99.5",
            "--score-top-k",
            "4",
        ],
    )

    orchestrator.main()

    assert captured["seeds"] == (42,)
    assert captured["threshold_percentile"] == 99.5
    assert captured["score_top_k"] == 4
    assert "payload" not in capsys.readouterr().out


@pytest.mark.integracao
def test_generate_all_comparison_figures_from_tabular_sources(tmp_path):
    metric_names = [
        "recall",
        "f1",
        "precision",
        "auc_roc",
        "auc_pr",
        "false_positive_rate",
    ]
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "metric": metric,
                "estimate": 0.75,
                "ci95_low": 0.65,
                "ci95_high": 0.85,
            }
            for model in ("ae_denso", "ae_lstm")
            for metric in metric_names
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "model": model,
                "y_true": target,
                "anomaly_index": score,
                "auc_roc_macro": 0.8,
                "auc_pr_macro": 0.8,
            }
            for model in ("ae_denso", "ae_lstm")
            for target, score in ((0, 0.1), (0, 0.3), (1, 0.7), (1, 0.9))
        ]
    )
    confusion = pd.DataFrame(
        [
            {
                "model": model,
                "tn": 8,
                "fp": 2,
                "fn": 3,
                "tp": 7,
                "threshold_requested_percentile": 99.9,
            }
            for model in ("ae_denso", "ae_lstm")
        ]
    )
    scenarios = pd.DataFrame(
        [
            {
                "model": model,
                "experiment": f"F{fault}{mode}",
                "is_reference": True,
                "recall": 0.7,
                "f1": 0.65,
            }
            for fault in range(1, 8)
            for mode in "LM"
            for model in ("ae_denso", "ae_lstm")
        ]
    )
    temporal_paired = pd.DataFrame(
        [
            {
                "seed": seed,
                "is_reference": seed == 42,
                "analysis": analysis,
                "metric": metric,
                "difference_lstm_minus_dense": 0.05,
                "ci95_low": 0.01,
                "ci95_high": 0.09,
            }
            for seed in (13, 29, 42, 71, 101)
            for analysis in evaluation.TEMPORAL_ANALYSES
            for metric in ("recall", "f1", "precision")
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "model": model,
                "seed": 42,
                "is_reference": True,
                "score_top_k": top_k,
                "threshold_requested_percentile": percentile,
                "healthy_test_false_positive_rate": 0.02,
                "macro_recall": 0.7,
                "macro_f1": 0.65,
                "macro_precision": 0.8,
            }
            for model in ("ae_denso", "ae_lstm")
            for top_k in (1, 3, 5, 8, 12, 24)
            for percentile in (95.0, 97.5, 99.0, 99.5, 99.9)
        ]
    )

    paths = charts.generate_all(
        tmp_path,
        e3_summary=summary,
        e3_scores=scores,
        e3_confusion=confusion,
        e3_scenarios=scenarios,
        temporal_ablation_paired=temporal_paired,
        score_threshold_sensitivity=sensitivity,
    )

    assert len(paths) == 12
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
