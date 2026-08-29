from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from src.ml.estatistica_comparacao import binary_metrics, bootstrap_mean
from src.ml.treino_comparacao import calibrate_threshold, empirical_threshold


torch = pytest.importorskip("torch")

from src.ml.modelos_autoencoder import score_dense, score_lstm  # noqa: E402


class _ZeroReconstruction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, values):
        return torch.zeros_like(values) + self.anchor


@pytest.mark.integracao
def test_score_dense_is_mean_of_five_largest_feature_errors():
    values = np.arange(1, 25, dtype=np.float32)[None, :]
    expected = np.mean(np.square(np.arange(20, 25, dtype=float)))

    scores = score_dense(_ZeroReconstruction(), values, top_k=5)

    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(expected)


@pytest.mark.integracao
def test_score_lstm_uses_top_k_only_on_last_time_step():
    sequences = np.zeros((1, 8, 24), dtype=np.float32)
    sequences[:, 0, :] = 1_000.0
    sequences[:, -1, :] = np.arange(1, 25, dtype=np.float32)
    expected = np.mean(np.square(np.arange(20, 25, dtype=float)))

    scores = score_lstm(_ZeroReconstruction(), sequences, top_k=5)

    assert scores[0] == pytest.approx(expected)


@pytest.mark.integracao
@pytest.mark.parametrize("top_k", [0, 25, 1.5, True])
def test_score_rejects_invalid_top_k(top_k):
    values = np.zeros((2, 24), dtype=np.float32)
    model = _ZeroReconstruction()
    with pytest.raises(ValueError, match="top_k"):
        score_dense(model, values, top_k=top_k)


@pytest.mark.leve
def test_p999_threshold_records_order_statistic_and_empirical_resolution():
    scores = np.arange(210, dtype=float)

    calibration = calibrate_threshold(scores, 99.9)

    assert calibration.value == 209.0
    assert calibration.requested_percentile == 99.9
    assert calibration.selected_rank == 210
    assert calibration.selected_order_index == 209
    assert calibration.effective_percentile == 100.0
    assert calibration.percentile_resolution == pytest.approx(100.0 / 210.0)
    assert empirical_threshold(scores, 99.9) == calibration.value


@pytest.mark.leve
@pytest.mark.parametrize("percentile", [0.0, -1.0, 100.1])
def test_threshold_rejects_invalid_percentile(percentile):
    with pytest.raises(ValueError, match="percentil"):
        calibrate_threshold(np.arange(10, dtype=float), percentile)


@pytest.mark.leve
def test_precision_is_undefined_without_predicted_positives():
    metrics = binary_metrics(
        np.asarray([0, 1], dtype=int),
        np.asarray([0.1, 0.2], dtype=float),
        threshold=1.0,
    )

    assert np.isnan(metrics["precision"])
    assert metrics["precision_defined"] is False
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


@pytest.mark.leve
def test_auc_pr_is_trapezoidal_area_not_average_precision():
    target = np.asarray([1, 0, 1, 0], dtype=int)
    scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=float)
    metrics = binary_metrics(target, scores, threshold=0.75)
    precision, recall, _ = precision_recall_curve(target, scores)

    assert metrics["auc_pr"] == pytest.approx(auc(recall, precision))
    assert metrics["auc_pr"] != pytest.approx(average_precision_score(target, scores))


@pytest.mark.leve
def test_bootstrap_uses_only_finite_experiments_and_reports_count():
    estimate, low, high, n_valid = bootstrap_mean(
        np.asarray([np.nan, 0.5, 1.0]),
        seed=42,
        n_resamples=500,
    )

    assert estimate == pytest.approx(0.75)
    assert 0.5 <= low <= estimate <= high <= 1.0
    assert n_valid == 2
