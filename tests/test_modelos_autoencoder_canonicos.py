from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.modelos_autoencoder import (  # noqa: E402
    AutoencoderDenso,
    AutoencoderLSTM,
    score_dense,
    score_lstm,
    sequences_for_blocks,
    train_dense,
    train_lstm,
)


@pytest.mark.integracao
def test_forward_shapes_match_dense_and_lstm_inputs():
    dense = AutoencoderDenso(24)
    lstm = AutoencoderLSTM(24)
    assert tuple(dense(torch.zeros(5, 24)).shape) == (5, 24)
    assert tuple(lstm(torch.zeros(5, 8, 24)).shape) == (5, 8, 24)


@pytest.mark.integracao
def test_sequences_never_cross_role_or_experiment_boundaries():
    values = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    sequences, targets = sequences_for_blocks(
        values, [np.arange(0, 5), np.arange(10, 15)], length=3
    )
    np.testing.assert_array_equal(targets, np.r_[0:5, 10:15])
    np.testing.assert_array_equal(sequences[5, -1], values[10])
    assert not np.any(sequences[5] == values[4])


@pytest.mark.integracao
def test_real_torch_training_scoring_and_serialization(tmp_path):
    rng = np.random.default_rng(42)
    train = rng.normal(size=(40, 24)).astype(np.float32)
    validation = rng.normal(size=(16, 24)).astype(np.float32)
    dense, dense_history = train_dense(
        train,
        validation,
        seed=42,
        max_epochs=3,
        patience=2,
        batch_size=16,
    )
    train_sequences = np.stack([train[:8]] * 20).astype(np.float32)
    validation_sequences = np.stack([validation[:8]] * 8).astype(np.float32)
    lstm, lstm_history = train_lstm(
        train_sequences,
        validation_sequences,
        seed=42,
        max_epochs=3,
        patience=2,
        batch_size=8,
    )
    assert np.isfinite(score_dense(dense, validation)).all()
    assert np.isfinite(score_lstm(lstm, validation_sequences)).all()
    assert dense_history.best_epoch >= 1
    assert lstm_history.best_epoch >= 1
    checkpoint = tmp_path / "models.pt"
    torch.save({"dense": dense.state_dict(), "lstm": lstm.state_dict()}, checkpoint)
    restored = torch.load(checkpoint, weights_only=True)
    assert set(restored) == {"dense", "lstm"}

