"""Contrato mínimo com PyTorch real para os dois modelos canônicos."""

from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.ml.modelos_autoencoder import (  # noqa: E402
    LATENT_DIM,
    SEQUENCE_LENGTH,
    AutoencoderDenso,
    AutoencoderLSTM,
    parameter_count,
    score_dense,
    score_lstm,
)


@pytest.mark.integracao
def test_dense_and_lstm_forward_encode_and_score():
    dense = AutoencoderDenso(24).eval()
    lstm = AutoencoderLSTM(24).eval()
    dense_values = np.zeros((6, 24), dtype=np.float32)
    sequences = np.zeros((6, SEQUENCE_LENGTH, 24), dtype=np.float32)

    assert tuple(dense(torch.from_numpy(dense_values)).shape) == (6, 24)
    assert tuple(lstm(torch.from_numpy(sequences)).shape) == (6, SEQUENCE_LENGTH, 24)
    assert tuple(dense.encoder(torch.from_numpy(dense_values)).shape) == (6, LATENT_DIM)
    _, (hidden, _) = lstm.encoder(torch.from_numpy(sequences))
    assert tuple(lstm.to_latent(hidden[-1]).shape) == (6, LATENT_DIM)
    assert score_dense(dense, dense_values).shape == (6,)
    assert score_lstm(lstm, sequences).shape == (6,)


@pytest.mark.integracao
def test_models_have_distinct_nonzero_parameter_budgets():
    dense_count = parameter_count(AutoencoderDenso(24))
    lstm_count = parameter_count(AutoencoderLSTM(24))
    assert dense_count > 0
    assert lstm_count > dense_count


@pytest.mark.integracao
def test_checkpoint_round_trip_preserves_both_state_dicts(tmp_path):
    dense = AutoencoderDenso(24)
    lstm = AutoencoderLSTM(24)
    checkpoint = tmp_path / "models.pt"
    torch.save({"ae_denso": dense.state_dict(), "ae_lstm": lstm.state_dict()}, checkpoint)
    restored = torch.load(checkpoint, weights_only=True)
    assert set(restored) == {"ae_denso", "ae_lstm"}
