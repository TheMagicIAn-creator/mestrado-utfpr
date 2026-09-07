from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.modelos_autoencoder import (  # noqa: E402
    LATENT_DIM,
    AutoencoderDenso,
    AutoencoderLSTM,
    AutoencoderLSTMAtencao,
    parameter_count,
    score_dense,
    score_lstm,
    sequences_for_blocks,
    train_dense,
    train_lstm,
    train_lstm_atencao,
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


# ── AE-LSTM com decoder de atenção (arquitetura exploratória E0) ────────────

@pytest.mark.integracao
def test_atencao_reconstroi_a_sequencia_e_pontua():
    """Reconstrói (B, T, F) e pontua pelo último passo, como o AE-LSTM."""
    modelo = AutoencoderLSTMAtencao(24)
    modelo.eval()
    entrada = torch.randn(5, 8, 24)
    saida = modelo(entrada)
    assert tuple(saida.shape) == (5, 8, 24)
    assert bool(torch.isfinite(saida).all())
    escores = score_lstm(modelo, entrada.numpy())
    assert escores.shape == (5,)
    assert np.isfinite(escores).all()


@pytest.mark.integracao
def test_atencao_mantem_gargalo_e_acrescenta_parametros():
    """Mesmo latente do AE-LSTM (comparação honesta), porém com mais parâmetros."""
    base = AutoencoderLSTM(24)
    atencao = AutoencoderLSTMAtencao(24)
    assert atencao.latent_dim == LATENT_DIM == base.latent_dim
    assert parameter_count(atencao) > parameter_count(base)


@pytest.mark.integracao
def test_atencao_treina_e_reduz_a_perda():
    """O treino determinístico reduz a perda de validação em poucas épocas."""
    rng = np.random.default_rng(0)
    treino = rng.normal(size=(40, 8, 24)).astype("float32")
    validacao = rng.normal(size=(12, 8, 24)).astype("float32")
    modelo, historico = train_lstm_atencao(
        treino, validacao, seed=42, max_epochs=8, patience=8, batch_size=8
    )
    assert historico.best_epoch >= 1
    assert historico.best_validation_loss <= historico.validation_loss[0]
    assert np.isfinite(score_lstm(modelo, validacao)).all()
