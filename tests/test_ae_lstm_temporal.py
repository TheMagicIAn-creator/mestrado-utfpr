"""AE-LSTM TEMPORAL (Ibrahim): a LSTM percorre o TEMPO, não o eixo das features.

Valida a MECÂNICA (sem dataset): forma das sequências, contexto correto e que
uma anomalia no último passo eleva o escore acima do da sequência limpa.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.ml.modelos_anomalia as ma


def test_sequencias_deslizantes_forma():
    Xn = np.arange(10 * 3, dtype=float).reshape(10, 3)   # 10 janelas, 3 features
    seq = ma.sequencias_deslizantes(Xn, L=4)
    assert seq.shape == (7, 4, 3)                          # 10-4+1 = 7
    # última sequência termina na última janela
    assert np.allclose(seq[-1, -1], Xn[-1])
    # cada sequência é contígua no tempo
    assert np.allclose(seq[0], Xn[0:4])


def test_sequencias_deslizantes_padding_quando_curto():
    Xn = np.ones((2, 3), dtype=float)
    seq = ma.sequencias_deslizantes(Xn, L=5)
    assert seq.shape == (1, 5, 3)                          # padding garante 1 seq


def test_sequencias_com_contexto_ultimo_passo_e_o_item():
    ctx = np.arange(6 * 2, dtype=float).reshape(6, 2)      # histórico normal
    itens = ctx + 100.0                                    # itens "marcados"
    seq = ma.sequencias_com_contexto(ctx, itens, L=3)
    assert seq.shape == (6, 3, 2)
    # último passo de cada sequência é o ITEM, não o contexto
    assert np.allclose(seq[:, -1, :], itens)
    # o passo anterior (i-1) é o contexto normal do tempo i-1
    assert np.allclose(seq[3, -2, :], ctx[2])
    # no início, padding com a primeira janela normal
    assert np.allclose(seq[0, 0, :], ctx[0])


def test_lstm_roda_sobre_o_tempo_nao_sobre_features():
    # Se a LSTM percorresse as features, a saída não dependeria da ORDEM
    # temporal. Aqui garantimos apenas que o modelo consome (B, L, F) e devolve
    # um score por sequência — a forma temporal correta.
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    seq_fit = rng.normal(0, 1, size=(40, 6, 4)).astype(np.float32)
    seq_eval = rng.normal(0, 1, size=(5, 6, 4)).astype(np.float32)
    score = ma._score_ae_lstm(seq_fit, seq_eval, epochs=5)
    assert score.shape == (5,)
    assert np.all(np.isfinite(score))


def test_anomalia_no_ultimo_passo_eleva_o_escore():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    # normal: sequências suaves em torno de 0
    seq_fit = rng.normal(0, 0.2, size=(80, 6, 4)).astype(np.float32)
    base = rng.normal(0, 0.2, size=(6, 4)).astype(np.float32)
    limpa = base.copy()[None]                              # (1, 6, 4)
    anom = base.copy()[None]
    anom[0, -1, :] += 5.0                                  # spike no ÚLTIMO passo
    scores = ma._score_ae_lstm(seq_fit, np.vstack([limpa, anom]), epochs=40)
    # a sequência com anomalia no passo atual deve pontuar mais alto
    assert scores[1] > scores[0]
