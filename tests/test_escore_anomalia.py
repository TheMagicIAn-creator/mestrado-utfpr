"""Escore de anomalia (fonte única): MSE médio × localizado (top-k)."""

from __future__ import annotations

import numpy as np

import src.ml.escore_anomalia as ea


def _regua_saudavel(seed: int = 0, n: int = 200, f: int = 109):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 0.1, size=(n, f))
    return R, ea.ajustar_estatistica_residuo(R)


def test_escore_mse_e_a_media_dos_quadrados():
    r = np.array([[3.0, 4.0]])          # média de 9 e 16 = 12.5
    assert ea.escore_mse_medio(r)[0] == 12.5


def test_localizado_detecta_falha_que_o_mse_dilui():
    R_sau, stats = _regua_saudavel()
    lim_loc = np.percentile(ea.escore_localizado(R_sau, stats, k=5), 99)
    # falha LOCALIZADA: só 3 de 109 features estouram
    rng = np.random.default_rng(99)
    r_falha = rng.normal(0.0, 0.1, size=(1, 109))
    r_falha[0, [5, 7, 11]] += 2.0
    assert ea.escore_localizado(r_falha, stats, k=5)[0] > lim_loc


def test_localizado_nao_alarma_saudavel():
    R_sau, stats = _regua_saudavel()
    lim_loc = np.percentile(ea.escore_localizado(R_sau, stats, k=5), 99)
    rng = np.random.default_rng(7)
    r_ok = rng.normal(0.0, 0.1, size=(1, 109))
    assert ea.escore_localizado(r_ok, stats, k=5)[0] < lim_loc


def test_pontuar_sem_regua_cai_para_mse():
    R_sau, stats = _regua_saudavel()
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.1, size=(1, 109))
    # sem stats → MSE (comportamento histórico), nunca quebra
    assert ea.pontuar(r, None, metodo="localizado")[0] == ea.escore_mse_medio(r)[0]
    assert ea.pontuar(r, stats, metodo="mse")[0] == ea.escore_mse_medio(r)[0]


def test_k_e_limitado_ao_numero_de_features():
    R_sau, stats = _regua_saudavel(f=4)
    r = np.array([[0.1, 0.2, 0.3, 0.4]])
    # k acima de F não deve quebrar (é limitado internamente)
    assert np.isfinite(ea.escore_localizado(r, stats, k=99)[0])


def test_persistencia_da_regua_roundtrip(tmp_path):
    _, stats = _regua_saudavel()
    ea.salvar_estatistica(stats, tmp_path)
    lido = ea.carregar_estatistica(tmp_path)
    assert lido is not None
    assert np.allclose(lido["mu"], stats["mu"])
    assert np.allclose(lido["sigma"], stats["sigma"])
    # pasta sem o artefato → None (dispara fallback para MSE no pipeline)
    assert ea.carregar_estatistica(tmp_path, "inexistente.npz") is None
