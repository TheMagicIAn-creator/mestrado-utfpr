"""
Sprint 1 — integridade metodológica.

Garante que a divisão temporal com purga não vaza:
- sem overlap entre treino e validação;
- sem overlap entre validação e teste;
- ordenação temporal preservada (treino < val < teste);
- purga respeitada na fronteira;
- determinística (reprodutível).
"""

import numpy as np
import pytest

from src.ml.split_temporal import split_temporal_com_purga


def test_sem_overlap_treino_val_teste():
    sp = split_temporal_com_purga(457, 0.6, 0.2, 0.2, purge_janelas=2)
    tr, val, te = set(sp["treino"]), set(sp["val"]), set(sp["teste"])
    assert tr.isdisjoint(val)
    assert val.isdisjoint(te)
    assert tr.isdisjoint(te)


def test_ordenacao_temporal_preservada():
    sp = split_temporal_com_purga(457, 0.6, 0.2, 0.2, purge_janelas=2)
    assert sp["treino"].max() < sp["val"].min()
    assert sp["val"].max() < sp["teste"].min()
    # blocos contíguos e crescentes
    for chave in ("treino", "val", "teste"):
        bloco = sp[chave]
        assert np.all(np.diff(bloco) == 1)


def test_purga_respeitada_na_fronteira():
    purge = 3
    sp = split_temporal_com_purga(500, 0.6, 0.2, 0.2, purge_janelas=purge)
    # ao menos `purge` janelas descartadas entre treino e val
    gap_tr_val = sp["val"].min() - sp["treino"].max() - 1
    gap_val_te = sp["teste"].min() - sp["val"].max() - 1
    assert gap_tr_val >= purge
    assert gap_val_te >= purge


def test_reprodutivel():
    a = split_temporal_com_purga(300, 0.7, 0.15, 0.15, purge_janelas=2)
    b = split_temporal_com_purga(300, 0.7, 0.15, 0.15, purge_janelas=2)
    assert np.array_equal(a["treino"], b["treino"])
    assert np.array_equal(a["val"], b["val"])
    assert np.array_equal(a["teste"], b["teste"])


def test_ratios_invalidos_levantam():
    with pytest.raises(ValueError):
        split_temporal_com_purga(100, 0.6, 0.6, 0.2)  # soma != 1


def test_janelas_insuficientes_levantam():
    with pytest.raises(ValueError):
        split_temporal_com_purga(5, 0.6, 0.2, 0.2, purge_janelas=10)
