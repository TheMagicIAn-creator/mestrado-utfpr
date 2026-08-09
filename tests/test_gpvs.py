from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.gpvs import (
    FEATURE_COLUMNS,
    extrair_features_gpvs,
    identificar_ensaio,
    inferir_taxa_amostragem,
    split_adaptativo,
    split_f0,
)


def _ensaio_sintetico(n: int = 12_000, fs: float = 10_000.0) -> pd.DataFrame:
    t = np.arange(n, dtype=float) / fs
    fase = 2 * np.pi * 50.0 * t
    return pd.DataFrame({
        "Time": t,
        "Ipv": 2.0 + 0.02 * np.sin(2 * np.pi * 1.0 * t),
        "Vpv": 100.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t),
        "Vdc": 145.0 + 0.2 * np.sin(2 * np.pi * 100.0 * t),
        "ia": np.sin(fase),
        "ib": np.sin(fase - 2 * np.pi / 3),
        "ic": np.sin(fase + 2 * np.pi / 3),
        "va": 150 * np.sin(fase),
        "vb": 150 * np.sin(fase - 2 * np.pi / 3),
        "vc": 150 * np.sin(fase + 2 * np.pi / 3),
        "Iabc": np.ones(n),
        "If": np.full(n, 50.0),
        "Vabc": np.ones(n),
        "Vf": np.full(n, 50.0),
    })


def test_identificar_ensaio():
    assert identificar_ensaio("F4M.csv") == (4, "M")
    assert identificar_ensaio("F0L") == (0, "L")
    with pytest.raises(ValueError):
        identificar_ensaio("bearing.csv")


def test_taxa_e_inferida_do_tempo():
    info = inferir_taxa_amostragem(np.arange(2_000) / 10_000)
    assert info["fs_hz"] == pytest.approx(10_000)
    assert info["sampling_period_us"] == pytest.approx(100.0)


def test_tempo_nao_monotono_e_rejeitado():
    tempo = np.arange(2_000) / 10_000
    tempo[1_000] = tempo[999]
    with pytest.raises(ValueError, match="estritamente crescente"):
        inferir_taxa_amostragem(tempo)


def test_features_usam_um_ciclo_sem_sobreposicao():
    features, meta = extrair_features_gpvs(_ensaio_sintetico(), "F1L")
    assert meta["window_samples"] == 200
    assert len(features) == 60
    assert list(features[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()
    assert set(features["fase"]) == {"pre_falha", "pos_falha"}
    assert (features["amostra_inicio"].diff().dropna() == 200).all()


def test_estimativas_pmu_nao_entram_nas_features():
    features, _ = extrair_features_gpvs(_ensaio_sintetico(), "F0M")
    assert all(not coluna.startswith(("Iabc", "If", "Vabc", "Vf")) for coluna in FEATURE_COLUMNS)
    assert features[FEATURE_COLUMNS].shape[1] == 24


def _assert_split_temporal(split):
    blocos = [split.treino, split.validacao, split.calibracao, split.teste]
    concatenado = np.concatenate(blocos)
    assert len(np.unique(concatenado)) == len(concatenado)
    assert all(np.all(np.diff(bloco) == 1) for bloco in blocos)
    assert split.treino[-1] < split.validacao[0] < split.calibracao[0] < split.teste[0]
    assert split.validacao[0] - split.treino[-1] >= 3


def test_splits_temporais_tem_purga_e_blocos_disjuntos():
    _assert_split_temporal(split_f0(700))
    features, _ = extrair_features_gpvs(_ensaio_sintetico(140_000), "F2M")
    split = split_adaptativo(features)
    _assert_split_temporal(split)
    assert features.iloc[split.teste]["fase"].eq("pre_falha").all()
