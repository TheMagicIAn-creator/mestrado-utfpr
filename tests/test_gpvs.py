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
from src.ml.gpvs_principal import (
    ajustar_normalizacao_f0,
    carregar_normalizacao_baseline,
    extrair_janela,
    normalizar_comissionamento,
    normalizar_vetores_f0,
    salvar_normalizacao_baseline,
    split_features_gpvs,
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


def test_extrator_da_injecao_repete_exatamente_as_features_gpvs():
    ensaio = _ensaio_sintetico()
    features, meta = extrair_features_gpvs(ensaio, "F0L")
    janela = ensaio.iloc[:meta["window_samples"]]
    unitario = extrair_janela(janela)
    np.testing.assert_allclose(
        np.array([unitario[c] for c in FEATURE_COLUMNS]),
        features.iloc[0][FEATURE_COLUMNS].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


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


def test_split_principal_separa_papeis_em_cada_f0():
    df = pd.DataFrame({
        "ensaio": np.repeat(["F0L", "F0M"], 700),
    })
    split = split_features_gpvs(df)
    blocos = [split[k] for k in ("treino", "validacao", "calibracao", "teste")]
    unidos = np.concatenate(blocos)
    assert len(unidos) == len(np.unique(unidos))
    assert split["estrategia"] == "temporal_por_ensaio_F0L_F0M"
    for ensaio in ("F0L", "F0M"):
        papeis = split["por_ensaio"][ensaio]
        assert max(papeis["treino"]) < min(papeis["validacao"])
        assert max(papeis["validacao"]) < min(papeis["calibracao"])
        assert max(papeis["calibracao"]) < min(papeis["teste"])


def test_metricas_e3_usam_limiar_congelado_e_separam_pre_pos():
    from src.ml.validacao_gpvs_principal import _metricas_cenario

    features = pd.DataFrame({
        "fase": ["pre_falha"] * 5 + ["transicao"] + ["pos_falha"] * 5,
        "tempo_inicio_s": np.arange(11, dtype=float) * 0.02,
    })
    scores = np.array([0.1, 0.2, 0.3, 1.2, 0.4, 50.0, 1.1, 1.2, 1.3, 0.2, 1.4])
    resultado = _metricas_cenario(features, scores, 1.0, persistencia=3)

    assert resultado["false_positives_pre"] == 1
    assert resultado["true_positives_post"] == 4
    assert resultado["specificity"] == pytest.approx(0.8)
    assert resultado["sensitivity"] == pytest.approx(0.8)
    assert resultado["sustained_detection"] is True
    assert resultado["detection_delay_s"] == pytest.approx(0.0)


def test_macro_e3_reamostra_ensaios_e_nao_janelas():
    from src.ml.validacao_gpvs_principal import _resumir_macros

    cenarios = []
    for indice, modo in enumerate("LMLM"):
        cenarios.append({
            "mode": modo,
            "auc": 0.6 + indice * 0.1,
            "sensitivity": 0.5 + indice * 0.1,
            "specificity": 0.9 - indice * 0.1,
            "balanced_accuracy": 0.7,
        })
    macro = _resumir_macros(cenarios)["canonical_ae"]["all"]

    assert macro["auc"]["n_experiments"] == 4
    assert macro["auc"]["bootstrap_resamples"] == 20_000
    assert macro["auc"]["mean"] == pytest.approx(0.75)


def test_normalizacao_baseline_roundtrip_e_aplicacao_f0(tmp_path):
    rng = np.random.default_rng(9)
    n = 700
    df = pd.DataFrame({
        "ensaio": np.repeat(["F0L", "F0M"], n),
        **{
            coluna: np.r_[
                rng.normal(10 + i, 1 + i / 20, n),
                rng.normal(30 + i, 2 + i / 20, n),
            ]
            for i, coluna in enumerate(FEATURE_COLUMNS)
        },
    })
    split = split_features_gpvs(df)
    matriz, normalizacao = ajustar_normalizacao_f0(df, split)
    caminho = salvar_normalizacao_baseline(normalizacao, tmp_path)
    recarregada = carregar_normalizacao_baseline(tmp_path)

    assert caminho.is_file()
    assert np.isfinite(matriz).all()
    np.testing.assert_allclose(
        recarregada["iqr_floor"], normalizacao["iqr_floor"]
    )
    amostra = df.iloc[[0]][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    aplicada = normalizar_vetores_f0(amostra, ["F0L"], recarregada)
    np.testing.assert_allclose(aplicada[0], matriz[0], rtol=1e-6, atol=1e-6)


def test_comissionamento_nao_reutiliza_pre_teste_na_normalizacao():
    rng = np.random.default_rng(10)
    n_pre, n_post = 100, 80
    features = pd.DataFrame({
        "fase": ["pre_falha"] * n_pre + ["pos_falha"] * n_post,
        **{
            coluna: rng.normal(i + 1, 0.5, n_pre + n_post)
            for i, coluna in enumerate(FEATURE_COLUMNS)
        },
    })
    normalizacao = {
        "iqr_floor": np.full(len(FEATURE_COLUMNS), 0.01),
        "baseline_fraction": 0.5,
        "baseline_min_windows": 30,
    }
    matriz, pre_teste, post, meta = normalizar_comissionamento(
        features, normalizacao
    )

    assert np.array_equal(pre_teste, np.arange(50, 100))
    assert np.array_equal(post, np.arange(100, 180))
    assert meta == {
        "n_baseline": 50,
        "n_pre_test": 50,
        "n_post_test": 80,
        "baseline_fraction_of_pre": 0.5,
    }
    assert np.isfinite(matriz).all()
