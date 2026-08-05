from __future__ import annotations

import numpy as np
import pandas as pd


def test_intervalo_wilson_nao_declara_certeza_com_amostra_pequena():
    from src.ml.estatistica import intervalo_wilson

    low, high = intervalo_wilson(40, 40)
    assert 0.90 < low < 0.95
    assert high == 1.0


def test_bootstrap_auc_limita_intervalos_ao_dominio_da_metrica():
    from src.ml.estatistica import bootstrap_auc_ci

    resultado = bootstrap_auc_ci(
        np.zeros(40), np.ones(40), n_boot=40, seed=7
    )
    assert resultado["auc_roc_ci_high"] == 1.0
    assert resultado["auc_pr_ci_high"] == 1.0
    assert all(
        0.0 <= resultado[chave] <= 1.0
        for chave in (
            "auc_roc_ci_low", "auc_roc_ci_high",
            "auc_pr_ci_low", "auc_pr_ci_high",
        )
    )


def test_holdout_temporal_retorna_janelas_sem_sobreposicao(tmp_path):
    from src.ml.dados_avaliacao import preparar_janelas_holdout
    from src.ml.features_ca import JANELA

    n_features = 100
    inicios = np.arange(n_features) * (JANELA // 2)
    features = pd.DataFrame({
        "janela_idx": np.arange(n_features),
        "amostra_inicio": inicios,
        "tempo_s": inicios / 10_000,
        "f": np.ones(n_features),
    })
    caminho = tmp_path / "features.parquet"
    features.to_parquet(caminho, index=False)
    bruto = pd.DataFrame({"sinal": np.arange(int(inicios[-1] + JANELA))})

    janelas, meta = preparar_janelas_holdout(
        bruto, arquivo_features=caminho, janela=JANELA
    )
    starts = meta["amostras_inicio"]
    assert all(b - a >= JANELA for a, b in zip(starts, starts[1:]))
    assert len(janelas) == meta["n_janelas_usadas"]
    assert min(meta["indices_features"]) >= 80


def test_estimador_f0_reduz_confusao_com_segundo_harmonico():
    from src.ml.features_ca import calcular_espectro, estimar_f0

    fs = 10_000
    n = 1024
    t = np.arange(n) / fs
    sinal = np.sin(2 * np.pi * 50 * t) + 1.4 * np.sin(2 * np.pi * 100 * t)
    freqs, amps = calcular_espectro(sinal, fs)
    f0 = estimar_f0(freqs, amps, f0_nominal=60, faixa_hz=40)
    assert abs(f0 - 50) < 5


def test_weibull_preserva_censura_e_retorna_intervalos():
    from src.ml.rul_weibull import ajustar_weibull

    rng = np.random.default_rng(7)
    tempos_reais = 12 * rng.weibull(2.2, size=120)
    horizonte = 14.0
    eventos = tempos_reais <= horizonte
    observados = np.minimum(tempos_reais, horizonte)

    ajuste = ajustar_weibull(observados, eventos, n_boot=30, seed=7)
    assert ajuste["fit_converged"]
    assert ajuste["n_censurados"] == int((~eventos).sum())
    assert ajuste["n_eventos"] == int(eventos.sum())
    assert ajuste["beta"] > 0
    assert ajuste["beta_ci95"][0] < ajuste["beta_ci95"][1]


def test_rul_restrita_km_permanece_disponivel_sem_ajuste_weibull():
    from src.ml.rul_weibull import ajustar_weibull, rul_restrita_km

    tempos = np.full(30, 120.0)
    eventos = np.zeros(30, dtype=bool)

    ajuste = ajustar_weibull(tempos, eventos, n_boot=0)
    assert not ajuste["fit_converged"]
    assert ajuste["rul_restrita_disponivel"]
    assert ajuste["rul_restrita_inicial"] == 120.0
    assert rul_restrita_km(30.0, tempos, eventos) == 90.0


def test_alta_censura_sinaliza_incerteza_sem_ocultar_rul_parametrica():
    from src.ml.rul_weibull import ajustar_weibull

    tempos = np.concatenate([np.linspace(5.0, 15.0, 12), np.full(48, 20.0)])
    eventos = np.concatenate([
        np.ones(12, dtype=bool), np.zeros(48, dtype=bool)
    ])

    ajuste = ajustar_weibull(tempos, eventos, n_boot=0)
    assert ajuste["fit_converged"]
    assert ajuste["rul_reportavel"]
    assert ajuste["rul_parametrica_alta_incerteza"]
    assert ajuste["rul_restrita_disponivel"]


def test_rul_declara_passos_sinteticos_sem_calibracao_fisica():
    from src.ml.rul_weibull import ajustar_weibull, metadados_tempo_rul

    ajuste = ajustar_weibull(
        np.linspace(5.0, 25.0, 30),
        np.ones(30, dtype=bool),
        n_boot=0,
    )
    tempo = metadados_tempo_rul()

    assert ajuste["ttf_unidade"] == "passo_sintetico_de_degradacao"
    assert ajuste["rul_unidade"] == "passo_sintetico_de_degradacao"
    assert ajuste["tempo_fisico_calibrado"] is False
    assert tempo["tempo_fisico_calibrado"] is False
    assert tempo["passo_tempo_fisico_horas"] is None
    assert tempo["janela_aquisicao_s"] == 0.1024
