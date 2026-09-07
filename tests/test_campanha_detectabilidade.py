"""A orquestração E2 reusa o detector congelado sem treinar nem recalibrar.

Nada aqui precisa de torch, dataset bruto ou checkpoint: o scorer é injetado
como função, e as janelas são sintéticas com valores conhecidos. O que se fixa é
o CABEAMENTO — normalização espelhando a E3, sequência estacionária do AE-LSTM,
injetor por trajetória na interpolação e a âncora — não a matemática de `a_det`,
que é testada em `test_detectabilidade.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

import src.ml.campanha_detectabilidade as campanha
from src.ml.dados_gpvs import (
    FEATURE_COLUMNS,
    PRIMARY_COLUMNS,
    WINDOW_SAMPLES,
    feature_vector,
)
from src.ml.injecao_e2 import ESPECIFICACOES, GRADE_SEVERIDADE, POR_ID
from src.ml.modelos_autoencoder import SEQUENCE_LENGTH

pytestmark = pytest.mark.leve


def _janela(seed: int = 0, amplitude: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW_SAMPLES) / WINDOW_SAMPLES
    fases = {"a": 0.0, "b": -2 * np.pi / 3, "c": 2 * np.pi / 3}
    dados = {
        "Ipv": np.full(WINDOW_SAMPLES, 8.0) + 0.01 * rng.normal(size=WINDOW_SAMPLES),
        "Vpv": np.full(WINDOW_SAMPLES, 320.0) + 0.01 * rng.normal(size=WINDOW_SAMPLES),
        "Vdc": np.full(WINDOW_SAMPLES, 400.0) + 0.01 * rng.normal(size=WINDOW_SAMPLES),
    }
    for nome, fase in fases.items():
        dados[f"i{nome}"] = amplitude * 10.0 * np.sin(2 * np.pi * t + fase)
        dados[f"v{nome}"] = 220.0 * np.sin(2 * np.pi * t + fase)
    return pd.DataFrame(dados)[list(PRIMARY_COLUMNS)]


def _janela_falha(seed: int) -> pd.DataFrame:
    """Uma janela 'de falha': saudável mais um desvio grande e conhecido."""
    janela = _janela(seed=seed, amplitude=0.4)
    janela = janela.copy()
    janela["ia"] = janela["ia"].to_numpy() * 0.3
    return janela


def _scorer_distancia(referencia: np.ndarray):
    """Escore = distância da feature à referência saudável; cresce com `a`."""

    def scorer(janelas):
        return np.asarray(
            [float(np.linalg.norm(feature_vector(j) - referencia)) for j in janelas],
            dtype=float,
        )

    return scorer


# ── sequência estacionária do AE-LSTM ──────────────────────────────────────

def test_sequencia_estacionaria_repete_o_quadro():
    escaladas = np.arange(2 * len(FEATURE_COLUMNS), dtype=np.float32).reshape(
        2, len(FEATURE_COLUMNS)
    )
    sequencias = campanha._sequencias_estacionarias(escaladas)
    assert sequencias.shape == (2, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    for i in range(2):
        for passo in range(SEQUENCE_LENGTH):
            np.testing.assert_array_equal(sequencias[i, passo], escaladas[i])


# ── normalização espelhando a E3 ───────────────────────────────────────────

def test_escalar_janelas_aplica_baseline_e_scaler():
    baseline = {
        "baselines": {
            "F0L": {
                "median": np.zeros(len(FEATURE_COLUMNS)),
                "scale": np.ones(len(FEATURE_COLUMNS)),
            }
        }
    }
    scaler = RobustScaler().fit(
        np.random.default_rng(0).normal(size=(64, len(FEATURE_COLUMNS)))
    )
    escaladas = campanha._escalar_janelas([_janela(), _janela(1)], "F0L", scaler, baseline)
    assert escaladas.shape == (2, len(FEATURE_COLUMNS))
    assert np.isfinite(escaladas).all()


# ── escala saudável para a âncora ──────────────────────────────────────────

def test_escala_saudavel_e_estritamente_positiva():
    rng = np.random.default_rng(3)
    features = pd.DataFrame(
        rng.normal(size=(40, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS)
    )
    prepared = type("P", (), {"split": {"train": np.arange(40)}, "features": features})()
    escala = campanha._escala_saudavel(prepared)
    assert escala.shape == (len(FEATURE_COLUMNS),)
    assert np.all(escala > 0.0)


# ── carregamento de janelas de falha ───────────────────────────────────────

def test_carregar_janelas_falha_fatia_o_bruto(tmp_path, monkeypatch):
    bruto = pd.concat([_janela(seed=0), _janela_falha(seed=1)], ignore_index=True)
    caminho = tmp_path / "F1L.csv"
    bruto.to_csv(caminho, index=False)
    monkeypatch.setattr(campanha, "dataset_files", lambda directory=None: {"F1L": caminho})
    fault_features = pd.DataFrame(
        [
            {"experiment": "F1L", "phase": "pre_fault", "window_index": 0,
             "sample_start": 0, "sample_end": WINDOW_SAMPLES},
            {"experiment": "F1L", "phase": "post_fault", "window_index": 1,
             "sample_start": WINDOW_SAMPLES, "sample_end": 2 * WINDOW_SAMPLES},
        ]
    )
    janelas, registros = campanha._carregar_janelas_falha(fault_features, ("F1L",))
    assert len(janelas) == 1
    assert list(janelas[0].columns) == list(PRIMARY_COLUMNS)
    assert len(janelas[0]) == WINDOW_SAMPLES
    assert registros == [{"experiment": "F1L", "window_index": 1}]


def test_carregar_janelas_falha_sem_fase_estoura(tmp_path, monkeypatch):
    caminho = tmp_path / "F1L.csv"
    _janela().to_csv(caminho, index=False)
    monkeypatch.setattr(campanha, "dataset_files", lambda directory=None: {"F1L": caminho})
    fault_features = pd.DataFrame(
        [{"experiment": "F1L", "phase": "pre_fault", "window_index": 0,
          "sample_start": 0, "sample_end": WINDOW_SAMPLES}]
    )
    with pytest.raises(ValueError, match="post_fault"):
        campanha._carregar_janelas_falha(fault_features, ("F1L",))


# ── varredura de um item, com scorer sintético ─────────────────────────────

def _execucoes_falsas(limiar: float) -> dict:
    return {
        model_id: {
            "model_id": model_id,
            "modelo": None,
            "scaler": None,
            "limiar": limiar,
            "score_top_k": 5,
        }
        for model_id in campanha.MODEL_IDS
    }


def test_varrer_injecao_assinatura_detecta_e_ancora_positiva(monkeypatch):
    referencia = feature_vector(_janela())
    monkeypatch.setattr(
        campanha, "_pontuador", lambda *a, **k: _scorer_distancia(referencia)
    )
    holdout = [_janela(seed=s) for s in range(6)]
    experimentos = ["F0L"] * 6
    janelas_falha = [_janela_falha(seed=100 + s) for s in range(4)]
    especificacao = POR_ID["igbt"]

    # Limiar derivado do próprio scorer para garantir detecção parcial.
    grade = GRADE_SEVERIDADE
    scores_ref = _scorer_distancia(referencia)(
        [campanha.injetor_de(especificacao)(holdout[0], a) for a in grade]
    )
    limiar = float(scores_ref[len(grade) // 2])

    por_modelo, injetores = campanha._varrer_injecao(
        especificacao,
        _execucoes_falsas(limiar),
        holdout,
        experimentos,
        janelas_falha,
        {"baselines": {"F0L": {}}},
        grade=grade,
    )
    assert set(por_modelo) == set(campanha.MODEL_IDS)
    for det in por_modelo.values():
        assert det.n_trajetorias == 6
        assert det.n_detectadas > 0
    ancora = campanha._ancora_da_injecao(injetores, holdout, janelas_falha, np.ones(len(FEATURE_COLUMNS)))
    assert ancora["distancia_euclidiana_iqr"] > 0.0


def test_varrer_injecao_controle_usa_injetor_por_trajetoria(monkeypatch):
    referencia = feature_vector(_janela())
    monkeypatch.setattr(
        campanha, "_pontuador", lambda *a, **k: _scorer_distancia(referencia)
    )
    holdout = [_janela(seed=s) for s in range(5)]
    experimentos = ["F0M"] * 5
    janelas_falha = [_janela_falha(seed=200 + s) for s in range(3)]
    especificacao = POR_ID["controle"]

    por_modelo, injetores = campanha._varrer_injecao(
        especificacao,
        _execucoes_falsas(1.0),
        holdout,
        experimentos,
        janelas_falha,
        {"baselines": {"F0M": {}}},
        grade=GRADE_SEVERIDADE,
    )
    # Interpolação: em a=1 cada trajetória retorna o ensaio real pareado, então a
    # âncora cai perto de zero — o que confirma "estado medido, não síntese".
    ancora = campanha._ancora_da_injecao(
        injetores, holdout, janelas_falha, np.ones(len(FEATURE_COLUMNS))
    )
    assert ancora["distancia_euclidiana_iqr"] == pytest.approx(0.0, abs=1e-6)
    assert len(por_modelo["ae_denso"].a_dets) == 5


# ── run() completo com stubs, sem torch nem dataset ────────────────────────

def test_run_sem_publicar_percorre_os_tres_itens(monkeypatch):
    referencia = feature_vector(_janela())
    healthy = pd.DataFrame(
        np.random.default_rng(1).normal(size=(40, len(FEATURE_COLUMNS))),
        columns=list(FEATURE_COLUMNS),
    )
    prepared = type(
        "P",
        (),
        {
            "split": {"train": np.arange(40)},
            "features": healthy,
            "baseline_normalization": {"baselines": {"F0L": {}}},
        },
    )()
    holdout = []
    for s in range(6):
        janela = _janela(seed=s)
        janela.attrs["experiment"] = "F0L"
        holdout.append(janela)
    holdout_meta = {"dataset": "GPVS-Faults", "n_windows": 6}

    monkeypatch.setattr(
        campanha, "load_or_extract_features",
        lambda directory=None: (healthy, pd.DataFrame(), {"windows": 320}),
    )
    monkeypatch.setattr(campanha, "prepare_healthy_data", lambda healthy: prepared)
    monkeypatch.setattr(
        campanha, "load_holdout_windows",
        lambda prepared, directory=None: (holdout, holdout_meta),
    )
    monkeypatch.setattr(
        campanha, "carregar_execucao_congelada",
        lambda model_id: {
            "model_id": model_id, "modelo": None, "scaler": None,
            "limiar": 5.0, "score_top_k": 5,
        },
    )
    monkeypatch.setattr(
        campanha, "_carregar_janelas_falha",
        lambda faults, ensaios, directory=None: (
            [_janela_falha(seed=300 + i) for i in range(3)],
            [{"experiment": ensaios[0], "window_index": i} for i in range(3)],
        ),
    )
    monkeypatch.setattr(
        campanha, "_pontuador", lambda *a, **k: _scorer_distancia(referencia)
    )

    resultado = campanha.run(publicar=False)
    assert resultado["ok"] is True
    assert len(resultado["resultados"]) == len(ESPECIFICACOES)
    for item in resultado["resultados"]:
        assert set(item.por_modelo) == set(campanha.MODEL_IDS)
        assert "distancia_euclidiana_iqr" in item.ancora
    assert resultado["parametros"]["sequence_policy"] == campanha.SEQUENCIA_POLICY
