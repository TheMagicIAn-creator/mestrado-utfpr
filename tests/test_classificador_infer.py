"""
Sprint 3 — persistência e inferência do classificador PV Farms (6.2/6.4).

Treina/salva um modelo (fixture sintética), classifica uma amostra compatível
e rejeita amostras com colunas ausentes/extras. Sempre com aviso de domínio CC.
"""

import json

import numpy as np
import pandas as pd

from src.ml.classificador_pv_infer import classificar, treinar_e_salvar_de


def _dados():
    rng = np.random.default_rng(0)
    n = 60
    x0 = rng.normal(0.0, 1.0, (n, 3))
    x1 = rng.normal(5.0, 1.0, (n, 3))
    X = pd.DataFrame(np.vstack([x0, x1]), columns=["f1", "f2", "f3"])
    y = pd.Series([0] * n + [1] * n)
    return X, y


def test_treina_salva_e_classifica(tmp_path):
    X, y = _dados()
    treinar_e_salvar_de(X, y, X, y, pasta=tmp_path)
    assert (tmp_path / "modelo_classificador.pkl").exists()
    assert (tmp_path / "scaler.pkl").exists()
    assert (tmp_path / "feature_columns.json").exists()
    assert (tmp_path / "class_mapping.json").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    assert (tmp_path / "training_manifest.json").exists()
    assert (tmp_path / "metricas.json").exists()
    assert (tmp_path / "metricas.csv").exists()
    assert (tmp_path / "matriz_confusao.png").exists()
    assert (tmp_path / "importancia_features.png").exists()

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dominio"] == "CC"
    assert manifest["evidence_level"] == "E1"
    assert manifest["n_features"] == 3

    r = classificar({"f1": 5.0, "f2": 5.0, "f3": 5.0}, pasta=tmp_path)
    assert r["ok"] and r["dominio"] == "CC"
    assert "diagnostica falhas ca" in r["aviso"].lower()
    assert 0.0 <= r["probabilidade"] <= 1.0


def test_rejeita_coluna_ausente(tmp_path):
    X, y = _dados()
    treinar_e_salvar_de(X, y, X, y, pasta=tmp_path)
    r = classificar({"f1": 1.0, "f2": 1.0}, pasta=tmp_path)  # falta f3
    assert not r["ok"] and "ausente" in r["erro"].lower()


def test_rejeita_coluna_extra(tmp_path):
    X, y = _dados()
    treinar_e_salvar_de(X, y, X, y, pasta=tmp_path)
    r = classificar({"f1": 1.0, "f2": 1.0, "f3": 1.0, "extra": 9.0}, pasta=tmp_path)
    assert not r["ok"] and "extra" in r["erro"].lower()


def test_sem_modelo_avisa(tmp_path):
    r = classificar({"f1": 1.0}, pasta=tmp_path)  # nada treinado
    assert not r["ok"] and "treinad" in r["erro"].lower()
