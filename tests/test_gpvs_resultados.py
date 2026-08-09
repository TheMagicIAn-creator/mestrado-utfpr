from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.config import RAIZ_PROJETO
from src.ml.proveniencia import (
    SUFIXOS_TEXTO_PORTAVEL,
    sha256_arquivo,
    sha256_arquivo_texto_normalizado,
)


RAIZ = Path(RAIZ_PROJETO)
PASTA = RAIZ / "resultados" / "gpvs"
JSON_RESULTADO = PASTA / "validacao_gpvs_e3.json"
CSV_CENARIOS = PASTA / "validacao_gpvs_cenarios.csv"
MANIFESTO = RAIZ / "resultados" / "manifestos" / "validacao_gpvs_e3.json"


def _carregar():
    return (
        json.loads(JSON_RESULTADO.read_text(encoding="utf-8")),
        pd.read_csv(CSV_CENARIOS),
    )


def test_gpvs_publica_os_14_ensaios_sem_selecao():
    resultado, cenarios = _carregar()
    esperados = {f"F{i}{modo}" for i in range(1, 8) for modo in "LM"}
    assert resultado["evidence_level"] == "E3"
    assert "bancada" in resultado["evidence_scope"]
    assert set(cenarios["experiment"]) == esperados
    assert len(cenarios) == 14
    assert cenarios["adaptive_ae_auc"].between(0, 1).all()
    assert cenarios["adaptive_ae_post_tpr"].between(0, 1).all()
    assert cenarios["adaptive_ae_specificity"].between(0, 1).all()


def test_macros_gpvs_reconciliam_com_tabela():
    resultado, cenarios = _carregar()
    macros = resultado["macro_summary"]
    for protocolo in ("strict_ae", "adaptive_ae", "adaptive_pca"):
        for metrica in ("auc", "post_tpr", "specificity", "balanced_accuracy"):
            publicado = macros[protocolo]["all"][metrica]
            assert publicado["mean"] == pytest.approx(
                cenarios[f"{protocolo}_{metrica}"].mean(), abs=1e-12,
            )
            assert publicado["n_experiments"] == 14
            assert publicado["ci95_low"] <= publicado["mean"] <= publicado["ci95_high"]


def test_gpvs_preserva_resultado_negativo_e_limite_de_campo():
    resultado, cenarios = _carregar()
    assert cenarios["strict_ae_specificity"].mean() < 0.10
    assert cenarios["adaptive_ae_specificity"].mean() > 0.90
    assert (cenarios["adaptive_ae_post_tpr"] < 0.10).any()
    texto = (PASTA / "relatorio_validacao_gpvs.md").read_text(encoding="utf-8").lower()
    assert "nao e validacao de\ncampo" in texto
    assert "weibull/rul fisico" in texto
    assert resultado["dataset"]["observed_sampling_period_us_min"] == pytest.approx(100, rel=0.01)
    assert resultado["dataset"]["manual_sampling_period_us"] == pytest.approx(9.9989)


def test_manifesto_gpvs_v2_confere_outputs_versionados():
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    assert manifesto["manifest_version"] == 2
    assert manifesto["evidence_level"] == "E3"
    assert len(manifesto["input_artifacts"]) == 16
    assert len(manifesto["outputs"]) == 9
    for relativo, esperado in manifesto["output_artifacts"].items():
        caminho = RAIZ / relativo
        assert caminho.exists(), relativo
        funcao = (
            sha256_arquivo_texto_normalizado
            if caminho.suffix.lower() in SUFIXOS_TEXTO_PORTAVEL
            else sha256_arquivo
        )
        assert funcao(caminho) == esperado, relativo
