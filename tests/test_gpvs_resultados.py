from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.config import RAIZ_PROJETO
from src.ml.proveniencia import funcao_de_hash_para


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
    assert cenarios["auc"].between(0, 1).all()
    assert cenarios["sensitivity"].between(0, 1).all()
    assert cenarios["specificity"].between(0, 1).all()


def test_macros_gpvs_reconciliam_com_tabela():
    resultado, cenarios = _carregar()
    macros = resultado["macro_summary"]["canonical_ae"]["all"]
    for metrica in ("auc", "sensitivity", "specificity", "balanced_accuracy"):
        publicado = macros[metrica]
        assert publicado["mean"] == pytest.approx(
            cenarios[metrica].mean(), abs=1e-12,
        )
        assert publicado["n_experiments"] == 14
        assert publicado["ci95_low"] <= publicado["mean"] <= publicado["ci95_high"]


def test_gpvs_preserva_resultado_negativo_e_limite_de_campo():
    resultado, cenarios = _carregar()
    assert cenarios["specificity"].mean() > 0.90
    assert (cenarios["sensitivity"] < 0.10).any()
    assert resultado["detector"]["canonical"] is True
    assert resultado["detector"]["model_retraining_per_experiment"] is False
    assert resultado["detector"]["threshold_recalibration_per_experiment"] is False
    assert resultado["detector"]["commissioning_normalization_per_experiment"] is True
    texto = (PASTA / "relatorio_validacao_gpvs.md").read_text(encoding="utf-8").lower()
    assert "não validação de campo" in texto
    assert "weibull físico" in texto
    assert resultado["dataset"]["observed_sampling_period_us_min"] == pytest.approx(100, rel=0.01)
    assert resultado["dataset"]["manual_sampling_period_us"] == pytest.approx(9.9989)


def test_manifesto_gpvs_v2_confere_outputs_versionados():
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    assert manifesto["manifest_version"] == 2
    assert manifesto["evidence_level"] == "E3"
    assert len(manifesto["input_artifacts"]) == 20
    assert len(manifesto["outputs"]) == 9
    for relativo, esperado in manifesto["output_artifacts"].items():
        caminho = RAIZ / relativo
        assert caminho.exists(), relativo
        # Delegado, nao replicado: esta era a TERCEIRA copia da regra de hash
        # (as outras em verificar_resultados_fmeca e auditar_artefatos). Copia
        # de regra deriva -- quando JSON passou a ser hasheado sem os campos de
        # data, as tres copias passaram a acusar divergencia inexistente.
        assert funcao_de_hash_para(caminho)(caminho) == esperado, relativo
