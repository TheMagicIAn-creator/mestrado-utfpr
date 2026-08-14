from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "resultados" / "v2" / "autoencoder"


def _json(nome: str) -> dict:
    return json.loads((PASTA / nome).read_text(encoding="utf-8"))


def test_selecao_publica_tres_arquiteturas_e_cinco_seeds():
    execucoes = pd.read_csv(PASTA / "selecao_execucoes.csv")
    resumo = pd.read_csv(PASTA / "selecao_resumo.csv")
    assert len(execucoes) == 15
    assert execucoes["arquitetura"].nunique() == 3
    assert execucoes["seed"].nunique() == 5
    assert resumo["selecionada"].sum() == 1
    assert resumo.loc[resumo["selecionada"], "arquitetura"].item() == "simetrico_16_8"


def test_limiar_v2_reconcilia_calibracao_e_teste_saudavel():
    info = _json("limiar_v2.json")
    canonico = info["canonical"]
    assert canonico["method"] == "finite_sample_order_statistic"
    assert canonico["order_one_based"] == 209
    assert canonico["n_calibration"] == 210
    assert canonico["calibration"]["n_above"] == 1
    assert canonico["healthy_test"]["n"] == 281
    assert canonico["healthy_test"]["n_above"] == 3
    assert canonico["healthy_test"]["taxa_pct"] == pytest.approx(300 / 281)


def test_avaliacao_publica_todos_os_ensaios_e_metodos():
    cenarios = pd.read_csv(PASTA / "avaliacao_cenarios.csv")
    esperados = {f"F{i}{modo}" for i in range(1, 8) for modo in "LM"}
    assert set(cenarios["experiment"]) == esperados
    assert set(cenarios["method"]) == {
        "autoencoder_v2",
        "autoencoder_ensemble",
        "pca",
    }
    assert cenarios.groupby("method")["experiment"].nunique().eq(14).all()
    assert "detection_delay_s" not in cenarios
    assert "detection_delay_from_nominal_midpoint_s" in cenarios


def test_macros_reconciliam_com_csv_por_ensaio():
    cenarios = pd.read_csv(PASTA / "avaliacao_cenarios.csv")
    resultado = _json("avaliacao_experimental.json")
    for metodo, bloco in cenarios.groupby("method"):
        for metrica in (
            "auc_roc",
            "average_precision",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "mcc",
        ):
            publicado = resultado["macro_summary"][metodo][metrica]
            assert publicado["mean"] == pytest.approx(bloco[metrica].mean())
            assert publicado["n_experiments"] == 14


def test_comparacao_pareada_nao_exagera_superioridade_global():
    resultado = _json("avaliacao_experimental.json")
    diferencas = resultado["paired_comparison_autoencoder_v2_minus_pca"]
    assert diferencas["auc_roc"]["mean_difference"] < 0
    assert diferencas["sensitivity"]["mean_difference"] > 0
    assert diferencas["specificity"]["mean_difference"] < 0
    assert diferencas["balanced_accuracy"]["ci_includes_zero"] is True
    assert diferencas["mcc"]["ci_includes_zero"] is True


def test_fronteira_temporal_e_declarada_como_nominal():
    resultado = _json("avaliacao_experimental.json")
    protocolo = resultado["protocol"]
    assert protocolo["fault_boundary_fraction"] == 0.5
    assert protocolo["fault_boundary_semantics"] == (
        "nominal_record_midpoint_not_instrumented_trigger"
    )
    assert protocolo["architecture_selected_without_fault_trials"] is True


def test_figuras_v2_existentes_nao_sao_vazias():
    figuras = (
        "selecao_arquitetura.png",
        "calibracao_limiar.png",
        "desempenho_por_ensaio.png",
        "mapa_ponto_operacional.png",
        "matrizes_confusao.png",
        "curvas_roc_pr_macro.png",
        "series_temporais.png",
        "contribuicoes_familias.png",
    )
    for nome in figuras:
        caminho = PASTA / nome
        assert caminho.exists(), nome
        assert caminho.stat().st_size > 50_000, nome
