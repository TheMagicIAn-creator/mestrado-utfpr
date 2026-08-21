"""Consistência dos artefatos canônicos Denso versus AE-LSTM publicados."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.config import RAIZ_PROJETO
from src.ml.proveniencia import funcao_de_hash_para, sha256_arquivo_texto_normalizado


ROOT = Path(RAIZ_PROJETO)
RESULTS = ROOT / "resultados" / "comparacao"
MANIFEST_PATH = ROOT / "resultados" / "manifestos" / "comparacao_autoencoders.json"
MODELS = {"ae_denso", "ae_lstm"}
COMPONENTS = {"contator_ac", "igbt", "fusivel_ac"}
SEEDS = [13, 29, 42, 71, 101]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_non_finite(value: str):
    raise ValueError(f"Constante JSON não finita: {value}")


@pytest.mark.integracao
def test_contrato_publicado_e_canonico():
    payload = _json(RESULTS / "comparacao_autoencoders.json")

    assert payload["schema_version"] == 2
    assert payload["dataset"]["dataset"] == "GPVS-Faults"
    assert payload["dataset"]["active_dataset_count"] == 1
    assert len(payload["dataset"]["experiments"]) == 16
    assert payload["e3"]["evidence_level"] == "E3_bench"
    assert payload["e3"]["primary_metric"] == "auc_pr"
    assert payload["e3"]["confusion_matrix_unit"] == "window_descriptive_only"
    assert payload["e2"]["evidence_level"] == "E2_synthetic"
    assert payload["e2"]["axis_is_time"] is False
    assert payload["e2"]["magnitude_steps"] == 101
    assert "descritivos" in payload["e2"]["interval_caveat"]
    assert {item["stability_seeds"][0] for item in payload["models"].values()} == {13}
    assert all(item["stability_seeds"] == SEEDS for item in payload["models"].values())


@pytest.mark.integracao
def test_contrato_publicado_e_json_estrito():
    payload = json.loads(
        (RESULTS / "comparacao_autoencoders.json").read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite,
    )
    fuse_lstm = next(
        item
        for item in payload["e2"]["summary"]
        if item["model"] == "ae_lstm" and item["component"] == "fusivel_ac"
    )
    assert fuse_lstm["smd95_status"] == "not_reached"
    assert fuse_lstm["smd95"] is None


def test_publicador_converte_nao_finitos_para_null(tmp_path):
    from src.ml.publicacao_comparacao import _write_json

    path = _write_json(
        tmp_path / "contrato.json",
        {"nan": float("nan"), "infinito": float("inf")},
    )
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite,
    )
    assert payload == {"nan": None, "infinito": None}


@pytest.mark.integracao
def test_tabelas_e2_e3_reconciliam():
    curves = pd.read_csv(RESULTS / "e2_deteccao_por_magnitude.csv")
    summary = pd.read_csv(RESULTS / "e2_resumo.csv")
    scenarios = pd.read_csv(RESULTS / "e3_metricas_por_ensaio.csv")
    stability = pd.read_csv(RESULTS / "e3_estabilidade_sementes.csv")
    confusion = pd.read_csv(RESULTS / "e3_matrizes_confusao.csv")
    scores = pd.read_csv(RESULTS / "e3_escores_referencia.csv")

    assert set(curves["model"]) == MODELS
    assert set(curves["component"]) == COMPONENTS
    assert curves.groupby(["model", "component"])["magnitude"].nunique().eq(101).all()
    assert len(summary) == len(MODELS) * len(COMPONENTS)
    assert set(summary["smd95_status"]) <= {"reached", "not_reached"}
    assert len(scenarios) == 14 * len(MODELS) * len(SEEDS)
    assert set(scenarios["seed"]) == set(SEEDS)
    assert len(stability) == len(MODELS) * len(SEEDS) * 9
    assert confusion["unit"].eq("window_descriptive_only").all()
    score_counts = scores.groupby("model").size()
    for row in confusion.itertuples(index=False):
        assert row.tn + row.fp + row.fn + row.tp == score_counts[row.model]


@pytest.mark.integracao
def test_manifesto_reconcilia_os_30_outputs():
    manifest = _json(MANIFEST_PATH)

    assert manifest["manifest_version"] == 2
    assert manifest["parameters"]["stability_seeds"] == SEEDS
    assert manifest["parameters"]["e2_steps"] == 101
    assert len(manifest["outputs"]) == 30
    assert set(manifest["outputs"]) == set(manifest["output_artifacts"])
    for relative_path, expected_hash in manifest["output_artifacts"].items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert funcao_de_hash_para(path)(path) == expected_hash, relative_path

    dependencies = {
        "dataset": "dados_gpvs.py",
        "models": "modelos_autoencoder.py",
        "training": "treino_comparacao.py",
        "evaluation": "avaliacao_comparativa.py",
        "statistics": "estatistica_comparacao.py",
        "fmeca_signatures": "assinaturas_fmeca.py",
        "detectability": "detectabilidade.py",
        "plots": "graficos_comparacao.py",
        "publication": "publicacao_comparacao.py",
    }
    for key, filename in dependencies.items():
        path = ROOT / "src" / "ml" / filename
        assert sha256_arquivo_texto_normalizado(path) == manifest["code_dependencies"][key]


@pytest.mark.integracao
def test_figuras_tem_png_300_dpi_e_pdf_vetorial():
    pngs = sorted(RESULTS.glob("*.png"))
    pdfs = sorted(RESULTS.glob("*.pdf"))

    assert len(pngs) == 8
    assert len(pdfs) == 8
    for path in pngs:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert b"pHYs" in data
    for path in pdfs:
        assert path.read_bytes().startswith(b"%PDF-")
