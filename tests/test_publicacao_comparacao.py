"""Consistência dos artefatos canônicos Denso versus AE-LSTM publicados."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.core.config import RAIZ_PROJETO
from src.ml.proveniencia import funcao_de_hash_para, sha256_arquivo_texto_normalizado


ROOT = Path(RAIZ_PROJETO)
RESULTS = ROOT / "resultados" / "comparacao"
MANIFEST_PATH = ROOT / "resultados" / "manifestos" / "comparacao_autoencoders.json"
MODELS = {"ae_denso", "ae_lstm"}
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
    assert payload["e3"]["primary_metrics"] == ["recall", "f1", "precision"]
    assert payload["e3"]["complementary_metrics"] == ["auc_roc", "auc_pr"]
    assert payload["e3"]["confusion_matrix_unit"] == "window_descriptive_only"
    ablation = payload["e3"]["temporal_ablation"]
    assert ablation["role"] == "supplementary_diagnostic"
    assert ablation["sequence_length"] == 8
    assert ablation["transition_post_windows"] == 7
    assert ablation["decision_target"] == "W_t"
    assert ablation["context"] == "causal_continuous_W_t_minus_7_to_W_t"
    assert ablation["conclusion"]["status"] in {
        "survives",
        "does_not_survive",
        "inconclusive",
    }
    sensitivity = payload["e3"]["score_threshold_sensitivity"]
    assert sensitivity["role"] == "supplementary_no_model_selection"
    assert sensitivity["top_k_values"] == [5, 10, 20]
    assert sensitivity["requested_percentiles"] == [99.0, 99.5, 99.9]
    assert sensitivity["configuration_count_per_model_seed"] == 9
    assert sensitivity["historical_reference_configuration"] == {
        "score_top_k": 5,
        "threshold_requested_percentile": 99.9,
        "role": "reproducibility_reference_not_universal_optimum",
    }
    assert sensitivity["uses_fault_data_for_selection"] is False
    assert "e2" not in payload
    assert "GPVS-Faults" not in payload["title"]
    assert {item["stability_seeds"][0] for item in payload["models"].values()} == {13}
    assert all(item["stability_seeds"] == SEEDS for item in payload["models"].values())
    for model in payload["models"].values():
        assert model["score_top_k"] == 5
        assert model["score_dimension"] == "feature"
        assert model["threshold_requested_percentile"] == 99.9
        assert model["threshold_effective_percentile"] == 100.0
        assert model["threshold_selected_rank"] == model["calibration_n"] == 210
        assert model["threshold_percentile_resolution"] == pytest.approx(100 / 210)


@pytest.mark.integracao
def test_contrato_publicado_e_json_estrito():
    payload = json.loads(
        (RESULTS / "comparacao_autoencoders.json").read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite,
    )
    assert set(payload["models"]) == MODELS
    assert payload["e3"]["primary_metrics"] == ["recall", "f1", "precision"]


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


def test_relatorio_deriva_tabelas_do_contrato_publicado(tmp_path):
    from src.ml.publicacao_comparacao import _write_report

    payload = _json(RESULTS / "comparacao_autoencoders.json")
    report = _write_report(payload, tmp_path / "relatorio.md")
    text = report.read_text(encoding="utf-8")

    assert "Recall | F1 | Precision | ROC-AUC | PR-AUC" in text
    assert "Matrizes de confusão agregadas" in text
    assert "Percentil efetivo" in text
    assert "Ablação temporal do AE-LSTM" in text
    assert "Sensibilidade do escore e do limiar" in text


def test_publicador_rejeita_configuracao_desigual_entre_modelos(monkeypatch, tmp_path):
    import src.ml.publicacao_comparacao as publication

    monkeypatch.setattr(publication, "RESULTS_DIR", tmp_path)
    runs = {
        "ae_denso": [
            SimpleNamespace(
                seed=42,
                score_top_k=5,
                threshold_calibration=SimpleNamespace(requested_percentile=99.9),
            )
        ],
        "ae_lstm": [
            SimpleNamespace(
                seed=42,
                score_top_k=4,
                threshold_calibration=SimpleNamespace(requested_percentile=99.9),
            )
        ],
    }

    with pytest.raises(ValueError, match="mesmo percentil e top-k"):
        publication.save_results({}, object(), runs, {}, seeds=(42,))


@pytest.mark.integracao
def test_tabelas_e3_reconciliam():
    scenarios = pd.read_csv(RESULTS / "e3_metricas_por_ensaio.csv")
    stability = pd.read_csv(RESULTS / "e3_estabilidade_sementes.csv")
    confusion = pd.read_csv(RESULTS / "e3_matrizes_confusao.csv")
    scores = pd.read_csv(RESULTS / "e3_escores_referencia.csv")
    ablation = pd.read_csv(RESULTS / "e3_ablacao_temporal_por_ensaio.csv")
    ablation_paired = pd.read_csv(RESULTS / "e3_ablacao_temporal.csv")
    sensitivity = pd.read_csv(RESULTS / "e3_sensibilidade_escore_limiar.csv")

    assert len(scenarios) == 14 * len(MODELS) * len(SEEDS)
    assert set(scenarios["seed"]) == set(SEEDS)
    assert len(stability) == len(MODELS) * len(SEEDS) * 10
    assert confusion["unit"].eq("window_descriptive_only").all()
    assert confusion["normalization"].eq("within_actual_class").all()
    assert len(ablation) == 14 * len(MODELS) * len(SEEDS) * 4
    assert set(ablation["analysis"]) == {
        "current_full",
        "transition",
        "sustained",
        "post_fault_reset",
    }
    assert len(ablation_paired) == len(SEEDS) * 4 * 10
    assert len(sensitivity) == len(MODELS) * len(SEEDS) * 3 * 3
    assert sensitivity["uses_fault_data_for_selection"].eq(False).all()  # noqa: E712
    assert sensitivity["calibration_role"].eq("healthy_calibration_only").all()
    assert sensitivity["fault_evaluation_role"].eq(
        "post_freeze_descriptive_only"
    ).all()
    assert sensitivity["is_historical_reference_configuration"].sum() == (
        len(MODELS) * len(SEEDS)
    )
    assert sensitivity["threshold_selected_rank"].le(
        sensitivity["calibration_n"]
    ).all()
    assert sensitivity["threshold_effective_percentile"].le(100.0).all()
    assert sensitivity["threshold_percentile_resolution"].gt(0.0).all()
    np_columns = {
        "tn_rate_actual_healthy",
        "fp_rate_actual_healthy",
        "fn_rate_actual_fault",
        "tp_rate_actual_fault",
    }
    assert np_columns.issubset(confusion.columns)
    score_counts = scores.groupby("model").size()
    for row in confusion.itertuples(index=False):
        assert row.tn + row.fp + row.fn + row.tp == score_counts[row.model]


@pytest.mark.integracao
def test_manifesto_reconcilia_os_23_outputs():
    manifest = _json(MANIFEST_PATH)

    assert manifest["manifest_version"] == 2
    assert manifest["parameters"]["stability_seeds"] == SEEDS
    assert manifest["parameters"]["threshold_percentile"] == 99.9
    assert manifest["parameters"]["score_top_k"] == 5
    assert "e2_steps" not in manifest["parameters"]
    assert manifest["evidence_level"] == "E3_bench"
    assert len(manifest["outputs"]) == 23
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
        "plots": "graficos_comparacao.py",
        "plot_style": "estilo_graficos.py",
        "publication": "publicacao_comparacao.py",
        "sensitivity": "sensibilidade_escore.py",
    }
    for key, filename in dependencies.items():
        path = ROOT / "src" / "ml" / filename
        assert sha256_arquivo_texto_normalizado(path) == manifest["code_dependencies"][key]


@pytest.mark.integracao
def test_figuras_tem_png_300_dpi_e_pdf_vetorial():
    pngs = sorted(RESULTS.glob("*.png"))
    pdfs = sorted(RESULTS.glob("*.pdf"))

    assert len(pngs) == 6
    assert len(pdfs) == 6
    for path in pngs:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert b"pHYs" in data
    for path in pdfs:
        assert path.read_bytes().startswith(b"%PDF-")
