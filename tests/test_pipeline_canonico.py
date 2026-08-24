from __future__ import annotations

from pathlib import Path

import src.ml.pipeline as pipeline


def test_pipeline_exposes_only_two_canonical_publications():
    assert tuple(pipeline.STAGES) == ("comparacao", "confiabilidade")
    assert set(pipeline.NOMES_ETAPAS) == {"comparacao", "confiabilidade"}
    assert "Denso" in pipeline.NOMES_ETAPAS["comparacao"]
    assert "Confiabilidade" in pipeline.NOMES_ETAPAS["confiabilidade"]


def test_gpvs_capacity_requires_exactly_sixteen_trials(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATASET_DIR", tmp_path)
    initial = pipeline.capacidade_recalculo_pipeline()
    assert initial["disponivel"] is False
    assert initial["arquivos_esperados"] == 16

    for name in pipeline.ALL_EXPERIMENTS:
        (tmp_path / f"{name}.csv").write_text("sample\n", encoding="utf-8")
    complete = pipeline.capacidade_recalculo_pipeline()
    assert complete["disponivel"] is True
    assert complete["arquivos_ausentes"] == []


def test_comparison_refuses_training_without_raw_gpvs(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATASET_DIR", tmp_path)
    result = pipeline.executar_etapa("comparacao")
    assert result["ok"] is False
    assert "16 CSVs" in result["mensagem"]


def test_reliability_runner_does_not_depend_on_gpvs(monkeypatch):
    calls = []
    stage = pipeline.PipelineStage(
        key="confiabilidade",
        label="Confiabilidade",
        manifest_name="confiabilidade_componentes",
        runner_module="unused",
        runner_function="unused",
        required_outputs=(),
    )
    monkeypatch.setitem(pipeline.STAGES, "confiabilidade", stage)
    monkeypatch.setattr(stage.__class__, "load_runner", lambda self: lambda: calls.append(True) or {})
    result = pipeline.executar_etapa("confiabilidade")
    assert result["ok"] is True
    assert calls == [True]


def test_published_state_uses_manifest_output_inventory():
    state = pipeline.estado_resultados_publicados()
    assert set(state) == {"comparacao", "confiabilidade"}
    assert state["comparacao"]["esperados"] == 17
    assert state["confiabilidade"]["esperados"] == 15
    assert all(item["disponivel"] for item in state.values())


def test_cleanup_inventory_never_leaves_results_root():
    root = pipeline.RESULTS_ROOT.resolve()
    for key in pipeline.STAGES:
        for path in pipeline.artefatos_a_partir(key):
            assert root in Path(path).resolve().parents
