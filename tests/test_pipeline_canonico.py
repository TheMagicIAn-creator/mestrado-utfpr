from __future__ import annotations

from pathlib import Path

import src.ml.pipeline as pipeline

# Saídas listadas no manifesto da E2, mais o próprio manifesto: seis dados-fonte
# e relatório, mais as quatro figuras publicadas pela etapa.
DETECTABILIDADE_ESPERADOS = 11


def test_pipeline_exposes_only_three_canonical_publications():
    # A ordem é a de dependência: a E2 consome os modelos congelados pela
    # comparação e por isso vem depois dela.
    assert tuple(pipeline.STAGES) == (
        "comparacao",
        "detectabilidade",
        "confiabilidade",
    )
    assert set(pipeline.NOMES_ETAPAS) == {
        "comparacao",
        "detectabilidade",
        "confiabilidade",
    }
    assert "Denso" in pipeline.NOMES_ETAPAS["comparacao"]
    assert "E2" in pipeline.NOMES_ETAPAS["detectabilidade"]
    assert "Confiabilidade" in pipeline.NOMES_ETAPAS["confiabilidade"]


def test_detectability_stage_requires_gpvs_and_never_recalibrates():
    """A E2 exige os dados brutos e reusa o limiar congelado da comparação."""
    stage = pipeline.get_stage("detectabilidade")
    assert stage.requires_gpvs is True
    assert stage.runner_module == "src.ml.campanha_detectabilidade"
    capacity = pipeline.capacidade_recalculo_pipeline()
    if capacity["disponivel"]:
        assert capacity["etapas_executaveis"] == [
            "comparacao",
            "detectabilidade",
            "confiabilidade",
        ]
    else:
        assert "detectabilidade" not in capacity["etapas_executaveis"]


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
    assert set(state) == {"comparacao", "detectabilidade", "confiabilidade"}
    assert state["comparacao"]["esperados"] == 24
    assert state["detectabilidade"]["esperados"] == DETECTABILIDADE_ESPERADOS
    assert state["confiabilidade"]["esperados"] == 15
    assert all(item["disponivel"] for item in state.values())


def test_cleanup_inventory_never_leaves_results_root():
    root = pipeline.RESULTS_ROOT.resolve()
    for key in pipeline.STAGES:
        for path in pipeline.artefatos_a_partir(key):
            assert root in Path(path).resolve().parents
