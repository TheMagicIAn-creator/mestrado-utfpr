"""Separação explícita entre cálculo local e consulta de artefatos na nuvem."""

from pathlib import Path

def test_capacidade_recalculo_depende_do_dataset(monkeypatch, tmp_path):
    import src.ml.pipeline as pipeline

    dataset = tmp_path / "gpvs"
    monkeypatch.setattr(pipeline, "DATASET_GPVS", dataset)
    assert pipeline.capacidade_recalculo_pipeline()["disponivel"] is False

    dataset.mkdir()
    for relativo in pipeline.GPVS_ALL_INPUTS:
        (dataset / Path(relativo).name).write_text("sample", encoding="utf-8")
    assert pipeline.capacidade_recalculo_pipeline()["disponivel"] is True


def test_estado_publicacao_exige_todos_os_artefatos(monkeypatch, tmp_path):
    import src.ml.pipeline as pipeline

    monkeypatch.setattr(pipeline, "RAIZ_PROJETO", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "ARTEFATOS_PUBLICADOS",
        {"etapa": ("resultado.json", "grafico.png")},
    )
    (tmp_path / "resultado.json").write_text("{}", encoding="utf-8")

    estado = pipeline.estado_resultados_publicados()["etapa"]
    assert estado == {"disponivel": False, "presentes": 1, "esperados": 2}

    (tmp_path / "grafico.png").write_bytes(b"png")
    assert pipeline.estado_resultados_publicados()["etapa"]["disponivel"] is True


def test_ferramenta_na_nuvem_nao_inicia_pipeline(monkeypatch):
    import src.conhecimento.ferramentas as ferramentas

    monkeypatch.setattr(
        ferramentas,
        "capacidade_recalculo_pipeline",
        lambda: {"disponivel": False},
    )
    monkeypatch.setattr(
        ferramentas,
        "consultar_status_pipeline",
        lambda **_kwargs: {"mensagem": "ARTEFATOS PUBLICADOS"},
    )
    monkeypatch.setattr(
        ferramentas,
        "executar_pipeline_ml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("não executar")),
    )

    resposta = ferramentas.rodar_pipeline_completo(pergunta="rode o pipeline")

    assert resposta["ok"] is True
    assert "modo de consulta" in resposta["mensagem"].lower()
    assert "ARTEFATOS PUBLICADOS" in resposta["mensagem"]
