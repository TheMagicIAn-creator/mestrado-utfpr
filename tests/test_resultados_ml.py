"""Adaptador unidirecional entre resultados científicos e memória."""

from src.conhecimento import indexador
from src.conhecimento import resultados_ml
from src.ml import resultados


def test_indexar_resultados_ml_salva_e_indexa(monkeypatch, tmp_path):
    resumo = tmp_path / "resumo.md"
    chamadas = []
    monkeypatch.setattr(resultados, "salvar_resumo_resultados_ml", lambda: resumo)
    monkeypatch.setattr(
        indexador,
        "indexar_sessao",
        lambda caminho, modelo, pasta: chamadas.append((caminho, modelo, pasta)),
    )
    monkeypatch.setattr(resultados_ml, "PASTA_CHROMADB", tmp_path / "chroma")

    mensagem = resultados_ml.indexar_resultados_ml("embeddings")

    assert "Resultados indexados" in mensagem
    assert chamadas == [(resumo, "embeddings", tmp_path / "chroma")]


def test_indexar_resultados_ml_informa_falha_recuperavel(monkeypatch, tmp_path):
    monkeypatch.setattr(resultados, "salvar_resumo_resultados_ml", lambda: tmp_path / "r.md")
    monkeypatch.setattr(
        indexador,
        "indexar_sessao",
        lambda *args: (_ for _ in ()).throw(RuntimeError("indice ocupado")),
    )

    assert "indice ocupado" in resultados_ml.indexar_resultados_ml("embeddings")
