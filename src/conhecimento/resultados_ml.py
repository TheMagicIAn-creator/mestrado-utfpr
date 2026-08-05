"""Adaptador entre os resultados científicos e a memória do agente."""

from __future__ import annotations

from src.core.config import PASTA_CHROMADB


def indexar_resultados_ml(modelo_embeddings) -> str:
    """Gera a nota de resultados e a indexa na coleção de sessões."""
    from src.conhecimento.indexador import indexar_sessao
    from src.ml.resultados import salvar_resumo_resultados_ml

    saida = salvar_resumo_resultados_ml()
    try:
        indexar_sessao(saida, modelo_embeddings, PASTA_CHROMADB)
        return "Resultados indexados. O agente já pode discuti-los no chat."
    except Exception as exc:  # integração best-effort, com diagnóstico explícito
        return f"Resumo salvo, mas houve erro ao indexar: {exc}"
