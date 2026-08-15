"""Carregamento da base de conhecimento independente da interface.

O runtime web consulta resultados cientificos ja versionados e inicializa o
RAG somente no primeiro turno. Este modulo concentra a restauracao dos indices
portateis, a escolha do backend de embeddings e a preparacao do BM25 sem
importar Streamlit ou disparar treinamento automaticamente.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseConhecimento:
    perfil: str
    modelo_embeddings: object
    literatura: object
    sessoes: object
    obsidian: object
    indice_lexical: object
    modo_consulta: bool
    relatorio: tuple[str, ...]


def _versao_indice_lexical(colecao, *, modo_consulta: bool) -> str:
    from src.core.config import ARQUIVO_INDICE_LITERATURA

    versao = f"chroma:{colecao.count()}"
    if modo_consulta and ARQUIVO_INDICE_LITERATURA.is_file():
        try:
            from src.conhecimento.indice_portatil import ler_manifesto

            manifesto = ler_manifesto(ARQUIVO_INDICE_LITERATURA)
            return str(manifesto.get("hash_corpus_sha256") or versao)
        except Exception:
            return versao

    if not colecao.count():
        return versao

    ids_amostra: list[str] = []
    for offset in sorted({0, colecao.count() // 2, colecao.count() - 1}):
        try:
            lote = colecao.get(limit=1, offset=offset, include=["metadatas"])
            ids_amostra.extend(str(item) for item in (lote.get("ids") or []))
        except Exception:
            continue
    return versao + ":" + ":".join(ids_amostra)


def carregar_base_conhecimento(
    *,
    sincronizar_obsidian_local: bool = True,
    embeddings_baixo_consumo: bool = False,
) -> BaseConhecimento:
    """Monta o runtime RAG sem acoplar ciclo de vida a uma biblioteca de UI.

    Interfaces interativas podem adiar a varredura incremental do Obsidian e
    trabalhar imediatamente com a colecao persistente ja disponivel. O valor
    padrao preserva o comportamento dos fluxos em lote e da interface legada.
    O encoder leve ONNX usa o mesmo espaco vetorial e evita carregar PyTorch
    quando a interface somente consulta indices existentes.
    """
    import chromadb

    from src.conhecimento.agente import carregar_perfil
    from src.conhecimento.embeddings import backend_embeddings, criar_modelo_embeddings
    from src.conhecimento.indice_lexical import IndiceLexicalSQLite
    from src.core.config import (
        ARQUIVO_INDICE_LITERATURA,
        ARQUIVO_INDICE_OBSIDIAN,
        ARQUIVO_MEMORIA_VALIDADA,
        NOME_COLECAO,
        NOME_COLECAO_OBSIDIAN,
        NOME_COLECAO_SESSOES,
        PASTA_CHROMADB,
    )
    from src.ml.pipeline import capacidade_recalculo_pipeline

    relatorio: list[str] = []
    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    literatura = cliente.get_or_create_collection(name=NOME_COLECAO)
    sessoes = cliente.get_or_create_collection(name=NOME_COLECAO_SESSOES)
    obsidian = cliente.get_or_create_collection(
        name=NOME_COLECAO_OBSIDIAN,
        metadata={"hnsw:space": "cosine"},
    )

    total_obsidian = obsidian.count()
    if total_obsidian == 0 and ARQUIVO_INDICE_OBSIDIAN.is_file():
        try:
            from src.conhecimento.indice_portatil import importar_colecao

            restauracao = importar_colecao(
                obsidian,
                ARQUIVO_INDICE_OBSIDIAN,
                mesclar=True,
            )
            relatorio.append(
                f"Obsidian: {restauracao['n_chunks']} chunks portateis disponiveis."
            )
        except Exception as exc:
            relatorio.append(f"Obsidian: snapshot indisponivel ({exc}).")
    elif total_obsidian:
        relatorio.append(
            f"Obsidian: {total_obsidian} chunks persistentes disponiveis."
        )

    if literatura.count() == 0 and ARQUIVO_INDICE_LITERATURA.is_file():
        try:
            from src.conhecimento.indice_portatil import importar_colecao

            restauracao = importar_colecao(literatura, ARQUIVO_INDICE_LITERATURA)
            relatorio.append(
                f"Literatura: {restauracao['n_chunks']} chunks restaurados."
            )
        except Exception as exc:
            relatorio.append(f"Literatura: snapshot indisponivel ({exc}).")

    capacidade = capacidade_recalculo_pipeline()
    modo_consulta = not bool(capacidade["disponivel"])
    modo_embeddings_consulta = modo_consulta or embeddings_baixo_consumo
    modelo = criar_modelo_embeddings(modo_consulta=modo_embeddings_consulta)
    relatorio.append(
        "Embeddings: backend "
        f"{backend_embeddings(modo_consulta=modo_embeddings_consulta)}."
    )

    try:
        from src.conhecimento.obsidian import (
            espelhar_memoria_validada,
            sincronizar_obsidian,
        )

        espelhar_memoria_validada(ARQUIVO_MEMORIA_VALIDADA)
        if not modo_consulta and sincronizar_obsidian_local:
            estado = sincronizar_obsidian(obsidian, modelo)
            relatorio.append(
                f"Obsidian: {estado['notas_ativas']} notas locais sincronizadas."
            )
        elif not modo_consulta:
            relatorio.append(
                "Obsidian: colecao persistente carregada; sincronizacao incremental adiada."
            )
    except Exception as exc:
        relatorio.append(f"Obsidian: sincronizacao indisponivel ({exc}).")

    indice_lexical = IndiceLexicalSQLite()
    try:
        estado_lexical = indice_lexical.sincronizar(
            literatura,
            versao=_versao_indice_lexical(
                literatura,
                modo_consulta=modo_consulta,
            ),
        )
        relatorio.append(
            f"Busca lexical: {estado_lexical.get('n_chunks', 0)} chunks preparados."
        )
    except Exception as exc:
        relatorio.append(f"Busca lexical: indisponivel ({exc}).")

    return BaseConhecimento(
        perfil=carregar_perfil(),
        modelo_embeddings=modelo,
        literatura=literatura,
        sessoes=sessoes,
        obsidian=obsidian,
        indice_lexical=indice_lexical,
        modo_consulta=modo_consulta,
        relatorio=tuple(relatorio),
    )
