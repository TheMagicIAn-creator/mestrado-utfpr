"""Reconstrói a coleção curada do Obsidian e seu snapshot portátil."""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import chromadb

from src.conhecimento.embeddings import criar_modelo_embeddings
from src.conhecimento.indice_portatil import exportar_colecao
from src.conhecimento.obsidian import (
    contar_notas_indexadas,
    hash_corpus_obsidian,
    sincronizar_obsidian,
)
from src.core.config import (
    ARQUIVO_INDICE_OBSIDIAN,
    MODELO_EMBEDDINGS,
    NOME_COLECAO_OBSIDIAN,
    PASTA_CHROMADB,
)


def main() -> None:
    modelo = criar_modelo_embeddings(modo_consulta=False)
    client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = client.get_or_create_collection(
        NOME_COLECAO_OBSIDIAN,
        metadata={"hnsw:space": "cosine"},
    )
    estado = sincronizar_obsidian(colecao, modelo)
    notas = contar_notas_indexadas(colecao)
    manifesto = exportar_colecao(
        colecao,
        ARQUIVO_INDICE_OBSIDIAN,
        modelo_embeddings=MODELO_EMBEDDINGS,
        hash_corpus=hash_corpus_obsidian(),
        n_documentos=notas,
    )
    print(
        f"Obsidian pronto: {estado['notas_ativas']} notas, "
        f"{manifesto['n_chunks']} chunks, {manifesto['tamanho_bytes']} bytes."
    )


if __name__ == "__main__":
    main()
