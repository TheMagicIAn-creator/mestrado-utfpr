"""Gera o snapshot portátil da literatura para o deploy Streamlit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import chromadb

from src.conhecimento.indice_portatil import exportar_colecao, hash_corpus_pdfs
from src.core.config import (
    ARQUIVO_INDICE_LITERATURA,
    MODELO_EMBEDDINGS,
    NOME_COLECAO,
    PASTA_CHROMADB,
    PASTA_LITERATURA,
)

def main() -> int:
    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = cliente.get_or_create_collection(NOME_COLECAO)
    hash_corpus, n_documentos = hash_corpus_pdfs(PASTA_LITERATURA)
    resultado = exportar_colecao(
        colecao,
        ARQUIVO_INDICE_LITERATURA,
        modelo_embeddings=MODELO_EMBEDDINGS,
        hash_corpus=hash_corpus,
        n_documentos=n_documentos,
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
