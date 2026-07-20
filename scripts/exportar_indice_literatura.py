"""Gera o snapshot portátil da literatura para o deploy Streamlit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import chromadb

from src.conhecimento.indice_portatil import exportar_colecao
from src.conhecimento.indexador import calcular_hash_arquivo
from src.core.config import (
    ARQUIVO_INDICE_LITERATURA,
    MODELO_EMBEDDINGS,
    NOME_COLECAO,
    PASTA_CHROMADB,
    PASTA_LITERATURA,
)


def _manifesto_corpus() -> tuple[str, int]:
    registros = []
    for pdf in sorted(PASTA_LITERATURA.rglob("*.pdf")):
        registros.append({
            "arquivo": pdf.relative_to(PASTA_LITERATURA).as_posix(),
            "sha256": calcular_hash_arquivo(pdf),
        })
    serializado = json.dumps(registros, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serializado).hexdigest(), len(registros)


def main() -> int:
    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = cliente.get_or_create_collection(NOME_COLECAO)
    hash_corpus, n_documentos = _manifesto_corpus()
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
