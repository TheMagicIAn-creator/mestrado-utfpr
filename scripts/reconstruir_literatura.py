"""Reconstrói somente a coleção de literatura do ChromaDB.

Não remove a coleção de sessões/memória.

Execute pela raiz do projeto, com Streamlit fechado:
    python scripts/reconstruir_literatura.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
import chromadb

from src.core.config import PASTA_CHROMADB, PASTA_LITERATURA, NOME_COLECAO, MODELO_EMBEDDINGS
from src.conhecimento.indexador import (
    indexar_pdf_unico,
    TAMANHO_CHUNK_LITERATURA,
    SOBREPOSICAO_LITERATURA,
    EXTRAIR_TABELAS_LITERATURA,
)


def main():
    client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))

    try:
        client.delete_collection(NOME_COLECAO)
        print(f"Coleção removida: {NOME_COLECAO}")
    except Exception as exc:
        print(f"Coleção não removida ou ainda inexistente: {NOME_COLECAO}")
        print(f"Detalhe: {exc}")

    client.get_or_create_collection(
        name=NOME_COLECAO,
        metadata={"hnsw:space": "cosine"},
    )

    pdfs = sorted(PASTA_LITERATURA.rglob("*.pdf"))
    print(f"PDFs encontrados: {len(pdfs)}")
    print(f"Chunk literatura: {TAMANHO_CHUNK_LITERATURA}")
    print(f"Sobreposição literatura: {SOBREPOSICAO_LITERATURA}")
    print(f"Extração de tabelas: {'ativada' if EXTRAIR_TABELAS_LITERATURA else 'desativada'}")

    modelo = SentenceTransformer(MODELO_EMBEDDINGS)

    total = 0
    pulados = 0
    erros = 0

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        res = indexar_pdf_unico(pdf, modelo, PASTA_CHROMADB)

        if not res.get("sucesso"):
            erros += 1
            print(f"  ERRO: {res.get('erro')}")
            continue

        if res.get("pulou"):
            pulados += 1
            print(f"  SKIP: {res.get('motivo')}")
            continue

        n_chunks = int(res.get("n_chunks", 0))
        total += n_chunks
        print(f"  OK: {n_chunks} chunks")

    print("=" * 72)
    print(f"Chunks novos : {total}")
    print(f"PDFs pulados : {pulados}")
    print(f"Erros        : {erros}")
    print("=" * 72)


if __name__ == "__main__":
    main()
