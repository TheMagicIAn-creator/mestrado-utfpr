"""Reconstrói o índice de todo o vault Obsidian e seu snapshot portátil."""

from __future__ import annotations

import argparse
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
    PASTA_VAULT_OBSIDIAN,
)


def main(vault: Path = PASTA_VAULT_OBSIDIAN) -> None:
    vault = Path(vault).expanduser().resolve()
    if not vault.is_dir():
        raise SystemExit(f"Vault não encontrado: {vault}")
    modelo = criar_modelo_embeddings(modo_consulta=False)
    client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = client.get_or_create_collection(
        NOME_COLECAO_OBSIDIAN,
        metadata={"hnsw:space": "cosine"},
    )
    estado = sincronizar_obsidian(colecao, modelo, raiz=vault)
    notas = contar_notas_indexadas(colecao)
    manifesto = exportar_colecao(
        colecao,
        ARQUIVO_INDICE_OBSIDIAN,
        modelo_embeddings=MODELO_EMBEDDINGS,
        hash_corpus=hash_corpus_obsidian(vault),
        n_documentos=notas,
    )
    print(
        f"Vault Obsidian pronto: {estado['notas_ativas']} notas, "
        f"{manifesto['n_chunks']} chunks, {manifesto['tamanho_bytes']} bytes."
    )
    for classe, quantidade in sorted(estado.get("fontes_por_classe", {}).items()):
        print(f"  - {classe}: {quantidade}")
    if estado.get("erros"):
        print(f"  - erros: {len(estado['erros'])}")
        for erro in estado["erros"][:10]:
            print(f"    * {erro}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vault",
        type=Path,
        default=PASTA_VAULT_OBSIDIAN,
        help="Pasta raiz do vault; padrão: AL_IADO_OBSIDIAN_VAULT_DIR ou notas/.",
    )
    args = parser.parse_args()
    main(args.vault)
