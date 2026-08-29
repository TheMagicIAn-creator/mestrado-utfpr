"""Manutencao explicita das bases de literatura, sessoes e Obsidian."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def _exportar_literatura() -> dict:
    import chromadb

    from src.conhecimento.indice_portatil import exportar_colecao, hash_corpus_pdfs
    from src.core.config import (
        ARQUIVO_INDICE_LITERATURA,
        MODELO_EMBEDDINGS,
        NOME_COLECAO,
        PASTA_CHROMADB,
        PASTA_LITERATURA,
    )

    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = cliente.get_or_create_collection(NOME_COLECAO)
    corpus_hash, documentos = hash_corpus_pdfs(PASTA_LITERATURA)
    return exportar_colecao(
        colecao,
        ARQUIVO_INDICE_LITERATURA,
        modelo_embeddings=MODELO_EMBEDDINGS,
        hash_corpus=corpus_hash,
        n_documentos=documentos,
    )


def reconstruir_literatura() -> int:
    import chromadb

    from src.conhecimento.embeddings import criar_modelo_embeddings
    from src.conhecimento.indexador import indexar_literatura
    from src.core.config import NOME_COLECAO, PASTA_CHROMADB

    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    try:
        cliente.delete_collection(NOME_COLECAO)
    except Exception:
        pass
    cliente.get_or_create_collection(NOME_COLECAO, metadata={"hnsw:space": "cosine"})
    modelo = criar_modelo_embeddings(modo_consulta=False)
    resultado = indexar_literatura(modelo=modelo, pasta_chromadb=PASTA_CHROMADB)
    resultado["snapshot"] = _exportar_literatura()
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return int(resultado.get("erros", 0) > 0)


def exportar_literatura() -> int:
    print(json.dumps(_exportar_literatura(), ensure_ascii=False, indent=2))
    return 0


def migrar_snapshot_v2() -> int:
    from src.conhecimento.indice_portatil import migrar_snapshot_v2 as migrar
    from src.core.config import ARQUIVO_INDICE_LITERATURA

    resultado = migrar(ARQUIVO_INDICE_LITERATURA)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


def reindexar_sessoes() -> int:
    from src.conhecimento.embeddings import criar_modelo_embeddings
    from src.conhecimento.indexador import indexar_sessao
    from src.core.config import PASTA_CHROMADB, PASTA_MEMORIAS, PASTA_SESSOES

    modelo = criar_modelo_embeddings(modo_consulta=False)
    pastas = (
        PASTA_SESSOES,
        RAIZ / "notas" / "sessoes_arquivadas",
        PASTA_MEMORIAS,
    )
    total_arquivos = 0
    total_chunks = 0
    erros: list[str] = []
    for pasta in pastas:
        for arquivo in sorted(pasta.glob("*.md")) if pasta.is_dir() else ():
            total_arquivos += 1
            try:
                total_chunks += indexar_sessao(arquivo, modelo, PASTA_CHROMADB)
            except Exception as exc:
                erros.append(f"{arquivo.name}: {exc}")
    print(
        json.dumps(
            {"arquivos": total_arquivos, "chunks": total_chunks, "erros": erros},
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(bool(erros))


def sincronizar_obsidian(vault: Path | None = None) -> int:
    import chromadb

    from src.conhecimento.embeddings import criar_modelo_embeddings
    from src.conhecimento.indice_portatil import exportar_colecao
    from src.conhecimento.obsidian import (
        contar_notas_indexadas,
        hash_corpus_obsidian,
    )
    from src.conhecimento.obsidian import (
        sincronizar_obsidian as sincronizar,
    )
    from src.core.config import (
        ARQUIVO_INDICE_OBSIDIAN,
        MODELO_EMBEDDINGS,
        NOME_COLECAO_OBSIDIAN,
        PASTA_CHROMADB,
        PASTA_VAULT_OBSIDIAN,
    )

    raiz = Path(vault or PASTA_VAULT_OBSIDIAN).expanduser().resolve()
    if not raiz.is_dir():
        print(f"Vault nao encontrado: {raiz}")
        return 1
    modelo = criar_modelo_embeddings(modo_consulta=False)
    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = cliente.get_or_create_collection(
        NOME_COLECAO_OBSIDIAN,
        metadata={"hnsw:space": "cosine"},
    )
    estado = sincronizar(colecao, modelo, raiz=raiz)
    manifesto = exportar_colecao(
        colecao,
        ARQUIVO_INDICE_OBSIDIAN,
        modelo_embeddings=MODELO_EMBEDDINGS,
        hash_corpus=hash_corpus_obsidian(raiz),
        n_documentos=contar_notas_indexadas(colecao),
        schema_version=1,
    )
    print(json.dumps({"estado": estado, "snapshot": manifesto}, ensure_ascii=False, indent=2))
    return int(bool(estado.get("erros")))


def verificar_autores() -> int:
    import chromadb

    from src.conhecimento.agente import buscar_contexto
    from src.conhecimento.embeddings import criar_modelo_embeddings
    from src.core.config import NOME_COLECAO, PASTA_CHROMADB

    modelo = criar_modelo_embeddings(modo_consulta=True)
    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = cliente.get_or_create_collection(NOME_COLECAO)
    metadados = colecao.get(include=["metadatas"]).get("metadatas", [])
    arquivos = sorted({str(item.get("arquivo", "")) for item in metadados if item.get("arquivo")})
    falhas: list[dict] = []
    for arquivo in arquivos:
        sobrenome = arquivo.split("_", 1)[0].replace("-", " ")
        _, citacoes = buscar_contexto(
            f"o que {sobrenome} diz sobre o tema da dissertacao?",
            modelo,
            colecao,
            n_pool=120,
            n_resultados=16,
            n_resultados_revisao=28,
            max_chunks_por_fonte=2,
            contexto_chars=14_000,
            sessao_chars=1_500,
            consultar_literatura=True,
        )
        if arquivo not in citacoes:
            falhas.append({"arquivo": arquivo, "recuperados": list(citacoes)[:3]})
    print(
        json.dumps(
            {"documentos": len(arquivos), "recuperados": len(arquivos) - len(falhas), "falhas": falhas},
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(bool(falhas))


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("reconstruir-literatura", help="recria a colecao e o snapshot")
    sub.add_parser("exportar-literatura", help="atualiza somente o snapshot portatil")
    sub.add_parser(
        "migrar-snapshot-v2",
        help="migra atomicamente o snapshot de literatura vigente para o schema v2",
    )
    sub.add_parser("reindexar-sessoes", help="reindexa sessoes e memorias Markdown")
    obsidian = sub.add_parser("sincronizar-obsidian", help="sincroniza o vault e seu snapshot")
    obsidian.add_argument("--vault", type=Path)
    sub.add_parser("verificar-autores", help="testa recuperacao bibliografica por autor")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    if args.comando == "reconstruir-literatura":
        return reconstruir_literatura()
    if args.comando == "exportar-literatura":
        return exportar_literatura()
    if args.comando == "migrar-snapshot-v2":
        return migrar_snapshot_v2()
    if args.comando == "reindexar-sessoes":
        return reindexar_sessoes()
    if args.comando == "sincronizar-obsidian":
        return sincronizar_obsidian(args.vault)
    return verificar_autores()


if __name__ == "__main__":
    raise SystemExit(main())
