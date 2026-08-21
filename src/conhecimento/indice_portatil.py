"""Snapshot portátil de coleções curadas do ChromaDB.

O diretório interno do ChromaDB não é um artefato adequado para Git: ele
acumula segmentos, mistura coleções locais e depende da implementação do
banco. Este módulo exporta somente ids, chunks, metadados e embeddings para
JSONL comprimido, permitindo restauração determinística em implantação ASGI.
O schema antigo da literatura continua aceito para preservar os snapshots.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
TIPO_MANIFESTO = "manifesto_indice_portatil"
TIPO_CHUNK = "chunk_indice_portatil"
TIPOS_MANIFESTO_COMPATIVEIS = {TIPO_MANIFESTO, "manifesto_indice_literatura"}
TIPOS_CHUNK_COMPATIVEIS = {TIPO_CHUNK, "chunk_literatura"}


class IndicePortatilInvalido(ValueError):
    """Indica snapshot ausente, incompatível ou incompleto."""


def _linha_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def ler_manifesto(caminho: Path) -> dict:
    """Lê e valida o cabeçalho do snapshot sem descompactá-lo por inteiro."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise IndicePortatilInvalido(f"Snapshot não encontrado: {caminho}")

    try:
        with gzip.open(caminho, "rt", encoding="utf-8") as arquivo:
            manifesto = json.loads(arquivo.readline())
    except (OSError, json.JSONDecodeError) as exc:
        raise IndicePortatilInvalido(f"Snapshot ilegível: {exc}") from exc

    if manifesto.get("tipo") not in TIPOS_MANIFESTO_COMPATIVEIS:
        raise IndicePortatilInvalido("Cabeçalho do snapshot não reconhecido.")
    if manifesto.get("schema_version") != SCHEMA_VERSION:
        raise IndicePortatilInvalido(
            f"Schema {manifesto.get('schema_version')} incompatível; "
            f"esperado {SCHEMA_VERSION}."
        )
    if int(manifesto.get("n_chunks", 0)) <= 0:
        raise IndicePortatilInvalido("Snapshot sem chunks declarados.")
    return manifesto


def hash_corpus_pdfs(raiz: Path) -> tuple[str, int]:
    """Calcula a identidade do corpus usando caminho relativo e SHA-256."""
    raiz = Path(raiz)
    registros = [
        {
            "arquivo": pdf.relative_to(raiz).as_posix(),
            "sha256": _sha256(pdf),
        }
        for pdf in sorted(raiz.rglob("*.pdf"))
    ]
    serializado = json.dumps(
        registros, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(serializado).hexdigest(), len(registros)


def atualizar_metadados_snapshot(
    caminho: Path,
    atualizacoes: dict[str, dict],
    *,
    hash_corpus: str | None = None,
    n_documentos: int | None = None,
) -> dict:
    """Atualiza somente metadados de chunks, preservando texto e embeddings.

    As chaves de ``atualizacoes`` são nomes antigos de PDF. A reescrita é
    streaming e atômica, adequada ao snapshot versionado de dezenas de MB.
    """
    caminho = Path(caminho)
    manifesto = ler_manifesto(caminho)
    temporario = caminho.with_name(caminho.name + ".tmp")
    chunks_lidos = 0
    chunks_atualizados = 0
    documentos_atualizados: set[str] = set()

    try:
        with gzip.open(caminho, "rt", encoding="utf-8") as origem, gzip.open(
            temporario, "wt", encoding="utf-8", compresslevel=6
        ) as destino:
            cabecalho = json.loads(origem.readline())
            if hash_corpus is not None:
                cabecalho["hash_corpus_sha256"] = str(hash_corpus)
            if n_documentos is not None:
                cabecalho["n_documentos"] = int(n_documentos)
            cabecalho["gerado_em_utc"] = datetime.now(timezone.utc).isoformat()
            destino.write(_linha_json(cabecalho))

            for numero_linha, linha in enumerate(origem, 2):
                try:
                    registro = json.loads(linha)
                except json.JSONDecodeError as exc:
                    raise IndicePortatilInvalido(
                        f"JSON inválido na linha {numero_linha}: {exc}"
                    ) from exc
                if registro.get("tipo") not in TIPOS_CHUNK_COMPATIVEIS:
                    raise IndicePortatilInvalido(
                        f"Registro desconhecido na linha {numero_linha}."
                    )

                chunks_lidos += 1
                metadata = registro.get("metadata") or {}
                nome_antigo = str(metadata.get("arquivo", ""))
                novos = atualizacoes.get(nome_antigo)
                if novos:
                    registro["metadata"] = {**metadata, **novos}
                    chunks_atualizados += 1
                    documentos_atualizados.add(nome_antigo)
                destino.write(_linha_json(registro))

        esperado = int(manifesto["n_chunks"])
        if chunks_lidos != esperado:
            raise IndicePortatilInvalido(
                f"Reescrita incompleta: {chunks_lidos}/{esperado} chunks."
            )
        os.replace(temporario, caminho)
    finally:
        temporario.unlink(missing_ok=True)

    return {
        "chunks_atualizados": chunks_atualizados,
        "documentos_atualizados": len(documentos_atualizados),
        "arquivo_sha256": _sha256(caminho),
    }


def exportar_colecao(
    colecao,
    destino: Path,
    *,
    modelo_embeddings: str,
    hash_corpus: str,
    n_documentos: int,
    tamanho_lote: int = 250,
) -> dict:
    """Exporta uma coleção Chroma para JSONL gzip de forma atômica."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    n_chunks = int(colecao.count())
    if n_chunks <= 0:
        raise IndicePortatilInvalido("A coleção de literatura está vazia.")

    manifesto = {
        "tipo": TIPO_MANIFESTO,
        "schema_version": SCHEMA_VERSION,
        "colecao": getattr(colecao, "name", "literatura_pv"),
        "modelo_embeddings": modelo_embeddings,
        "n_documentos": int(n_documentos),
        "n_chunks": n_chunks,
        "hash_corpus_sha256": hash_corpus,
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
    }

    escritos = 0
    try:
        with gzip.open(temporario, "wt", encoding="utf-8", compresslevel=6) as arquivo:
            arquivo.write(_linha_json(manifesto))
            for offset in range(0, n_chunks, tamanho_lote):
                lote = colecao.get(
                    limit=tamanho_lote,
                    offset=offset,
                    include=["documents", "metadatas", "embeddings"],
                )
                ids = lote.get("ids") or []
                documentos = lote.get("documents") or []
                metadados = lote.get("metadatas") or []
                embeddings = lote.get("embeddings")
                if embeddings is None:
                    embeddings = []

                if not (len(ids) == len(documentos) == len(metadados) == len(embeddings)):
                    raise IndicePortatilInvalido(
                        f"Lote inconsistente no offset {offset}: "
                        f"ids={len(ids)}, docs={len(documentos)}, "
                        f"metadados={len(metadados)}, embeddings={len(embeddings)}."
                    )

                for chunk_id, documento, metadata, embedding in zip(
                    ids, documentos, metadados, embeddings
                ):
                    vetor = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                    arquivo.write(_linha_json({
                        "tipo": TIPO_CHUNK,
                        "id": chunk_id,
                        "documento": documento,
                        "metadata": metadata,
                        "embedding": vetor,
                    }))
                    escritos += 1

        if escritos != n_chunks:
            raise IndicePortatilInvalido(
                f"Exportação incompleta: {escritos}/{n_chunks} chunks."
            )
        os.replace(temporario, destino)
    finally:
        temporario.unlink(missing_ok=True)

    return {
        **manifesto,
        "arquivo": str(destino),
        "tamanho_bytes": destino.stat().st_size,
        "arquivo_sha256": _sha256(destino),
    }


def _ids_existentes(colecao, tamanho_lote: int) -> set[str]:
    """Lê os IDs atuais em lotes, sem carregar documentos ou embeddings."""
    total = int(colecao.count())
    existentes: set[str] = set()
    for offset in range(0, total, tamanho_lote):
        lote = colecao.get(
            limit=tamanho_lote,
            offset=offset,
            include=["metadatas"],
        )
        existentes.update(str(item) for item in (lote.get("ids") or []))
    return existentes


def importar_colecao(
    colecao,
    origem: Path,
    *,
    tamanho_lote: int = 250,
    mesclar: bool = False,
) -> dict:
    """Restaura um snapshot sob o mesmo lock das demais escritas."""
    from src.conhecimento.index_lock import lock_indexacao

    with lock_indexacao():
        return _importar_colecao_sem_lock(
            colecao,
            origem,
            tamanho_lote=tamanho_lote,
            mesclar=mesclar,
        )


def _importar_colecao_sem_lock(
    colecao,
    origem: Path,
    *,
    tamanho_lote: int = 250,
    mesclar: bool = False,
) -> dict:
    """Restaura ou mescla um snapshot e valida todos os IDs declarados.

    O modo estrito continua exigindo uma coleção vazia. ``mesclar=True`` é
    destinado a coleções de memória: preserva registros criados em runtime e
    importa apenas os chunks históricos ausentes do snapshot.
    """
    origem = Path(origem)
    manifesto = ler_manifesto(origem)
    esperado = int(manifesto["n_chunks"])
    existentes = int(colecao.count())
    if existentes == esperado and not mesclar:
        return {**manifesto, "importados": 0, "ja_estava_pronto": True}
    if existentes and not mesclar:
        raise IndicePortatilInvalido(
            f"Coleção parcialmente preenchida ({existentes}/{esperado}); "
            "a restauração automática exige coleção vazia."
        )

    ids_antes = _ids_existentes(colecao, tamanho_lote) if mesclar else set()
    ids_snapshot: set[str] = set()
    ids: list[str] = []
    documentos: list[str] = []
    metadados: list[dict] = []
    embeddings: list[list[float]] = []
    importados = 0

    def enviar_lote() -> None:
        nonlocal importados
        if not ids:
            return
        colecao.upsert(
            ids=list(ids),
            documents=list(documentos),
            metadatas=list(metadados),
            embeddings=list(embeddings),
        )
        importados += len(ids)
        ids.clear()
        documentos.clear()
        metadados.clear()
        embeddings.clear()

    with gzip.open(origem, "rt", encoding="utf-8") as arquivo:
        next(arquivo)  # manifesto já validado
        for numero_linha, linha in enumerate(arquivo, 2):
            try:
                item = json.loads(linha)
            except json.JSONDecodeError as exc:
                raise IndicePortatilInvalido(
                    f"JSON inválido na linha {numero_linha}."
                ) from exc
            if item.get("tipo") not in TIPOS_CHUNK_COMPATIVEIS:
                raise IndicePortatilInvalido(
                    f"Registro inesperado na linha {numero_linha}."
                )
            chunk_id = str(item["id"])
            ids_snapshot.add(chunk_id)
            if chunk_id in ids_antes:
                continue
            ids.append(chunk_id)
            documentos.append(str(item["documento"]))
            metadados.append(dict(item["metadata"]))
            embeddings.append([float(v) for v in item["embedding"]])
            if len(ids) >= tamanho_lote:
                enviar_lote()
        enviar_lote()

    total_final = int(colecao.count())
    if len(ids_snapshot) != esperado:
        raise IndicePortatilInvalido(
            f"Snapshot incompleto: IDs únicos={len(ids_snapshot)}, esperado={esperado}."
        )
    if mesclar:
        ausentes = ids_snapshot - _ids_existentes(colecao, tamanho_lote)
        if ausentes:
            raise IndicePortatilInvalido(
                f"Mesclagem incompleta: {len(ausentes)} chunks continuam ausentes."
            )
    elif importados != esperado or total_final != esperado:
        raise IndicePortatilInvalido(
            f"Restauração incompleta: importados={importados}, "
            f"coleção={total_final}, esperado={esperado}."
        )
    return {
        **manifesto,
        "importados": importados,
        "preservados": len(ids_antes - ids_snapshot) if mesclar else 0,
        "ja_estava_pronto": importados == 0,
    }
