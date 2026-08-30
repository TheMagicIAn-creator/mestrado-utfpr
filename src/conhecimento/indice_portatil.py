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
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2
SCHEMAS_COMPATIVEIS = {1, SCHEMA_VERSION}
TIPO_MANIFESTO = "manifesto_indice_portatil"
TIPO_CHUNK = "chunk_indice_portatil"
TIPOS_MANIFESTO_COMPATIVEIS = {TIPO_MANIFESTO, "manifesto_indice_literatura"}
TIPOS_CHUNK_COMPATIVEIS = {TIPO_CHUNK, "chunk_literatura"}
ESTRATEGIA_TEXTO_R2 = "identity_raw_text"
ESTRATEGIA_DOCUMENT_ID = "doc:sha256"
ESTRATEGIA_CHUNK_ID = "sha256__chunk_ordinal_v1"


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


def _erro_registro(mensagem: str, numero_linha: int | None = None) -> None:
    sufixo = f" na linha {numero_linha}" if numero_linha is not None else ""
    raise IndicePortatilInvalido(f"{mensagem}{sufixo}.")


def _hash_documento(metadata: dict) -> str:
    return str(
        metadata.get("arquivo_sha256") or metadata.get("arquivo_hash") or ""
    ).strip().lower()


def _id_vizinho(
    chunk_id: str,
    indice: int,
    total: int,
    deslocamento: int,
) -> str | None:
    alvo = indice + deslocamento
    if alvo < 0 or alvo >= total or "__chunk_" not in chunk_id:
        return None
    prefixo, sufixo = chunk_id.rsplit("__chunk_", 1)
    if not sufixo.isdigit():
        return None
    return f"{prefixo}__chunk_{alvo:05d}"


def _normalizar_metadados_v2(
    metadata: dict,
    *,
    chunk_id: str,
    raw_text: str,
) -> dict:
    normalizados = dict(metadata or {})
    arquivo_sha256 = _hash_documento(normalizados)
    if arquivo_sha256:
        normalizados["arquivo_sha256"] = arquivo_sha256
        normalizados.setdefault("arquivo_hash", arquivo_sha256)

    pagina_inicio = int(
        normalizados.get("pagina_inicio") or normalizados.get("pagina") or 0
    )
    pagina_fim = int(normalizados.get("pagina_fim") or pagina_inicio)
    normalizados["pagina"] = pagina_inicio
    normalizados["pagina_inicio"] = pagina_inicio
    normalizados["pagina_fim"] = pagina_fim
    normalizados["rotulo_pagina"] = str(
        normalizados.get("rotulo_pagina")
        or normalizados.get("pagina_rotulo")
        or ""
    )
    normalizados.setdefault("pagina_rotulo", normalizados["rotulo_pagina"])

    indice = int(normalizados.get("chunk_index", 0))
    total = int(normalizados.get("total_chunks", 0))
    normalizados["chunk_index"] = indice
    normalizados["prev_chunk_id"] = normalizados.get("prev_chunk_id") or _id_vizinho(
        chunk_id, indice, total, -1
    )
    normalizados["next_chunk_id"] = normalizados.get("next_chunk_id") or _id_vizinho(
        chunk_id, indice, total, 1
    )

    autores = normalizados.get("autores")
    if isinstance(autores, list):
        normalizados["autores"] = [str(item).strip() for item in autores if str(item).strip()]
    else:
        autor = str(normalizados.get("autor") or "").strip()
        normalizados["autores"] = [autor] if autor else []
    normalizados["idioma"] = str(normalizados.get("idioma") or "desconhecido")
    normalizados["tipo_conteudo"] = (
        "tabela" if raw_text.lstrip().startswith("[TABELA") else "texto"
    )
    return normalizados


def _registro_v2(registro: dict) -> dict:
    chunk_id = str(registro.get("chunk_id") or registro.get("id") or "")
    raw_text = str(registro.get("raw_text", registro.get("documento", "")))
    retrieval_text = str(registro.get("retrieval_text", raw_text))
    metadata = _normalizar_metadados_v2(
        dict(registro.get("metadata") or {}),
        chunk_id=chunk_id,
        raw_text=raw_text,
    )
    arquivo_sha256 = _hash_documento(metadata)
    return {
        "tipo": TIPO_CHUNK,
        "schema_version": SCHEMA_VERSION,
        "id": chunk_id,
        "document_id": str(
            registro.get("document_id") or f"doc:{arquivo_sha256}"
        ),
        "chunk_id": chunk_id,
        "raw_text": raw_text,
        "retrieval_text": retrieval_text,
        "metadata": metadata,
        "embedding": registro.get("embedding") or [],
    }


def _registro_v1(registro: dict) -> dict:
    return {
        "tipo": TIPO_CHUNK,
        "id": str(registro.get("id") or registro.get("chunk_id") or ""),
        "documento": str(
            registro.get("documento", registro.get("retrieval_text", ""))
        ),
        "metadata": dict(registro.get("metadata") or {}),
        "embedding": registro.get("embedding") or [],
    }


def _validar_registro_v1(registro: dict, numero_linha: int | None) -> None:
    if not registro.get("id") or not isinstance(registro.get("documento"), str):
        _erro_registro("Chunk v1 incompleto", numero_linha)
    if not isinstance(registro.get("metadata"), dict):
        _erro_registro("Metadados v1 inválidos", numero_linha)
    if not isinstance(registro.get("embedding"), list):
        _erro_registro("Embedding v1 inválido", numero_linha)


def _validar_textos_v2(
    registro: dict,
    numero_linha: int | None,
    estrategia_texto: str | None,
) -> None:
    raw_text = registro.get("raw_text")
    retrieval_text = registro.get("retrieval_text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        _erro_registro("raw_text ausente", numero_linha)
    if not isinstance(retrieval_text, str) or not retrieval_text.strip():
        _erro_registro("retrieval_text ausente", numero_linha)
    if estrategia_texto == ESTRATEGIA_TEXTO_R2 and retrieval_text != raw_text:
        _erro_registro("R2 exige retrieval_text idêntico a raw_text", numero_linha)


def _validar_identidade_v2(registro: dict, numero_linha: int | None) -> None:
    chunk_id = str(registro.get("chunk_id") or "")
    if not chunk_id or str(registro.get("id") or "") != chunk_id:
        _erro_registro("IDs de chunk ausentes ou divergentes", numero_linha)
    metadata = registro.get("metadata")
    if not isinstance(metadata, dict):
        _erro_registro("Metadados v2 inválidos", numero_linha)
    arquivo_sha256 = _hash_documento(metadata)
    if len(arquivo_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in arquivo_sha256
    ):
        _erro_registro("SHA-256 documental inválido", numero_linha)
    if registro.get("document_id") != f"doc:{arquivo_sha256}":
        _erro_registro("document_id divergente do SHA-256", numero_linha)
    indice = int(metadata.get("chunk_index", -1))
    esperado = f"{arquivo_sha256}__chunk_{indice:05d}"
    if indice < 0 or chunk_id != esperado:
        _erro_registro("chunk_id não segue a identidade determinística", numero_linha)


def _validar_metadados_v2(metadata: dict, numero_linha: int | None) -> None:
    if not str(metadata.get("arquivo") or "").strip():
        _erro_registro("Arquivo de origem ausente", numero_linha)
    if int(metadata.get("pagina_inicio") or 0) <= 0:
        _erro_registro("Página física ausente", numero_linha)
    if "prev_chunk_id" not in metadata or "next_chunk_id" not in metadata:
        _erro_registro("IDs de vizinhança ausentes", numero_linha)
    if not str(metadata.get("idioma") or "").strip():
        _erro_registro("Idioma ausente", numero_linha)


def validar_registro(
    registro: dict,
    schema_version: int,
    *,
    numero_linha: int | None = None,
    estrategia_texto: str | None = None,
) -> None:
    """Valida um chunk sem reinterpretar seu conteúdo científico."""
    if registro.get("tipo") not in TIPOS_CHUNK_COMPATIVEIS:
        _erro_registro("Tipo de registro desconhecido", numero_linha)
    if schema_version == 1:
        _validar_registro_v1(registro, numero_linha)
        return
    if schema_version != SCHEMA_VERSION:
        _erro_registro(f"Schema de chunk incompatível: {schema_version}", numero_linha)
    if int(registro.get("schema_version", 0)) != SCHEMA_VERSION:
        _erro_registro("Chunk sem marcação de schema v2", numero_linha)

    _validar_textos_v2(registro, numero_linha, estrategia_texto)
    _validar_identidade_v2(registro, numero_linha)
    _validar_metadados_v2(registro["metadata"], numero_linha)
    if not isinstance(registro.get("embedding"), list) or not registro["embedding"]:
        _erro_registro("Embedding v2 inválido", numero_linha)


def _atualizar_hash_conteudo(digest, registro: dict) -> None:
    payload = {
        "chunk_id": registro.get("chunk_id") or registro.get("id"),
        "raw_text": registro.get("raw_text", registro.get("documento", "")),
        "retrieval_text": registro.get(
            "retrieval_text", registro.get("documento", "")
        ),
        "embedding": registro.get("embedding") or [],
    }
    digest.update(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _manifesto_v2(manifesto: dict, hash_conteudo: str) -> dict:
    atualizado = {
        **manifesto,
        "tipo": TIPO_MANIFESTO,
        "schema_version": SCHEMA_VERSION,
        "document_id_strategy": ESTRATEGIA_DOCUMENT_ID,
        "chunk_id_strategy": ESTRATEGIA_CHUNK_ID,
        "raw_text_field": "raw_text",
        "retrieval_text_field": "retrieval_text",
        "retrieval_text_strategy": ESTRATEGIA_TEXTO_R2,
        "embedding_input_field": "retrieval_text",
        "hash_conteudo_retrieval_sha256": hash_conteudo,
    }
    return atualizado


def _metadata_para_chroma(metadata: dict) -> dict:
    """Remove estruturas que o contrato escalar do ChromaDB não aceita."""
    saida = {
        chave: valor
        for chave, valor in metadata.items()
        if isinstance(valor, (str, int, float, bool))
    }
    autores = metadata.get("autores")
    if "autor" not in saida and isinstance(autores, list):
        nomes = [str(item).strip() for item in autores if str(item).strip()]
        if nomes:
            saida["autor"] = "; ".join(nomes)
    return saida


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
    schema_version = int(manifesto.get("schema_version", 0))
    if schema_version not in SCHEMAS_COMPATIVEIS:
        raise IndicePortatilInvalido(
            f"Schema {manifesto.get('schema_version')} incompatível; "
            f"aceitos {sorted(SCHEMAS_COMPATIVEIS)}."
        )
    if int(manifesto.get("n_chunks", 0)) <= 0:
        raise IndicePortatilInvalido("Snapshot sem chunks declarados.")
    if schema_version == SCHEMA_VERSION:
        obrigatorios = {
            "document_id_strategy": ESTRATEGIA_DOCUMENT_ID,
            "chunk_id_strategy": ESTRATEGIA_CHUNK_ID,
            "raw_text_field": "raw_text",
            "retrieval_text_field": "retrieval_text",
            "embedding_input_field": "retrieval_text",
        }
        divergentes = [
            chave
            for chave, valor in obrigatorios.items()
            if manifesto.get(chave) != valor
        ]
        if divergentes:
            raise IndicePortatilInvalido(
                "Manifesto v2 incompleto: " + ", ".join(divergentes) + "."
            )
        if not str(manifesto.get("retrieval_text_strategy") or "").strip():
            raise IndicePortatilInvalido(
                "Manifesto v2 sem estratégia de retrieval_text."
            )
        hash_conteudo = str(manifesto.get("hash_conteudo_retrieval_sha256") or "")
        if len(hash_conteudo) != 64:
            raise IndicePortatilInvalido("Manifesto v2 sem hash de conteúdo válido.")
    return manifesto


def validar_snapshot(caminho: Path) -> dict:
    """Valida todos os registros, unicidade e hash sem carregar o índice em RAM."""
    caminho = Path(caminho)
    manifesto = ler_manifesto(caminho)
    schema_version = int(manifesto["schema_version"])
    ids: set[str] = set()
    digest = hashlib.sha256()
    lidos = 0

    with gzip.open(caminho, "rt", encoding="utf-8") as arquivo:
        next(arquivo)
        for numero_linha, linha in enumerate(arquivo, 2):
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError as exc:
                raise IndicePortatilInvalido(
                    f"JSON inválido na linha {numero_linha}: {exc}"
                ) from exc
            validar_registro(
                registro,
                schema_version,
                numero_linha=numero_linha,
                estrategia_texto=manifesto.get("retrieval_text_strategy"),
            )
            chunk_id = str(registro.get("chunk_id") or registro.get("id"))
            if chunk_id in ids:
                raise IndicePortatilInvalido(
                    f"Chunk duplicado na linha {numero_linha}: {chunk_id}."
                )
            ids.add(chunk_id)
            _atualizar_hash_conteudo(digest, registro)
            lidos += 1

    esperado = int(manifesto["n_chunks"])
    if lidos != esperado:
        raise IndicePortatilInvalido(
            f"Snapshot incompleto: {lidos}/{esperado} chunks."
        )
    hash_conteudo = digest.hexdigest()
    if schema_version == SCHEMA_VERSION and hash_conteudo != manifesto.get(
        "hash_conteudo_retrieval_sha256"
    ):
        raise IndicePortatilInvalido("Hash de conteúdo do snapshot v2 divergente.")
    return {
        **manifesto,
        "chunks_validados": lidos,
        "ids_unicos": len(ids),
        "hash_conteudo_retrieval_sha256": hash_conteudo,
        "arquivo_sha256": _sha256(caminho),
    }


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


def _decodificar_registro(linha: str, numero_linha: int) -> dict:
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
    return registro


def _aplicar_atualizacao_metadados(
    registro: dict,
    novos: dict | None,
    *,
    schema_version: int,
    numero_linha: int,
    estrategia_texto: str | None,
) -> tuple[dict, bool]:
    if not novos:
        return registro, False

    metadata = registro.get("metadata") or {}
    metadata_atualizada = {**metadata, **novos}
    if "autor" in novos and "autores" not in novos:
        metadata_atualizada.pop("autores", None)
    registro["metadata"] = metadata_atualizada
    if schema_version == SCHEMA_VERSION:
        registro = _registro_v2(registro)
        validar_registro(
            registro,
            schema_version,
            numero_linha=numero_linha,
            estrategia_texto=estrategia_texto,
        )
    return registro, True


def _reescrever_registros_snapshot(
    origem,
    destino,
    *,
    atualizacoes: dict[str, dict],
    schema_version: int,
    estrategia_texto: str | None,
) -> tuple[int, int, set[str]]:
    chunks_lidos = 0
    chunks_atualizados = 0
    documentos_atualizados: set[str] = set()
    for numero_linha, linha in enumerate(origem, 2):
        registro = _decodificar_registro(linha, numero_linha)
        validar_registro(
            registro,
            schema_version,
            numero_linha=numero_linha,
            estrategia_texto=estrategia_texto,
        )
        nome_antigo = str((registro.get("metadata") or {}).get("arquivo", ""))
        registro, atualizado = _aplicar_atualizacao_metadados(
            registro,
            atualizacoes.get(nome_antigo),
            schema_version=schema_version,
            numero_linha=numero_linha,
            estrategia_texto=estrategia_texto,
        )
        if atualizado:
            chunks_atualizados += 1
            documentos_atualizados.add(nome_antigo)
        destino.write(_linha_json(registro))
        chunks_lidos += 1
    return chunks_lidos, chunks_atualizados, documentos_atualizados


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
    schema_version = int(manifesto["schema_version"])
    temporario = caminho.with_name(caminho.name + ".tmp")

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
            (
                chunks_lidos,
                chunks_atualizados,
                documentos_atualizados,
            ) = _reescrever_registros_snapshot(
                origem,
                destino,
                atualizacoes=atualizacoes,
                schema_version=schema_version,
                estrategia_texto=manifesto.get("retrieval_text_strategy"),
            )

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


def _extrair_lote_exportacao(colecao, offset: int, tamanho_lote: int) -> tuple:
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
    tamanhos = (len(ids), len(documentos), len(metadados), len(embeddings))
    if len(set(tamanhos)) != 1:
        raise IndicePortatilInvalido(
            f"Lote inconsistente no offset {offset}: "
            f"ids={tamanhos[0]}, docs={tamanhos[1]}, "
            f"metadados={tamanhos[2]}, embeddings={tamanhos[3]}."
        )
    return zip(ids, documentos, metadados, embeddings)


def _preparar_registro_exportacao(
    chunk_id,
    documento,
    metadata,
    embedding,
    *,
    schema_version: int,
) -> dict:
    vetor = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    registro_base = {
        "tipo": TIPO_CHUNK,
        "id": chunk_id,
        "documento": documento,
        "metadata": metadata,
        "embedding": vetor,
    }
    registro = (
        _registro_v2(registro_base)
        if schema_version == SCHEMA_VERSION
        else _registro_v1(registro_base)
    )
    validar_registro(
        registro,
        schema_version,
        estrategia_texto=(
            ESTRATEGIA_TEXTO_R2 if schema_version == SCHEMA_VERSION else None
        ),
    )
    return registro


def _exportar_registros_colecao(
    colecao,
    registros,
    digest,
    *,
    n_chunks: int,
    tamanho_lote: int,
    schema_version: int,
) -> int:
    escritos = 0
    for offset in range(0, n_chunks, tamanho_lote):
        lote = _extrair_lote_exportacao(colecao, offset, tamanho_lote)
        for chunk_id, documento, metadata, embedding in lote:
            registro = _preparar_registro_exportacao(
                chunk_id,
                documento,
                metadata,
                embedding,
                schema_version=schema_version,
            )
            registros.write(_linha_json(registro))
            _atualizar_hash_conteudo(digest, registro)
            escritos += 1
    return escritos


def exportar_colecao(
    colecao,
    destino: Path,
    *,
    modelo_embeddings: str,
    hash_corpus: str,
    n_documentos: int,
    tamanho_lote: int = 250,
    schema_version: int = SCHEMA_VERSION,
) -> dict:
    """Exporta uma coleção Chroma para JSONL gzip de forma atômica."""
    if schema_version not in SCHEMAS_COMPATIVEIS:
        raise IndicePortatilInvalido(
            f"Schema de exportação inválido: {schema_version}."
        )
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    n_chunks = int(colecao.count())
    if n_chunks <= 0:
        raise IndicePortatilInvalido("A coleção de literatura está vazia.")

    manifesto_base = {
        "tipo": TIPO_MANIFESTO,
        "schema_version": schema_version,
        "colecao": getattr(colecao, "name", "literatura_pv"),
        "modelo_embeddings": modelo_embeddings,
        "n_documentos": int(n_documentos),
        "n_chunks": n_chunks,
        "hash_corpus_sha256": hash_corpus,
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
    }

    digest = hashlib.sha256()
    try:
        with tempfile.SpooledTemporaryFile(
            mode="w+t",
            encoding="utf-8",
            newline="\n",
            max_size=8 * 1024 * 1024,
        ) as registros:
            escritos = _exportar_registros_colecao(
                colecao,
                registros,
                digest,
                n_chunks=n_chunks,
                tamanho_lote=tamanho_lote,
                schema_version=schema_version,
            )
            if escritos != n_chunks:
                raise IndicePortatilInvalido(
                    f"Exportação incompleta: {escritos}/{n_chunks} chunks."
                )
            manifesto = (
                _manifesto_v2(manifesto_base, digest.hexdigest())
                if schema_version == SCHEMA_VERSION
                else manifesto_base
            )
            registros.seek(0)
            with gzip.open(
                temporario, "wt", encoding="utf-8", compresslevel=6
            ) as arquivo:
                arquivo.write(_linha_json(manifesto))
                shutil.copyfileobj(registros, arquivo)
        os.replace(temporario, destino)
    finally:
        temporario.unlink(missing_ok=True)

    return {
        **manifesto,
        "arquivo": str(destino),
        "tamanho_bytes": destino.stat().st_size,
        "arquivo_sha256": _sha256(destino),
    }


def migrar_snapshot_v2(origem: Path, destino: Path | None = None) -> dict:
    """Migra o JSONL existente para v2 sem recalcular texto ou embeddings."""
    origem = Path(origem)
    destino = Path(destino) if destino is not None else origem
    manifesto_origem = ler_manifesto(origem)
    schema_origem = int(manifesto_origem["schema_version"])
    if schema_origem == SCHEMA_VERSION:
        validacao = validar_snapshot(origem)
        if origem.resolve() != destino.resolve():
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origem, destino)
            validacao = validar_snapshot(destino)
        return {
            **validacao,
            "hash_conteudo_origem_sha256": validacao[
                "hash_conteudo_retrieval_sha256"
            ],
            "conteudo_preservado": True,
            "ja_estava_pronto": True,
        }

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    escritos = 0
    digest_origem = hashlib.sha256()
    digest = hashlib.sha256()
    try:
        with tempfile.SpooledTemporaryFile(
            mode="w+t",
            encoding="utf-8",
            newline="\n",
            max_size=8 * 1024 * 1024,
        ) as registros, gzip.open(origem, "rt", encoding="utf-8") as arquivo:
            next(arquivo)
            for numero_linha, linha in enumerate(arquivo, 2):
                try:
                    registro_origem = json.loads(linha)
                except json.JSONDecodeError as exc:
                    raise IndicePortatilInvalido(
                        f"JSON inválido na linha {numero_linha}: {exc}"
                    ) from exc
                validar_registro(
                    registro_origem,
                    schema_origem,
                    numero_linha=numero_linha,
                    estrategia_texto=manifesto_origem.get(
                        "retrieval_text_strategy"
                    ),
                )
                _atualizar_hash_conteudo(digest_origem, registro_origem)
                registro_v2 = _registro_v2(registro_origem)
                validar_registro(
                    registro_v2,
                    SCHEMA_VERSION,
                    numero_linha=numero_linha,
                    estrategia_texto=ESTRATEGIA_TEXTO_R2,
                )
                registros.write(_linha_json(registro_v2))
                _atualizar_hash_conteudo(digest, registro_v2)
                escritos += 1

            esperado = int(manifesto_origem["n_chunks"])
            if escritos != esperado:
                raise IndicePortatilInvalido(
                    f"Migração incompleta: {escritos}/{esperado} chunks."
                )
            hash_origem = digest_origem.hexdigest()
            hash_destino = digest.hexdigest()
            if hash_origem != hash_destino:
                raise IndicePortatilInvalido(
                    "A migração alterou texto, retrieval_text ou embeddings."
                )
            manifesto_v2 = _manifesto_v2(
                {
                    **manifesto_origem,
                    "migrado_de_schema_version": schema_origem,
                    "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
                },
                hash_destino,
            )
            registros.seek(0)
            with gzip.open(
                temporario, "wt", encoding="utf-8", compresslevel=6
            ) as arquivo_destino:
                arquivo_destino.write(_linha_json(manifesto_v2))
                shutil.copyfileobj(registros, arquivo_destino)
        os.replace(temporario, destino)
    finally:
        temporario.unlink(missing_ok=True)

    validacao = validar_snapshot(destino)
    return {
        **validacao,
        "migrado_de_schema_version": schema_origem,
        "hash_conteudo_origem_sha256": hash_origem,
        "conteudo_preservado": True,
        "ja_estava_pronto": False,
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
            schema_version = int(manifesto["schema_version"])
            validar_registro(
                item,
                schema_version,
                numero_linha=numero_linha,
                estrategia_texto=manifesto.get("retrieval_text_strategy"),
            )
            chunk_id = str(item.get("chunk_id") or item["id"])
            ids_snapshot.add(chunk_id)
            if chunk_id in ids_antes:
                continue
            ids.append(chunk_id)
            documentos.append(
                str(
                    item.get("retrieval_text", item.get("documento", ""))
                )
            )
            metadados.append(_metadata_para_chroma(dict(item["metadata"])))
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
