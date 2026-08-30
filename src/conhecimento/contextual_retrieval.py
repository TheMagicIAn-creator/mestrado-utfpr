"""Build a deterministic contextual candidate without changing cited text."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.conhecimento.embeddings import REPOSITORIO_MODELO, REVISAO_MODELO
from src.conhecimento.indice_portatil import (
    ESTRATEGIA_TEXTO_R2,
    SCHEMA_VERSION,
    atualizar_hash_conteudo,
    criar_manifesto_v2,
    ler_manifesto,
    validar_registro,
    validar_snapshot,
)

ESTRATEGIA_CONTEXTO_R3 = "deterministic_document_context_v1"
VERSAO_TEMPLATE_R3 = "r3-context-v1"
CAMPOS_CONTEXTUAIS_R3 = (
    "titulo",
    "autores",
    "ano",
    "pasta",
    "pagina_inicio",
    "pagina_fim",
    "idioma",
)
LIMITES_CAMPOS_R3 = {
    "titulo": 180,
    "autores": 120,
    "pasta": 60,
    "idioma": 24,
}


class ContextualizacaoInvalida(ValueError):
    """Indica que o candidato R3 nao preserva o contrato do snapshot R2."""


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _linha_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _texto_curto(valor, limite: int) -> str:
    texto = " ".join(str(valor or "").split())
    if len(texto) <= limite:
        return texto
    return texto[: limite - 1].rstrip() + "..."


def _autores(metadata: dict) -> str:
    autores = metadata.get("autores")
    if isinstance(autores, list):
        nomes = [str(item).strip() for item in autores if str(item).strip()]
        if nomes:
            return "; ".join(nomes)
    return str(metadata.get("autor") or "").strip()


def _pagina(metadata: dict) -> str:
    inicio = int(metadata.get("pagina_inicio") or 0)
    fim = int(metadata.get("pagina_fim") or inicio)
    if inicio <= 0:
        return ""
    return str(inicio) if fim <= inicio else f"{inicio}-{fim}"


def construir_prefixo_contextual(metadata: dict) -> str:
    """Cria contexto curto apenas com metadados observados no corpus."""
    valores = (
        ("Documento", _texto_curto(metadata.get("titulo"), LIMITES_CAMPOS_R3["titulo"])),
        ("Autores", _texto_curto(_autores(metadata), LIMITES_CAMPOS_R3["autores"])),
        ("Ano", _texto_curto(metadata.get("ano"), 12)),
        ("Colecao", _texto_curto(metadata.get("pasta"), LIMITES_CAMPOS_R3["pasta"])),
        ("Pagina fisica", _pagina(metadata)),
        ("Idioma", _texto_curto(metadata.get("idioma"), LIMITES_CAMPOS_R3["idioma"])),
    )
    linhas = [f"{rotulo}: {valor}" for rotulo, valor in valores if valor]
    if not linhas:
        return ""
    return "Contexto documental deterministico:\n" + "\n".join(linhas)


def construir_retrieval_text(raw_text: str, metadata: dict) -> str:
    prefixo = construir_prefixo_contextual(metadata)
    if not prefixo:
        return raw_text
    return f"{prefixo}\n\nTrecho original:\n{raw_text}"


def _normalizar_vetores(vetores) -> list[list[float]]:
    valores = vetores.tolist() if hasattr(vetores, "tolist") else list(vetores)
    return [[float(valor) for valor in vetor] for vetor in valores]


def _hash_identidade(digest_raw, digest_ids, registro: dict) -> None:
    chunk_id = str(registro["chunk_id"])
    raw_text = str(registro["raw_text"])
    digest_raw.update(f"{chunk_id}\0{raw_text}\n".encode("utf-8"))
    digest_ids.update(f"{chunk_id}\n".encode("utf-8"))


def _contextualizar_lote(
    registros: list[dict],
    modelo_embeddings,
    destino,
    digest_conteudo,
    digest_raw,
    digest_ids,
    *,
    tamanho_lote: int,
) -> tuple[int, int]:
    textos = [
        construir_retrieval_text(item["raw_text"], item["metadata"])
        for item in registros
    ]
    vetores = _normalizar_vetores(
        modelo_embeddings.encode(
            textos,
            batch_size=tamanho_lote,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
    )
    if len(vetores) != len(registros):
        raise ContextualizacaoInvalida(
            f"Encoder retornou {len(vetores)} vetores para {len(registros)} chunks."
        )

    prefixos = 0
    caracteres_prefixo = 0
    for registro, retrieval_text, vetor in zip(registros, textos, vetores):
        contextualizado = {
            **registro,
            "retrieval_text": retrieval_text,
            "embedding": vetor,
        }
        validar_registro(
            contextualizado,
            SCHEMA_VERSION,
            estrategia_texto=ESTRATEGIA_CONTEXTO_R3,
        )
        destino.write(_linha_json(contextualizado))
        atualizar_hash_conteudo(digest_conteudo, contextualizado)
        _hash_identidade(digest_raw, digest_ids, contextualizado)
        tamanho_prefixo = len(retrieval_text) - len(str(registro["raw_text"]))
        if tamanho_prefixo > 0:
            prefixos += 1
            caracteres_prefixo += tamanho_prefixo
    return prefixos, caracteres_prefixo


def _candidato_pronto(destino: Path, origem_sha256: str) -> dict | None:
    if not destino.is_file():
        return None
    try:
        manifesto = ler_manifesto(destino)
        contexto = manifesto.get("contextual_retrieval", {})
        if (
            manifesto.get("retrieval_text_strategy") != ESTRATEGIA_CONTEXTO_R3
            or contexto.get("template_version") != VERSAO_TEMPLATE_R3
            or contexto.get("source_snapshot_sha256") != origem_sha256
        ):
            return None
        return {**validar_snapshot(destino), "ja_estava_pronto": True}
    except (OSError, ValueError):
        return None


def contextualizar_snapshot(
    origem: Path,
    destino: Path,
    modelo_embeddings,
    *,
    tamanho_lote: int = 32,
) -> dict:
    """Gera snapshot R3 paralelo e reembeda somente ``retrieval_text``."""
    origem = Path(origem).resolve()
    destino = Path(destino).resolve()
    if origem == destino:
        raise ContextualizacaoInvalida("R3 exige snapshot candidato separado do R2.")
    tamanho_lote = max(1, int(tamanho_lote))
    manifesto_origem = ler_manifesto(origem)
    if int(manifesto_origem["schema_version"]) != SCHEMA_VERSION:
        raise ContextualizacaoInvalida("R3 exige snapshot de origem no schema v2.")
    if manifesto_origem.get("retrieval_text_strategy") != ESTRATEGIA_TEXTO_R2:
        raise ContextualizacaoInvalida(
            "R3 deve partir do contrato identity_raw_text publicado em R2."
        )

    origem_sha256 = _sha256(origem)
    pronto = _candidato_pronto(destino, origem_sha256)
    if pronto is not None:
        return pronto

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    digest_conteudo = hashlib.sha256()
    digest_raw = hashlib.sha256()
    digest_ids = hashlib.sha256()
    lidos = 0
    prefixos = 0
    caracteres_prefixo = 0

    try:
        with tempfile.SpooledTemporaryFile(
            mode="w+t",
            encoding="utf-8",
            newline="\n",
            max_size=8 * 1024 * 1024,
        ) as registros_destino, gzip.open(origem, "rt", encoding="utf-8") as arquivo:
            next(arquivo)
            lote: list[dict] = []
            for numero_linha, linha in enumerate(arquivo, 2):
                registro = json.loads(linha)
                validar_registro(
                    registro,
                    SCHEMA_VERSION,
                    numero_linha=numero_linha,
                    estrategia_texto=ESTRATEGIA_TEXTO_R2,
                )
                lote.append(registro)
                if len(lote) < tamanho_lote:
                    continue
                n_prefixos, n_caracteres = _contextualizar_lote(
                    lote,
                    modelo_embeddings,
                    registros_destino,
                    digest_conteudo,
                    digest_raw,
                    digest_ids,
                    tamanho_lote=tamanho_lote,
                )
                prefixos += n_prefixos
                caracteres_prefixo += n_caracteres
                lidos += len(lote)
                lote.clear()
            if lote:
                n_prefixos, n_caracteres = _contextualizar_lote(
                    lote,
                    modelo_embeddings,
                    registros_destino,
                    digest_conteudo,
                    digest_raw,
                    digest_ids,
                    tamanho_lote=tamanho_lote,
                )
                prefixos += n_prefixos
                caracteres_prefixo += n_caracteres
                lidos += len(lote)

            esperado = int(manifesto_origem["n_chunks"])
            if lidos != esperado:
                raise ContextualizacaoInvalida(
                    f"Contextualizacao incompleta: {lidos}/{esperado} chunks."
                )
            contexto = {
                "stage": "R3",
                "mode": "deterministic_metadata_prefix",
                "template_version": VERSAO_TEMPLATE_R3,
                "fields": list(CAMPOS_CONTEXTUAIS_R3),
                "field_limits_chars": dict(LIMITES_CAMPOS_R3),
                "llm_used": False,
                "raw_text_unchanged": True,
                "embedding_recomputed": True,
                "embedding_backend": "sentence-transformers",
                "embedding_repository": REPOSITORIO_MODELO,
                "embedding_revision": REVISAO_MODELO,
                "source_snapshot_sha256": origem_sha256,
                "source_content_hash_sha256": manifesto_origem.get(
                    "hash_conteudo_retrieval_sha256"
                ),
                "contextualized_chunks": prefixos,
                "mean_prefix_chars": round(
                    caracteres_prefixo / prefixos if prefixos else 0.0,
                    3,
                ),
            }
            manifesto = criar_manifesto_v2(
                {
                    **manifesto_origem,
                    "colecao": "literatura_pv_contextual_r3",
                    "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
                },
                digest_conteudo.hexdigest(),
                retrieval_text_strategy=ESTRATEGIA_CONTEXTO_R3,
                extras={
                    "hash_raw_text_sha256": digest_raw.hexdigest(),
                    "hash_chunk_ids_sha256": digest_ids.hexdigest(),
                    "contextual_retrieval": contexto,
                },
            )
            registros_destino.seek(0)
            with gzip.open(
                temporario,
                "wt",
                encoding="utf-8",
                compresslevel=6,
            ) as arquivo_destino:
                arquivo_destino.write(_linha_json(manifesto))
                shutil.copyfileobj(registros_destino, arquivo_destino)
        os.replace(temporario, destino)
    finally:
        temporario.unlink(missing_ok=True)

    return {
        **validar_snapshot(destino),
        "arquivo": str(destino),
        "arquivo_sha256": _sha256(destino),
        "tamanho_bytes": destino.stat().st_size,
        "ja_estava_pronto": False,
    }
