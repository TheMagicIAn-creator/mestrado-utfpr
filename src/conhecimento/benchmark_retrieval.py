"""Benchmark versionado do retrieval bibliográfico vigente do ALIAdo.

Este módulo mede o pipeline atual sem alterar expansão, fusão RRF, reranking ou
diversificação. O conjunto-ouro é validado contra o snapshot portátil para que
arquivo, SHA-256, página e chunk existam antes de qualquer métrica ser aceita.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from src.conhecimento.benchmark_relatorio import (
    relatorio_markdown as relatorio_markdown,
)
from src.conhecimento.indice_portatil import (
    TIPOS_CHUNK_COMPATIVEIS,
    ler_manifesto,
    validar_registro,
)
from src.conhecimento.retrieval_metrics import metricas_por_evidencias

GOLD_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_KS = (5, 8)
GOLD_STATUS_PROVISORIO = "provisional_pending_researcher_review"
RETRIEVAL_PROFILE_BASELINE = "baseline"
RETRIEVAL_PROFILE_R4 = "r4_hybrid"
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CATEGORIAS_OBRIGATORIAS = frozenset(
    {
        "localizacao_direta",
        "conceito",
        "comparacao_autores",
        "metodo",
        "componente",
        "fmeca",
        "rcm",
        "confiabilidade",
        "autoencoders",
        "gpvs_faults",
        "multilingue",
        "sinonimo",
        "revisao_ampla",
        "multi_hop",
    }
)


def resolver_caminho_no_projeto(
    caminho: str | Path,
    *,
    deve_existir: bool = False,
) -> Path:
    """Resolve um caminho sem permitir escape da raiz versionada."""
    raiz = RAIZ_PROJETO.resolve()
    candidato = Path(caminho)
    if not candidato.is_absolute():
        candidato = raiz / candidato
    resolvido = candidato.resolve(strict=False)
    try:
        resolvido.relative_to(raiz)
    except ValueError as exc:
        raise ValueError(f"Caminho fora da raiz do projeto: {caminho}") from exc
    if deve_existir and not resolvido.is_file():
        raise FileNotFoundError(f"Arquivo local não encontrado: {resolvido}")
    return resolvido


def _json_canonico(valor: object) -> bytes:
    return (
        json.dumps(
            valor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def hash_json_sha256(valor: object) -> str:
    return hashlib.sha256(_json_canonico(valor)).hexdigest()


def hash_arquivo_sha256(caminho: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def carregar_gold_set(caminho: str | Path, *, validar_campanha: bool = True) -> dict:
    arquivo = resolver_caminho_no_projeto(caminho, deve_existir=True)
    dados = json.loads(arquivo.read_text(encoding="utf-8"))
    validar_gold_set(dados, validar_campanha=validar_campanha)
    return dados


def _validar_evidencia_gold(evidencia: dict, query_id: str) -> None:
    chunks = evidencia.get("chunk_ids", [])
    paginas = evidencia.get("pages", [])
    if not evidencia.get("document_id") or not evidencia.get("file"):
        raise ValueError(f"Evidência sem identidade documental em {query_id}.")
    if not chunks or not paginas:
        raise ValueError(f"Evidência sem página ou chunk em {query_id}.")
    if int(evidencia.get("relevance", 0)) not in (1, 2, 3):
        raise ValueError(f"Relevância fora da escala 1–3 em {query_id}.")


def _validar_pergunta_gold(pergunta: dict, ids: set[str]) -> tuple[str, str]:
    query_id = str(pergunta.get("id", "")).strip()
    texto = str(pergunta.get("question", "")).strip()
    categoria = str(pergunta.get("category", "")).strip()
    comportamento = pergunta.get("expected_behavior", "retrieve")
    evidencias = pergunta.get("expected_evidence", [])

    if not query_id or query_id in ids:
        raise ValueError(f"ID de pergunta ausente ou duplicado: {query_id!r}.")
    if not texto:
        raise ValueError(f"Pergunta vazia em {query_id}.")
    if categoria not in CATEGORIAS_OBRIGATORIAS:
        raise ValueError(f"Categoria inválida em {query_id}: {categoria!r}.")
    if comportamento not in {"retrieve", "abstain"}:
        raise ValueError(f"Comportamento inválido em {query_id}.")
    if comportamento == "retrieve" and not evidencias:
        raise ValueError(f"Pergunta recuperável sem evidência em {query_id}.")
    if comportamento == "abstain" and evidencias:
        raise ValueError(f"Pergunta de abstenção não pode ter evidência em {query_id}.")
    if pergunta.get("curation_status") != "provisional_verified_in_snapshot":
        raise ValueError(f"Estado de curadoria inválido em {query_id}.")

    for evidencia in evidencias:
        _validar_evidencia_gold(evidencia, query_id)
    return query_id, categoria


def validar_gold_set(dados: dict, *, validar_campanha: bool = True) -> None:
    if int(dados.get("schema_version", 0)) != GOLD_SCHEMA_VERSION:
        raise ValueError("Schema do gold set incompatível.")
    if dados.get("status") != GOLD_STATUS_PROVISORIO:
        raise ValueError("O gold set inicial deve permanecer provisório até a revisão R6.")

    perguntas = dados.get("queries")
    if not isinstance(perguntas, list) or not perguntas:
        raise ValueError("O gold set não contém perguntas.")
    if validar_campanha and not 30 <= len(perguntas) <= 50:
        raise ValueError("A campanha R1 exige entre 30 e 50 perguntas.")

    ids: set[str] = set()
    categorias: set[str] = set()
    for pergunta in perguntas:
        query_id, categoria = _validar_pergunta_gold(pergunta, ids)
        ids.add(query_id)
        categorias.add(categoria)

    if validar_campanha:
        ausentes = sorted(CATEGORIAS_OBRIGATORIAS - categorias)
        if ausentes:
            raise ValueError(f"Categorias obrigatórias ausentes: {', '.join(ausentes)}.")


def carregar_snapshot(caminho: str | Path) -> tuple[dict, dict[str, dict]]:
    registros: dict[str, dict] = {}
    arquivo_snapshot = resolver_caminho_no_projeto(caminho, deve_existir=True)
    manifesto = ler_manifesto(arquivo_snapshot)
    schema_version = int(manifesto["schema_version"])
    with gzip.open(arquivo_snapshot, "rt", encoding="utf-8") as arquivo:
        next(arquivo)
        for numero_linha, linha in enumerate(arquivo, 2):
            registro = json.loads(linha)
            if registro.get("tipo") not in TIPOS_CHUNK_COMPATIVEIS:
                continue
            validar_registro(
                registro,
                schema_version,
                numero_linha=numero_linha,
                estrategia_texto=manifesto.get("retrieval_text_strategy"),
            )
            chunk_id = str(registro.get("chunk_id") or registro.get("id") or "")
            if not chunk_id or chunk_id in registros:
                raise ValueError(f"Chunk ausente ou duplicado no snapshot: {chunk_id!r}.")
            registros[chunk_id] = registro
    return manifesto, registros


def _validar_chunk_gold(
    chunk_id: str,
    *,
    documento: str,
    arquivo: str,
    paginas: set[int],
    registros_snapshot: dict[str, dict],
) -> None:
    registro = registros_snapshot.get(str(chunk_id))
    if registro is None:
        raise ValueError(f"Chunk do gold set não existe: {chunk_id}.")
    metadata = registro.get("metadata", {})
    hash_documento = str(
        metadata.get("arquivo_sha256")
        or metadata.get("arquivo_hash")
        or str(registro.get("document_id") or "").removeprefix("doc:")
    )
    if hash_documento != documento:
        raise ValueError(f"Hash documental divergente em {chunk_id}.")
    if metadata.get("arquivo") != arquivo:
        raise ValueError(f"Arquivo divergente em {chunk_id}.")
    inicio = int(metadata.get("pagina_inicio", 0) or 0)
    fim = int(metadata.get("pagina_fim", inicio) or inicio)
    if not paginas.intersection(range(inicio, fim + 1)):
        raise ValueError(f"Página do gold set não pertence ao chunk {chunk_id}.")


def _validar_evidencia_contra_snapshot(
    evidencia: dict,
    registros_snapshot: dict[str, dict],
) -> int:
    documento = str(evidencia["document_id"])
    arquivo = str(evidencia["file"])
    paginas = {int(item) for item in evidencia["pages"]}
    chunks = [str(item) for item in evidencia["chunk_ids"]]
    for chunk_id in chunks:
        _validar_chunk_gold(
            chunk_id,
            documento=documento,
            arquivo=arquivo,
            paginas=paginas,
            registros_snapshot=registros_snapshot,
        )
    return len(chunks)


def validar_gold_contra_snapshot(
    gold_set: dict,
    manifesto_snapshot: dict,
    registros_snapshot: dict[str, dict],
) -> dict:
    hash_esperado = str(gold_set.get("corpus_hash_sha256", ""))
    hash_observado = str(manifesto_snapshot.get("hash_corpus_sha256", ""))
    if not hash_esperado or hash_esperado != hash_observado:
        raise ValueError(
            "O gold set não corresponde ao corpus do snapshot: "
            f"esperado={hash_esperado or 'ausente'}, observado={hash_observado or 'ausente'}."
        )

    evidencias = (
        evidencia
        for pergunta in gold_set["queries"]
        for evidencia in pergunta.get("expected_evidence", [])
    )
    validados = sum(
        _validar_evidencia_contra_snapshot(evidencia, registros_snapshot)
        for evidencia in evidencias
    )
    return {
        "n_queries": len(gold_set["queries"]),
        "n_chunks_gold": validados,
        "n_chunks_snapshot": len(registros_snapshot),
        "corpus_hash_sha256": hash_observado,
    }


def _chunk_id(metadata: dict) -> str:
    explicito = metadata.get("chunk_id") or metadata.get("_chunk_id")
    if explicito:
        return str(explicito)
    arquivo_hash = str(metadata.get("arquivo_hash", ""))
    indice = metadata.get("chunk_index")
    if arquivo_hash and indice is not None:
        return f"{arquivo_hash}__chunk_{int(indice):05d}"
    return ""


def recuperar_baseline(
    pergunta: str,
    modelo_embeddings,
    colecao,
    indice_lexical,
    *,
    n_pool: int = 80,
    n_resultados: int = 12,
    n_resultados_revisao: int = 20,
    max_chunks_por_fonte: int = 2,
) -> list[dict]:
    """Executa exatamente a expansão, busca híbrida e reranking atuais."""
    from src.conhecimento.agente_recuperacao import (
        _busca_hibrida,
        _expandir_query,
        _rerankar,
    )

    expansao = _expandir_query(pergunta)
    variacoes = list(expansao.get("variacoes") or [pergunta])
    if pergunta not in variacoes:
        variacoes.insert(0, pergunta)
    termos = list(expansao.get("termos") or [])
    revisao = bool(expansao.get("revisao"))
    candidatos = _busca_hibrida(
        variacoes,
        termos,
        colecao,
        modelo_embeddings,
        n_pool=n_pool,
        indice_lexical=indice_lexical,
    )
    melhores = _rerankar(
        candidatos,
        pergunta,
        n_final=n_resultados_revisao if revisao else n_resultados,
        max_por_fonte=1 if revisao else max_chunks_por_fonte,
        termos_extra=termos,
    )

    return _serializar_recuperados(melhores)


def _serializar_recuperados(melhores: list) -> list[dict]:
    saida = []
    for rank, (documento, metadata) in enumerate(melhores, start=1):
        inicio = int(metadata.get("pagina_inicio", 0) or 0)
        fim = int(metadata.get("pagina_fim", inicio) or inicio)
        saida.append(
            {
                "rank": rank,
                "chunk_id": _chunk_id(metadata),
                "document_id": str(metadata.get("arquivo_hash", "")),
                "file": str(metadata.get("arquivo", "")),
                "pages": list(range(inicio, fim + 1)) if inicio else [],
                "rrf_score": round(float(metadata.get("_rrf_score", 0.0) or 0.0), 8),
                "context_chars": len(str(documento or "")),
            }
        )
    return saida


def recuperar_refinado_r4(
    pergunta: str,
    modelo_embeddings,
    colecao,
    indice_lexical,
    *,
    n_pool: int = 80,
    n_resultados: int = 12,
    n_resultados_revisao: int = 20,
    max_chunks_por_fonte: int = 2,
    peso_filtro_semantico: float = 0.25,
) -> list[dict]:
    """Executa o candidato R4 sem substituir o caminho vigente."""
    from src.conhecimento.agente_recuperacao import (
        _busca_hibrida,
        _expandir_query,
        _expandir_vizinhanca,
        _filtros_metadata_explicitos,
        _rerankar,
    )

    expansao = _expandir_query(pergunta)
    variacoes = list(expansao.get("variacoes") or [pergunta])
    if pergunta not in variacoes:
        variacoes.insert(0, pergunta)
    termos = list(expansao.get("termos") or [])
    revisao = bool(expansao.get("revisao"))
    n_final = n_resultados_revisao if revisao else n_resultados
    filtros = _filtros_metadata_explicitos(pergunta, colecao)
    candidatos = _busca_hibrida(
        variacoes,
        termos,
        colecao,
        modelo_embeddings,
        n_pool=n_pool,
        indice_lexical=indice_lexical,
        rrf_constant=60.0,
        filtros_metadata=filtros,
        n_filtrado=3,
        peso_filtro_semantico=peso_filtro_semantico,
        peso_scan_metadata=1.0,
        preferir_filtro_semantico=False,
    )
    melhores = _rerankar(
        candidatos,
        pergunta,
        n_final=n_final,
        max_por_fonte=1 if revisao else max_chunks_por_fonte,
        termos_extra=termos,
    )
    com_vizinhanca = _expandir_vizinhanca(
        melhores,
        colecao,
        max_sementes=min(20, len(melhores)),
        decaimento_rrf=0.85,
    )
    return _serializar_recuperados(com_vizinhanca)


def _percentil(valores: list[float], probabilidade: float) -> float:
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    posicao = (len(ordenados) - 1) * probabilidade
    inferior = math.floor(posicao)
    superior = math.ceil(posicao)
    if inferior == superior:
        return float(ordenados[inferior])
    peso = posicao - inferior
    return float(ordenados[inferior] * (1 - peso) + ordenados[superior] * peso)


def _agregar_metricas(resultados: list[dict], ks: tuple[int, ...]) -> dict:
    recuperaveis = [
        item for item in resultados if item["expected_behavior"] == "retrieve"
    ]
    resumo: dict[str, object] = {
        "n_queries": len(resultados),
        "n_retrieval_queries": len(recuperaveis),
        "n_future_abstention_queries": len(resultados) - len(recuperaveis),
    }
    for k in ks:
        chaves = (
            f"recall@{k}",
            f"precision@{k}",
            f"hit_rate@{k}",
            f"ndcg@{k}",
            f"strict_chunk_recall@{k}",
            f"document_recall@{k}",
        )
        for chave in chaves:
            resumo[chave] = round(
                statistics.fmean(item["metrics"][str(k)][chave] for item in recuperaveis),
                6,
            )
        resumo[f"mrr@{k}"] = round(
            statistics.fmean(item["metrics"][str(k)]["mrr"] for item in recuperaveis),
            6,
        )

    latencias = [float(item["latency_ms"]) for item in resultados]
    contextos = [int(item["context_chars_at_max_k"]) for item in resultados]
    resumo.update(
        {
            "latency_ms_mean": round(statistics.fmean(latencias), 3),
            "latency_ms_p50": round(_percentil(latencias, 0.50), 3),
            "latency_ms_p95": round(_percentil(latencias, 0.95), 3),
            "context_chars_mean_at_max_k": round(statistics.fmean(contextos), 1),
        }
    )
    return resumo


def executar_benchmark(
    gold_set: dict,
    *,
    modelo_embeddings,
    colecao,
    indice_lexical,
    manifesto_snapshot: dict,
    ks: Iterable[int] = DEFAULT_KS,
    relogio: Callable[[], float] = time.perf_counter,
    warmup: bool = True,
    git_revision: str | None = None,
    benchmark_id: str = "evidence-rag-baseline-v1",
    stage: str = "R0-R1",
    variant: str = "baseline_current",
    retrieval_profile: str = RETRIEVAL_PROFILE_BASELINE,
) -> dict:
    from src.conhecimento.embeddings import REPOSITORIO_MODELO, REVISAO_MODELO

    contexto_deterministico = manifesto_snapshot.get("contextual_retrieval", {})
    ks_ordenados = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not ks_ordenados:
        raise ValueError("Informe ao menos um valor positivo de k.")

    if retrieval_profile == RETRIEVAL_PROFILE_R4:
        recuperador = recuperar_refinado_r4
    elif retrieval_profile == RETRIEVAL_PROFILE_BASELINE:
        recuperador = recuperar_baseline
    else:
        raise ValueError(f"Perfil de retrieval desconhecido: {retrieval_profile}.")

    inicio_warmup = relogio()
    if warmup:
        recuperador(
            "confiabilidade de inversores fotovoltaicos",
            modelo_embeddings,
            colecao,
            indice_lexical,
        )
    warmup_ms = (relogio() - inicio_warmup) * 1000 if warmup else 0.0

    resultados = []
    max_k = max(ks_ordenados)
    for item in gold_set["queries"]:
        inicio = relogio()
        recuperados = recuperador(
            item["question"],
            modelo_embeddings,
            colecao,
            indice_lexical,
        )
        latencia_ms = (relogio() - inicio) * 1000
        metricas = {}
        if item.get("expected_behavior", "retrieve") == "retrieve":
            for k in ks_ordenados:
                metricas_pagina = metricas_por_evidencias(
                    recuperados,
                    item["expected_evidence"],
                    k,
                    nivel="page",
                )
                metricas_chunk = metricas_por_evidencias(
                    recuperados,
                    item["expected_evidence"],
                    k,
                    nivel="chunk",
                )
                metricas_documento = metricas_por_evidencias(
                    recuperados,
                    item["expected_evidence"],
                    k,
                    nivel="document",
                )
                metricas[str(k)] = {
                    **metricas_pagina,
                    f"strict_chunk_recall@{k}": metricas_chunk[f"recall@{k}"],
                    f"document_recall@{k}": metricas_documento[f"recall@{k}"],
                }

        resultados.append(
            {
                "query_id": item["id"],
                "category": item["category"],
                "language": item.get("language", "pt-BR"),
                "question": item["question"],
                "expected_behavior": item.get("expected_behavior", "retrieve"),
                "latency_ms": round(latencia_ms, 3),
                "context_chars_at_max_k": sum(
                    registro["context_chars"] for registro in recuperados[:max_k]
                ),
                "distinct_sources_at_max_k": len(
                    {registro["file"] for registro in recuperados[:max_k]}
                ),
                "metrics": metricas,
                "retrieved": recuperados[:max_k],
            }
        )

    por_categoria = {}
    for categoria in sorted({item["category"] for item in resultados}):
        grupo = [item for item in resultados if item["category"] == categoria]
        if any(item["expected_behavior"] == "retrieve" for item in grupo):
            por_categoria[categoria] = _agregar_metricas(grupo, ks_ordenados)

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "stage": stage,
        "variant": variant,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision,
        "gold_set": {
            "id": gold_set["gold_set_id"],
            "status": gold_set["status"],
            "sha256": hash_json_sha256(gold_set),
            "researcher_review_required_at": "R6",
        },
        "corpus": {
            "snapshot_schema_version": int(manifesto_snapshot.get("schema_version", 0)),
            "snapshot_file_sha256": manifesto_snapshot.get(
                "arquivo_snapshot_sha256"
            ),
            "snapshot_size_bytes": manifesto_snapshot.get(
                "arquivo_snapshot_tamanho_bytes"
            ),
            "hash_sha256": manifesto_snapshot.get("hash_corpus_sha256"),
            "snapshot_content_hash_sha256": manifesto_snapshot.get(
                "hash_conteudo_retrieval_sha256"
            ),
            "raw_text_hash_sha256": manifesto_snapshot.get(
                "hash_raw_text_sha256"
            ),
            "chunk_ids_hash_sha256": manifesto_snapshot.get(
                "hash_chunk_ids_sha256"
            ),
            "source_snapshot_sha256": contexto_deterministico.get(
                "source_snapshot_sha256"
            ),
            "source_content_hash_sha256": contexto_deterministico.get(
                "source_content_hash_sha256"
            ),
            "n_documents": int(manifesto_snapshot.get("n_documentos", 0)),
            "n_chunks": int(manifesto_snapshot.get("n_chunks", 0)),
            "collection_count": int(colecao.count()),
        },
        "retrieval": {
            "profile": retrieval_profile,
            "embedding_model": manifesto_snapshot.get("modelo_embeddings"),
            "embedding_repository": REPOSITORIO_MODELO,
            "embedding_revision": REVISAO_MODELO,
            "index_embedding_backend": contexto_deterministico.get(
                "embedding_backend", "sentence-transformers"
            ),
            "query_embedding_backend": "onnxruntime",
            "semantic_index": "ChromaDB",
            "lexical_index": "SQLite FTS5 BM25",
            "lexical_available": bool(getattr(indice_lexical, "disponivel", False)),
            "fusion": "reciprocal_rank_fusion",
            "rrf_constant": 60,
            "reranker": (
                "deterministic_local_v2"
                if retrieval_profile == RETRIEVAL_PROFILE_R4
                else "deterministic_local_v1"
            ),
            "source_diversification": True,
            "n_pool": 80,
            "n_results": 12,
            "n_results_review": 20,
            "max_chunks_per_source": 2,
            "ks": list(ks_ordenados),
            "warmup_ms": round(warmup_ms, 3),
            "raw_text_separated_from_retrieval_text": int(
                manifesto_snapshot.get("schema_version", 0)
            ) >= 2,
            "retrieval_text_strategy": manifesto_snapshot.get(
                "retrieval_text_strategy", "legacy_documento"
            ),
            "contextual_retrieval": stage in {"R3", "R4"},
            "context_template_version": contexto_deterministico.get(
                "template_version"
            ),
            "context_fields": contexto_deterministico.get("fields", []),
            "context_field_limits_chars": contexto_deterministico.get(
                "field_limits_chars", {}
            ),
            "contextualized_chunks": contexto_deterministico.get(
                "contextualized_chunks", 0
            ),
            "mean_prefix_chars": contexto_deterministico.get(
                "mean_prefix_chars", 0.0
            ),
            "llm_contextualization_used": contexto_deterministico.get(
                "llm_used", False
            ),
            "parallel_candidate_index": stage in {"R3", "R4"},
            "explicit_metadata_filtered_search": (
                retrieval_profile == RETRIEVAL_PROFILE_R4
            ),
            "metadata_filters_are_advisory": (
                retrieval_profile == RETRIEVAL_PROFILE_R4
            ),
            "metadata_filtered_results_per_query": (
                3 if retrieval_profile == RETRIEVAL_PROFILE_R4 else 0
            ),
            "metadata_filtered_rrf_weight": (
                0.25 if retrieval_profile == RETRIEVAL_PROFILE_R4 else 0.0
            ),
            "neighborhood_expansion": (
                retrieval_profile == RETRIEVAL_PROFILE_R4
            ),
            "neighborhood_radius": (
                1 if retrieval_profile == RETRIEVAL_PROFILE_R4 else 0
            ),
            "neighborhood_seed_limit": (
                20 if retrieval_profile == RETRIEVAL_PROFILE_R4 else 0
            ),
            "evidence_package": False,
            "evidence_guard": False,
        },
        "metric_definition": {
            "primary_level": "page",
            "strict_level": "chunk",
            "upper_bound_level": "document",
            "duplicate_evidence_groups_score_once": True,
        },
        "summary": _agregar_metricas(resultados, ks_ordenados),
        "by_category": por_categoria,
        "queries": resultados,
    }


def executar_baseline_local(
    gold_path: str | Path,
    snapshot_path: str | Path,
    *,
    git_revision: str | None = None,
    candidate_runtime: str | Path | None = None,
    retrieval_profile: str = "auto",
) -> dict:
    import chromadb

    from src.conhecimento.base_runtime import _versao_indice_lexical
    from src.conhecimento.contextual_retrieval import ESTRATEGIA_CONTEXTO_R3
    from src.conhecimento.embeddings import criar_modelo_embeddings
    from src.conhecimento.indice_lexical import IndiceLexicalSQLite
    from src.conhecimento.indice_portatil import importar_colecao
    from src.core.config import NOME_COLECAO, PASTA_CHROMADB

    gold_set = carregar_gold_set(gold_path)
    arquivo_snapshot = resolver_caminho_no_projeto(snapshot_path, deve_existir=True)
    manifesto, registros = carregar_snapshot(arquivo_snapshot)
    manifesto = {
        **manifesto,
        "arquivo_snapshot_sha256": hash_arquivo_sha256(arquivo_snapshot),
        "arquivo_snapshot_tamanho_bytes": arquivo_snapshot.stat().st_size,
    }
    validar_gold_contra_snapshot(gold_set, manifesto, registros)

    identificacao = {}
    estrategia = manifesto.get("retrieval_text_strategy")
    if estrategia == ESTRATEGIA_CONTEXTO_R3:
        if retrieval_profile not in {"auto", RETRIEVAL_PROFILE_R4}:
            raise ValueError(
                "Snapshot contextual aceita apenas os perfis auto ou r4_hybrid."
            )
        runtime = resolver_caminho_no_projeto(
            candidate_runtime
            or RAIZ_PROJETO
            / "base_conhecimento"
            / "candidatos"
            / (
                "r4_hybrid"
                if retrieval_profile == RETRIEVAL_PROFILE_R4
                else "r3_contextual"
            )
        )
        runtime.mkdir(parents=True, exist_ok=True)
        conteudo_hash = str(manifesto["hash_conteudo_retrieval_sha256"])
        nome_colecao = f"literatura_contextual_r3_{conteudo_hash[:12]}"
        cliente = chromadb.PersistentClient(path=str(runtime / "chromadb"))
        colecao = cliente.get_or_create_collection(
            name=nome_colecao,
            metadata={"hnsw:space": "cosine"},
        )
        esperado = int(manifesto["n_chunks"])
        if colecao.count() not in {0, esperado}:
            cliente.delete_collection(nome_colecao)
            colecao = cliente.get_or_create_collection(
                name=nome_colecao,
                metadata={"hnsw:space": "cosine"},
            )
        importar_colecao(colecao, arquivo_snapshot)
        indice_lexical = IndiceLexicalSQLite(
            runtime / f"{nome_colecao}_fts.sqlite3"
        )
        indice_lexical.sincronizar(colecao, versao=conteudo_hash)
        if retrieval_profile == RETRIEVAL_PROFILE_R4:
            identificacao = {
                "benchmark_id": "evidence-rag-r4-hybrid-refinement",
                "stage": "R4",
                "variant": "filtered_hybrid_neighborhood_v1",
                "retrieval_profile": RETRIEVAL_PROFILE_R4,
            }
        else:
            identificacao = {
                "benchmark_id": "evidence-rag-r3-contextual-deterministic",
                "stage": "R3",
                "variant": "deterministic_document_context_v1",
            }
    else:
        if retrieval_profile == RETRIEVAL_PROFILE_R4:
            raise ValueError("O perfil R4 exige o snapshot contextual R3.")
        cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = cliente.get_or_create_collection(name=NOME_COLECAO)
        if colecao.count() != int(manifesto.get("n_chunks", -1)):
            raise RuntimeError(
                "A coleção ChromaDB local não corresponde ao snapshot auditado. "
                "Restaure a base antes de medir o baseline."
            )
        indice_lexical = IndiceLexicalSQLite()
        indice_lexical.sincronizar(
            colecao,
            versao=_versao_indice_lexical(colecao, modo_consulta=True),
        )
    if int(manifesto.get("schema_version", 0)) >= 2 and not identificacao:
        identificacao = {
            "benchmark_id": "evidence-rag-r2-schema-v2",
            "stage": "R2",
            "variant": "jsonl_schema_v2_identity",
        }
    modelo = criar_modelo_embeddings(modo_consulta=True)
    return executar_benchmark(
        gold_set,
        modelo_embeddings=modelo,
        colecao=colecao,
        indice_lexical=indice_lexical,
        manifesto_snapshot=manifesto,
        git_revision=git_revision,
        **identificacao,
    )


METRICAS_RETRIEVAL_COMPARAVEIS = (
    "recall@5",
    "recall@8",
    "precision@5",
    "precision@8",
    "hit_rate@5",
    "hit_rate@8",
    "mrr@5",
    "mrr@8",
    "ndcg@5",
    "ndcg@8",
    "strict_chunk_recall@5",
    "strict_chunk_recall@8",
    "document_recall@5",
    "document_recall@8",
    "context_chars_mean_at_max_k",
)


def comparar_benchmarks(baseline: dict, candidato: dict) -> dict:
    """Compara candidatos sem misturar latência com qualidade científica."""
    resumo_base = baseline["summary"]
    resumo_candidato = candidato["summary"]
    deltas = {
        metrica: round(
            float(resumo_candidato[metrica]) - float(resumo_base[metrica]),
            9,
        )
        for metrica in METRICAS_RETRIEVAL_COMPARAVEIS
    }
    metricas = {
        metrica: {
            "baseline": resumo_base[metrica],
            "candidate": resumo_candidato[metrica],
            "delta": deltas[metrica],
        }
        for metrica in METRICAS_RETRIEVAL_COMPARAVEIS
    }
    campos_ranking = (
        "fusion",
        "rrf_constant",
        "reranker",
        "source_diversification",
        "n_pool",
        "n_results",
        "n_results_review",
        "max_chunks_per_source",
        "ks",
    )
    ranking_inalterado = all(
        baseline["retrieval"].get(campo) == candidato["retrieval"].get(campo)
        for campo in campos_ranking
    )
    corpus_inalterado = all(
        baseline["corpus"].get(campo) == candidato["corpus"].get(campo)
        for campo in ("hash_sha256", "n_documents", "n_chunks", "collection_count")
    )
    base_por_id = {item["query_id"]: item for item in baseline["queries"]}
    candidato_por_id = {item["query_id"]: item for item in candidato["queries"]}
    regressions = []
    regression_details = []
    improvements = []
    simple_categories = {"localizacao_direta", "conceito", "metodo"}
    critical_simple_regressions = []
    for query_id in sorted(base_por_id.keys() & candidato_por_id.keys()):
        base_item = base_por_id[query_id]
        candidate_item = candidato_por_id[query_id]
        if base_item["expected_behavior"] != "retrieve":
            continue
        base_recall = float(base_item["metrics"]["5"]["recall@5"])
        candidate_recall = float(candidate_item["metrics"]["5"]["recall@5"])
        if candidate_recall < base_recall:
            regressions.append(query_id)
            base_top = (base_item.get("retrieved") or [{}])[0]
            candidate_top = (candidate_item.get("retrieved") or [{}])[0]
            regression_details.append(
                {
                    "query_id": query_id,
                    "category": base_item["category"],
                    "question": base_item["question"],
                    "baseline_page_recall_at_5": base_recall,
                    "candidate_page_recall_at_5": candidate_recall,
                    "candidate_document_recall_at_5": float(
                        candidate_item["metrics"]["5"]["document_recall@5"]
                    ),
                    "baseline_top_1": {
                        "file": base_top.get("file"),
                        "pages": base_top.get("pages", []),
                    },
                    "candidate_top_1": {
                        "file": candidate_top.get("file"),
                        "pages": candidate_top.get("pages", []),
                    },
                }
            )
            if base_item["category"] in simple_categories:
                critical_simple_regressions.append(query_id)
        elif candidate_recall > base_recall:
            improvements.append(query_id)
    quality_metrics = (
        "recall@5",
        "recall@8",
        "mrr@5",
        "mrr@8",
        "ndcg@5",
        "ndcg@8",
    )
    quality_gain = any(deltas[metrica] > 1e-12 for metrica in quality_metrics)
    promotion_eligible = all(
        (
            corpus_inalterado,
            ranking_inalterado or candidato.get("stage") == "R4",
            quality_gain,
            not critical_simple_regressions,
        )
    )
    return {
        "baseline_benchmark_id": baseline.get("benchmark_id"),
        "candidate_benchmark_id": candidato.get("benchmark_id"),
        "corpus_identity_preserved": corpus_inalterado,
        "ranking_contract_preserved": ranking_inalterado,
        "ranking_change_expected": candidato.get("stage") == "R4",
        "scientific_metrics_identical": all(
            math.isclose(valor, 0.0, abs_tol=1e-12) for valor in deltas.values()
        ),
        "quality_gain_observed": quality_gain,
        "regressed_queries_at_5": regressions,
        "regression_details": regression_details,
        "improved_queries_at_5": improvements,
        "critical_simple_regressions": critical_simple_regressions,
        "promotion_eligible_after_quality_stages": promotion_eligible,
        "promotion_decision": (
            "deferred_to_r4_r5_r6"
            if candidato.get("stage") == "R3"
            else (
                "deferred_to_r5_r6"
                if candidato.get("stage") == "R4"
                else "not_applicable"
            )
        ),
        "metrics": metricas,
        "metric_deltas": deltas,
        "latency_ms": {
            "baseline_mean": resumo_base["latency_ms_mean"],
            "candidate_mean": resumo_candidato["latency_ms_mean"],
            "informational_only": True,
        },
    }


