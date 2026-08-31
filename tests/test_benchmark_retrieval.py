from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from src.conhecimento import benchmark_retrieval as benchmark

DOCUMENT_ID = "a" * 64
CHUNK_ID = f"{DOCUMENT_ID}__chunk_00000"


def _evidencia(*, chunk_id: str = CHUNK_ID) -> dict:
    return {
        "document_id": DOCUMENT_ID,
        "file": "fonte.pdf",
        "pages": [7],
        "chunk_ids": [chunk_id],
        "relevance": 3,
    }


def _pergunta(
    query_id: str = "q1",
    *,
    categoria: str = "conceito",
    comportamento: str = "retrieve",
) -> dict:
    return {
        "id": query_id,
        "question": f"Pergunta {query_id}",
        "category": categoria,
        "language": "pt-BR",
        "expected_behavior": comportamento,
        "curation_status": "provisional_verified_in_snapshot",
        "expected_evidence": [] if comportamento == "abstain" else [_evidencia()],
    }


def _gold(perguntas: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "gold_set_id": "teste",
        "status": benchmark.GOLD_STATUS_PROVISORIO,
        "corpus_hash_sha256": "corpus",
        "queries": perguntas or [_pergunta()],
    }


def _manifesto() -> dict:
    return {
        "tipo": "manifesto_indice_portatil",
        "schema_version": 1,
        "hash_corpus_sha256": "corpus",
        "modelo_embeddings": "modelo",
        "n_documentos": 1,
        "n_chunks": 1,
    }


def test_validar_gold_set_aceita_fixture_minima_e_rejeita_duplicidade():
    benchmark.validar_gold_set(_gold(), validar_campanha=False)

    duplicado = _gold([_pergunta(), _pergunta()])
    with pytest.raises(ValueError, match="duplicado"):
        benchmark.validar_gold_set(duplicado, validar_campanha=False)


def test_validar_gold_set_aceita_aprovacao_humana_r6():
    gold = _gold()
    gold["status"] = benchmark.GOLD_STATUS_APROVADO_R6
    gold["curation"] = {"researcher_review": "approved_R6_2026-08-30"}
    gold["queries"][0]["curation_status"] = "researcher_approved_R6"

    benchmark.validar_gold_set(gold, validar_campanha=False)


@pytest.mark.parametrize(
    ("mutacao", "mensagem"),
    [
        (lambda gold: gold.update(schema_version=99), "Schema"),
        (lambda gold: gold.update(status="final"), "provisório"),
        (lambda gold: gold.update(queries=[]), "não contém"),
        (
            lambda gold: gold["queries"][0].update(category="inexistente"),
            "Categoria inválida",
        ),
        (
            lambda gold: gold["queries"][0].update(expected_behavior="talvez"),
            "Comportamento inválido",
        ),
    ],
)
def test_validar_gold_set_rejeita_contratos_invalidos(mutacao, mensagem):
    gold = _gold()
    mutacao(gold)
    with pytest.raises(ValueError, match=mensagem):
        benchmark.validar_gold_set(gold, validar_campanha=False)


def test_carregar_e_validar_snapshot(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(benchmark, "RAIZ_PROJETO", tmp_path)
    caminho = tmp_path / "snapshot.jsonl.gz"
    registro = {
        "tipo": "chunk_indice_portatil",
        "id": CHUNK_ID,
        "documento": "trecho",
        "embedding": [0.1, 0.2],
        "metadata": {
            "arquivo_hash": DOCUMENT_ID,
            "arquivo": "fonte.pdf",
            "pagina_inicio": 7,
            "pagina_fim": 7,
            "chunk_index": 0,
        },
    }
    with gzip.open(caminho, "wt", encoding="utf-8", newline="\n") as arquivo:
        arquivo.write(json.dumps(_manifesto()) + "\n")
        arquivo.write(json.dumps(registro) + "\n")

    manifesto, registros = benchmark.carregar_snapshot(caminho)
    validacao = benchmark.validar_gold_contra_snapshot(_gold(), manifesto, registros)

    assert validacao["n_queries"] == 1
    assert validacao["n_chunks_gold"] == 1
    assert validacao["n_chunks_snapshot"] == 1


def test_validacao_snapshot_detecta_hash_e_chunk_ausente():
    registro = {
        "metadata": {
            "arquivo_hash": DOCUMENT_ID,
            "arquivo": "fonte.pdf",
            "pagina_inicio": 7,
            "pagina_fim": 7,
        }
    }
    gold = _gold()
    manifesto_divergente = {**_manifesto(), "hash_corpus_sha256": "outro"}
    registros = {CHUNK_ID: registro}
    with pytest.raises(ValueError, match="não corresponde"):
        benchmark.validar_gold_contra_snapshot(
            gold,
            manifesto_divergente,
            registros,
        )
    manifesto = _manifesto()
    with pytest.raises(ValueError, match="não existe"):
        benchmark.validar_gold_contra_snapshot(gold, manifesto, {})


def test_resolver_caminho_rejeita_escape_da_raiz(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(benchmark, "RAIZ_PROJETO", tmp_path / "projeto")

    with pytest.raises(ValueError, match="fora da raiz"):
        benchmark.resolver_caminho_no_projeto(tmp_path / "externo.json")


def test_recuperar_baseline_preserva_pipeline_vigente(monkeypatch):
    from src.conhecimento import agente_recuperacao

    chamadas = {}

    monkeypatch.setattr(
        agente_recuperacao,
        "_expandir_query",
        lambda pergunta: {
            "variacoes": [pergunta, "variação"],
            "termos": ["termo"],
            "revisao": False,
        },
    )

    def busca(variacoes, termos, colecao, modelo, **kwargs):
        chamadas["busca"] = (variacoes, termos, colecao, modelo, kwargs)
        return [("documento", {})]

    def rerank(candidatos, pergunta, **kwargs):
        chamadas["rerank"] = (candidatos, pergunta, kwargs)
        return [
            (
                "documento",
                {
                    "arquivo_hash": DOCUMENT_ID,
                    "arquivo": "fonte.pdf",
                    "chunk_index": 0,
                    "pagina_inicio": 7,
                    "pagina_fim": 8,
                    "_rrf_score": 0.25,
                },
            )
        ]

    monkeypatch.setattr(agente_recuperacao, "_busca_hibrida", busca)
    monkeypatch.setattr(agente_recuperacao, "_rerankar", rerank)

    resultado = benchmark.recuperar_baseline(
        "pergunta", "modelo", "colecao", "lexical"
    )

    assert chamadas["busca"][4]["n_pool"] == 80
    assert chamadas["rerank"][2]["n_final"] == 12
    assert resultado == [
        {
            "rank": 1,
            "chunk_id": CHUNK_ID,
            "document_id": DOCUMENT_ID,
            "file": "fonte.pdf",
            "pages": [7, 8],
            "rrf_score": 0.25,
            "context_chars": 9,
        }
    ]


class _Colecao:
    def count(self):
        return 1


class _Lexical:
    disponivel = True


def test_executar_benchmark_calcula_metricas_e_abstencao_futura(monkeypatch):
    recuperado = {
        "rank": 1,
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "file": "fonte.pdf",
        "pages": [7],
        "rrf_score": 0.1,
        "context_chars": 120,
    }
    monkeypatch.setattr(
        benchmark,
        "recuperar_baseline",
        lambda *args, **kwargs: [recuperado],
    )
    tempos = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.4])
    gold = _gold(
        [
            _pergunta("q1"),
            _pergunta("q2", categoria="gpvs_faults", comportamento="abstain"),
        ]
    )

    resultado = benchmark.executar_benchmark(
        gold,
        modelo_embeddings="modelo",
        colecao=_Colecao(),
        indice_lexical=_Lexical(),
        manifesto_snapshot=_manifesto(),
        relogio=lambda: next(tempos),
        git_revision="abc123",
    )

    assert resultado["summary"]["recall@5"] == 1.0
    assert resultado["summary"]["n_future_abstention_queries"] == 1
    assert resultado["summary"]["latency_ms_mean"] == 300.0
    assert resultado["retrieval"]["warmup_ms"] == 100.0
    assert resultado["queries"][1]["metrics"] == {}
    assert resultado["git_revision"] == "abc123"


def test_comparacao_r2_exige_metricas_e_ranking_idênticos(monkeypatch):
    recuperado = {
        "rank": 1,
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "file": "fonte.pdf",
        "pages": [7],
        "rrf_score": 0.1,
        "context_chars": 120,
    }
    monkeypatch.setattr(
        benchmark,
        "recuperar_baseline",
        lambda *args, **kwargs: [recuperado],
    )
    baseline = benchmark.executar_benchmark(
        _gold(),
        modelo_embeddings="modelo",
        colecao=_Colecao(),
        indice_lexical=_Lexical(),
        manifesto_snapshot=_manifesto(),
        relogio=lambda: 0.0,
        warmup=False,
    )
    manifesto_v2 = {
        **_manifesto(),
        "schema_version": 2,
        "retrieval_text_strategy": "identity_raw_text",
        "hash_conteudo_retrieval_sha256": "c" * 64,
    }
    candidato = benchmark.executar_benchmark(
        _gold(),
        modelo_embeddings="modelo",
        colecao=_Colecao(),
        indice_lexical=_Lexical(),
        manifesto_snapshot=manifesto_v2,
        relogio=lambda: 0.0,
        warmup=False,
        benchmark_id="evidence-rag-r2-schema-v2",
        stage="R2",
        variant="jsonl_schema_v2_identity",
    )

    comparacao = benchmark.comparar_benchmarks(baseline, candidato)

    assert comparacao["corpus_identity_preserved"] is True
    assert comparacao["ranking_contract_preserved"] is True
    assert comparacao["scientific_metrics_identical"] is True
    assert comparacao["metrics"]["recall@5"]["delta"] == 0.0
    assert candidato["retrieval"]["raw_text_separated_from_retrieval_text"] is True
    candidato["comparison_to_baseline"] = comparacao
    relatorio = benchmark.relatorio_markdown(candidato)
    assert "Comparação baseline x candidato" in relatorio
    assert "Gate R2: APROVADO" in relatorio


def test_comparacao_r3_registra_ganho_sem_promover(monkeypatch):
    recuperado = {
        "rank": 1,
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "file": "fonte.pdf",
        "pages": [7],
        "rrf_score": 0.1,
        "context_chars": 120,
    }
    monkeypatch.setattr(
        benchmark,
        "recuperar_baseline",
        lambda *args, **kwargs: [recuperado],
    )
    baseline = benchmark.executar_benchmark(
        _gold(),
        modelo_embeddings="modelo",
        colecao=_Colecao(),
        indice_lexical=_Lexical(),
        manifesto_snapshot={
            **_manifesto(),
            "schema_version": 2,
            "retrieval_text_strategy": "identity_raw_text",
        },
        relogio=lambda: 0.0,
        warmup=False,
        benchmark_id="evidence-rag-r2-schema-v2",
        stage="R2",
        variant="jsonl_schema_v2_identity",
    )
    candidato = copy.deepcopy(baseline)
    candidato.update(
        benchmark_id="evidence-rag-r3-contextual-deterministic",
        stage="R3",
        variant="deterministic_document_context_v1",
    )
    baseline["summary"]["recall@5"] = 0.0
    baseline["queries"][0]["metrics"]["5"]["recall@5"] = 0.0

    comparacao = benchmark.comparar_benchmarks(baseline, candidato)

    assert comparacao["quality_gain_observed"] is True
    assert comparacao["improved_queries_at_5"] == ["q1"]
    assert comparacao["critical_simple_regressions"] == []
    assert comparacao["promotion_eligible_after_quality_stages"] is True
    assert comparacao["promotion_decision"] == "deferred_to_r4_r5_r6"
    candidato["comparison_to_baseline"] = comparacao
    texto = benchmark.relatorio_markdown(candidato)
    assert "Contextual Retrieval determinístico R3" in texto
    assert "candidato não promovido" in texto


def test_comparacao_r3_detalha_regressao_de_pagina_com_documento_correto(
    monkeypatch,
):
    recuperado = {
        "rank": 1,
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "file": "fonte.pdf",
        "pages": [7],
        "rrf_score": 0.1,
        "context_chars": 120,
    }
    monkeypatch.setattr(
        benchmark,
        "recuperar_baseline",
        lambda *args, **kwargs: [recuperado],
    )
    baseline = benchmark.executar_benchmark(
        _gold(),
        modelo_embeddings="modelo",
        colecao=_Colecao(),
        indice_lexical=_Lexical(),
        manifesto_snapshot=_manifesto(),
        relogio=lambda: 0.0,
        warmup=False,
    )
    candidato = copy.deepcopy(baseline)
    candidato.update(stage="R3", benchmark_id="r3")
    candidato["summary"]["recall@5"] = 0.0
    candidato["queries"][0]["metrics"]["5"]["recall@5"] = 0.0

    comparacao = benchmark.comparar_benchmarks(baseline, candidato)

    detalhe = comparacao["regression_details"][0]
    assert detalhe["query_id"] == "q1"
    assert detalhe["candidate_page_recall_at_5"] == 0.0
    assert detalhe["candidate_document_recall_at_5"] == 1.0
    assert detalhe["candidate_top_1"] == {"file": "fonte.pdf", "pages": [7]}


def test_r4_usa_perfil_refinado_e_adia_promocao_para_r5_r6(monkeypatch):
    recuperado = {
        "rank": 1,
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "file": "fonte.pdf",
        "pages": [7],
        "rrf_score": 0.1,
        "context_chars": 120,
    }
    monkeypatch.setattr(
        benchmark,
        "recuperar_refinado_r4",
        lambda *args, **kwargs: [recuperado],
    )
    candidato = benchmark.executar_benchmark(
        _gold(),
        modelo_embeddings="modelo",
        colecao=_Colecao(),
        indice_lexical=_Lexical(),
        manifesto_snapshot={
            **_manifesto(),
            "schema_version": 2,
            "retrieval_text_strategy": "deterministic_document_context_v1",
        },
        relogio=lambda: 0.0,
        warmup=False,
        benchmark_id="evidence-rag-r4-hybrid-refinement",
        stage="R4",
        variant="filtered_hybrid_neighborhood_v1",
        retrieval_profile=benchmark.RETRIEVAL_PROFILE_R4,
    )
    baseline = copy.deepcopy(candidato)
    baseline.update(stage="R3", benchmark_id="r3")
    baseline["retrieval"].update(
        profile="baseline",
        reranker="deterministic_local_v1",
        explicit_metadata_filtered_search=False,
        neighborhood_expansion=False,
    )
    baseline["summary"]["recall@5"] = 0.0
    baseline["queries"][0]["metrics"]["5"]["recall@5"] = 0.0

    comparacao = benchmark.comparar_benchmarks(baseline, candidato)

    assert candidato["retrieval"]["explicit_metadata_filtered_search"] is True
    assert candidato["retrieval"]["neighborhood_expansion"] is True
    assert comparacao["ranking_contract_preserved"] is False
    assert comparacao["ranking_change_expected"] is True
    assert comparacao["promotion_eligible_after_quality_stages"] is True
    assert comparacao["promotion_decision"] == "deferred_to_r5_r6"
    candidato["comparison_to_baseline"] = comparacao
    texto = benchmark.relatorio_markdown(candidato)
    assert "Refinamento híbrido R4" in texto
    assert "avaliação continua em R5–R6" in texto


def test_relatorio_markdown_explicita_estado_provisorio(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "recuperar_baseline",
        lambda *args, **kwargs: [],
    )
    tempos = iter([0.0, 0.0, 1.0, 1.1])
    resultado = benchmark.executar_benchmark(
        _gold(),
        modelo_embeddings="modelo",
        colecao=_Colecao(),
        indice_lexical=_Lexical(),
        manifesto_snapshot=_manifesto(),
        relogio=lambda: next(tempos),
        warmup=False,
    )

    texto = benchmark.relatorio_markdown(resultado)

    assert "baseline R0–R1" in texto
    assert "pendente de revisão" in texto
    assert "Nenhum parâmetro de ranking foi modificado" in texto
    assert "Consultas sem acerto no top-5" in texto


def test_hash_json_independe_da_ordem_das_chaves():
    assert benchmark.hash_json_sha256({"a": 1, "b": 2}) == benchmark.hash_json_sha256(
        {"b": 2, "a": 1}
    )
