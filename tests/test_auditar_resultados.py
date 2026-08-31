"""Execução independente do auditor da publicação científica."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.auditar_resultados import (
    _audit_evidence_graph,
    _audit_evidence_guard,
    _audit_retrieval_baseline,
    _audit_retrieval_r3,
    _audit_retrieval_r4,
)


def test_auditor_executa_fora_da_raiz(tmp_path):
    raiz = Path(__file__).resolve().parents[1]
    processo = subprocess.run(
        [sys.executable, str(raiz / "scripts" / "auditar_resultados.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert processo.returncode == 0, processo.stdout + processo.stderr
    assert "APROVADO" in processo.stdout
    assert "34 artefatos" in processo.stdout


def test_auditor_rejeita_contrato_retrieval_divergente(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stage": "R2",
                "variant": "experimental",
                "gold_set": {
                    "status": "final",
                    "researcher_review_required_at": "R7",
                },
                "corpus": {
                    "n_documents": 1,
                    "n_chunks": 2,
                    "collection_count": 3,
                },
                "metric_definition": {"primary_level": "document"},
                "summary": {
                    "n_queries": 1,
                    "n_retrieval_queries": 1,
                    "n_future_abstention_queries": 0,
                },
                "queries": [{"query_id": "duplicada"}] * 40,
            }
        ),
        encoding="utf-8",
    )
    errors = []

    _audit_retrieval_baseline(path, errors)

    assert len(errors) == 9
    assert any("schema_version" in error for error in errors)
    assert any("query_id duplicado" in error for error in errors)


def test_auditor_rejeita_resultados_incompletos_e_json_invalido(tmp_path):
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"queries": []}), encoding="utf-8")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    errors = []

    _audit_retrieval_baseline(incomplete, errors)
    _audit_retrieval_baseline(invalid, errors)

    assert any("resultados por pergunta incompletos" in error for error in errors)
    assert any("manifesto inválido" in error for error in errors)


def test_auditor_rejeita_candidato_r3_sem_gates_cientificos(tmp_path):
    errors = []

    _audit_retrieval_r3(
        {
            "comparison_to_baseline": {
                "baseline_benchmark_id": "baseline-incorreto",
                "corpus_identity_preserved": False,
                "ranking_contract_preserved": False,
                "promotion_decision": "promoted",
            },
            "retrieval": {
                "retrieval_text_strategy": "estrategia-incorreta",
                "contextual_retrieval": False,
                "llm_contextualization_used": True,
                "parallel_candidate_index": False,
            },
        },
        errors,
        tmp_path / "r3.json",
    )

    assert len(errors) == 7
    assert any("comparado diretamente ao R2" in error for error in errors)
    assert any("promovido antes dos gates" in error for error in errors)
    assert any("usou contextualização por LLM" in error for error in errors)


def test_auditor_rejeita_candidato_r4_sem_gates_cientificos(tmp_path):
    errors = []
    _audit_retrieval_r4(
        {
            "comparison_to_baseline": {
                "baseline_benchmark_id": "incorreto",
                "corpus_identity_preserved": False,
                "quality_gain_observed": False,
                "regressed_queries_at_5": ["q"],
                "critical_simple_regressions": ["q"],
                "promotion_decision": "promoted",
            },
            "retrieval": {},
        },
        errors,
        tmp_path / "r4.json",
    )
    assert any("comparado diretamente ao R3" in error for error in errors)
    assert any("ganho de qualidade" in error for error in errors)
    assert any("antes dos gates R5-R6" in error for error in errors)


def test_auditor_rejeita_evidence_guard_sem_integridade(tmp_path):
    path = tmp_path / "r5.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "R5",
                "variant": "deterministic_claim_evidence_guard_v1",
                "guard": {},
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    errors = []
    _audit_evidence_guard(path, errors)
    assert any("external_model_required" in error for error in errors)
    assert any("citation_validity" in error for error in errors)
    assert any("claims sem suporte" in error for error in errors)


def test_auditor_rejeita_evidence_graph_sem_ancoragem(tmp_path):
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "r7.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "R7",
                "variant": "lightweight_evidence_anchored_graph_v1",
                "contracts": {
                    "primary_retrieval": True,
                    "full_graphrag": False,
                    "raptor_enabled": False,
                    "llm_relation_extraction": False,
                    "literal_entity_match_required": True,
                    "evidence_id_required_per_edge": False,
                    "chunk_id_required_per_edge": False,
                    "memory_is_scientific_source": False,
                },
                "taxonomy": {
                    "path": "taxonomy.json",
                    "sha256": "hash-incorreto",
                },
            }
        ),
        encoding="utf-8",
    )
    errors = []

    _audit_evidence_graph(manifest_path, tmp_path, errors)

    assert any("primary_retrieval" in error for error in errors)
    assert any("evidence_id_required_per_edge" in error for error in errors)
    assert any("chunk_id_required_per_edge" in error for error in errors)
    assert any("hash da taxonomia" in error for error in errors)
