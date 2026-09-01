"""Audita a publicação canônica sem treinar modelos nem ler dados brutos."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import RAIZ_PROJETO
from src.ml.proveniencia import funcao_de_hash_para

RESULT_DIRS = {"comparacao", "confiabilidade", "manifestos"}
SCIENTIFIC_MANIFEST_NAMES = {
    "comparacao_autoencoders.json",
    "confiabilidade_componentes.json",
}
EVALUATION_MANIFESTS = {
    "evidence_rag_baseline_v1.json": ("R0-R1", "baseline_current", 1),
    "evidence_rag_contextual_r3.json": (
        "R3",
        "deterministic_document_context_v1",
        2,
    ),
    "evidence_rag_hybrid_r4.json": (
        "R4",
        "filtered_hybrid_neighborhood_v1",
        2,
    ),
    "evidence_rag_promotion_r6.json": (
        "R6",
        "promoted_hybrid_evidence_guard_v1",
        2,
    ),
    "evidence_rag_schema_v2_r2.json": ("R2", "jsonl_schema_v2_identity", 2),
}
MANIFEST_NAMES = SCIENTIFIC_MANIFEST_NAMES | set(EVALUATION_MANIFESTS)
EVIDENCE_GUARD_MANIFEST = "evidence_rag_guard_r5.json"
MANIFEST_NAMES.add(EVIDENCE_GUARD_MANIFEST)
EVIDENCE_GRAPH_MANIFEST = "evidence_graph_pilot_r7.json"
MANIFEST_NAMES.add(EVIDENCE_GRAPH_MANIFEST)
LEGACY_DIRS = {"auditoria", "autoencoder", "gpvs", "macro", "qualidade", "v2"}


def _reject_constant(value: str):
    raise ValueError(f"constante JSON não finita: {value}")


def _read_json(path: Path) -> dict:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"o JSON raiz deve ser um objeto: {path}")
    return payload


def _audit_manifest(root: Path, path: Path, errors: list[str]) -> int:
    try:
        manifest = _read_json(path)
    except (OSError, ValueError) as exc:
        errors.append(f"manifesto inválido {path.name}: {exc}")
        return 0

    if manifest.get("manifest_version") != 2:
        errors.append(f"{path.name}: manifest_version deve ser 2")

    outputs = manifest.get("outputs")
    hashes = manifest.get("output_artifacts")
    if not isinstance(outputs, list) or not isinstance(hashes, dict):
        errors.append(f"{path.name}: outputs/output_artifacts inválidos")
        return 0
    if set(outputs) != set(hashes):
        errors.append(f"{path.name}: lista de outputs diverge dos hashes")

    checked = 0
    for relative, expected_hash in hashes.items():
        artifact = root / relative
        if not artifact.is_file():
            errors.append(f"artefato ausente: {relative}")
            continue
        if artifact.stat().st_size == 0:
            errors.append(f"artefato vazio: {relative}")
            continue
        actual_hash = funcao_de_hash_para(artifact)(artifact)
        if actual_hash != expected_hash:
            errors.append(f"hash divergente: {relative}")
            continue
        checked += 1
    return checked


def _audit_retrieval_identity(
    manifest: dict,
    errors: list[str],
    *,
    path: Path,
    expected_stage: str,
    expected_variant: str,
) -> None:
    if manifest.get("schema_version") != 1:
        errors.append(f"{path.name}: schema_version deve ser 1")
    if (
        manifest.get("stage") != expected_stage
        or manifest.get("variant") != expected_variant
    ):
        errors.append(f"{path.name}: etapa ou variante de retrieval inválida")


def _audit_retrieval_gold(
    manifest: dict,
    errors: list[str],
    path: Path,
    expected_stage: str,
) -> None:
    gold = manifest.get("gold_set", {})
    status_esperado = (
        "researcher_approved_R6"
        if expected_stage == "R6"
        else "provisional_pending_researcher_review"
    )
    if gold.get("status") != status_esperado:
        errors.append(f"{path.name}: estado do gold set divergente")
    if gold.get("researcher_review_required_at") != "R6":
        errors.append(f"{path.name}: revisão humana R6 não está registrada")


def _audit_retrieval_corpus(
    manifest: dict,
    errors: list[str],
    path: Path,
    expected_snapshot_schema: int | None,
) -> None:
    corpus = manifest.get("corpus", {})
    if (
        expected_snapshot_schema is not None
        and corpus.get("snapshot_schema_version") != expected_snapshot_schema
    ):
        errors.append(f"{path.name}: schema do snapshot divergente")
    if corpus.get("n_documents") != 44 or corpus.get("n_chunks") != 12556:
        errors.append(f"{path.name}: inventário do corpus divergente")
    if corpus.get("collection_count") != corpus.get("n_chunks"):
        errors.append(f"{path.name}: coleção medida não corresponde ao snapshot")


def _audit_retrieval_metrics(
    manifest: dict,
    errors: list[str],
    path: Path,
) -> list | None:
    metricas = manifest.get("metric_definition", {})
    if metricas.get("primary_level") != "page":
        errors.append(f"{path.name}: métrica primária deve usar evidência por página")
    resumo = manifest.get("summary", {})
    if (
        resumo.get("n_queries") != 40
        or resumo.get("n_retrieval_queries") != 39
        or resumo.get("n_future_abstention_queries") != 1
    ):
        errors.append(f"{path.name}: contagem de perguntas do gold set divergente")
    consultas = manifest.get("queries")
    if not isinstance(consultas, list) or len(consultas) != 40:
        errors.append(f"{path.name}: resultados por pergunta incompletos")
        return None
    if len({item.get("query_id") for item in consultas}) != len(consultas):
        errors.append(f"{path.name}: query_id duplicado")
    return consultas


def _audit_retrieval_r2(manifest: dict, errors: list[str], path: Path) -> None:
    comparacao = manifest.get("comparison_to_baseline", {})
    gates = (
        "corpus_identity_preserved",
        "ranking_contract_preserved",
        "scientific_metrics_identical",
    )
    if not all(comparacao.get(campo) is True for campo in gates):
        errors.append(f"{path.name}: gate de regressão R2 não foi aprovado")
    deltas = comparacao.get("metric_deltas", {}).values()
    if any(not math.isclose(float(valor), 0.0, abs_tol=1e-12) for valor in deltas):
        errors.append(f"{path.name}: métricas científicas divergiram do baseline")
    retrieval = manifest.get("retrieval", {})
    if retrieval.get("retrieval_text_strategy") != "identity_raw_text":
        errors.append(f"{path.name}: R2 não usa estratégia de texto identidade")
    if retrieval.get("raw_text_separated_from_retrieval_text") is not True:
        errors.append(f"{path.name}: contrato raw/retrieval não está separado")


def _audit_retrieval_r3(manifest: dict, errors: list[str], path: Path) -> None:
    comparacao = manifest.get("comparison_to_baseline", {})
    if comparacao.get("baseline_benchmark_id") != "evidence-rag-r2-schema-v2":
        errors.append(f"{path.name}: R3 não foi comparado diretamente ao R2")
    if not all(
        comparacao.get(campo) is True
        for campo in ("corpus_identity_preserved", "ranking_contract_preserved")
    ):
        errors.append(f"{path.name}: identidade de corpus ou ranking mudou em R3")
    if comparacao.get("promotion_decision") != "deferred_to_r4_r5_r6":
        errors.append(f"{path.name}: candidato R3 foi promovido antes dos gates")
    retrieval = manifest.get("retrieval", {})
    if retrieval.get("retrieval_text_strategy") != (
        "deterministic_document_context_v1"
    ):
        errors.append(f"{path.name}: estratégia contextual R3 divergente")
    if retrieval.get("contextual_retrieval") is not True:
        errors.append(f"{path.name}: Contextual Retrieval R3 não está declarado")
    if retrieval.get("llm_contextualization_used") is not False:
        errors.append(f"{path.name}: R3 usou contextualização por LLM")
    if retrieval.get("parallel_candidate_index") is not True:
        errors.append(f"{path.name}: R3 não foi medido em índice paralelo")


def _audit_retrieval_r4(manifest: dict, errors: list[str], path: Path) -> None:
    comparacao = manifest.get("comparison_to_baseline", {})
    if comparacao.get("baseline_benchmark_id") != (
        "evidence-rag-r3-contextual-deterministic"
    ):
        errors.append(f"{path.name}: R4 não foi comparado diretamente ao R3")
    if comparacao.get("corpus_identity_preserved") is not True:
        errors.append(f"{path.name}: identidade do corpus mudou em R4")
    if comparacao.get("quality_gain_observed") is not True:
        errors.append(f"{path.name}: R4 não demonstrou ganho de qualidade")
    if comparacao.get("regressed_queries_at_5"):
        errors.append(f"{path.name}: R4 contém regressões em recall de página@5")
    if comparacao.get("critical_simple_regressions"):
        errors.append(f"{path.name}: R4 contém regressões simples críticas")
    if comparacao.get("promotion_decision") != "deferred_to_r5_r6":
        errors.append(f"{path.name}: candidato R4 avançou antes dos gates R5-R6")
    retrieval = manifest.get("retrieval", {})
    gates = {
        "parallel_candidate_index": True,
        "explicit_metadata_filtered_search": True,
        "metadata_filters_are_advisory": True,
        "neighborhood_expansion": True,
        "evidence_package": False,
        "evidence_guard": False,
    }
    for campo, esperado in gates.items():
        if retrieval.get(campo) is not esperado:
            errors.append(f"{path.name}: contrato R4 divergente em {campo}")
    if retrieval.get("metadata_filtered_results_per_query") != 3:
        errors.append(f"{path.name}: limite filtrado R4 divergente")
    if not math.isclose(
        float(retrieval.get("metadata_filtered_rrf_weight", -1.0)),
        0.25,
        abs_tol=1e-12,
    ):
        errors.append(f"{path.name}: peso filtrado R4 divergente")


def _audit_retrieval_r6(manifest: dict, errors: list[str], path: Path) -> None:
    comparacao = manifest.get("comparison_to_baseline", {})
    gates = {
        "corpus_identity_preserved": True,
        "quality_gain_observed": True,
        "promotion_eligible_after_quality_stages": True,
    }
    if comparacao.get("baseline_benchmark_id") != (
        "evidence-rag-r3-contextual-deterministic"
    ):
        errors.append(f"{path.name}: promoção R6 não foi comparada ao R3")
    for campo, esperado in gates.items():
        if comparacao.get(campo) is not esperado:
            errors.append(f"{path.name}: gate R6 divergente em {campo}")
    if comparacao.get("regressed_queries_at_5"):
        errors.append(f"{path.name}: promoção R6 contém regressões")
    if comparacao.get("promotion_decision") != "promoted_R6":
        errors.append(f"{path.name}: decisão de promoção R6 ausente")
    promocao = manifest.get("promotion", {})
    if promocao.get("researcher_review") != "approved_R6_2026-08-30":
        errors.append(f"{path.name}: aprovação do pesquisador não registrada")
    if promocao.get("snapshot_restored_from_empty_runtime") is not True:
        errors.append(f"{path.name}: snapshot promovido não foi restaurado")
    if promocao.get("local_and_deploy_contract_identical") is not True:
        errors.append(f"{path.name}: contrato local/deploy divergente")
    if promocao.get("retrieval_profile_default") != "r4_hybrid":
        errors.append(f"{path.name}: perfil promovido divergente")
    if promocao.get("rollback_profile") != "baseline":
        errors.append(f"{path.name}: rollback de ranking ausente")
    if not promocao.get("previous_snapshot_sha256"):
        errors.append(f"{path.name}: hash do snapshot de rollback ausente")
    if promocao.get("promoted_snapshot_sha256") != manifest.get("corpus", {}).get(
        "snapshot_file_sha256"
    ):
        errors.append(f"{path.name}: hash do snapshot promovido divergente")
    resumo_guard = promocao.get("evidence_guard_summary", {})
    for campo in (
        "citation_validity",
        "invalid_claim_rejection_rate",
        "abstention_accuracy",
        "memory_rejection_accuracy",
    ):
        if not math.isclose(float(resumo_guard.get(campo, -1.0)), 1.0):
            errors.append(f"{path.name}: gate do Evidence Guard divergente em {campo}")


def _audit_retrieval_result(
    path: Path,
    errors: list[str],
    *,
    expected_stage: str,
    expected_variant: str,
    expected_snapshot_schema: int | None,
) -> None:
    try:
        manifest = _read_json(path)
    except (OSError, ValueError) as exc:
        errors.append(f"manifesto inválido {path.name}: {exc}")
        return

    _audit_retrieval_identity(
        manifest,
        errors,
        path=path,
        expected_stage=expected_stage,
        expected_variant=expected_variant,
    )
    _audit_retrieval_gold(manifest, errors, path, expected_stage)
    _audit_retrieval_corpus(manifest, errors, path, expected_snapshot_schema)
    _audit_retrieval_metrics(manifest, errors, path)
    if expected_stage == "R2":
        _audit_retrieval_r2(manifest, errors, path)
    elif expected_stage == "R3":
        _audit_retrieval_r3(manifest, errors, path)
    elif expected_stage == "R4":
        _audit_retrieval_r4(manifest, errors, path)
    elif expected_stage == "R6":
        _audit_retrieval_r6(manifest, errors, path)


def _audit_retrieval_baseline(path: Path, errors: list[str]) -> None:
    """Compatibilidade com chamadas que auditam apenas o baseline histórico."""
    _audit_retrieval_result(
        path,
        errors,
        expected_stage="R0-R1",
        expected_variant="baseline_current",
        expected_snapshot_schema=None,
    )


def _audit_evidence_guard(path: Path, errors: list[str]) -> None:
    try:
        manifest = _read_json(path)
    except (OSError, ValueError) as exc:
        errors.append(f"manifesto inválido {path.name}: {exc}")
        return
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "R5"
        or manifest.get("variant") != "deterministic_claim_evidence_guard_v1"
    ):
        errors.append(f"{path.name}: identidade do Evidence Guard divergente")
    guard = manifest.get("guard", {})
    gates = {
        "external_model_required": False,
        "claim_evidence_chain": True,
        "quote_uses_raw_text": True,
        "memory_is_scientific_source": False,
        "abstention_enabled": True,
    }
    for campo, esperado in gates.items():
        if guard.get(campo) is not esperado:
            errors.append(f"{path.name}: contrato divergente em {campo}")
    resumo = manifest.get("summary", {})
    for campo in (
        "citation_validity",
        "invalid_claim_rejection_rate",
        "abstention_accuracy",
        "memory_rejection_accuracy",
    ):
        if not math.isclose(float(resumo.get(campo, -1.0)), 1.0, abs_tol=1e-12):
            errors.append(f"{path.name}: gate R5 reprovado em {campo}")
    if not math.isclose(
        float(resumo.get("unsupported_claim_rate_after_guard", -1.0)),
        0.0,
        abs_tol=1e-12,
    ):
        errors.append(f"{path.name}: claims sem suporte passaram pelo guard")


def _audit_evidence_graph(path: Path, root: Path, errors: list[str]) -> None:
    try:
        manifest = _read_json(path)
    except (OSError, ValueError) as exc:
        errors.append(f"manifesto inválido {path.name}: {exc}")
        return
    if (
        manifest.get("stage") != "R7"
        or manifest.get("variant") != "lightweight_evidence_anchored_graph_v1"
    ):
        errors.append(f"{path.name}: identidade do piloto R7 divergente")
    contracts = manifest.get("contracts", {})
    expected = {
        "primary_retrieval": False,
        "full_graphrag": False,
        "raptor_enabled": False,
        "llm_relation_extraction": False,
        "literal_entity_match_required": True,
        "evidence_id_required_per_edge": True,
        "chunk_id_required_per_edge": True,
        "memory_is_scientific_source": False,
    }
    for campo, valor in expected.items():
        if contracts.get(campo) is not valor:
            errors.append(f"{path.name}: contrato R7 divergente em {campo}")
    taxonomy = manifest.get("taxonomy", {})
    taxonomy_path = root / str(taxonomy.get("path", ""))
    if not taxonomy_path.is_file():
        errors.append(f"{path.name}: taxonomia R7 ausente")
    elif funcao_de_hash_para(taxonomy_path)(taxonomy_path) != taxonomy.get("sha256"):
        errors.append(f"{path.name}: hash da taxonomia R7 divergente")


def _audit_result_layout(results: Path, errors: list[str]) -> None:
    present_dirs = {path.name for path in results.iterdir() if path.is_dir()}
    extra_dirs = present_dirs - RESULT_DIRS
    missing_dirs = RESULT_DIRS - present_dirs
    if extra_dirs:
        errors.append(f"pastas de resultado não canônicas: {sorted(extra_dirs)}")
    if missing_dirs:
        errors.append(f"pastas de resultado ausentes: {sorted(missing_dirs)}")
    for legacy in sorted(LEGACY_DIRS):
        if (results / legacy).exists():
            errors.append(f"pasta legada ainda presente: resultados/{legacy}")


def _audit_scientific_contracts(results: Path, errors: list[str]) -> None:
    comparison = _read_json(results / "comparacao" / "comparacao_autoencoders.json")
    if comparison.get("dataset", {}).get("dataset") != "GPVS-Faults":
        errors.append("a comparação não declara GPVS-Faults como dataset único")
    if comparison.get("dataset", {}).get("active_dataset_count") != 1:
        errors.append("active_dataset_count deve ser 1")
    if set(comparison.get("models", {})) != {"ae_denso", "ae_lstm"}:
        errors.append("a comparação deve conter somente AE Denso e AE-LSTM")
    sensitivity = comparison.get("e3", {}).get("score_threshold_sensitivity", {})
    if sensitivity.get("top_k_values") != [5, 10, 20]:
        errors.append("a sensibilidade não usa a grade top-k canônica 5/10/20")
    if sensitivity.get("requested_percentiles") != [99.0, 99.5, 99.9]:
        errors.append("a sensibilidade não usa os percentis 99/99,5/99,9")
    if sensitivity.get("configuration_count_per_model_seed") != 9:
        errors.append("a sensibilidade não declara nove configurações por modelo/semente")
    if sensitivity.get("uses_fault_data_for_selection") is not False:
        errors.append("a publicação permite seleção com os ensaios F1-F7")
    temporal = comparison.get("e3", {}).get("temporal_ablation", {})
    if temporal.get("decision_target") != "W_t":
        errors.append("a análise temporal não declara decisão causal em W_t")

    reliability = _read_json(results / "confiabilidade" / "metodologia.json")
    if reliability.get("evidence_scope") != "bibliographic_reliability_only":
        errors.append("a confiabilidade não declara escopo exclusivamente bibliográfico")
    if "dataset_role" in reliability or "experimental_dataset" in reliability:
        errors.append("a confiabilidade física ainda depende de metadados experimentais")
    physical_weibull = reliability.get("physical_weibull", {})
    if physical_weibull.get("beta") is not None or physical_weibull.get("eta") is not None:
        errors.append("a publicação fabricou parâmetros Weibull físicos")
    fmeca = reliability.get("fmeca", {})
    if fmeca.get("status") != "awaiting_user_fmeca":
        errors.append("a FMECA não declara que aguarda os valores do pesquisador")
    component_ids = {
        item.get("component_id") for item in fmeca.get("components", [])
    }
    if component_ids != {
        "igbt",
        "sensor_feedback_system",
        "inverter_control_system",
    }:
        errors.append("a FMECA publicada não usa o trio metodológico vigente")
    if any(
        item.get(field) is not None
        for item in fmeca.get("components", [])
        for field in ("severity", "occurrence", "detectability", "npr")
    ):
        errors.append("a FMECA publicou S/O/D/NPR sem dados aprovados")
    serialized = json.dumps(reliability, ensure_ascii=False).lower()
    if any(field in serialized for field in ("pod_mon", "d_mon", "d_proj", "npr_proj")):
        errors.append("a publicação ainda contém campos de projeção revogados")


def auditar_publicacao(root: Path | str = RAIZ_PROJETO) -> dict:
    """Retorna um relatório determinístico dos resultados publicados."""
    root = Path(root).resolve()
    results = root / "resultados"
    manifests = results / "manifestos"
    errors: list[str] = []

    _audit_result_layout(results, errors)

    manifest_names = {path.name for path in manifests.glob("*.json")}
    if manifest_names != MANIFEST_NAMES:
        errors.append(
            "manifestos divergentes: "
            f"esperados={sorted(MANIFEST_NAMES)}, encontrados={sorted(manifest_names)}"
        )

    artifact_count = 0
    for name in sorted(SCIENTIFIC_MANIFEST_NAMES):
        artifact_count += _audit_manifest(root, manifests / name, errors)
    for name, (stage, variant, snapshot_schema) in sorted(
        EVALUATION_MANIFESTS.items()
    ):
        _audit_retrieval_result(
            manifests / name,
            errors,
            expected_stage=stage,
            expected_variant=variant,
            expected_snapshot_schema=snapshot_schema,
        )
    _audit_evidence_guard(manifests / EVIDENCE_GUARD_MANIFEST, errors)
    _audit_evidence_graph(manifests / EVIDENCE_GRAPH_MANIFEST, root, errors)

    _audit_scientific_contracts(results, errors)

    return {
        "ok": not errors,
        "errors": errors,
        "manifests": len(manifest_names),
        "artifacts": artifact_count,
    }


def main() -> int:
    report = auditar_publicacao()
    if report["ok"]:
        print(
            "APROVADO - publicação canônica íntegra: "
            f"{report['manifests']} manifestos, {report['artifacts']} artefatos."
        )
        return 0
    print("REPROVADO - inconsistências na publicação canônica:")
    for error in report["errors"]:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
