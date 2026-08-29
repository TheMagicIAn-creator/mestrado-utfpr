"""Audita a publicação canônica sem treinar modelos nem ler dados brutos."""

from __future__ import annotations

import json
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
EVALUATION_MANIFEST_NAMES = {"evidence_rag_baseline_v1.json"}
MANIFEST_NAMES = SCIENTIFIC_MANIFEST_NAMES | EVALUATION_MANIFEST_NAMES
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
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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


def _audit_retrieval_baseline(path: Path, errors: list[str]) -> None:
    try:
        manifest = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"manifesto inválido {path.name}: {exc}")
        return

    if manifest.get("schema_version") != 1:
        errors.append(f"{path.name}: schema_version deve ser 1")
    if manifest.get("stage") != "R0-R1" or manifest.get("variant") != "baseline_current":
        errors.append(f"{path.name}: etapa ou variante baseline inválida")
    gold = manifest.get("gold_set", {})
    if gold.get("status") != "provisional_pending_researcher_review":
        errors.append(f"{path.name}: gold set não está marcado como provisório")
    if gold.get("researcher_review_required_at") != "R6":
        errors.append(f"{path.name}: revisão humana R6 não está registrada")

    corpus = manifest.get("corpus", {})
    if corpus.get("n_documents") != 44 or corpus.get("n_chunks") != 12556:
        errors.append(f"{path.name}: inventário do corpus divergente")
    if corpus.get("collection_count") != corpus.get("n_chunks"):
        errors.append(f"{path.name}: coleção medida não corresponde ao snapshot")

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
    elif len({item.get("query_id") for item in consultas}) != len(consultas):
        errors.append(f"{path.name}: query_id duplicado")


def auditar_publicacao(root: Path | str = RAIZ_PROJETO) -> dict:
    """Retorna um relatório determinístico dos resultados publicados."""
    root = Path(root).resolve()
    results = root / "resultados"
    manifests = results / "manifestos"
    errors: list[str] = []

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

    manifest_names = {path.name for path in manifests.glob("*.json")}
    if manifest_names != MANIFEST_NAMES:
        errors.append(
            "manifestos divergentes: "
            f"esperados={sorted(MANIFEST_NAMES)}, encontrados={sorted(manifest_names)}"
        )

    artifact_count = 0
    for name in sorted(SCIENTIFIC_MANIFEST_NAMES):
        artifact_count += _audit_manifest(root, manifests / name, errors)
    for name in sorted(EVALUATION_MANIFEST_NAMES):
        _audit_retrieval_baseline(manifests / name, errors)

    comparison = _read_json(results / "comparacao" / "comparacao_autoencoders.json")
    if comparison.get("dataset", {}).get("dataset") != "GPVS-Faults":
        errors.append("a comparação não declara GPVS-Faults como dataset único")
    if comparison.get("dataset", {}).get("active_dataset_count") != 1:
        errors.append("active_dataset_count deve ser 1")
    if set(comparison.get("models", {})) != {"ae_denso", "ae_lstm"}:
        errors.append("a comparação deve conter somente AE Denso e AE-LSTM")

    reliability = _read_json(results / "confiabilidade" / "metodologia.json")
    if reliability.get("evidence_scope") != "bibliographic_reliability_only":
        errors.append("a confiabilidade não declara escopo exclusivamente bibliográfico")
    if "dataset_role" in reliability or "experimental_dataset" in reliability:
        errors.append("a confiabilidade física ainda depende de metadados experimentais")
    physical_weibull = reliability.get("physical_weibull", {})
    if physical_weibull.get("beta") is not None or physical_weibull.get("eta") is not None:
        errors.append("a publicação fabricou parâmetros Weibull físicos")

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
