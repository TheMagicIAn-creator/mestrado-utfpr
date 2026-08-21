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
MANIFEST_NAMES = {
    "comparacao_autoencoders.json",
    "confiabilidade_componentes.json",
}
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
    for name in sorted(MANIFEST_NAMES):
        artifact_count += _audit_manifest(root, manifests / name, errors)

    comparison = _read_json(results / "comparacao" / "comparacao_autoencoders.json")
    if comparison.get("dataset", {}).get("dataset") != "GPVS-Faults":
        errors.append("a comparação não declara GPVS-Faults como dataset único")
    if comparison.get("dataset", {}).get("active_dataset_count") != 1:
        errors.append("active_dataset_count deve ser 1")
    if set(comparison.get("models", {})) != {"ae_denso", "ae_lstm"}:
        errors.append("a comparação deve conter somente AE Denso e AE-LSTM")

    reliability = _read_json(results / "confiabilidade" / "metodologia.json")
    expected_role = "detector_evaluation_only_not_physical_reliability"
    if reliability.get("dataset_role") != expected_role:
        errors.append("o papel do GPVS na confiabilidade física está ambíguo")
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
