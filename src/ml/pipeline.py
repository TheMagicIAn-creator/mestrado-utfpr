"""Orquestra as duas publicações científicas canônicas do projeto."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

from src.core.config import RAIZ_PROJETO
from src.ml.dados_gpvs import ALL_EXPERIMENTS, DATASET_DIR
from src.ml.proveniencia import (
    carregar_manifesto,
    comparar,
    funcao_de_hash_para,
    gerar_manifesto,
)


ROOT = Path(RAIZ_PROJETO)
RESULTS_ROOT = ROOT / "resultados"
MANIFEST_ROOT = RESULTS_ROOT / "manifestos"


@dataclass(frozen=True)
class PipelineStage:
    key: str
    label: str
    manifest_name: str
    runner_module: str
    runner_function: str
    required_outputs: tuple[str, ...]
    requires_gpvs: bool = False

    @property
    def manifest_path(self) -> Path:
        return MANIFEST_ROOT / f"{self.manifest_name}.json"

    def load_runner(self) -> Callable:
        return getattr(import_module(self.runner_module), self.runner_function)

    def output_paths(self) -> list[Path]:
        manifesto = _read_json(self.manifest_path)
        listed = manifesto.get("outputs", []) if manifesto else []
        outputs = [ROOT / item for item in listed if isinstance(item, str)]
        if outputs:
            return outputs
        return [ROOT / item for item in self.required_outputs]

    def paths(self) -> list[Path]:
        return [self.manifest_path, *self.output_paths()]

    def is_complete(self) -> bool:
        return all(path.is_file() for path in self.paths())


STAGES: dict[str, PipelineStage] = {
    "comparacao": PipelineStage(
        key="comparacao",
        label="Comparação Denso versus AE-LSTM (E3)",
        manifest_name="comparacao_autoencoders",
        runner_module="src.ml.comparacao_autoencoders",
        runner_function="run",
        required_outputs=(
            "resultados/comparacao/comparacao_autoencoders.json",
            "resultados/comparacao/relatorio_comparacao.md",
        ),
        requires_gpvs=True,
    ),
    "confiabilidade": PipelineStage(
        key="confiabilidade",
        label="Confiabilidade física bibliográfica",
        manifest_name="confiabilidade_componentes",
        runner_module="src.ml.publicacao_confiabilidade",
        runner_function="generate",
        required_outputs=(
            "resultados/confiabilidade/metodologia.json",
            "resultados/confiabilidade/relatorio.md",
        ),
    ),
}

ORDEM_ETAPAS_ML = tuple(STAGES)
NOMES_ETAPAS = {key: stage.label for key, stage in STAGES.items()}

STAGE_ALIASES = {
    "comparacao_autoencoders": "comparacao",
    "confiabilidade_componentes": "confiabilidade",
}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


ARTEFATOS_ML = {
    key: [str(path.relative_to(ROOT)).replace("\\", "/") for path in stage.paths()]
    for key, stage in STAGES.items()
}


def _canonical_key(key: str) -> str:
    return STAGE_ALIASES.get(key, key)


def get_stage(key: str) -> PipelineStage:
    canonical = _canonical_key(key)
    try:
        return STAGES[canonical]
    except KeyError as exc:
        raise ValueError(f"Etapa desconhecida: {key}") from exc


def _gpvs_paths() -> dict[str, Path]:
    return {name: DATASET_DIR / f"{name}.csv" for name in ALL_EXPERIMENTS}


def capacidade_recalculo_pipeline() -> dict:
    """Informa se o ambiente possui os 16 ensaios exigidos pelo treino."""

    expected = _gpvs_paths()
    missing = [str(path) for path in expected.values() if not path.is_file()]
    available = not missing
    executable = ["confiabilidade"]
    if available:
        executable.insert(0, "comparacao")
    return {
        "disponivel": available,
        "modo": "calculo_local" if available else "consulta_publicada",
        "dataset": str(DATASET_DIR),
        "arquivos_esperados": len(expected),
        "arquivos_ausentes": missing,
        "etapas_executaveis": executable,
    }


def estado_resultados_publicados() -> dict[str, dict]:
    states: dict[str, dict] = {}
    for key, stage in STAGES.items():
        paths = stage.paths()
        present = sum(path.is_file() for path in paths)
        states[key] = {
            "disponivel": present == len(paths),
            "presentes": present,
            "esperados": len(paths),
        }
    return states


def _comparison_manifest(outputs: list[Path]) -> dict:
    from src.ml.estatistica_comparacao import BOOTSTRAP_RESAMPLES
    from src.ml.publicacao_comparacao import RESULTS_DIR
    from src.ml.modelos_autoencoder import SCORE_TOP_K
    from src.ml.treino_comparacao import (
        MODEL_IDS,
        MODEL_ROOT,
        REFERENCE_SEED,
        STABILITY_SEEDS,
        THRESHOLD_PERCENTILE,
    )

    source = ROOT / "src" / "ml"
    model_inputs = {
        f"{model_id}_{name}": MODEL_ROOT / model_id / name
        for model_id in MODEL_IDS
        for name in (
            "modelo.pt",
            "scaler.pkl",
            "normalizacao_baseline_gpvs.npz",
            "historico_treino.csv",
            "contrato.json",
        )
    }
    inputs = {f"raw_{name}": path for name, path in _gpvs_paths().items()}
    inputs.update(model_inputs)
    saved = carregar_manifesto("comparacao_autoencoders") or {}
    parameters = saved.get("parameters") or {
        "dataset": "GPVS-Faults",
        "models": list(MODEL_IDS),
        "reference_seed": REFERENCE_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "score_top_k": SCORE_TOP_K,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    return gerar_manifesto(
        "comparacao_autoencoders",
        source / "comparacao_autoencoders.py",
        parameters,
        inputs,
        outputs or [RESULTS_DIR / "comparacao_autoencoders.json"],
        code_dependencies={
            "dataset": source / "dados_gpvs.py",
            "models": source / "modelos_autoencoder.py",
            "training": source / "treino_comparacao.py",
            "evaluation": source / "avaliacao_comparativa.py",
            "statistics": source / "estatistica_comparacao.py",
            "plots": source / "graficos_comparacao.py",
            "publication": source / "publicacao_comparacao.py",
        },
        evidence_level="E3_bench",
    )


def _reliability_manifest(outputs: list[Path]) -> dict:
    from src.ml.confiabilidade_componentes import HOURS_PER_YEAR, SCENARIOS, SOURCE_PDF
    from src.ml.publicacao_confiabilidade import HORIZON_YEARS, N_POINTS

    source = ROOT / "src" / "ml"
    return gerar_manifesto(
        "confiabilidade_componentes",
        source / "publicacao_confiabilidade.py",
        {
            "model": "exponential_constant_hazard",
            "hours_per_year": HOURS_PER_YEAR,
            "horizon_years": HORIZON_YEARS,
            "n_points": N_POINTS,
            "scenarios": [scenario.scenario_id for scenario in SCENARIOS],
        },
        {"torres_tcc": ROOT / SOURCE_PDF},
        outputs,
        code_dependencies={
            "reliability": source / "confiabilidade_componentes.py",
            "plots": source / "graficos_confiabilidade.py",
            "style": source / "estilo_graficos.py",
        },
        evidence_level="bibliographic_sensitivity",
    )


def _current_manifest(stage: PipelineStage) -> dict:
    outputs = stage.output_paths()
    if stage.key == "comparacao":
        return _comparison_manifest(outputs)
    return _reliability_manifest(outputs)


def _output_hash_reasons(saved: dict) -> list[str]:
    reasons: list[str] = []
    for relative, expected in (saved.get("output_artifacts") or {}).items():
        path = ROOT / relative
        if not path.is_file():
            reasons.append(f"saída ausente: {relative}")
            continue
        current = funcao_de_hash_para(path)(path)
        if current != expected:
            reasons.append(f"hash divergente: {relative}")
    return reasons


def estado_etapa_completo(key: str) -> dict:
    stage = get_stage(key)
    if not stage.is_complete():
        return {"estado": "pending", "motivos": ["artefato(s) ausente(s)"]}
    saved = carregar_manifesto(stage.manifest_name)
    if not saved:
        return {"estado": "pending", "motivos": ["sem manifesto v2"]}
    reasons = _output_hash_reasons(saved)
    try:
        reasons.extend(
            comparar(saved, _current_manifest(stage), permitir_inputs_ausentes=True)
        )
    except Exception as exc:  # diagnóstico deve continuar disponível
        reasons.append(f"não foi possível validar a proveniência: {exc}")
    return {"estado": "stale" if reasons else "ready", "motivos": reasons}


def estado_publicacao(key: str) -> dict:
    """Estado rápido dos artefatos publicados, sem re-hashear o dataset bruto."""

    stage = get_stage(key)
    if not stage.is_complete():
        return {
            "estado": "pending",
            "motivos": ["artefato(s) ausente(s)"],
            "escopo_verificacao": "published_outputs",
        }
    saved = carregar_manifesto(stage.manifest_name)
    if not saved:
        return {
            "estado": "pending",
            "motivos": ["sem manifesto v2"],
            "escopo_verificacao": "published_outputs",
        }
    reasons = _output_hash_reasons(saved)
    return {
        "estado": "stale" if reasons else "ready",
        "motivos": reasons,
        "escopo_verificacao": "published_outputs",
    }


def estado_pipeline() -> dict[str, dict]:
    return {key: estado_etapa_completo(key) for key in ORDEM_ETAPAS_ML}


def pipeline_status() -> dict[str, bool]:
    return {
        key: state["estado"] == "ready"
        for key, state in estado_pipeline().items()
    }


def etapa_pendente(key: str) -> bool:
    return estado_etapa_completo(key)["estado"] != "ready"


def status_markdown() -> str:
    labels = {"ready": "pronto", "stale": "desatualizado", "pending": "pendente"}
    lines = ["## Estado das publicações científicas", ""]
    for key, state in estado_pipeline().items():
        detail = "; ".join(state["motivos"])
        suffix = f" ({detail})" if detail else ""
        lines.append(f"- **{NOMES_ETAPAS[key]}:** {labels[state['estado']]}{suffix}")
    return "\n".join(lines)


def artefatos_a_partir(etapa_inicial: str) -> list[Path]:
    """Retorna apenas os artefatos da publicação escolhida."""

    return get_stage(etapa_inicial).paths()


def limpar_artefatos(etapa_inicial: str) -> list[Path]:
    removed: list[Path] = []
    for path in artefatos_a_partir(etapa_inicial):
        resolved = path.resolve()
        if RESULTS_ROOT.resolve() not in resolved.parents or not resolved.is_file():
            continue
        resolved.unlink()
        removed.append(resolved)
    return removed


def dependencias_pendentes(etapa: str) -> list[str]:
    get_stage(etapa)
    return []


def executar_etapa(
    etapa: str,
    *,
    force: bool = False,
    progresso=None,
) -> dict:
    stage = get_stage(etapa)
    if stage.requires_gpvs and not capacidade_recalculo_pipeline()["disponivel"]:
        return {
            "ok": False,
            "etapa": stage.label,
            "mensagem": "Os 16 CSVs GPVS-Faults não estão disponíveis neste ambiente.",
        }
    if progresso:
        progresso(f"Executando {stage.label}...")
    try:
        runner = stage.load_runner()
        result = runner(force_features=force) if stage.requires_gpvs else runner()
    except Exception as exc:
        return {"ok": False, "etapa": stage.label, "mensagem": str(exc)}
    return {
        "ok": bool(result.get("ok", True)) if isinstance(result, dict) else True,
        "etapa": stage.label,
        "mensagem": f"{stage.label} concluída e publicada.",
        "resultado": result,
    }


def executar_pipeline_ml(
    etapa_inicial: str = "comparacao",
    *,
    force: bool = False,
    progresso=None,
) -> list[str]:
    start = ORDEM_ETAPAS_ML.index(get_stage(etapa_inicial).key)
    messages: list[str] = []
    for key in ORDEM_ETAPAS_ML[start:]:
        result = executar_etapa(key, force=force, progresso=progresso)
        prefix = "OK" if result["ok"] else "ERRO"
        messages.append(f"{prefix} - {result['etapa']}: {result['mensagem']}")
        if not result["ok"]:
            break
    return messages


def regenerar_pipeline(
    etapa_inicial: str = "comparacao",
    *,
    force: bool = True,
    progresso=None,
) -> list[str]:
    return executar_pipeline_ml(etapa_inicial, force=force, progresso=progresso)


__all__ = [
    "ARTEFATOS_ML",
    "NOMES_ETAPAS",
    "ORDEM_ETAPAS_ML",
    "STAGES",
    "PipelineStage",
    "artefatos_a_partir",
    "capacidade_recalculo_pipeline",
    "dependencias_pendentes",
    "estado_etapa_completo",
    "estado_pipeline",
    "estado_resultados_publicados",
    "etapa_pendente",
    "executar_etapa",
    "executar_pipeline_ml",
    "get_stage",
    "limpar_artefatos",
    "pipeline_status",
    "regenerar_pipeline",
    "status_markdown",
]
