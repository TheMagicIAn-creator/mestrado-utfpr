"""Contratos científicos somente leitura para a aplicação canônica."""

from __future__ import annotations

import csv
import json
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.ml.proveniencia import funcao_de_hash_para, sha256_arquivo
from src.webapp.chart_data import (
    e2_detection_series,
    e2_empirical_series,
    e3_discrimination_series,
    reliability_curve_series,
)

ROOT = Path(RAIZ_PROJETO)
COMPARISON = ROOT / "resultados" / "comparacao"
RELIABILITY = ROOT / "resultados" / "confiabilidade"
MANIFESTS = ROOT / "resultados" / "manifestos"
LITERATURE = ROOT / "literatura" / "inversores-pv"

COMPARISON_JSON = COMPARISON / "comparacao_autoencoders.json"
RELIABILITY_JSON = RELIABILITY / "metodologia.json"
COMPARISON_MANIFEST = MANIFESTS / "comparacao_autoencoders.json"
RELIABILITY_MANIFEST = MANIFESTS / "confiabilidade_componentes.json"

_CONTRACT_LOCK = threading.RLock()
_CONTRACT_STATUS = {
    "state": "iniciando",
    "detail": None,
    "updated_at": None,
}


class ContratoWebInvalido(RuntimeError):
    """Artefato ausente, ilegível ou incompatível com a aplicação."""


def _reject_constant(value: str):
    raise ValueError(f"constante JSON não finita: {value}")


def _json(path: Path) -> dict:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except FileNotFoundError as exc:
        raise ContratoWebInvalido(f"Artefato ausente: {path}") from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ContratoWebInvalido(f"Artefato JSON inválido: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContratoWebInvalido(f"Contrato JSON deve ser um objeto: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except FileNotFoundError as exc:
        raise ContratoWebInvalido(f"Artefato ausente: {path}") from exc
    except (OSError, csv.Error) as exc:
        raise ContratoWebInvalido(f"Artefato CSV inválido: {path}: {exc}") from exc


def _boolean(value) -> bool:
    return str(value).strip().casefold() in {"1", "true", "sim", "yes"}


def _signature(paths: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError as exc:
            raise ContratoWebInvalido(f"Artefato ausente: {path}") from exc
        signature.append((path.as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


@lru_cache(maxsize=8)
def _comparison_payload(_signature_value) -> dict:
    payload = _json(COMPARISON_JSON)
    if payload.get("dataset", {}).get("dataset") != "GPVS-Faults":
        raise ContratoWebInvalido("A comparação ativa deve usar exclusivamente GPVS-Faults")
    if set(payload.get("models", {})) != {"ae_denso", "ae_lstm"}:
        raise ContratoWebInvalido("A comparação ativa deve conter Denso e AE-LSTM")
    return payload


@lru_cache(maxsize=8)
def _reliability_payload(_signature_value) -> dict:
    payload = _json(RELIABILITY_JSON)
    if payload.get("dataset_role") != "detector_evaluation_only_not_physical_reliability":
        raise ContratoWebInvalido("O papel do GPVS na confiabilidade física está ambíguo")
    return payload


def _artifact(area: str, filename: str, title: str, note: str = "") -> dict:
    folder = {
        "comparison": COMPARISON,
        "reliability": RELIABILITY,
        "manifests": MANIFESTS,
    }[area]
    path = folder / filename
    if not path.is_file():
        raise ContratoWebInvalido(f"Artefato ausente: {path}")
    return {
        "title": title,
        "note": note,
        "filename": filename,
        "url": f"/artifacts/{area}/{filename}",
        "size_bytes": path.stat().st_size,
        "sha256": funcao_de_hash_para(path)(path),
    }


def _figure(area: str, stem: str, title: str, note: str) -> dict:
    image = _artifact(area, f"{stem}.png", title, note)
    pdf = _artifact(area, f"{stem}.pdf", title, note)
    image["pdf_url"] = pdf["url"]
    image["pdf_sha256"] = pdf["sha256"]
    return image


def _set_contract_status(state: str, detail: str | None = None) -> None:
    with _CONTRACT_LOCK:
        _CONTRACT_STATUS.update(
            state=state,
            detail=detail,
            updated_at=agora_local().isoformat(),
        )


def contracts_status() -> dict:
    """Devolve estado barato; não lê nem recalcula artefatos."""
    with _CONTRACT_LOCK:
        return dict(_CONTRACT_STATUS)


def _metric_map(payload: dict) -> dict:
    metrics: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in payload["e3"]["macro"]:
        metrics[item["model"]][item["metric"]] = {
            "estimate": float(item["estimate"]),
            "ci95_low": float(item["ci95_low"]),
            "ci95_high": float(item["ci95_high"]),
            "n_experiments": int(item["n_experiments"]),
            "bootstrap_resamples": int(item["bootstrap_resamples"]),
            "bootstrap_unit": item["bootstrap_unit"],
        }
    return dict(metrics)


def _reference_trials() -> list[dict]:
    output = []
    for row in _csv(COMPARISON / "e3_metricas_por_ensaio.csv"):
        if not _boolean(row["is_reference"]):
            continue
        output.append(
            {
                "model": row["model"],
                "model_name": row["model_name"],
                "experiment": row["experiment"],
                "fault": int(row["fault"]),
                "fault_type": row["fault_type"],
                "mode": row["mode"],
                "mode_name": row["mode_name"],
                "auc_pr": float(row["auc_pr"]),
                "auc_roc": float(row["auc_roc"]),
                "sensitivity": float(row["sensitivity"]),
                "specificity": float(row["specificity"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "mcc": float(row["mcc"]),
                "f1": float(row["f1"]),
                "precision": float(row["precision"]),
                "false_positive_rate": float(row["false_positive_rate"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tp": int(row["tp"]),
            }
        )
    if len(output) != 28:
        raise ContratoWebInvalido(
            f"Esperados 14 ensaios x 2 modelos na semente 42; recebidos {len(output)}"
        )
    return output


def _confusion_matrices(trials: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    )
    names = {}
    for item in trials:
        names[item["model"]] = item["model_name"]
        for key in ("tn", "fp", "fn", "tp"):
            grouped[item["model"]][key] += int(item[key])
    return [
        {"model": model, "model_name": names[model], **counts}
        for model, counts in sorted(grouped.items())
    ]


def _stability_summary(payload: dict) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in payload["e3"]["stability"]:
        if item["metric"] == "auc_pr":
            grouped[item["model"]].append(float(item["macro_mean"]))
    return [
        {
            "model": model,
            "metric": "auc_pr",
            "n_seeds": len(values),
            "mean": fmean(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for model, values in grouped.items()
    ]


@lru_cache(maxsize=8)
def _e3_contract_cached(signature_value) -> dict:
    payload = _comparison_payload((signature_value[0],))
    trials = _reference_trials()
    contract = {
        "contract_version": 2,
        "evidence_level": payload["e3"]["evidence_level"],
        "dataset": {
            "name": payload["dataset"]["dataset"],
            "doi": payload["dataset"]["doi"],
            "experiments": payload["dataset"]["fault_experiments"],
            "fault_boundary": payload["dataset"]["fault_boundary"],
        },
        "generated_at": payload["created_at"],
        "primary_metric": payload["e3"]["primary_metric"],
        "models": payload["models"],
        "protocol": payload["protocol"],
        "metrics": _metric_map(payload),
        "paired_differences": payload["e3"]["paired_differences"],
        "stability": _stability_summary(payload),
        "trials": trials,
        "confusion_matrices": _confusion_matrices(trials),
        "discrimination": e3_discrimination_series(
            COMPARISON / "e3_escores_referencia.csv"
        ),
        "confusion_matrix_unit": payload["e3"]["confusion_matrix_unit"],
        "limitations": payload["limitations"],
        "figures": [
            _figure(
                "comparison",
                "e3_metricas_macro",
                "Desempenho macro nos ensaios GPVS-Faults",
                "Estimativas macro e IC95% por bootstrap no nível do ensaio.",
            ),
            _figure(
                "comparison",
                "e3_curvas_discriminacao",
                "Curvas de discriminação dos autoencoders",
                "AUC-PR é a métrica primária; ROC-AUC é apresentada como complemento.",
            ),
            _figure(
                "comparison",
                "e3_matrizes_confusao",
                "Matrizes de confusão no ponto operacional",
                "Contagens por janela, descritivas diante da autocorrelação intraensaio.",
            ),
            _figure(
                "comparison",
                "e3_resultados_por_ensaio",
                "Resultados por ensaio e condição operacional",
                "Heterogeneidade nos 14 ensaios F1L-F7M sem retreino ou recalibração.",
            ),
        ],
        "tables": [
            _artifact("comparison", "e3_metricas_macro.csv", "Métricas macro"),
            _artifact("comparison", "e3_metricas_por_ensaio.csv", "Métricas por ensaio"),
            _artifact("comparison", "e3_diferencas_pareadas.csv", "Diferenças pareadas"),
            _artifact("comparison", "e3_estabilidade_sementes.csv", "Estabilidade por semente"),
        ],
    }
    _set_contract_status("pronto")
    return contract


def e3_contract() -> dict:
    paths = (
        COMPARISON_JSON,
        COMPARISON / "e3_metricas_por_ensaio.csv",
        COMPARISON / "e3_escores_referencia.csv",
        COMPARISON / "e3_metricas_macro.png",
        COMPARISON / "e3_curvas_discriminacao.png",
        COMPARISON / "e3_matrizes_confusao.png",
        COMPARISON / "e3_resultados_por_ensaio.png",
    )
    return _e3_contract_cached(_signature(paths))


@lru_cache(maxsize=8)
def _e2_contract_cached(signature_value) -> dict:
    payload = _comparison_payload((signature_value[0],))
    model_names = {key: value["name"] for key, value in payload["models"].items()}
    component_names = {
        item["component_id"]: item["component_name"]
        for item in payload["e2"]["signatures"]
    }
    summary = []
    for item in payload["e2"]["summary"]:
        summary.append(
            {
                **item,
                "model_name": model_names[item["model"]],
                "component_name": component_names[item["component"]],
            }
        )
    contract = {
        "contract_version": 2,
        "evidence_level": payload["e2"]["evidence_level"],
        "axis": payload["e2"]["axis"],
        "axis_is_time": payload["e2"]["axis_is_time"],
        "magnitude_steps": payload["e2"]["magnitude_steps"],
        "persistence_width": payload["e2"]["persistence_width"],
        "interval_method": payload["e2"]["interval_method"],
        "interval_caveat": payload["e2"]["interval_caveat"],
        "smd95_definition": payload["e2"]["smd95_definition"],
        "signatures": payload["e2"]["signatures"],
        "summary": summary,
        "detection_series": e2_detection_series(
            COMPARISON / "e2_deteccao_por_magnitude.csv", model_names
        ),
        "empirical_series": e2_empirical_series(
            COMPARISON / "e2_funcoes_empiricas.csv",
            model_names,
            component_names,
        ),
        "weibull_role": payload["e2"]["weibull_role"],
        "weibull_acceptance_scope": payload["e2"]["weibull_acceptance_scope"],
        "figures": [
            _figure(
                "comparison",
                "e2_deteccao_por_magnitude",
                "Probabilidade de detecção por magnitude sintética",
                "Curvas compartilhadas para os dois modelos e três assinaturas FMECA.",
            ),
            _figure(
                "comparison",
                "e2_smd95",
                "Limite de detectabilidade SMD95",
                "Menor magnitude cujo limite inferior do IC95% atinge 95%.",
            ),
            _figure(
                "comparison",
                "e2_funcoes_empiricas",
                "Funções empíricas de primeiro cruzamento",
                "Sobrevivência, incidência acumulada e risco discreto no eixo a_det.",
            ),
            _figure(
                "comparison",
                "e2_diagnostico_weibull",
                "Diagnóstico Weibull no papel de probabilidade",
                "Os ajustes são diagnósticos e não foram aceitos para síntese paramétrica.",
            ),
        ],
        "tables": [
            _artifact("comparison", "e2_resumo.csv", "Resumo SMD95"),
            _artifact(
                "comparison",
                "e2_deteccao_por_magnitude.csv",
                "Detecção por magnitude",
            ),
            _artifact("comparison", "e2_funcoes_empiricas.csv", "Funções empíricas"),
            _artifact("comparison", "e2_weibull_pontos.csv", "Pontos Weibull"),
            _artifact("comparison", "e2_weibull_ajustes.csv", "Diagnóstico dos ajustes"),
        ],
        "limitations": [
            item for item in payload["limitations"] if "E2" in item or "a_det" in item
        ],
    }
    _set_contract_status("pronto")
    return contract


def e2_contract() -> dict:
    paths = (
        COMPARISON_JSON,
        COMPARISON / "e2_resumo.csv",
        COMPARISON / "e2_deteccao_por_magnitude.csv",
        COMPARISON / "e2_funcoes_empiricas.csv",
        COMPARISON / "e2_deteccao_por_magnitude.png",
        COMPARISON / "e2_smd95.png",
        COMPARISON / "e2_funcoes_empiricas.png",
        COMPARISON / "e2_diagnostico_weibull.png",
    )
    return _e2_contract_cached(_signature(paths))


@lru_cache(maxsize=8)
def _reliability_contract_cached(signature_value) -> dict:
    payload = _reliability_payload((signature_value[0],))
    scenario_names = {
        item["scenario_id"]: item["plot_label"] for item in payload["scenarios"]
    }
    contract = {
        "contract_version": 2,
        "status": payload["status"],
        "dataset": payload["experimental_dataset"],
        "dataset_role": payload["dataset_role"],
        "time_unit_primary": payload["time_unit_primary"],
        "hours_per_year": payload["hours_per_year"],
        "formulas": payload["formulas"],
        "physical_weibull": payload["physical_weibull"],
        "scenarios": payload["scenarios"],
        "curve_series": reliability_curve_series(
            RELIABILITY / "curvas.csv", scenario_names
        ),
        "failure_rate_distribution": {
            "status": "not_estimable",
            "chart_available": False,
            "requested_model": "normal_histogram",
            "reason": (
                "Os quatro valores atuais misturam tres cenarios derivados e uma "
                "taxa bibliografica direta; eles nao constituem amostra homogenea "
                "para estimar distribuicao normal."
            ),
            "required_data": [
                "taxas observadas de uma populacao homogenea de componentes",
                "exposicao ou tempo de observacao por unidade",
                "criterio de censura e mesma definicao de falha",
                "tamanho amostral suficiente para avaliar a aderencia",
            ],
        },
        "source": payload["source"],
        "figures": [
            _figure(
                "reliability",
                "confiabilidade_probabilidade_falha",
                "Confiabilidade e probabilidade acumulada de falha",
                "Cenários exponenciais bibliográficos em eixo temporal explícito.",
            ),
            _figure(
                "reliability",
                "densidade_taxa_falha",
                "Densidade de falha e função de risco",
                "f(t) varia com o tempo; h(t) permanece constante no modelo exponencial.",
            ),
            _figure(
                "reliability",
                "taxas_componentes",
                "Taxas bibliográficas e cenários de sensibilidade",
                "Valores derivados permanecem separados da taxa direta do fusível.",
            ),
        ],
        "tables": [
            _artifact("reliability", "cenarios.csv", "Cenários de taxa de falha"),
            _artifact("reliability", "curvas.csv", "Dados-fonte das curvas"),
            _artifact("reliability", "metodologia.json", "Metodologia rastreável"),
            _artifact("reliability", "relatorio.md", "Relatório acadêmico"),
        ],
    }
    _set_contract_status("pronto")
    return contract


def reliability_contract() -> dict:
    paths = (
        RELIABILITY_JSON,
        RELIABILITY / "curvas.csv",
        RELIABILITY / "confiabilidade_probabilidade_falha.png",
        RELIABILITY / "densidade_taxa_falha.png",
        RELIABILITY / "taxas_componentes.png",
    )
    return _reliability_contract_cached(_signature(paths))


@lru_cache(maxsize=8)
def _sources_contract_cached(signature_value) -> dict:
    comparison = _comparison_payload((signature_value[0],))
    reliability = _reliability_payload((signature_value[1],))
    source_path = ROOT / reliability["source"]["artifact"]
    if not source_path.is_file():
        raise ContratoWebInvalido(f"Fonte bibliográfica ausente: {source_path}")
    source_relative = source_path.relative_to(LITERATURE).as_posix()
    contract = {
        "contract_version": 1,
        "dataset": {
            "name": comparison["dataset"]["dataset"],
            "doi": comparison["dataset"]["doi"],
            "url": f"https://doi.org/{comparison['dataset']['doi']}",
            "experiments": len(comparison["dataset"]["experiments"]),
            "raw_files_sha256": {
                key: value["sha256"]
                for key, value in comparison["dataset"]["raw_files"].items()
            },
        },
        "bibliography": [
            {
                "title": "Aplicação da metodologia Reliability-Centred Maintenance",
                "path": reliability["source"]["artifact"],
                "url": f"/sources/inversores-pv/{source_relative}",
                "sha256": sha256_arquivo(source_path),
                "pdf_page": reliability["source"]["pdf_page"],
                "printed_page": reliability["source"]["printed_page"],
                "tables": reliability["source"]["tables"],
            }
        ],
        "manifests": [
            _artifact(
                "manifests",
                COMPARISON_MANIFEST.name,
                "Manifesto da comparação Denso × AE-LSTM",
            ),
            _artifact(
                "manifests",
                RELIABILITY_MANIFEST.name,
                "Manifesto da confiabilidade física",
            ),
        ],
        "reports": [
            _artifact("comparison", "relatorio_comparacao.md", "Relatório comparativo"),
            _artifact("reliability", "relatorio.md", "Relatório de confiabilidade"),
        ],
        "separation_rules": [
            "GPVS-Faults sustenta a avaliação dos detectores, não taxas físicas de falha.",
            "E2 usa magnitude sintética adimensional; seu eixo não representa tempo.",
            "Weibull E2 é diagnóstico de detectabilidade e não modelo de vida útil.",
            "Taxas derivadas são cenários de sensibilidade, não medições de componente.",
        ],
    }
    _set_contract_status("pronto")
    return contract


def sources_contract() -> dict:
    paths = (
        COMPARISON_JSON,
        RELIABILITY_JSON,
        COMPARISON_MANIFEST,
        RELIABILITY_MANIFEST,
    )
    return _sources_contract_cached(_signature(paths))


def warm_contracts() -> dict:
    """Valida contratos em background sem bloquear a primeira pintura da UI."""
    _set_contract_status("iniciando")
    try:
        e3_contract()
        e2_contract()
        reliability_contract()
        sources_contract()
    except ContratoWebInvalido as exc:
        _set_contract_status("degradado", str(exc))
    return contracts_status()


def warm_contracts_background() -> threading.Thread:
    thread = threading.Thread(
        target=warm_contracts,
        name="aliado-contract-warmup",
        daemon=True,
    )
    thread.start()
    return thread


__all__ = [
    "COMPARISON",
    "LITERATURE",
    "MANIFESTS",
    "RELIABILITY",
    "ContratoWebInvalido",
    "contracts_status",
    "e2_contract",
    "e3_contract",
    "reliability_contract",
    "sources_contract",
    "warm_contracts",
    "warm_contracts_background",
]
