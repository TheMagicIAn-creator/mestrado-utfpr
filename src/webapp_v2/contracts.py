"""Contratos somente leitura consumidos pela aplicação web V2.

O frontend nunca recalcula métricas científicas. Este módulo valida e reduz os
artefatos versionados para um contrato de apresentação estável.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from src.core.config import RAIZ_PROJETO
from src.ml.proveniencia import sha256_arquivo

RAIZ = Path(RAIZ_PROJETO)
AUTOENCODER = RAIZ / "resultados" / "v2" / "autoencoder"
CONFIABILIDADE = RAIZ / "resultados" / "v2" / "confiabilidade"


class ContratoWebInvalido(RuntimeError):
    """Artefato ausente, ilegível ou incompatível com a aplicação."""


def _json(path: Path) -> dict:
    try:
        dados = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContratoWebInvalido(f"Artefato ausente: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ContratoWebInvalido(f"Artefato inválido: {path}: {exc}") from exc
    if not isinstance(dados, dict):
        raise ContratoWebInvalido(f"Contrato JSON deve ser um objeto: {path}")
    return dados


def _csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as arquivo:
            return list(csv.DictReader(arquivo))
    except FileNotFoundError as exc:
        raise ContratoWebInvalido(f"Artefato ausente: {path}") from exc
    except (OSError, csv.Error) as exc:
        raise ContratoWebInvalido(f"Artefato CSV inválido: {path}: {exc}") from exc


def _numero(valor, *, inteiro: bool = False):
    if valor in (None, "", "None", "nan"):
        return None
    return int(float(valor)) if inteiro else float(valor)


def _booleano(valor) -> bool:
    return str(valor).strip().lower() in {"1", "true", "sim", "yes"}


def _metrica(resumo: dict, nome: str) -> dict:
    item = resumo[nome]
    return {
        "mean": float(item["mean"]),
        "ci95_low": float(item["ci95_low"]),
        "ci95_high": float(item["ci95_high"]),
        "n_experiments": int(item["n_experiments"]),
    }


def _figura(area: str, arquivo: str, titulo: str, pergunta: str) -> dict:
    path = (AUTOENCODER if area == "autoencoder" else CONFIABILIDADE) / arquivo
    if not path.is_file():
        raise ContratoWebInvalido(f"Figura ausente: {path}")
    return {
        "title": titulo,
        "question": pergunta,
        "url": f"/artifacts/{area}/{arquivo}",
        "download_url": f"/artifacts/{area}/{arquivo}",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_arquivo(path),
    }


def _artefato(area: str, arquivo: str, rotulo: str, tipo: str) -> dict:
    path = (AUTOENCODER if area == "autoencoder" else CONFIABILIDADE) / arquivo
    if not path.is_file():
        raise ContratoWebInvalido(f"Artefato ausente: {path}")
    return {
        "label": rotulo,
        "type": tipo,
        "url": f"/artifacts/{area}/{arquivo}",
        "filename": arquivo,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_arquivo(path),
    }


def _trial_rows() -> list[dict]:
    saida = []
    for row in _csv(AUTOENCODER / "avaliacao_cenarios.csv"):
        if row["method"] not in {"autoencoder_v2", "pca"}:
            continue
        saida.append(
            {
                "experiment": row["experiment"],
                "fault": int(row["fault"]),
                "fault_type": row["fault_type"],
                "mode": row["mode"],
                "mode_name": row["mode_name"],
                "method": row["method"],
                "auc_roc": float(row["auc_roc"]),
                "average_precision": float(row["average_precision"]),
                "sensitivity": float(row["sensitivity"]),
                "specificity": float(row["specificity"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "mcc": float(row["mcc"]),
                "sustained_detection": _booleano(row["sustained_detection"]),
                "detection_delay_from_nominal_midpoint_s": _numero(
                    row["detection_delay_from_nominal_midpoint_s"]
                ),
            }
        )
    if len(saida) != 28:
        raise ContratoWebInvalido(
            f"Esperados 14 ensaios x 2 métodos; recebidos {len(saida)} registros"
        )
    return saida


def _selection_rows() -> list[dict]:
    linhas = []
    for row in _csv(AUTOENCODER / "selecao_resumo.csv"):
        linhas.append(
            {
                "architecture_id": row["arquitetura"],
                "median_validation_loss": float(row["mediana_validacao"]),
                "mean_validation_loss": float(row["media_validacao"]),
                "std_validation_loss": float(row["desvio_validacao"]),
                "trainable_parameters": int(row["n_parametros"]),
                "n_seeds": int(row["n_seeds"]),
                "within_two_percent": _booleano(row["dentro_faixa_2pct"]),
                "selected": _booleano(row["selecionada"]),
            }
        )
    if len(linhas) != 3 or sum(item["selected"] for item in linhas) != 1:
        raise ContratoWebInvalido("Seleção de arquitetura V2 incoerente")
    return linhas


def _reliability_contract() -> dict:
    resultado = _json(CONFIABILIDADE / "resultado.json")
    if int(resultado.get("schema_version", 0)) != 2:
        raise ContratoWebInvalido("Resultado de confiabilidade não está no schema v2")
    marcos = {item["scenario_id"]: item for item in resultado["milestones"]}
    cenarios = []
    for item in resultado["scenarios"]:
        cenarios.append(
            {
                "scenario_id": item["scenario_id"],
                "label": item["plot_label"],
                "source": item["source"],
                "doi": item.get("doi"),
                "source_location": item["source_location"],
                "source_type": item["source_type"],
                "scope": item["scope"],
                "original_value": float(item["original_value"]),
                "original_unit": item["original_unit"],
                "lambda_per_hour": float(item["lambda_per_hour"]),
                "lambda_per_year": float(item["lambda_per_year"]),
                "reciprocal_time_years": float(item["reciprocal_time_years"]),
                "b1_years": float(marcos[item["scenario_id"]]["b1_years"]),
                "b10_years": float(marcos[item["scenario_id"]]["b10_years"]),
                "median_years": float(marcos[item["scenario_id"]]["median_years"]),
                "mean_semantics": item["mean_semantics"],
                "caveat": item["caveat"],
                "source_artifact": item["source_artifact"],
                "source_sha256": item["source_sha256"],
            }
        )
    return {
        "status": resultado["status"],
        "dataset_role": resultado["dataset_role"],
        "analysis_date": resultado["analysis_date"],
        "model": resultado["model"],
        "physical_weibull": resultado["physical_weibull"],
        "dimensional_audit": resultado["dimensional_audit"],
        "scenarios": cenarios,
        "figures": [
            _figura(
                "reliability",
                "confiabilidade_cenarios.png",
                "Confiabilidade R(t)",
                "Qual a probabilidade de operação sem falha em cada cenário?",
            ),
            _figura(
                "reliability",
                "probabilidade_falha_cenarios.png",
                "Probabilidade acumulada F(t)",
                "Qual a probabilidade modelada de ao menos uma falha até t?",
            ),
            _figura(
                "reliability",
                "densidade_taxa_falha.png",
                "Densidade e taxa de falha",
                "Como f(t) difere da taxa instantânea h(t)?",
            ),
            _figura(
                "reliability",
                "marcos_confiabilidade.png",
                "Marcos B1, B10, mediana e 1/λ",
                "Como os horizontes de probabilidade variam entre fontes?",
            ),
        ],
    }


def _fmeca_contract() -> dict:
    return {
        "source": "Torres (2024), consolidação oficial do pesquisador",
        "status": "researcher_defined_not_cristaldi_rpn_reproduction",
        "evidence": "E1/E2",
        "components": [
            {
                "id": "contator_ac",
                "component": "Contator AC",
                "function": "Chavear e conectar a saída CA à rede",
                "s": 5,
                "o": 7,
                "d_field": 9,
                "npr": 315,
                "tickets_pct": 12,
                "energy_lost_pct": 13,
                "electrical_signature": "Transiente de comutação e chattering na corrente CA",
            },
            {
                "id": "igbt",
                "component": "IGBT",
                "function": "Comutar a conversão CC-CA por PWM",
                "s": 5,
                "o": 6,
                "d_field": 3,
                "npr": 90,
                "tickets_pct": 6,
                "energy_lost_pct": 6,
                "electrical_signature": "Harmônicos de chaveamento e aumento de THD",
            },
            {
                "id": "fusivel_ac",
                "component": "Fusível AC",
                "function": "Proteger o lado CA contra sobrecorrente",
                "s": 5,
                "o": 3,
                "d_field": 2,
                "npr": 30,
                "tickets_pct": 4,
                "energy_lost_pct": 12,
                "electrical_signature": "Perda parcial de fase e aumento de desbalanceamento",
            },
        ],
        "separation_note": (
            "D_campo é julgamento FMECA. A detectabilidade do monitor é medida "
            "separadamente e não substitui a tabela oficial."
        ),
    }


@lru_cache(maxsize=1)
def dashboard_contract() -> dict:
    experimento = _json(AUTOENCODER / "contrato_experimento.json")
    limiar = _json(AUTOENCODER / "limiar_v2.json")
    avaliacao = _json(AUTOENCODER / "avaliacao_experimental.json")
    if any(int(item.get("schema_version", 0)) != 2 for item in (experimento, limiar, avaliacao)):
        raise ContratoWebInvalido("Os artefatos do autoencoder devem usar schema v2")

    macro = avaliacao["macro_summary"]
    ae = macro["autoencoder_v2"]
    pca = macro["pca"]
    canonical = limiar["canonical"]
    selection = _selection_rows()
    selected = next(item for item in selection if item["selected"])

    autoencoder_figures = [
        _figura(
            "autoencoder",
            "selecao_arquitetura.png",
            "Seleção da arquitetura",
            "Qual arquitetura equilibra perda saudável e complexidade?",
        ),
        _figura(
            "autoencoder",
            "calibracao_limiar.png",
            "Calibração do limiar",
            "Qual a excedência em calibração e no teste saudável intocado?",
        ),
        _figura(
            "autoencoder",
            "desempenho_por_ensaio.png",
            "Desempenho por ensaio",
            "Quanta heterogeneidade existe entre os 14 ensaios?",
        ),
        _figura(
            "autoencoder",
            "curvas_roc_pr_macro.png",
            "Curvas ROC e PR macro",
            "Como a discriminação varia entre ensaios independentes?",
        ),
        _figura(
            "autoencoder",
            "matrizes_confusao.png",
            "Matrizes binárias agregadas",
            "Como erros pré e pós-falha se distribuem no ponto operacional?",
        ),
        _figura(
            "autoencoder",
            "series_temporais.png",
            "Resposta temporal nos 14 ensaios",
            "Quando o índice cruza o limiar após a fronteira nominal?",
        ),
        _figura(
            "autoencoder",
            "contribuicoes_familias.png",
            "Contribuição por família física",
            "Quais grupos de variáveis dominam o escore em cada ensaio?",
        ),
    ]

    artefatos = [
        _artefato("autoencoder", "contrato_experimento.json", "Contrato experimental", "JSON"),
        _artefato("autoencoder", "limiar_v2.json", "Limiar e robustez por semente", "JSON"),
        _artefato(
            "autoencoder", "avaliacao_experimental.json", "Avaliação experimental", "JSON"
        ),
        _artefato("autoencoder", "avaliacao_cenarios.csv", "Métricas por ensaio", "CSV"),
        _artefato("reliability", "resultado.json", "Confiabilidade física V2", "JSON"),
        _artefato("reliability", "cenarios.csv", "Cenários normalizados", "CSV"),
        _artefato("reliability", "manifesto_v2.json", "Manifesto de proveniência", "JSON"),
        _artefato("reliability", "relatorio.md", "Relatório acadêmico", "Markdown"),
    ]

    return {
        "schema_version": 2,
        "project": {
            "name": "ALIAdo PV",
            "researcher": "Rodolfo Torres",
            "advisor": "Prof. Fernanda Cristina Correa",
            "dataset": avaliacao["dataset"]["name"],
            "dataset_doi": avaliacao["dataset"]["doi"],
            "experimental_evidence": "E3 de bancada",
            "reliability_evidence": "sensibilidade bibliográfica",
            "generated_at": avaliacao["created_at"],
        },
        "overview": {
            "verdict": (
                "O detector denso V2 melhora a sensibilidade frente ao PCA, enquanto "
                "o PCA preserva maior AUC e especificidade. Não há superioridade global."
            ),
            "healthy_false_positive_pct": float(canonical["healthy_test"]["taxa_pct"]),
            "metrics": {
                "auc_roc": _metrica(ae, "auc_roc"),
                "average_precision": _metrica(ae, "average_precision"),
                "sensitivity": _metrica(ae, "sensitivity"),
                "specificity": _metrica(ae, "specificity"),
                "balanced_accuracy": _metrica(ae, "balanced_accuracy"),
                "mcc": _metrica(ae, "mcc"),
            },
            "method_comparison": [
                {
                    "method_id": "autoencoder_v2",
                    "label": "Autoencoder V2",
                    **{name: _metrica(ae, name) for name in (
                        "auc_roc", "sensitivity", "specificity", "balanced_accuracy", "mcc"
                    )},
                },
                {
                    "method_id": "pca",
                    "label": "PCA (8 componentes)",
                    **{name: _metrica(pca, name) for name in (
                        "auc_roc", "sensitivity", "specificity", "balanced_accuracy", "mcc"
                    )},
                },
            ],
        },
        "autoencoder": {
            "architecture": {
                "display": "24-16-8-16-24",
                "architecture_id": selected["architecture_id"],
                "trainable_parameters": selected["trainable_parameters"],
                "canonical_seed": int(experimento["selection"]["canonical_seed"]),
                "activation": experimento["training"]["activation"],
                "loss": experimento["training"]["loss"],
            },
            "sample_counts": experimento["sample_counts"],
            "threshold": {
                "value": float(canonical["threshold"]),
                "method": canonical["method"],
                "order_one_based": int(canonical["order_one_based"]),
                "n_calibration": int(canonical["n_calibration"]),
                "nominal_tail_pct": float(canonical["tail_nominal_pct"]),
                "healthy_test": canonical["healthy_test"],
            },
            "selection": selection,
            "trials": _trial_rows(),
            "figures": autoencoder_figures,
            "boundary_semantics": avaliacao["protocol"]["fault_boundary_semantics"],
        },
        "reliability": _reliability_contract(),
        "fmeca": _fmeca_contract(),
        "evidence": {
            "artifacts": artefatos,
            "rules": [
                "GPVS-Faults é o único dataset experimental dos resultados V2.",
                "F1-F7 permanecem classes do dataset e não são renomeadas como componentes FMECA.",
                "Confiabilidade física usa cenários bibliográficos identificados.",
                "Weibull físico e RUL permanecem não estimáveis sem dados de vida.",
            ],
        },
    }


@lru_cache(maxsize=1)
def reliability_curves_contract() -> dict:
    rows = []
    for row in _csv(CONFIABILIDADE / "curvas.csv"):
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "time_years": float(row["time_years"]),
                "reliability": float(row["reliability"]),
                "cumulative_failure_probability": float(
                    row["cumulative_failure_probability"]
                ),
                "failure_density_per_year": float(row["failure_density_per_year"]),
                "hazard_per_year": float(row["hazard_per_year"]),
            }
        )
    if len(rows) != 5 * 401:
        raise ContratoWebInvalido(f"Grade de confiabilidade incompleta: {len(rows)}")
    return {
        "schema_version": 2,
        "time_unit": "year",
        "rows": rows,
    }


def clear_contract_cache() -> None:
    dashboard_contract.cache_clear()
    reliability_curves_contract.cache_clear()
