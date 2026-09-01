"""Publicação rastreável da comparação Denso versus AE-LSTM."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.ml.dados_gpvs import (
    FAULT_CONTRACTS,
    FEATURE_COLUMNS,
    PreparedData,
    dataset_files,
)
from src.ml.estatistica_comparacao import BOOTSTRAP_RESAMPLES
from src.ml.graficos_comparacao import generate_all
from src.ml.modelos_autoencoder import SEQUENCE_LENGTH
from src.ml.proveniencia import gerar_manifesto, salvar_manifesto
from src.ml.sensibilidade_escore import (
    SENSITIVITY_PERCENTILES,
    SENSITIVITY_TOP_K,
)
from src.ml.treino_comparacao import (
    MODEL_IDS,
    MODEL_NAMES,
    MODEL_ROOT,
    REFERENCE_SEED,
    STABILITY_SEEDS,
    ModelRun,
)


ROOT = Path(RAIZ_PROJETO)
RESULTS_DIR = ROOT / "resultados" / "comparacao"


def _json_safe(value):
    """Converte o contrato para tipos compatíveis com JSON estrito."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    return path


def results_payload(
    dataset_manifest: dict,
    prepared: PreparedData,
    runs: dict[str, list[ModelRun]],
    e3: dict[str, pd.DataFrame],
    *,
    seeds: tuple[int, ...],
) -> dict:
    sensitivity = e3["score_threshold_sensitivity"]
    reference_sensitivity = sensitivity[sensitivity["is_reference"]]
    model_contract = {}
    for model_id in MODEL_IDS:
        reference = next(run for run in runs[model_id] if run.seed == REFERENCE_SEED)
        model_contract[model_id] = {
            "name": MODEL_NAMES[model_id],
            "reference_seed": REFERENCE_SEED,
            "stability_seeds": list(seeds),
            "architecture": (
                "24-16-8-16-24"
                if model_id == "ae_denso"
                else "AE-LSTM temporal: L=8, hidden=32, latent=8"
            ),
            "n_parameters": reference.n_parameters,
            "score_method": "mean_of_top_k_feature_squared_reconstruction_errors",
            "score_top_k": reference.score_top_k,
            "score_dimension": "feature",
            "lstm_scored_time_step": "last" if model_id == "ae_lstm" else None,
            **reference.threshold_calibration.as_dict(),
            "healthy_test_false_positive_rate": float(
                np.mean(reference.healthy_test_scores > reference.threshold)
            ),
            "best_epoch": reference.history.best_epoch,
            "stopped_epoch": reference.history.stopped_epoch,
            "best_validation_loss": reference.history.best_validation_loss,
        }
    split_counts = {
        role: int(len(prepared.split[role]))
        for role in ("train", "validation", "calibration", "test")
    }
    return {
        "schema_version": 2,
        "created_at": agora_local().isoformat(),
        "title": "Comparação entre Autoencoder Denso e AE-LSTM",
        "dataset": dataset_manifest,
        "protocol": {
            "healthy_roles_nominal": prepared.split["nominal_fractions"],
            "healthy_role_counts_after_purge": split_counts,
            "split_strategy": prepared.split["strategy"],
            "purge_windows": prepared.split["purge_windows"],
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "model_selection_uses_fault_data": False,
            "fault_data_role": "E3_evaluation_only_after_freeze",
            "synthetic_faults_used_in_experimental_core": False,
            "e3": (
                "14 ensaios reais de bancada; pesos, scaler e limiares congelados; "
                "bootstrap no nível do ensaio"
            ),
        },
        "models": model_contract,
        "e3": {
            "evidence_level": "E3_bench",
            "primary_metrics": ["recall", "f1", "precision"],
            "complementary_metrics": ["auc_roc", "auc_pr"],
            "accuracy_role": "auxiliary_only",
            "confusion_matrix_unit": "window_descriptive_only",
            "fault_boundary_method": "nominal_mid_record",
            "fault_boundary_caveat": (
                "Os CSVs não fornecem canal de disparo; 50% é fronteira nominal."
            ),
            "native_fault_catalog": {
                f"F{fault}": contract for fault, contract in FAULT_CONTRACTS.items()
            },
            "macro": e3["macro"].to_dict(orient="records"),
            "paired_differences": e3["paired"].to_dict(orient="records"),
            "stability": e3["stability"].to_dict(orient="records"),
            "confusion_matrices": e3["confusion"].to_dict(orient="records"),
            "temporal_ablation": {
                "role": "supplementary_diagnostic",
                "sequence_length": SEQUENCE_LENGTH,
                "transition_post_windows": SEQUENCE_LENGTH - 1,
                "decision_target": "W_t",
                "context": "causal_continuous_W_t_minus_7_to_W_t",
                "transition_definition": (
                    "primeiras sete janelas após a fronteira nominal"
                ),
                "sustained_definition": (
                    "janelas posteriores, com contexto integralmente pós-fronteira"
                ),
                "analyses": [
                    "current_full",
                    "transition",
                    "sustained",
                    "post_fault_reset",
                ],
                "paired_differences": e3["temporal_ablation_paired"].to_dict(
                    orient="records"
                ),
                "conclusion": e3["temporal_ablation_conclusion"],
            },
            "score_threshold_sensitivity": {
                "role": "supplementary_no_model_selection",
                "top_k_values": list(SENSITIVITY_TOP_K),
                "requested_percentiles": list(SENSITIVITY_PERCENTILES),
                "reference_seed": REFERENCE_SEED,
                "configuration_count_per_model_seed": (
                    len(SENSITIVITY_TOP_K) * len(SENSITIVITY_PERCENTILES)
                ),
                "historical_reference_configuration": {
                    "score_top_k": 5,
                    "threshold_requested_percentile": 99.9,
                    "role": "reproducibility_reference_not_universal_optimum",
                },
                "calibration_source": "healthy_calibration_only",
                "uses_fault_data_for_selection": False,
                "table": "e3_sensibilidade_escore_limiar.csv",
                "reference_rows": reference_sensitivity.to_dict(orient="records"),
            },
        },
        "limitations": [
            "GPVS-Faults é evidência experimental de bancada, não validação de campo.",
            "Janelas do mesmo ensaio permanecem temporalmente autocorrelacionadas.",
            "As métricas publicadas descrevem os modelos, não o dataset isoladamente.",
        ],
    }


def _write_report(payload: dict, output: Path) -> Path:
    macro = pd.DataFrame(payload["e3"]["macro"])
    indexed = macro.set_index(["model", "metric"])
    confusion = pd.DataFrame(payload["e3"]["confusion_matrices"]).set_index("model")
    temporal = payload["e3"]["temporal_ablation"]
    temporal_paired = pd.DataFrame(temporal["paired_differences"])
    sustained = temporal_paired[
        temporal_paired["is_reference"]
        & temporal_paired["analysis"].eq("sustained")
        & temporal_paired["metric"].isin(["recall", "f1", "precision"])
    ].set_index("metric")
    sensitivity_contract = payload["e3"]["score_threshold_sensitivity"]
    sensitivity = pd.DataFrame(sensitivity_contract["reference_rows"])

    def formatted(model_id: str, metric: str) -> str:
        row = indexed.loc[(model_id, metric)]
        if not math.isfinite(float(row["estimate"])):
            return "N/A"
        return (
            f"{row['estimate']:.3f} "
            f"(IC95% {row['ci95_low']:.3f}-{row['ci95_high']:.3f})"
        )

    lines = [
        "# Comparação canônica: Autoencoder Denso versus AE-LSTM",
        "",
        "## Delineamento",
        "",
        "O GPVS-Faults é a única fonte experimental. F0L/F0M fornecem treino, ",
        "validação, calibração e teste saudável em blocos temporais disjuntos. ",
        "F1L-F7M permanecem fora do ajuste e formam a evidência E3 de bancada.",
        "",
        "## Resultado experimental E3",
        "",
        "| Modelo | Recall | F1 | Precision | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_IDS:
        lines.append(
            f"| {MODEL_NAMES[model_id]} | {formatted(model_id, 'recall')} | "
            f"{formatted(model_id, 'f1')} | {formatted(model_id, 'precision')} | "
            f"{formatted(model_id, 'auc_roc')} | {formatted(model_id, 'auc_pr')} |"
        )
    precision_valid = {
        model_id: indexed.loc[(model_id, "precision")]
        for model_id in MODEL_IDS
    }
    lines.extend(
        [
            "",
            "Recall, F1 e Precision formam a camada principal. ROC-AUC e PR-AUC "
            "são medidas complementares de discriminação. Precision é N/A quando "
            "a execução não produz nenhum alarme positivo.",
            "",
            "Precision teve valor finito em "
            f"{int(precision_valid['ae_denso']['n_valid_experiments'])}/14 ensaios "
            "do Autoencoder Denso e "
            f"{int(precision_valid['ae_lstm']['n_valid_experiments'])}/14 do AE-LSTM.",
            "",
            "## Matrizes de confusão agregadas",
            "",
            "| Modelo | TP | FP | TN | FN |",
            "|---|---:|---:|---:|---:|",
            *(
                f"| {MODEL_NAMES[model_id]} | {int(confusion.loc[model_id, 'tp'])} | "
                f"{int(confusion.loc[model_id, 'fp'])} | "
                f"{int(confusion.loc[model_id, 'tn'])} | "
                f"{int(confusion.loc[model_id, 'fn'])} |"
                for model_id in MODEL_IDS
            ),
            "",
            "As contagens são agregadas por janela e têm uso descritivo devido à "
            "autocorrelação dentro de cada ensaio.",
            "",
            "## Ponto operacional",
            "",
            "| Modelo | Top-k | Limiar | Percentil solicitado | Percentil efetivo | Ordem | Resolução | FP no teste saudável |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *(
                f"| {model['name']} | {model['score_top_k']} | "
                f"{model['score_threshold']:.6f} | "
                f"p{model['threshold_requested_percentile']:.1f} | "
                f"p{model['threshold_effective_percentile']:.3f} | "
                f"{model['threshold_selected_rank']}/{model['calibration_n']} | "
                f"{model['threshold_percentile_resolution']:.3f} p.p. | "
                f"{model['healthy_test_false_positive_rate']:.3%} |"
                for model in payload["models"].values()
            ),
            "",
            "A unidade inferencial do intervalo é o ensaio, não a janela. A ",
            "semente 42 é pré-fixada; cinco sementes descrevem estabilidade sem ",
            "selecionar o modelo pelo desempenho nas falhas.",
            "",
            "A fronteira de falha é nominalmente 50% do registro porque os CSVs ",
            "não contêm canal instrumentado de disparo.",
            "",
            "O ponto operacional reproduzido usa a média dos cinco maiores erros "
            "quadráticos por feature e p99,9 solicitado. Ele é uma referência "
            "histórica pré-fixada, não um ótimo universal. Cada modelo calibra seu "
            "próprio limiar somente no bloco saudável; o contrato registra o order "
            "statistic e o percentil empírico efetivamente alcançável.",
            "",
            "## Ablação temporal do AE-LSTM",
            "",
            "A análise canônica usa a sequência causal contínua [W_(t-7), ..., W_t] "
            "e decide em W_t. Ela separa as sete primeiras janelas pós-fronteira "
            "da falha sustentada, cujo contexto já é integralmente pós-fronteira. "
            "O reinício pós-falha permanece apenas como diagnóstico auxiliar. Treino, "
            "scaler, escore e limiares permanecem congelados.",
            "",
            "| Métrica | AE-LSTM − Denso na falha sustentada | IC95% |",
            "|---|---:|---:|",
            *(
                f"| {metric.title()} | "
                f"{float(sustained.loc[metric, 'difference_lstm_minus_dense']):.3f} | "
                f"{float(sustained.loc[metric, 'ci95_low']):.3f} a "
                f"{float(sustained.loc[metric, 'ci95_high']):.3f} |"
                for metric in ("recall", "f1", "precision")
            ),
            "",
            f"Conclusão pré-especificada: `{temporal['conclusion']['status']}`. "
            f"{temporal['conclusion']['reason']}",
            "",
            "## Sensibilidade do escore e do limiar",
            "",
            "A grade complementar usa `k = 5, 10, 20` e percentis solicitados "
            "p99, p99,5 e p99,9, totalizando nove configurações por modelo e "
            "semente. Cada limiar é derivado "
            "exclusivamente da calibração saudável; as falhas não selecionam a "
            "configuração.",
            "",
            "| Modelo | FP saudável mínimo-máximo | Recall E3 mínimo-máximo | F1 E3 mínimo-máximo | Precision E3 mínimo-máximo |",
            "|---|---:|---:|---:|---:|",
            *(
                f"| {MODEL_NAMES[model_id]} | "
                f"{sensitivity[sensitivity['model'].eq(model_id)]['healthy_test_false_positive_rate'].min():.3%}–"
                f"{sensitivity[sensitivity['model'].eq(model_id)]['healthy_test_false_positive_rate'].max():.3%} | "
                f"{sensitivity[sensitivity['model'].eq(model_id)]['macro_recall'].min():.3f}–"
                f"{sensitivity[sensitivity['model'].eq(model_id)]['macro_recall'].max():.3f} | "
                f"{sensitivity[sensitivity['model'].eq(model_id)]['macro_f1'].min():.3f}–"
                f"{sensitivity[sensitivity['model'].eq(model_id)]['macro_f1'].max():.3f} | "
                f"{sensitivity[sensitivity['model'].eq(model_id)]['macro_precision'].min():.3f}–"
                f"{sensitivity[sensitivity['model'].eq(model_id)]['macro_precision'].max():.3f} |"
                for model_id in MODEL_IDS
            ),
            "",
            "k=5 com p99,9 solicitado permanece somente como referência histórica "
            "de reprodutibilidade. A tabela registra também o percentil empírico "
            "efetivo, a estatística de ordem e a resolução da calibração; esta "
            "análise não promove uma configuração a partir das falhas.",
        ]
    )
    output.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )
    return output


def save_results(
    dataset_manifest: dict,
    prepared: PreparedData,
    runs: dict[str, list[ModelRun]],
    e3: dict[str, pd.DataFrame],
    *,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reference_configurations = {
        (
            run.threshold_calibration.requested_percentile,
            run.score_top_k,
        )
        for model_id in MODEL_IDS
        for run in runs[model_id]
        if run.seed == REFERENCE_SEED
    }
    if len(reference_configurations) != 1:
        raise ValueError("Os modelos de referência devem usar o mesmo percentil e top-k")
    threshold_percentile, score_top_k = reference_configurations.pop()
    outputs: list[Path] = []
    tables = {
        "e3_metricas_por_ensaio.csv": e3["scenarios"],
        "e3_metricas_macro.csv": e3["macro"],
        "e3_escores_referencia.csv": e3["scores"],
        "e3_matrizes_confusao.csv": e3["confusion"],
        "e3_estabilidade_sementes.csv": e3["stability"],
        "e3_diferencas_pareadas.csv": e3["paired"],
        "e3_ablacao_temporal_por_ensaio.csv": e3["temporal_ablation"],
        "e3_ablacao_temporal.csv": e3["temporal_ablation_paired"],
        "e3_sensibilidade_escore_limiar.csv": e3[
            "score_threshold_sensitivity"
        ],
    }
    for name, frame in tables.items():
        outputs.append(_write_csv(RESULTS_DIR / name, frame))

    payload = results_payload(
        dataset_manifest,
        prepared,
        runs,
        e3,
        seeds=seeds,
    )
    outputs.append(_write_json(RESULTS_DIR / "comparacao_autoencoders.json", payload))
    outputs.append(_write_report(payload, RESULTS_DIR / "relatorio_comparacao.md"))
    outputs.extend(
        generate_all(
            RESULTS_DIR,
            e3_summary=e3["macro"],
            e3_scores=e3["scores"],
            e3_confusion=e3["confusion"],
            e3_scenarios=e3["scenarios"],
            temporal_ablation_paired=e3["temporal_ablation_paired"],
            score_threshold_sensitivity=e3["score_threshold_sensitivity"],
        )
    )

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
    inputs = {f"raw_{name}": path for name, path in dataset_files().items()}
    inputs.update(model_inputs)
    manifest = gerar_manifesto(
        "comparacao_autoencoders",
        Path(__file__).with_name("comparacao_autoencoders.py"),
        {
            "dataset": "GPVS-Faults",
            "models": list(MODEL_IDS),
            "reference_seed": REFERENCE_SEED,
            "stability_seeds": list(seeds),
            "threshold_percentile": threshold_percentile,
            "score_top_k": score_top_k,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "temporal_ablation": {
                "sequence_length": SEQUENCE_LENGTH,
                "transition_post_windows": SEQUENCE_LENGTH - 1,
                "reference_seed": REFERENCE_SEED,
            },
            "score_threshold_sensitivity": {
                "top_k_values": list(SENSITIVITY_TOP_K),
                "requested_percentiles": list(SENSITIVITY_PERCENTILES),
                "configuration_count_per_model_seed": (
                    len(SENSITIVITY_TOP_K) * len(SENSITIVITY_PERCENTILES)
                ),
                "reference_seed": REFERENCE_SEED,
                "uses_fault_data_for_selection": False,
            },
        },
        inputs,
        outputs,
        code_dependencies={
            "dataset": Path(__file__).with_name("dados_gpvs.py"),
            "models": Path(__file__).with_name("modelos_autoencoder.py"),
            "training": Path(__file__).with_name("treino_comparacao.py"),
            "evaluation": Path(__file__).with_name("avaliacao_comparativa.py"),
            "statistics": Path(__file__).with_name("estatistica_comparacao.py"),
            "plots": Path(__file__).with_name("graficos_comparacao.py"),
            "plot_style": Path(__file__).with_name("estilo_graficos.py"),
            "publication": Path(__file__),
            "sensitivity": Path(__file__).with_name("sensibilidade_escore.py"),
        },
        evidence_level="E3_bench",
    )
    manifest_path = salvar_manifesto(manifest)
    return {"outputs": outputs, "manifest": manifest_path, "payload": payload}


__all__ = ["RESULTS_DIR", "results_payload", "save_results"]
