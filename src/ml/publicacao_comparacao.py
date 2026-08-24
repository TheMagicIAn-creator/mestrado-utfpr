"""Publicação rastreável da comparação Denso versus AE-LSTM."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.ml.dados_gpvs import FEATURE_COLUMNS, PreparedData, dataset_files
from src.ml.estatistica_comparacao import BOOTSTRAP_RESAMPLES
from src.ml.graficos_comparacao import generate_all
from src.ml.proveniencia import gerar_manifesto, salvar_manifesto
from src.ml.treino_comparacao import (
    MODEL_IDS,
    MODEL_NAMES,
    MODEL_ROOT,
    REFERENCE_SEED,
    STABILITY_SEEDS,
    THRESHOLD_PERCENTILE,
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
            "score_method": "mean_squared_reconstruction_error",
            "threshold_method": "empirical_p99_higher",
            "score_threshold": reference.threshold,
            "threshold_effective_percentile": THRESHOLD_PERCENTILE,
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
            "e3": (
                "14 ensaios reais de bancada; pesos, scaler e limiares congelados; "
                "bootstrap no nível do ensaio"
            ),
        },
        "models": model_contract,
        "e3": {
            "evidence_level": "E3_bench",
            "primary_metric": "auc_pr",
            "confusion_matrix_unit": "window_descriptive_only",
            "fault_boundary_method": "nominal_mid_record",
            "fault_boundary_caveat": (
                "Os CSVs não fornecem canal de disparo; 50% é fronteira nominal."
            ),
            "macro": e3["macro"].to_dict(orient="records"),
            "paired_differences": e3["paired"].to_dict(orient="records"),
            "stability": e3["stability"].to_dict(orient="records"),
        },
        "limitations": [
            "GPVS-Faults é evidência experimental de bancada, não validação de campo.",
            "Janelas do mesmo ensaio permanecem temporalmente autocorrelacionadas.",
            "As métricas publicadas descrevem os modelos, não o dataset isoladamente.",
        ],
    }


def _write_report(payload: dict, output: Path) -> Path:
    macro = pd.DataFrame(payload["e3"]["macro"])
    primary = macro[macro["metric"].eq("auc_pr")].set_index("model")
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
    ]
    for model_id in MODEL_IDS:
        row = primary.loc[model_id]
        lines.append(
            f"- **{MODEL_NAMES[model_id]}:** AUC-PR macro "
            f"{row['estimate']:.3f} (IC95% {row['ci95_low']:.3f}-{row['ci95_high']:.3f})."
        )
    lines.extend(
        [
            "",
            "A unidade inferencial do intervalo é o ensaio, não a janela. A ",
            "semente 42 é pré-fixada; cinco sementes descrevem estabilidade sem ",
            "selecionar o modelo pelo desempenho nas falhas.",
            "",
            "A fronteira de falha é nominalmente 50% do registro porque os CSVs ",
            "não contêm canal instrumentado de disparo.",
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
    outputs: list[Path] = []
    tables = {
        "e3_metricas_por_ensaio.csv": e3["scenarios"],
        "e3_metricas_macro.csv": e3["macro"],
        "e3_escores_referencia.csv": e3["scores"],
        "e3_matrizes_confusao.csv": e3["confusion"],
        "e3_estabilidade_sementes.csv": e3["stability"],
        "e3_diferencas_pareadas.csv": e3["paired"],
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
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
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
            "publication": Path(__file__),
        },
        evidence_level="E3_bench",
    )
    manifest_path = salvar_manifesto(manifest)
    return {"outputs": outputs, "manifest": manifest_path, "payload": payload}


__all__ = ["RESULTS_DIR", "results_payload", "save_results"]
