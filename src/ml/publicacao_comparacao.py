"""Publicação rastreável dos resultados canônicos E2 e E3."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.ml.assinaturas_fmeca import SIGNATURES
from src.ml.avaliacao_comparativa import E2_PERSISTENCE_MAGNITUDE
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
    e2: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    e2_steps: int,
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
        "title": "Autoencoder Denso versus AE-LSTM no GPVS-Faults",
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
            "e2": (
                "janelas F0 compartilhadas, mesmas perturbações e mesma grade "
                "de magnitude para ambos os modelos"
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
        "e2": {
            "evidence_level": "E2_synthetic",
            "axis": "a_det, fração da assinatura sintética nominal",
            "axis_is_time": False,
            "magnitude_steps": int(e2_steps),
            "persistence_width": E2_PERSISTENCE_MAGNITUDE,
            "interval_method": "Wilson 95% at the window-trajectory level",
            "interval_caveat": (
                "As trajetórias são janelas dos dois ensaios F0 e podem manter "
                "autocorrelação intraensaio; os intervalos E2 são descritivos."
            ),
            "smd95_definition": (
                "menor magnitude cujo limite inferior do IC95% Wilson atinge 0,95"
            ),
            "signatures": [signature.as_dict() for signature in SIGNATURES],
            "summary": e2["summary"].to_dict(orient="records"),
            "weibull_role": (
                "diagnóstico no papel de probabilidade; síntese paramétrica "
                "somente quando o teste formal quantizado é aceito"
            ),
            "weibull_acceptance_scope": (
                "A aceitação ou rejeição se refere somente ao ajuste Weibull; "
                "não classifica a qualidade dos detectores."
            ),
        },
        "limitations": [
            "GPVS-Faults é evidência experimental de bancada, não validação de campo.",
            "Janelas do mesmo ensaio permanecem temporalmente autocorrelacionadas.",
            "As assinaturas E2 são proxies sintéticos orientados pela FMECA.",
            (
                "Os IC95% Wilson de E2 usam janelas-trajetórias e são "
                "descritivos diante da autocorrelação intraensaio."
            ),
            "Curvas em a_det não são vida útil, RUL ou confiabilidade física.",
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
            "",
            "## Detectabilidade E2",
            "",
            "Contator AC, IGBT e Fusível AC usam as mesmas janelas, magnitudes e ",
            "realizações nos dois detectores. SMD95 exige limite inferior Wilson ",
            "de 95%; quando a condição não ocorre até a_det=1, registra-se ",
            "`não atingido`.",
            "",
            "Sobrevivência empírica, incidência acumulada e risco discreto vivem ",
            "no eixo de magnitude. O Weibull 2P é apenas diagnóstico formal e ",
            "nunca produz RUL, MTTF ou confiabilidade física.",
            "",
            "A não aceitação de um ajuste Weibull rejeita apenas a síntese ",
            "paramétrica correspondente; não reprova nenhum dos detectores.",
            "",
            "Os IC95% Wilson de E2 tratam cada janela-trajetória como unidade ",
            "Bernoulli e são apresentados como descritivos, pois janelas do ",
            "mesmo ensaio podem permanecer autocorrelacionadas.",
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
    e2: dict[str, Any],
    *,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
    e2_steps: int,
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
        "e2_deteccao_por_magnitude.csv": e2["curves"],
        "e2_primeiro_cruzamento.csv": e2["crossings"],
        "e2_funcoes_empiricas.csv": e2["empirical"],
        "e2_weibull_ajustes.csv": e2["fits"],
        "e2_weibull_pontos.csv": e2["probability_points"],
        "e2_resumo.csv": e2["summary"],
    }
    for name, frame in tables.items():
        outputs.append(_write_csv(RESULTS_DIR / name, frame))

    payload = results_payload(
        dataset_manifest,
        prepared,
        runs,
        e3,
        e2,
        seeds=seeds,
        e2_steps=e2_steps,
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
            e2_curves=e2["curves"],
            e2_summary=e2["summary"],
            e2_empirical=e2["empirical"],
            e2_probability_points=e2["probability_points"],
            e2_fits=e2["fits"],
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
            "e2_steps": int(e2_steps),
            "e2_persistence_magnitude": E2_PERSISTENCE_MAGNITUDE,
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
            "fmeca_signatures": Path(__file__).with_name("assinaturas_fmeca.py"),
            "detectability": Path(__file__).with_name("detectabilidade.py"),
            "plots": Path(__file__).with_name("graficos_comparacao.py"),
            "publication": Path(__file__),
        },
        evidence_level="E2+E3_bench",
    )
    manifest_path = salvar_manifesto(manifest)
    return {"outputs": outputs, "manifest": manifest_path, "payload": payload}


__all__ = ["RESULTS_DIR", "results_payload", "save_results"]
