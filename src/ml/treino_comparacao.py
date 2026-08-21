"""Treino congelado dos dois braços canônicos no holdout saudável GPVS."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import RAIZ_PROJETO
from src.ml.dados_gpvs import (
    FEATURE_COLUMNS,
    PreparedData,
    role_blocks,
    save_baseline_normalization,
)
from src.ml.modelos_autoencoder import (
    BATCH_SIZE,
    DENSE_HIDDEN,
    DROPOUT,
    LATENT_DIM,
    LEARNING_RATE,
    LSTM_HIDDEN,
    MAX_EPOCHS,
    PATIENCE,
    SEQUENCE_LENGTH,
    TrainingHistory,
    parameter_count,
    score_dense,
    score_lstm,
    sequences_for_blocks,
    train_dense,
    train_lstm,
)


MODEL_ROOT = Path(RAIZ_PROJETO) / "artefatos" / "modelos"
MODEL_IDS = ("ae_denso", "ae_lstm")
MODEL_NAMES = {"ae_denso": "Autoencoder Denso", "ae_lstm": "AE-LSTM"}
REFERENCE_SEED = 42
STABILITY_SEEDS = (13, 29, 42, 71, 101)
THRESHOLD_PERCENTILE = 99.0


@dataclass
class ModelRun:
    model_id: str
    seed: int
    model: Any
    threshold: float
    calibration_scores: np.ndarray
    healthy_test_scores: np.ndarray
    history: TrainingHistory
    n_parameters: int


def empirical_threshold(
    values: np.ndarray,
    percentile: float = THRESHOLD_PERCENTILE,
) -> float:
    scores = np.asarray(values, dtype=float)
    if len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("A calibração exige ao menos dois escores finitos")
    return float(np.percentile(scores, float(percentile), method="higher"))


def score_role(
    model_id: str,
    model,
    prepared: PreparedData,
    role: str,
) -> np.ndarray:
    blocks = role_blocks(prepared.split, role)
    if model_id == "ae_denso":
        return score_dense(model, prepared.scaled_values[np.concatenate(blocks)])
    sequences, _ = sequences_for_blocks(prepared.scaled_values, blocks)
    return score_lstm(model, sequences)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def save_reference_run(run: ModelRun, prepared: PreparedData) -> list[Path]:
    import torch

    model_dir = MODEL_ROOT / run.model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "modelo.pt"
    scaler_path = model_dir / "scaler.pkl"
    history_path = model_dir / "historico_treino.csv"
    contract_path = model_dir / "contrato.json"
    torch.save(
        {
            "state_dict": run.model.state_dict(),
            "model_id": run.model_id,
            "seed": run.seed,
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "dense_hidden": DENSE_HIDDEN if run.model_id == "ae_denso" else None,
            "lstm_hidden": LSTM_HIDDEN if run.model_id == "ae_lstm" else None,
            "latent_dim": LATENT_DIM,
            "sequence_length": SEQUENCE_LENGTH if run.model_id == "ae_lstm" else None,
        },
        checkpoint,
    )
    with scaler_path.open("wb") as stream:
        pickle.dump(prepared.scaler, stream)
    normalization_path = save_baseline_normalization(
        prepared.baseline_normalization, model_dir
    )
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(run.history.train_loss) + 1),
            "train_loss": run.history.train_loss,
            "validation_loss": run.history.validation_loss,
        }
    ).to_csv(history_path, index=False, lineterminator="\n")
    _write_json(
        contract_path,
        {
            "model_id": run.model_id,
            "model_name": MODEL_NAMES[run.model_id],
            "seed": run.seed,
            "dataset": "GPVS-Faults",
            "feature_columns": list(FEATURE_COLUMNS),
            "score_method": "mean_squared_reconstruction_error",
            "score_threshold": run.threshold,
            "threshold_method": "empirical_p99_higher",
            "threshold_effective_percentile": THRESHOLD_PERCENTILE,
            "calibration_n": len(run.calibration_scores),
            "healthy_test_n": len(run.healthy_test_scores),
            "healthy_test_false_positive_rate": float(
                np.mean(run.healthy_test_scores > run.threshold)
            ),
            "n_parameters": run.n_parameters,
            "best_epoch": run.history.best_epoch,
            "stopped_epoch": run.history.stopped_epoch,
            "best_validation_loss": run.history.best_validation_loss,
            "training_budget": {
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            },
            "roles": {
                "train": "weights_and_shared_scaler",
                "validation": "early_stopping_only",
                "calibration": "own_p99_threshold_only",
                "test": "healthy_false_positive_estimate_only",
            },
        },
    )
    return [checkpoint, scaler_path, normalization_path, history_path, contract_path]


def train_models(
    prepared: PreparedData,
    *,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
) -> dict[str, list[ModelRun]]:
    import torch

    if REFERENCE_SEED not in seeds:
        raise ValueError(f"As sementes devem incluir a referência {REFERENCE_SEED}")
    device = torch.device("cpu")
    train_indices = np.asarray(prepared.split["train"], dtype=int)
    validation_indices = np.asarray(prepared.split["validation"], dtype=int)
    train_sequences, _ = sequences_for_blocks(
        prepared.scaled_values, role_blocks(prepared.split, "train")
    )
    validation_sequences, _ = sequences_for_blocks(
        prepared.scaled_values, role_blocks(prepared.split, "validation")
    )
    runs: dict[str, list[ModelRun]] = {model_id: [] for model_id in MODEL_IDS}
    for seed in tuple(int(value) for value in seeds):
        dense, dense_history = train_dense(
            prepared.scaled_values[train_indices],
            prepared.scaled_values[validation_indices],
            seed=seed,
            device=device,
        )
        dense_calibration = score_role("ae_denso", dense, prepared, "calibration")
        dense_test = score_role("ae_denso", dense, prepared, "test")
        runs["ae_denso"].append(
            ModelRun(
                model_id="ae_denso",
                seed=seed,
                model=dense,
                threshold=empirical_threshold(dense_calibration),
                calibration_scores=dense_calibration,
                healthy_test_scores=dense_test,
                history=dense_history,
                n_parameters=parameter_count(dense),
            )
        )

        lstm, lstm_history = train_lstm(
            train_sequences,
            validation_sequences,
            seed=seed,
            device=device,
        )
        lstm_calibration = score_role("ae_lstm", lstm, prepared, "calibration")
        lstm_test = score_role("ae_lstm", lstm, prepared, "test")
        runs["ae_lstm"].append(
            ModelRun(
                model_id="ae_lstm",
                seed=seed,
                model=lstm,
                threshold=empirical_threshold(lstm_calibration),
                calibration_scores=lstm_calibration,
                healthy_test_scores=lstm_test,
                history=lstm_history,
                n_parameters=parameter_count(lstm),
            )
        )
    for model_id in MODEL_IDS:
        reference = next(run for run in runs[model_id] if run.seed == REFERENCE_SEED)
        save_reference_run(reference, prepared)
    return runs


__all__ = [
    "MODEL_IDS",
    "MODEL_NAMES",
    "MODEL_ROOT",
    "ModelRun",
    "REFERENCE_SEED",
    "STABILITY_SEEDS",
    "THRESHOLD_PERCENTILE",
    "empirical_threshold",
    "save_reference_run",
    "score_role",
    "train_models",
]
