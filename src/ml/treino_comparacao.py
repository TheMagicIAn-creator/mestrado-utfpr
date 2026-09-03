"""Treino congelado dos dois braços canônicos no holdout saudável GPVS."""

from __future__ import annotations

import json
import math
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
    SCORE_TOP_K,
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

# Percentil canônico do limiar saudável, decidido pelo pesquisador em
# 2026-09-03. Era 99,9 até essa data.
#
# MOTIVO: a calibração saudável do GPVS tem 210 janelas. Pedir p99,9 com
# n=210 seleciona a ordem 210/210 — o limiar passa a ser o MÁXIMO da
# calibração, e o percentil declarado vira ficção (ver
# `minimum_n_for_percentile`: p99,9 exigiria n >= 1001). p99 é o maior
# percentil que 210 observações sustentam: ordem 208/210, p99,05 efetivo.
#
# O ponto histórico p99,9 continua reproduzível por `strict_threshold=False`
# e permanece na grade de sensibilidade, agora marcado como degenerado.
THRESHOLD_PERCENTILE = 99.0
HISTORICAL_THRESHOLD_PERCENTILE = 99.9


class DegenerateThresholdError(ValueError):
    """O percentil pedido não é distinguível do máximo amostral.

    Erro separado de `ValueError` genérico porque a saída é acionável: o
    chamador escolhe entre baixar o percentil ou aumentar a calibração, e a
    mensagem carrega os dois números necessários para decidir.
    """


def minimum_n_for_percentile(percentile: float) -> int | None:
    """Menor calibração em que `percentile` deixa de cair no máximo amostral.

    `calibrate_threshold` seleciona ``ceil((n-1) * p/100)``. Esse índice é o
    último (``n-1``) — isto é, o limiar É o maior escore visto — sempre que
    ``(n-1)*p/100 > n-2``. Resolvendo em ``n`` com ``q = p/100 < 1``::

        n >= (q - 2) / (q - 1)

    Para p99,9 isso dá 1001; para p99, dá 101. Com a calibração de 210 janelas
    do GPVS, portanto, p99 é representável e p99,9 não é.

    Devolve ``None`` para p100, que é o máximo amostral por definição e não
    tem tamanho de calibração que o conserte.
    """

    q = float(percentile) / 100.0
    if q >= 1.0:
        return None
    # O arredondamento evita que 1001,0000000000002 vire 1002 no teto.
    return int(math.ceil(round((q - 2.0) / (q - 1.0), 9)))


@dataclass(frozen=True)
class ThresholdCalibration:
    value: float
    requested_percentile: float
    effective_percentile: float
    selected_rank: int
    selected_order_index: int
    calibration_n: int
    percentile_resolution: float

    @property
    def is_sample_maximum(self) -> bool:
        """O limiar é o maior escore da calibração, não um percentil interior.

        Quando verdadeiro, `requested_percentile` é ficção: o valor publicado
        não separa a cauda saudável, ele a delimita. O ponto operacional fica
        no extremo conservador possível e sua variância é a variância de um
        máximo amostral, não a de um quantil.
        """

        return self.selected_order_index >= self.calibration_n - 1

    @property
    def minimum_n_for_request(self) -> int | None:
        return minimum_n_for_percentile(self.requested_percentile)

    def as_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            "threshold_method": "healthy_percentile_higher",
            "score_threshold": self.value,
            "threshold_requested_percentile": self.requested_percentile,
            "threshold_effective_percentile": self.effective_percentile,
            "threshold_selected_rank": self.selected_rank,
            "threshold_selected_order_index": self.selected_order_index,
            "calibration_n": self.calibration_n,
            "threshold_percentile_resolution": self.percentile_resolution,
            "threshold_is_sample_maximum": self.is_sample_maximum,
            "threshold_minimum_n_for_request": self.minimum_n_for_request,
        }


@dataclass
class ModelRun:
    model_id: str
    seed: int
    model: Any
    threshold_calibration: ThresholdCalibration
    score_top_k: int
    calibration_scores: np.ndarray
    healthy_test_scores: np.ndarray
    history: TrainingHistory
    n_parameters: int

    @property
    def threshold(self) -> float:
        return self.threshold_calibration.value


def calibrate_threshold(
    values: np.ndarray,
    percentile: float = THRESHOLD_PERCENTILE,
    *,
    strict: bool = False,
) -> ThresholdCalibration:
    """Seleciona um order statistic e explicita a resolução empírica disponível.

    Com ``strict``, recusa um percentil que degenere no máximo amostral. A
    varredura de sensibilidade percorre percentis degenerados de propósito e
    por isso não usa ``strict``; a publicação canônica usa, porque um limiar
    que é o máximo da calibração não sustenta o percentil que ela declara.
    """

    scores = np.asarray(values, dtype=float)
    requested = float(percentile)
    if len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("A calibração exige ao menos dois escores finitos")
    if not 0.0 < requested <= 100.0:
        raise ValueError("O percentil do limiar deve estar em (0, 100]")
    selected_index = int(np.ceil((len(scores) - 1) * requested / 100.0))
    selected_rank = selected_index + 1
    ordered = np.sort(scores)
    calibration = ThresholdCalibration(
        value=float(ordered[selected_index]),
        requested_percentile=requested,
        effective_percentile=100.0 * selected_rank / len(scores),
        selected_rank=selected_rank,
        selected_order_index=selected_index,
        calibration_n=len(scores),
        percentile_resolution=100.0 / len(scores),
    )
    if strict and calibration.is_sample_maximum:
        minimo = calibration.minimum_n_for_request
        saida = (
            f"aumente a calibração para n >= {minimo}"
            if minimo is not None
            else "peça um percentil menor que 100"
        )
        raise DegenerateThresholdError(
            f"p{requested:g} com n={len(scores)} seleciona a ordem "
            f"{selected_rank}/{len(scores)}: o limiar seria o MÁXIMO da "
            f"calibração, não o percentil pedido. Publicar assim declararia "
            f"um percentil que os dados não sustentam. Para corrigir, "
            f"{saida}, ou baixe o percentil pedido."
        )
    return calibration


def empirical_threshold(
    values: np.ndarray,
    percentile: float = THRESHOLD_PERCENTILE,
) -> float:
    """Compatibilidade: retorna somente o valor da calibração rastreável."""

    return calibrate_threshold(values, percentile).value


def score_role(
    model_id: str,
    model,
    prepared: PreparedData,
    role: str,
    *,
    score_top_k: int = SCORE_TOP_K,
) -> np.ndarray:
    blocks = role_blocks(prepared.split, role)
    if model_id == "ae_denso":
        return score_dense(
            model,
            prepared.scaled_values[np.concatenate(blocks)],
            top_k=score_top_k,
        )
    sequences, _ = sequences_for_blocks(prepared.scaled_values, blocks)
    return score_lstm(model, sequences, top_k=score_top_k)


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
            "score_top_k": run.score_top_k,
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
            "score_method": "mean_of_top_k_feature_squared_reconstruction_errors",
            "score_top_k": run.score_top_k,
            **run.threshold_calibration.as_dict(),
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
                # Orçamento IGUAL nos dois braços não significa capacidade
                # igual: o AE-LSTM tem cerca de 16x mais parâmetros. Publicar
                # `dropout` e `n_parameters` lado a lado deixa a assimetria
                # visível no artefato, em vez de só no código.
                "dropout": float(getattr(run.model, "dropout_p", DROPOUT)),
            },
            "roles": {
                "train": "weights_and_shared_scaler",
                "validation": "early_stopping_only",
                "calibration": "own_healthy_percentile_threshold_only",
                "test": "healthy_false_positive_estimate_only",
            },
        },
    )
    return [checkpoint, scaler_path, normalization_path, history_path, contract_path]


def train_models(
    prepared: PreparedData,
    *,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
    threshold_percentile: float = THRESHOLD_PERCENTILE,
    score_top_k: int = SCORE_TOP_K,
    strict_threshold: bool = True,
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
        dense_calibration = score_role(
            "ae_denso", dense, prepared, "calibration", score_top_k=score_top_k
        )
        dense_test = score_role(
            "ae_denso", dense, prepared, "test", score_top_k=score_top_k
        )
        runs["ae_denso"].append(
            ModelRun(
                model_id="ae_denso",
                seed=seed,
                model=dense,
                threshold_calibration=calibrate_threshold(
                    dense_calibration, threshold_percentile, strict=strict_threshold
                ),
                score_top_k=int(score_top_k),
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
        lstm_calibration = score_role(
            "ae_lstm", lstm, prepared, "calibration", score_top_k=score_top_k
        )
        lstm_test = score_role(
            "ae_lstm", lstm, prepared, "test", score_top_k=score_top_k
        )
        runs["ae_lstm"].append(
            ModelRun(
                model_id="ae_lstm",
                seed=seed,
                model=lstm,
                threshold_calibration=calibrate_threshold(
                    lstm_calibration, threshold_percentile, strict=strict_threshold
                ),
                score_top_k=int(score_top_k),
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
    "DegenerateThresholdError",
    "ModelRun",
    "REFERENCE_SEED",
    "STABILITY_SEEDS",
    "HISTORICAL_THRESHOLD_PERCENTILE",
    "THRESHOLD_PERCENTILE",
    "ThresholdCalibration",
    "calibrate_threshold",
    "empirical_threshold",
    "minimum_n_for_percentile",
    "save_reference_run",
    "score_role",
    "train_models",
]
