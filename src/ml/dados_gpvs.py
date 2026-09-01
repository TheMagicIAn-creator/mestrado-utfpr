"""Contrato único de dados do comparativo Denso versus AE-LSTM.

Este módulo é o único responsável por localizar os 16 ensaios GPVS-Faults,
validar os sinais, extrair as 24 features elétricas, criar os quatro papéis
saudáveis e aplicar as normalizações de baseline. Ele não gera gráficos nem
resultados científicos autônomos.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.core.config import RAIZ_PROJETO


ROOT = Path(RAIZ_PROJETO)
DEFAULT_DATASET_DIR = (
    ROOT / "dados" / "brutos" / "gpvs" / "csv" / "CSV_Files"
)
DATASET_DIR = Path(os.getenv("AL_IADO_DATASET_GPVS", DEFAULT_DATASET_DIR))
PROCESSED_DIR = ROOT / "dados" / "processados"
HEALTHY_FEATURES_PATH = PROCESSED_DIR / "features_gpvs.parquet"
FAULT_FEATURES_PATH = PROCESSED_DIR / "features_gpvs_falhas.parquet"
DATASET_MANIFEST_PATH = ROOT / "dados" / "dataset_manifest.json"

DATASET_NAME = "GPVS-Faults"
DATASET_DOI = "10.17632/n76t439f65.1"
OFFICIAL_ZIP_SHA256 = (
    "88cd20c848fee86752870cf9b198eab45568c31355685328dd75aba982bf1a63"
)
HEALTHY_EXPERIMENTS = ("F0L", "F0M")
FAULT_EXPERIMENTS = tuple(f"F{fault}{mode}" for fault in range(1, 8) for mode in "LM")
ALL_EXPERIMENTS = HEALTHY_EXPERIMENTS + FAULT_EXPERIMENTS

SOURCE_COLUMNS = (
    "Time",
    "Ipv",
    "Vpv",
    "Vdc",
    "ia",
    "ib",
    "ic",
    "va",
    "vb",
    "vc",
    "Iabc",
    "If",
    "Vabc",
    "Vf",
)
PRIMARY_COLUMNS = ("Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc")
DC_COLUMNS = ("Ipv", "Vpv", "Vdc")
CURRENT_COLUMNS = ("ia", "ib", "ic")
VOLTAGE_COLUMNS = ("va", "vb", "vc")
FEATURE_COLUMNS = (
    "Ipv_median",
    "Ipv_iqr",
    "Vpv_median",
    "Vpv_iqr",
    "Vdc_median",
    "Vdc_iqr",
    "ia_rms",
    "ib_rms",
    "ic_rms",
    "va_rms",
    "vb_rms",
    "vc_rms",
    "ia_thd",
    "ib_thd",
    "ic_thd",
    "va_thd",
    "vb_thd",
    "vc_thd",
    "i_rms_unbalance",
    "v_rms_unbalance",
    "p_ac_mean",
    "p_ac_std",
    "p_dc_median",
    "p_dc_iqr",
)
FAULT_NAMES = {
    1: "Falha completa de um IGBT",
    2: "Erro de 20% no sistema de sensor/realimentação",
    3: "Afundamentos intermitentes de tensão",
    4: "Sombreamento parcial não uniforme (10-20%)",
    5: "Circuito aberto em 15% do arranjo PV",
    6: "Ganho do controlador PI reduzido em 20%",
    7: "Constante de tempo do controlador PI elevada em 20%",
}
FAULT_CONTRACTS = {
    1: {
        "native_interpretation": "falha completa de um IGBT",
        "fmeca_scope": "igbt",
        "scope_relation": "direct_native_counterpart",
        "physical_component_failure": True,
    },
    2: {
        "native_interpretation": "erro de 20% no sistema de sensor/realimentação",
        "fmeca_scope": "sensor_feedback_system",
        "scope_relation": "direct_native_counterpart",
        "physical_component_failure": False,
    },
    3: {
        "native_interpretation": "afundamentos intermitentes de tensão da rede",
        "fmeca_scope": None,
        "scope_relation": "outside_canonical_fmeca_trio",
        "physical_component_failure": False,
    },
    4: {
        "native_interpretation": "sombreamento parcial não uniforme entre 10% e 20%",
        "fmeca_scope": None,
        "scope_relation": "outside_canonical_fmeca_trio",
        "physical_component_failure": False,
    },
    5: {
        "native_interpretation": "circuito aberto em 15% do arranjo fotovoltaico",
        "fmeca_scope": None,
        "scope_relation": "outside_canonical_fmeca_trio",
        "physical_component_failure": True,
    },
    6: {
        "native_interpretation": "redução de 20% no ganho do controlador PI",
        "fmeca_scope": "inverter_control_system",
        "scope_relation": "functional_control_anomaly",
        "physical_component_failure": False,
    },
    7: {
        "native_interpretation": "aumento de 20% na constante de tempo do controlador PI",
        "fmeca_scope": "inverter_control_system",
        "scope_relation": "functional_control_anomaly",
        "physical_component_failure": False,
    },
}
OPERATING_MODES = {
    "L": "IPPT (potência limitada)",
    "M": "MPPT (potência máxima)",
}

SAMPLING_RATE_HZ = 10_000.0
GRID_FREQUENCY_HZ = 50.0
WINDOW_SAMPLES = int(round(SAMPLING_RATE_HZ / GRID_FREQUENCY_HZ))
PURGE_WINDOWS = 2
BASELINE_FRACTION = 0.50
BASELINE_MIN_WINDOWS = 30
BASELINE_IQR_FLOOR_FRACTION = 0.10
NORMALIZATION_FILENAME = "normalizacao_baseline_gpvs.npz"


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


@dataclass
class PreparedData:
    features: pd.DataFrame
    split: dict
    normalized_values: np.ndarray
    scaled_values: np.ndarray
    scaler: RobustScaler
    baseline_normalization: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_files(directory: Path = DATASET_DIR) -> dict[str, Path]:
    """Localiza exatamente F0L-F7M e rejeita um conjunto incompleto."""

    directory = Path(directory)
    candidates = (directory, directory / "CSV_Files")
    base = next((path for path in candidates if (path / "F0L.csv").is_file()), directory)
    files = {name: base / f"{name}.csv" for name in ALL_EXPERIMENTS}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "GPVS-Faults incompleto; ensaios ausentes: " + ", ".join(missing)
        )
    unexpected = sorted(
        path.name
        for path in base.glob("*.csv")
        if path.stem.upper() not in set(ALL_EXPERIMENTS)
    )
    if unexpected:
        raise ValueError(
            "A pasta GPVS ativa contém CSVs fora do contrato F0L-F7M: "
            + ", ".join(unexpected)
        )
    return files


def parse_experiment(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"F([0-7])([LM])(?:\.csv)?", Path(name).name)
    if not match:
        raise ValueError(f"Nome de ensaio GPVS inválido: {name}")
    return int(match.group(1)), match.group(2)


def infer_sampling(time_values) -> dict[str, float]:
    time = np.asarray(time_values, dtype=float)
    if time.ndim != 1 or len(time) < 1000 or not np.isfinite(time).all():
        raise ValueError("Time deve ser finito, unidimensional e ter >= 1000 pontos")
    delta = np.diff(time)
    if np.any(delta <= 0):
        raise ValueError("Time deve ser estritamente crescente")
    median = float(np.median(delta))
    rate = 1.0 / median
    if not 9_000.0 <= rate <= 11_000.0:
        raise ValueError(f"Taxa de amostragem fora do contrato GPVS: {rate:.3f} Hz")
    return {
        "dt_median_s": median,
        "dt_min_s": float(delta.min()),
        "dt_max_s": float(delta.max()),
        "fs_hz": rate,
        "sampling_period_us": median * 1e6,
    }


def validate_frame(frame: pd.DataFrame, experiment: str) -> dict[str, float]:
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{experiment}: colunas ausentes: {missing}")
    values = frame[list(SOURCE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{experiment}: há valores NaN ou infinitos")
    return infer_sampling(frame["Time"].to_numpy(dtype=float))


def _iqr(values: np.ndarray) -> float:
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _thd_one_cycle(values: np.ndarray, max_harmonic: int = 40) -> float:
    signal = np.asarray(values, dtype=float)
    spectrum = np.abs(np.fft.rfft(signal - np.mean(signal)))
    fundamental = max(float(spectrum[1]), np.finfo(float).eps)
    limit = min(int(max_harmonic) + 1, len(spectrum))
    return float(np.sqrt(np.sum(spectrum[2:limit] ** 2)) / fundamental)


def feature_vector(window: pd.DataFrame) -> np.ndarray:
    missing = [column for column in PRIMARY_COLUMNS if column not in window.columns]
    if missing:
        raise ValueError(f"Janela GPVS sem colunas primárias: {missing}")
    matrix = window[list(PRIMARY_COLUMNS)].to_numpy(dtype=float)
    if len(matrix) != WINDOW_SAMPLES:
        raise ValueError(
            f"Janela GPVS deve ter {WINDOW_SAMPLES} amostras; recebeu {len(matrix)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Janela GPVS contém NaN ou infinito")
    signals = {column: matrix[:, index] for index, column in enumerate(PRIMARY_COLUMNS)}
    features: list[float] = []
    for column in DC_COLUMNS:
        features.extend([float(np.median(signals[column])), _iqr(signals[column])])
    current_rms = np.sqrt(np.mean(matrix[:, 3:6] ** 2, axis=0))
    voltage_rms = np.sqrt(np.mean(matrix[:, 6:9] ** 2, axis=0))
    features.extend(current_rms.tolist())
    features.extend(voltage_rms.tolist())
    for column in CURRENT_COLUMNS + VOLTAGE_COLUMNS:
        features.append(_thd_one_cycle(signals[column]))
    features.extend(
        [
            float(np.std(current_rms) / max(float(np.mean(current_rms)), np.finfo(float).eps)),
            float(np.std(voltage_rms) / max(float(np.mean(voltage_rms)), np.finfo(float).eps)),
        ]
    )
    ac_power = np.sum(matrix[:, 3:6] * matrix[:, 6:9], axis=1)
    dc_power = matrix[:, 0] * matrix[:, 1]
    features.extend(
        [
            float(np.mean(ac_power)),
            float(np.std(ac_power)),
            float(np.median(dc_power)),
            _iqr(dc_power),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def extract_experiment_features(
    frame: pd.DataFrame,
    experiment: str,
) -> tuple[pd.DataFrame, dict]:
    fault, mode = parse_experiment(experiment)
    sampling = validate_frame(frame, experiment)
    exact_window = sampling["fs_hz"] / GRID_FREQUENCY_HZ
    window_samples = int(round(exact_window))
    if abs(window_samples - exact_window) / exact_window > 0.01:
        raise ValueError("A amostragem não permite janela de um ciclo a 50 Hz")
    if window_samples != WINDOW_SAMPLES:
        raise ValueError(
            f"Janela observada {window_samples} diverge do contrato {WINDOW_SAMPLES}"
        )

    time = frame["Time"].to_numpy(dtype=float)
    complete_windows = len(frame) // window_samples
    nominal_fault_sample = len(frame) // 2 if fault else None
    rows = []
    for window_index in range(complete_windows):
        start = window_index * window_samples
        end = start + window_samples
        window = frame.iloc[start:end]
        if nominal_fault_sample is None:
            phase = "healthy"
        elif end <= nominal_fault_sample:
            phase = "pre_fault"
        elif start >= nominal_fault_sample:
            phase = "post_fault"
        else:
            phase = "transition"
        rows.append(
            {
                "experiment": experiment,
                "fault": fault,
                "mode": mode,
                "window_index": window_index,
                "sample_start": start,
                "sample_end": end,
                "time_start_s": float(time[start]),
                "time_end_s": float(time[end - 1]),
                "time_center_s": float((time[start] + time[end - 1]) / 2.0),
                "phase": phase,
                **dict(zip(FEATURE_COLUMNS, feature_vector(window), strict=True)),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or not np.isfinite(result[list(FEATURE_COLUMNS)].to_numpy()).all():
        raise ValueError(f"{experiment}: features vazias ou não finitas")
    metadata = {
        **sampling,
        "rows": int(len(frame)),
        "windows": int(len(result)),
        "window_samples": window_samples,
        "window_duration_s": window_samples / sampling["fs_hz"],
        "grid_frequency_hz": GRID_FREQUENCY_HZ,
        "fault_boundary_method": "nominal_mid_record" if fault else None,
        "fault_sample_nominal": nominal_fault_sample,
        "fault_time_nominal_s": (
            float(time[nominal_fault_sample]) if nominal_fault_sample is not None else None
        ),
        "discarded_tail_samples": int(len(frame) - complete_windows * window_samples),
    }
    return result, metadata


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_or_extract_features(
    *,
    force: bool = False,
    directory: Path = DATASET_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    files = dataset_files(directory)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    extraction_metadata: dict[str, dict] = {}

    if force or not HEALTHY_FEATURES_PATH.is_file():
        frames = []
        for experiment in HEALTHY_EXPERIMENTS:
            features, metadata = extract_experiment_features(
                pd.read_csv(files[experiment]), experiment
            )
            frames.append(features)
            extraction_metadata[experiment] = metadata
        pd.concat(frames, ignore_index=True).to_parquet(HEALTHY_FEATURES_PATH, index=False)

    if force or not FAULT_FEATURES_PATH.is_file():
        frames = []
        for experiment in FAULT_EXPERIMENTS:
            features, metadata = extract_experiment_features(
                pd.read_csv(files[experiment]), experiment
            )
            frames.append(features)
            extraction_metadata[experiment] = metadata
        pd.concat(frames, ignore_index=True).to_parquet(FAULT_FEATURES_PATH, index=False)

    healthy = pd.read_parquet(HEALTHY_FEATURES_PATH)
    faults = pd.read_parquet(FAULT_FEATURES_PATH)
    observed = set(healthy["experiment"]) | set(faults["experiment"])
    if observed != set(ALL_EXPERIMENTS):
        raise ValueError(
            f"Contrato GPVS incompleto: esperado={set(ALL_EXPERIMENTS)}, observado={observed}"
        )
    raw_files = {
        name: {
            "path": path.relative_to(ROOT).as_posix()
            if path.is_relative_to(ROOT)
            else str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for name, path in files.items()
    }
    manifest = {
        "dataset": DATASET_NAME,
        "doi": DATASET_DOI,
        "official_zip_sha256": OFFICIAL_ZIP_SHA256,
        "active_dataset_count": 1,
        "experiments": list(ALL_EXPERIMENTS),
        "healthy_experiments": list(HEALTHY_EXPERIMENTS),
        "fault_experiments": list(FAULT_EXPERIMENTS),
        "fault_catalog": {
            f"F{fault}": contract for fault, contract in FAULT_CONTRACTS.items()
        },
        "synthetic_faults_used_in_experimental_core": False,
        "fault_boundary": {
            "method": "nominal_mid_record",
            "instrumented_trigger_available": False,
            "caveat": (
                "A fronteira de 50% é nominal; os CSVs não fornecem canal de disparo."
            ),
        },
        "raw_files": raw_files,
        "processed": {
            "healthy_features": HEALTHY_FEATURES_PATH.relative_to(ROOT).as_posix(),
            "fault_features": FAULT_FEATURES_PATH.relative_to(ROOT).as_posix(),
            "n_healthy_windows": int(len(healthy)),
            "n_fault_windows": int(len(faults)),
            "feature_columns": list(FEATURE_COLUMNS),
        },
        "extraction_metadata": extraction_metadata,
    }
    _write_json(DATASET_MANIFEST_PATH, manifest)
    return healthy, faults, manifest


def _fractional_split(indices: np.ndarray, purge: int = PURGE_WINDOWS) -> TemporalSplit:
    values = np.asarray(indices, dtype=int)
    if len(values) < 60 or np.any(np.diff(values) != 1):
        raise ValueError("O split exige ao menos 60 índices consecutivos")
    n = len(values)
    boundary_50 = int(n * 0.50)
    boundary_65 = int(n * 0.65)
    boundary_80 = int(n * 0.80)
    split = TemporalSplit(
        train=values[:boundary_50],
        validation=values[boundary_50 + purge : boundary_65],
        calibration=values[boundary_65 + purge : boundary_80],
        test=values[boundary_80 + purge :],
    )
    blocks = (split.train, split.validation, split.calibration, split.test)
    if any(len(block) < 10 for block in blocks):
        raise ValueError("O split GPVS produziu papel com menos de 10 janelas")
    combined = np.concatenate(blocks)
    if len(np.unique(combined)) != len(combined):
        raise AssertionError("O split GPVS possui sobreposição")
    return split


def split_healthy_features(features: pd.DataFrame) -> dict:
    roles = {name: [] for name in ("train", "validation", "calibration", "test")}
    per_experiment: dict[str, dict[str, list[int]]] = {}
    boundaries = {name: [] for name in roles}
    labels = features["experiment"].astype(str).to_numpy()
    for experiment in HEALTHY_EXPERIMENTS:
        positions = np.flatnonzero(labels == experiment)
        local = _fractional_split(np.arange(len(positions)), purge=PURGE_WINDOWS)
        per_experiment[experiment] = {}
        for role in roles:
            global_indices = positions[getattr(local, role)]
            roles[role].append(global_indices)
            per_experiment[experiment][role] = global_indices.tolist()
            boundaries[role].append(
                [int(global_indices[0]), int(global_indices[-1]) + 1]
            )
    result = {
        role: np.concatenate(parts).astype(int) for role, parts in roles.items()
    }
    combined = np.concatenate([result[role] for role in roles])
    if len(np.unique(combined)) != len(combined):
        raise AssertionError("Papéis saudáveis compartilham janelas")
    result.update(
        {
            "strategy": "temporal_blocks_per_F0L_F0M",
            "nominal_fractions": {
                "train": 0.50,
                "validation": 0.15,
                "calibration": 0.15,
                "test": 0.20,
            },
            "purge_windows": PURGE_WINDOWS,
            "boundaries": boundaries,
            "per_experiment": per_experiment,
        }
    )
    return result


def _baseline_statistics(
    matrix: np.ndarray,
    indices: np.ndarray,
    iqr_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block = np.asarray(matrix, dtype=float)[np.asarray(indices, dtype=int)]
    if len(block) < BASELINE_MIN_WINDOWS:
        raise ValueError(f"Baseline exige ao menos {BASELINE_MIN_WINDOWS} janelas")
    median = np.median(block, axis=0)
    q25, q75 = np.percentile(block, (25, 75), axis=0)
    scale = np.maximum(q75 - q25, np.asarray(iqr_floor, dtype=float))
    return median, scale


def fit_baseline_normalization(features: pd.DataFrame, split: dict) -> tuple[np.ndarray, dict]:
    matrix = features[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    train = matrix[np.asarray(split["train"], dtype=int)]
    q25, q75 = np.percentile(train, (25, 75), axis=0)
    iqr_floor = np.maximum(
        (q75 - q25) * BASELINE_IQR_FLOOR_FRACTION,
        np.full(len(FEATURE_COLUMNS), 1e-6),
    )
    normalized = np.empty_like(matrix, dtype=np.float32)
    baselines = {}
    labels = features["experiment"].astype(str).to_numpy()
    for experiment in HEALTHY_EXPERIMENTS:
        positions = np.flatnonzero(labels == experiment)
        train_indices = np.asarray(
            split["per_experiment"][experiment]["train"], dtype=int
        )
        median, scale = _baseline_statistics(matrix, train_indices, iqr_floor)
        normalized[positions] = ((matrix[positions] - median) / scale).astype(np.float32)
        baselines[experiment] = {"median": median, "scale": scale}
    return normalized, {
        "feature_columns": list(FEATURE_COLUMNS),
        "iqr_floor": iqr_floor,
        "baselines": baselines,
        "baseline_fraction": BASELINE_FRACTION,
        "baseline_min_windows": BASELINE_MIN_WINDOWS,
        "iqr_floor_fraction": BASELINE_IQR_FLOOR_FRACTION,
    }


def prepare_healthy_data(features: pd.DataFrame) -> PreparedData:
    split = split_healthy_features(features)
    normalized, baseline = fit_baseline_normalization(features, split)
    scaler = RobustScaler()
    scaler.fit(normalized[np.asarray(split["train"], dtype=int)])
    scaled = scaler.transform(normalized).astype(np.float32)
    return PreparedData(
        features=features,
        split=split,
        normalized_values=normalized,
        scaled_values=scaled,
        scaler=scaler,
        baseline_normalization=baseline,
    )


def normalize_f0_vectors(
    vectors: np.ndarray,
    experiments: np.ndarray | list[str],
    normalization: dict,
) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    labels = np.asarray(experiments).astype(str)
    if len(matrix) != len(labels):
        raise ValueError("Cada vetor deve declarar seu ensaio GPVS")
    result = np.empty_like(matrix, dtype=np.float32)
    for experiment in np.unique(labels):
        if experiment not in normalization["baselines"]:
            raise ValueError(f"Baseline saudável desconhecido: {experiment}")
        mask = labels == experiment
        baseline = normalization["baselines"][experiment]
        result[mask] = (
            (matrix[mask] - baseline["median"]) / baseline["scale"]
        ).astype(np.float32)
    return result


def normalize_commissioning(
    features: pd.DataFrame,
    normalization: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    pre = np.flatnonzero(features["phase"].eq("pre_fault").to_numpy())
    post = np.flatnonzero(features["phase"].eq("post_fault").to_numpy())
    n_baseline = max(
        int(normalization["baseline_min_windows"]),
        int(np.floor(len(pre) * float(normalization["baseline_fraction"]))),
    )
    if n_baseline >= len(pre):
        raise ValueError("Pré-falha insuficiente para comissionamento e teste separados")
    baseline_indices = pre[:n_baseline]
    pre_test = pre[n_baseline:]
    matrix = features[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    median, scale = _baseline_statistics(
        matrix, baseline_indices, normalization["iqr_floor"]
    )
    transformed = ((matrix - median) / scale).astype(np.float32)
    return transformed, pre_test, post, {
        "n_baseline": int(len(baseline_indices)),
        "n_pre_test": int(len(pre_test)),
        "n_post_test": int(len(post)),
        "baseline_fraction_of_pre": float(len(baseline_indices) / len(pre)),
    }


def save_baseline_normalization(normalization: dict, directory: Path) -> Path:
    path = Path(directory) / NORMALIZATION_FILENAME
    experiments = list(normalization["baselines"])
    np.savez_compressed(
        path,
        feature_columns=np.asarray(normalization["feature_columns"], dtype="U64"),
        iqr_floor=np.asarray(normalization["iqr_floor"], dtype=np.float64),
        experiments=np.asarray(experiments, dtype="U8"),
        medians=np.vstack(
            [normalization["baselines"][item]["median"] for item in experiments]
        ).astype(np.float64),
        scales=np.vstack(
            [normalization["baselines"][item]["scale"] for item in experiments]
        ).astype(np.float64),
        baseline_fraction=np.asarray([normalization["baseline_fraction"]]),
        baseline_min_windows=np.asarray([normalization["baseline_min_windows"]]),
        iqr_floor_fraction=np.asarray([normalization["iqr_floor_fraction"]]),
    )
    return path


def role_blocks(split: dict, role: str) -> list[np.ndarray]:
    return [
        np.asarray(split["per_experiment"][experiment][role], dtype=int)
        for experiment in HEALTHY_EXPERIMENTS
    ]


def load_holdout_windows(
    prepared: PreparedData,
    *,
    directory: Path = DATASET_DIR,
) -> tuple[list[pd.DataFrame], dict]:
    files = dataset_files(directory)
    test_indices = np.asarray(prepared.split["test"], dtype=int)
    cache: dict[str, pd.DataFrame] = {}
    windows: list[pd.DataFrame] = []
    records = []
    for index in test_indices:
        row = prepared.features.iloc[int(index)]
        experiment = str(row["experiment"])
        if experiment not in cache:
            cache[experiment] = pd.read_csv(
                files[experiment],
                usecols=list(PRIMARY_COLUMNS),
                dtype={column: np.float32 for column in PRIMARY_COLUMNS},
            )
        start, end = int(row["sample_start"]), int(row["sample_end"])
        window = cache[experiment].iloc[start:end].copy().reset_index(drop=True)
        if len(window) != WINDOW_SAMPLES:
            raise ValueError(f"Holdout {experiment}/{index} possui janela incompleta")
        window.attrs.update({"experiment": experiment, "feature_index": int(index)})
        windows.append(window)
        records.append(
            {
                "experiment": experiment,
                "feature_index": int(index),
                "sample_start": start,
                "sample_end": end,
            }
        )
    if len(windows) != len(test_indices):
        raise ValueError("O holdout bruto e o holdout de features não estão alinhados")
    return windows, {
        "dataset": DATASET_NAME,
        "doi": DATASET_DOI,
        "protocol": "purged_temporal_holdout_F0L_F0M",
        "purge_windows": prepared.split["purge_windows"],
        "non_overlapping_raw_windows": True,
        "n_windows": len(windows),
        "records": records,
    }


__all__ = [
    "ALL_EXPERIMENTS",
    "DATASET_DIR",
    "DATASET_DOI",
    "DATASET_NAME",
    "FAULT_EXPERIMENTS",
    "FAULT_NAMES",
    "FAULT_CONTRACTS",
    "FEATURE_COLUMNS",
    "HEALTHY_EXPERIMENTS",
    "NORMALIZATION_FILENAME",
    "OFFICIAL_ZIP_SHA256",
    "OPERATING_MODES",
    "PRIMARY_COLUMNS",
    "PURGE_WINDOWS",
    "PreparedData",
    "WINDOW_SAMPLES",
    "dataset_files",
    "extract_experiment_features",
    "feature_vector",
    "fit_baseline_normalization",
    "load_holdout_windows",
    "load_or_extract_features",
    "normalize_commissioning",
    "normalize_f0_vectors",
    "prepare_healthy_data",
    "role_blocks",
    "save_baseline_normalization",
    "split_healthy_features",
]
