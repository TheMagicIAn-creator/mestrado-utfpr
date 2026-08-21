"""Assinaturas sintéticas E2 ligadas aos três componentes da FMECA.

O parâmetro ``magnitude`` é a fração adimensional ``a_det`` da assinatura
nominal. Ele não representa tempo, severidade S da FMECA, vida útil ou taxa de
falha. As funções operam sobre uma janela saudável GPVS de um ciclo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd


CURRENT_COLUMNS = ("ia", "ib", "ic")
SAMPLING_RATE_HZ = 10_000.0
GRID_FREQUENCY_HZ = 50.0


@dataclass(frozen=True)
class FailureSignature:
    component_id: str
    component_name: str
    severity: int
    occurrence: int
    detection: int
    npr: int
    signal_columns: tuple[str, ...]
    formula: str
    physical_hypothesis: str
    limitation: str

    def as_dict(self) -> dict:
        return asdict(self)


SIGNATURES = (
    FailureSignature(
        component_id="contator_ac",
        component_name="Contator AC",
        severity=5,
        occurrence=7,
        detection=9,
        npr=315,
        signal_columns=("ia",),
        formula="ia += N(0, a_det * std(ia) * 0.30)",
        physical_hypothesis=(
            "Chattering ou comutação deficiente introduz conteúdo transitório "
            "na corrente CA."
        ),
        limitation=(
            "Ruído gaussiano é proxy sintético e requer calibração em contator real."
        ),
    ),
    FailureSignature(
        component_id="igbt",
        component_name="IGBT",
        severity=5,
        occurrence=6,
        detection=3,
        npr=90,
        signal_columns=CURRENT_COLUMNS,
        formula=(
            "i_fase += a_det * std(i_fase) * "
            "[0.30 sen(5wt)+0.20 sen(7wt)+0.10 sen(11wt)+0.05 sen(13wt)]"
        ),
        physical_hypothesis=(
            "Degradação de chaveamento eleva THD e harmônicos característicos "
            "nas correntes de fase."
        ),
        limitation=(
            "As amplitudes harmônicas são proxies fundamentados, não medições "
            "de envelhecimento térmico no GPVS."
        ),
    ),
    FailureSignature(
        component_id="fusivel_ac",
        component_name="Fusível AC",
        severity=5,
        occurrence=3,
        detection=2,
        npr=30,
        signal_columns=("ia",),
        formula="ia *= 1 - 0.12 * a_det",
        physical_hypothesis=(
            "Perda parcial de uma fase reduz a corrente conduzida e aumenta o "
            "desbalanceamento entre fases."
        ),
        limitation=(
            "A perda parcial é proxy contínuo; não reproduz abertura abrupta "
            "nem a atuação completa da proteção."
        ),
    ),
)

SIGNATURE_BY_ID = {signature.component_id: signature for signature in SIGNATURES}


def _validate(window: pd.DataFrame, magnitude: float) -> float:
    value = float(magnitude)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("a_det deve ser finito e pertencer ao intervalo [0, 1]")
    missing = [column for column in CURRENT_COLUMNS if column not in window.columns]
    if missing:
        raise ValueError(f"Janela GPVS sem correntes de fase: {missing}")
    if len(window) < 2:
        raise ValueError("A assinatura exige uma janela com ao menos duas amostras")
    return value


def inject_contactor(
    window: pd.DataFrame,
    magnitude: float,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    value = _validate(window, magnitude)
    result = window.copy()
    if value == 0.0:
        return result
    signal = window["ia"].to_numpy(dtype=float)
    noise = np.random.default_rng(int(seed)).normal(
        0.0, value * float(np.std(signal)) * 0.30, size=len(signal)
    )
    result["ia"] = signal + noise
    return result


def inject_igbt(
    window: pd.DataFrame,
    magnitude: float,
    *,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    grid_frequency_hz: float = GRID_FREQUENCY_HZ,
) -> pd.DataFrame:
    value = _validate(window, magnitude)
    result = window.copy()
    if value == 0.0:
        return result
    time = np.arange(len(result), dtype=float) / float(sampling_rate_hz)
    harmonics = ((5, 0.30), (7, 0.20), (11, 0.10), (13, 0.05))
    for column in CURRENT_COLUMNS:
        signal = window[column].to_numpy(dtype=float)
        amplitude = float(np.std(signal))
        perturbation = sum(
            value
            * weight
            * amplitude
            * np.sin(2.0 * np.pi * order * float(grid_frequency_hz) * time)
            for order, weight in harmonics
        )
        result[column] = signal + perturbation
    return result


def inject_fuse(window: pd.DataFrame, magnitude: float) -> pd.DataFrame:
    value = _validate(window, magnitude)
    result = window.copy()
    result["ia"] = window["ia"].to_numpy(dtype=float) * (1.0 - 0.12 * value)
    return result


INJECTORS: dict[str, Callable[..., pd.DataFrame]] = {
    "contator_ac": inject_contactor,
    "igbt": inject_igbt,
    "fusivel_ac": inject_fuse,
}


__all__ = [
    "FailureSignature",
    "GRID_FREQUENCY_HZ",
    "INJECTORS",
    "SAMPLING_RATE_HZ",
    "SIGNATURES",
    "SIGNATURE_BY_ID",
    "inject_contactor",
    "inject_fuse",
    "inject_igbt",
]
