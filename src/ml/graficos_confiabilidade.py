"""Figuras acadêmicas dos cenários bibliográficos históricos de confiabilidade."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.confiabilidade_componentes import SCENARIOS
from src.ml.estilo_graficos import PALETA, TAM, adicionar_nota, aplicar_estilo


aplicar_estilo()

COLORS = {
    "contator_ac_derived": PALETA[0],
    "igbt_derived": PALETA[1],
    "fusivel_ac_derived": PALETA[2],
    "fusivel_ac_direct": "#8c5a2b",
}
LINESTYLES = {
    "derived_sensitivity": "-",
    "direct_bibliographic": "--",
}
LEGEND_LABELS = {
    "contator_ac_derived": "Contator AC (derivada)",
    "igbt_derived": "IGBT (derivada)",
    "fusivel_ac_derived": "Fusível AC (derivada)",
    "fusivel_ac_direct": "Fusível (direta, Tab. 3.4)",
}


def _save_pair(fig, base_path: Path, note: str) -> tuple[Path, Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    adicionar_nota(fig, note)
    fig.savefig(png, metadata={"Software": "ALIAdo - Matplotlib"})
    fig.savefig(
        pdf,
        metadata={
            "Creator": "ALIAdo - Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def _iter_curves(curves: pd.DataFrame):
    for scenario in SCENARIOS:
        yield scenario, curves[curves["scenario_id"].eq(scenario.scenario_id)]


def _plot_time_function(
    curves: pd.DataFrame,
    output: Path,
    *,
    column: str,
    title: str,
    ylabel: str,
    note: str,
    probability: bool = False,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    for scenario, block in _iter_curves(curves):
        style = {
            "color": COLORS[scenario.scenario_id],
            "linestyle": LINESTYLES[scenario.evidence_type],
            "label": LEGEND_LABELS[scenario.scenario_id],
        }
        ax.plot(block["time_years"], block[column], **style)
    ax.set_xlim(0, float(curves["time_years"].max()))
    if probability:
        ax.set_ylim(0, 1.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        maximum = float(curves[column].max())
        ax.set_ylim(0, maximum * 1.08 if maximum > 0 else 1.0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.set_xlabel("Tempo de operação (anos)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return _save_pair(fig, output, note)


def plot_reliability(curves: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    return _plot_time_function(
        curves,
        output,
        column="reliability",
        title="Curva de confiabilidade R(t) — cenários bibliográficos históricos",
        ylabel="R(t) - probabilidade de operação sem falha",
        note=r"Modelo exponencial: $R(t)=e^{-\lambda t}$. Eixos lineares; taxas bibliográficas direta/derivadas.",
        probability=True,
    )


def plot_cumulative_failure(curves: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    return _plot_time_function(
        curves,
        output,
        column="cumulative_failure_probability",
        title=(
            "Curva da probabilidade acumulada de falha F(t) — "
            "cenários bibliográficos históricos"
        ),
        ylabel="F(t) - probabilidade acumulada de falha",
        note=r"Modelo exponencial: $F(t)=1-e^{-\lambda t}$. Eixos lineares; não representa incidência observada de campo.",
        probability=True,
    )


def plot_failure_density(curves: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    return _plot_time_function(
        curves,
        output,
        column="failure_density_per_year",
        title=(
            "Curva da densidade de probabilidade de falha f(t) — "
            "cenários bibliográficos históricos"
        ),
        ylabel="f(t) (ano⁻¹)",
        note=r"Modelo exponencial: $f(t)=\lambda e^{-\lambda t}$. Escala linear; não é uma distribuição normal ajustada.",
    )


def plot_hazard(curves: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    return _plot_time_function(
        curves,
        output,
        column="hazard_per_year",
        title="Curva da taxa de falha h(t) — cenários bibliográficos históricos",
        ylabel="h(t) (ano⁻¹)",
        note=r"Modelo exponencial: $h(t)=\lambda$. Escala linear e taxa constante; não há curva de banheira estimável.",
    )


def plot_rates(scenarios: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    frame = scenarios.sort_values("lambda_per_hour", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    y = np.arange(len(frame))
    for index, row in frame.iterrows():
        ax.scatter(
            row["lambda_per_hour"],
            index,
            color=COLORS[row["scenario_id"]],
            marker="D" if row["evidence_type"] == "direct_bibliographic" else "o",
            s=62,
        )
    ax.set_xlim(0, float(frame["lambda_per_hour"].max()) * 1.08)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.set_yticks(y, frame["plot_label"])
    ax.set_xlabel(r"Taxa de falha, $\lambda$ (falhas h$^{-1}$)")
    ax.set_title("Comparação histórica das taxas bibliográficas de falha λ")
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", color="#555555", label="Derivada"
            ),
            Line2D(
                [], [], marker="D", linestyle="none", color="#555555", label="Direta"
            ),
        ],
        loc="lower right",
    )
    return _save_pair(
        fig,
        output,
        "Contator e IGBT não possuem taxa direta equivalente na fonte; a ausência permanece explícita.",
    )


def generate_all(
    output_dir: Path,
    *,
    curves: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> list[Path]:
    output_dir = Path(output_dir)
    paths: list[Path] = []
    for pair in (
        plot_reliability(curves, output_dir / "curva_confiabilidade"),
        plot_cumulative_failure(curves, output_dir / "curva_probabilidade_falha"),
        plot_failure_density(curves, output_dir / "curva_densidade_falha"),
        plot_hazard(curves, output_dir / "curva_taxa_falha"),
        plot_rates(scenarios, output_dir / "taxas_componentes"),
    ):
        paths.extend(pair)
    return paths


__all__ = [
    "generate_all",
    "plot_cumulative_failure",
    "plot_failure_density",
    "plot_hazard",
    "plot_rates",
    "plot_reliability",
]
