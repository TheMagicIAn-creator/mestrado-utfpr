"""Figuras acadêmicas dos cenários físicos de confiabilidade por componente."""

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


def plot_reliability_failure(curves: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")
    for scenario, block in _iter_curves(curves):
        style = {
            "color": COLORS[scenario.scenario_id],
            "linestyle": LINESTYLES[scenario.evidence_type],
            "label": LEGEND_LABELS[scenario.scenario_id],
        }
        axes[0].plot(block["time_years"], block["reliability"], **style)
        axes[1].plot(
            block["time_years"], block["cumulative_failure_probability"], **style
        )
    for ax, title, ylabel in (
        (axes[0], "Confiabilidade", "R(t) - probabilidade de operação sem falha"),
        (
            axes[1],
            "Probabilidade acumulada de falha",
            "F(t) - probabilidade de falha",
        ),
    ):
        ax.set_xlim(0, float(curves["time_years"].max()))
        ax.set_ylim(0, 1.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Tempo de operação (anos)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Confiabilidade física por componente sob cenários exponenciais")
    return _save_pair(
        fig,
        output,
        "Taxas bibliográficas direta/derivadas; o GPVS-Faults não fornece tempos de vida.",
    )


def plot_density_hazard(curves: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")
    for scenario, block in _iter_curves(curves):
        style = {
            "color": COLORS[scenario.scenario_id],
            "linestyle": LINESTYLES[scenario.evidence_type],
            "label": LEGEND_LABELS[scenario.scenario_id],
        }
        axes[0].plot(block["time_years"], block["failure_density_per_year"], **style)
        axes[1].plot(block["time_years"], block["hazard_per_year"], **style)
    positive_density = curves.loc[
        curves["failure_density_per_year"].gt(0), "failure_density_per_year"
    ]
    positive_hazard = curves.loc[curves["hazard_per_year"].gt(0), "hazard_per_year"]
    for ax, values, title, ylabel in (
        (
            axes[0],
            positive_density,
            "Densidade de probabilidade de falha",
            "f(t) (ano⁻¹)",
        ),
        (axes[1], positive_hazard, "Taxa de falha constante", "h(t) (ano⁻¹)"),
    ):
        lower = 10 ** np.floor(np.log10(values.min()))
        upper = 10 ** np.ceil(np.log10(values.max()))
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        ax.set_xlim(0, float(curves["time_years"].max()))
        ax.set_xlabel("Tempo de operação (anos)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Densidade f(t) e taxa de falha h(t): funções físicas distintas")
    return _save_pair(
        fig,
        output,
        r"Escalas ajustadas aos dados. No modelo exponencial $h(t)=\lambda$ e não há curva de banheira.",
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
    ax.set_xscale("log")
    ax.set_yticks(y, frame["plot_label"])
    ax.set_xlabel(r"Taxa de falha, $\lambda$ (falhas h$^{-1}$; escala logarítmica)")
    ax.set_title("Taxas usadas nos cenários de sensibilidade por componente")
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
        plot_reliability_failure(
            curves, output_dir / "confiabilidade_probabilidade_falha"
        ),
        plot_density_hazard(curves, output_dir / "densidade_taxa_falha"),
        plot_rates(scenarios, output_dir / "taxas_componentes"),
    ):
        paths.extend(pair)
    return paths


__all__ = [
    "generate_all",
    "plot_density_hazard",
    "plot_rates",
    "plot_reliability_failure",
]
