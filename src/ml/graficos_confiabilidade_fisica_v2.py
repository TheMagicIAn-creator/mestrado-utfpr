"""Figuras acadêmicas da confiabilidade física bibliográfica V2."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.confiabilidade_fisica_v2 import CENARIOS
from src.ml.estilo_graficos import (
    COR_GRADE,
    COR_TEXTO_SEC,
    PALETA,
    TAM,
    adicionar_nota,
    aplicar_estilo,
)

aplicar_estilo()

CORES = {
    "torres_colli_rate": "#c43d3d",
    "cristaldi_inverter_rate": PALETA[0],
    "obeidat_high_quality": PALETA[3],
    "obeidat_low_quality": PALETA[2],
    "dhople_markov_example": PALETA[4],
}

ESTILOS = {
    "secondary_bibliographic_rate": (0, (5, 2)),
    "literature_assumption": "-",
    "mil_hdbk_217f_prediction": "-.",
    "illustrative_markov_parameter": ":",
}


def _salvar_png_pdf(fig, caminho_base: Path, nota: str) -> tuple[Path, Path]:
    """Salva a mesma figura em raster de 300 dpi e PDF vetorial."""

    caminho_base = Path(caminho_base)
    caminho_base.parent.mkdir(parents=True, exist_ok=True)
    png = caminho_base.with_suffix(".png")
    pdf = caminho_base.with_suffix(".pdf")
    adicionar_nota(fig, nota)
    fig.savefig(png, metadata={"Software": "ALIAdo PV - Matplotlib"})
    fig.savefig(
        pdf,
        metadata={
            "Creator": "ALIAdo PV - Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def _iterar_curvas(curvas: pd.DataFrame):
    for cenario in CENARIOS:
        bloco = curvas[curvas["scenario_id"].eq(cenario.scenario_id)]
        yield cenario, bloco


def plotar_confiabilidade(curvas: pd.DataFrame, caminho_base: Path) -> tuple[Path, Path]:
    """Compara R(t) sem confundir fontes bibliográficas com dados de campo."""

    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    for cenario, bloco in _iterar_curvas(curvas):
        ax.plot(
            bloco["time_years"],
            bloco["reliability"],
            color=CORES[cenario.scenario_id],
            linestyle=ESTILOS[cenario.source_type],
            label=cenario.plot_label,
        )
    ax.set_xlim(0, float(curvas["time_years"].max()))
    ax.set_ylim(0, 1.01)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Tempo de operação sob o cenário (anos)")
    ax.set_ylabel("Confiabilidade, R(t)")
    ax.set_title("Probabilidade de operação sem falha ao longo do horizonte")
    ax.legend(loc="upper right", ncols=2)
    fig.suptitle(
        "Confiabilidade do inversor sob cenários bibliográficos de taxa constante"
    )
    return _salvar_png_pdf(
        fig,
        caminho_base,
        (
            "Modelo exponencial R(t)=exp(-λt). As linhas são cenários de "
            "sensibilidade de fontes distintas; não são estimativas do GPVS-Faults."
        ),
    )


def plotar_probabilidade_falha(
    curvas: pd.DataFrame, caminho_base: Path
) -> tuple[Path, Path]:
    """Compara a probabilidade acumulada F(t)=1-R(t)."""

    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    for cenario, bloco in _iterar_curvas(curvas):
        ax.plot(
            bloco["time_years"],
            bloco["cumulative_failure_probability"],
            color=CORES[cenario.scenario_id],
            linestyle=ESTILOS[cenario.source_type],
            label=cenario.plot_label,
        )
    ax.set_xlim(0, float(curvas["time_years"].max()))
    ax.set_ylim(0, 1.01)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Tempo de operação sob o cenário (anos)")
    ax.set_ylabel("Probabilidade acumulada de falha, F(t)")
    ax.set_title("Probabilidade de ao menos uma falha até o tempo t")
    ax.legend(loc="lower right", ncols=2)
    fig.suptitle(
        "Probabilidade acumulada de falha sob cenários bibliográficos"
    )
    return _salvar_png_pdf(
        fig,
        caminho_base,
        (
            "F(t)=1-exp(-λt). A probabilidade pertence ao modelo de cada cenário; "
            "não representa frequência empírica observada no dataset experimental."
        ),
    )


def plotar_densidade_e_taxa(
    curvas: pd.DataFrame, caminho_base: Path
) -> tuple[Path, Path]:
    """Distingue densidade de probabilidade e taxa instantânea de falha."""

    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")
    densidade = curvas[curvas["time_years"].le(10.0)]
    for cenario, bloco_completo in _iterar_curvas(curvas):
        bloco = densidade[densidade["scenario_id"].eq(cenario.scenario_id)]
        kwargs = {
            "color": CORES[cenario.scenario_id],
            "linestyle": ESTILOS[cenario.source_type],
        }
        axes[0].plot(
            bloco["time_years"],
            bloco["failure_density_per_year"],
            label=cenario.plot_label,
            **kwargs,
        )
        axes[1].plot(
            bloco_completo["time_years"],
            bloco_completo["hazard_per_year"],
            **kwargs,
        )

    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(1e-7, 2.0)
    axes[0].set_xlabel("Tempo de operação sob o cenário (anos)")
    axes[0].set_ylabel("Densidade de falha, f(t) (ano⁻¹)")
    axes[0].set_title("Densidade de probabilidade de falha")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_yscale("log")
    axes[1].set_xlim(0, float(curvas["time_years"].max()))
    axes[1].set_ylim(0.05, 2.0)
    axes[1].set_xlabel("Tempo de operação sob o cenário (anos)")
    axes[1].set_ylabel("Taxa instantânea de falha, h(t) (ano⁻¹)")
    axes[1].set_title("Risco constante do modelo exponencial")

    fig.suptitle("Densidade e taxa de falha: funções distintas do modelo")
    return _salvar_png_pdf(
        fig,
        caminho_base,
        (
            "Escala vertical logarítmica. f(t) é uma curva analítica suave; "
            "pontos dispersos exigiriam tempos de vida observados, indisponíveis no GPVS."
        ),
    )


def plotar_marcos(
    marcos: pd.DataFrame, caminho_base: Path
) -> tuple[Path, Path]:
    """Compara B1, B10, mediana e 1/lambda sem sugerir precisão inexistente."""

    dados = marcos.set_index("scenario_id")
    ordem = sorted(CENARIOS, key=lambda c: c.reciprocal_time_years)
    y = list(range(len(ordem)))
    marcadores = (
        ("b1_years", "B1", "|"),
        ("b10_years", "B10", "o"),
        ("median_years", "Mediana", "s"),
        ("reciprocal_time_years", "1/λ", "D"),
    )

    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")
    for posicao, cenario in zip(y, ordem, strict=True):
        valores = [float(dados.loc[cenario.scenario_id, coluna]) for coluna, _, _ in marcadores]
        ax.plot(
            [min(valores), max(valores)],
            [posicao, posicao],
            color=COR_GRADE,
            linewidth=2,
            zorder=1,
        )
        for indice, (valor, (_, _rotulo, marcador)) in enumerate(
            zip(valores, marcadores, strict=True)
        ):
            ax.scatter(
                valor,
                posicao,
                marker=marcador,
                s=70 if marcador != "|" else 130,
                linewidth=1.8,
                color=CORES[cenario.scenario_id],
                zorder=3 + indice,
            )

    ax.set_xscale("log")
    ax.set_yticks(y, [cenario.plot_label for cenario in ordem])
    ax.invert_yaxis()
    ax.set_xlabel("Tempo sob a hipótese exponencial (anos, escala logarítmica)")
    ax.set_title("Marcos de probabilidade e tempo recíproco da taxa")
    legenda = [
        Line2D(
            [],
            [],
            color=COR_TEXTO_SEC,
            marker=marcador,
            linestyle="none",
            markersize=10 if marcador != "|" else 13,
            markeredgewidth=1.8,
            label=rotulo,
        )
        for _, rotulo, marcador in marcadores
    ]
    ax.legend(handles=legenda, loc="upper right", ncols=4)
    ax.tick_params(axis="y", colors=COR_TEXTO_SEC)
    fig.suptitle("Horizontes B1, B10, mediana e 1/λ por cenário bibliográfico")
    return _salvar_png_pdf(
        fig,
        caminho_base,
        (
            "B1 e B10 indicam 1% e 10% de falhas no modelo. 1/λ é MTTF apenas "
            "no caso não reparável; fontes que usam MTBF mantêm essa distinção."
        ),
    )
