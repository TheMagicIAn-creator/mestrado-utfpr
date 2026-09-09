"""Figuras da campanha E2 de detectabilidade por magnitude.

O EIXO NÃO É TEMPO
==================
Toda figura daqui tem `a` no eixo — fração da assinatura nominal, adimensional,
em `[0, 1]`. As figuras da confiabilidade física têm `t` em horas ou anos. As
duas famílias podem sair no mesmo formato de gráfico e **não** podem ser lidas
na mesma escala; por isso cada figura carrega a ressalva no rodapé.

POR QUE ESTAS DUAS
==================
A POD responde a pergunta que a E2 existe para responder — a partir de que
magnitude cada modelo passa a detectar. A fidelidade responde se aquilo que foi
injetado se parece com o defeito medido: sem ela, alguém leria a POD como
capacidade de detector. Uma não se publica sem a outra.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.estilo_graficos import (
    COR_EIXO,
    COR_NAO_DETECTADO,
    COR_TEXTO_SEC,
    PALETA,
    TAM,
    adicionar_nota,
    aplicar_estilo,
)

aplicar_estilo()

# A cor segue a entidade, não o rank: mesmo modelo, mesma cor da família E3
# (`graficos_comparacao.MODEL_COLORS`). Ler a POD do Denso em azul e o recall do
# Denso em azul é o que permite cruzar as duas famílias sem legenda mental.
MODEL_COLORS = {"ae_denso": PALETA[0], "ae_lstm": PALETA[1]}
MODEL_LABELS = {"ae_denso": "Autoencoder Denso", "ae_lstm": "AE-LSTM"}

# Rótulos curtos: o id da injeção é chave de dado, não texto de figura.
INJECTION_LABELS = {
    "igbt": "IGBT",
    "sensor_realimentacao": "Sensor/realimentação",
    "controle": "Controle",
}

_RESSALVA_EIXO = (
    "`a` é fração da assinatura nominal, não tempo, ciclo ou vida consumida. "
    "Não leia estas curvas na escala das figuras de confiabilidade física."
)


def _rotulo_injecao(injection_id: str) -> str:
    return INJECTION_LABELS.get(injection_id, injection_id)


def _save_pair(fig, base_path: Path, nota: str) -> tuple[Path, Path]:
    """PNG e PDF do MESMO desenho, com metadados determinísticos.

    Igual ao par da confiabilidade: `CreationDate`/`ModDate` nulos no PDF, para
    o hash do manifesto não mudar só porque o relógio andou.
    """
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    adicionar_nota(fig, nota)
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


def plot_pod(pod: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """Curvas POD por item, um painel por item, uma linha por modelo.

    Painéis separados em vez de seis linhas num eixo: os três itens têm regimes
    de detecção muito distintos, e sobrepô-los esconderia justamente o contraste
    entre eles.
    """
    itens = [item for item in INJECTION_LABELS if item in set(pod["injection_id"])]
    itens += [item for item in sorted(set(pod["injection_id"])) if item not in itens]

    fig, axes = plt.subplots(1, len(itens), figsize=TAM["painel_3"], sharey=True)
    axes = [axes] if len(itens) == 1 else list(axes)

    for ax, injection_id in zip(axes, itens, strict=True):
        bloco = pod[pod["injection_id"] == injection_id]
        # Traço mais largo embaixo, mais fino em cima: no controle as duas curvas
        # coincidem em 50% até a~0,75 e larguras iguais apagariam a primeira.
        for ordem, model_id in enumerate(MODEL_COLORS):
            serie = bloco[bloco["model"] == model_id].sort_values("a")
            if serie.empty:
                continue
            ax.plot(
                serie["a"],
                serie["pod"],
                color=MODEL_COLORS[model_id],
                linewidth=2.8 if ordem == 0 else 1.6,
                label=MODEL_LABELS[model_id],
            )
        # Referência de meia detecção: onde a curva a cruza é o a50 visual.
        ax.axhline(0.5, color=COR_EIXO, linestyle=":", linewidth=1.0)
        ax.set_title(_rotulo_injecao(injection_id))
        ax.set_xlabel("a — fração da assinatura nominal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    axes[0].set_ylabel("POD — fração de trajetórias detectadas")
    axes[0].legend(loc="lower right", frameon=False)

    return _save_pair(
        fig,
        output,
        f"Linha pontilhada: POD 50%. {_RESSALVA_EIXO}",
    )


def plot_fidelidade(fidelidade: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """POD em `a=1` contra a faixa de recall da E3 nos ensaios de referência.

    Um eixo só, porque as duas grandezas são frações em `[0, 1]` — mas elas NÃO
    são a mesma grandeza: a POD vem de janelas F0 normalizadas pela baseline
    saudável e o recall vem dos ensaios reais normalizados por comissionamento.
    A distância entre o ponto e a faixa mede fidelidade da injeção, não erro de
    nenhuma das duas etapas.
    """
    linhas = list(fidelidade.itertuples(index=False))
    posicoes = list(range(len(linhas)))[::-1]

    altura = max(3.2, 0.62 * len(linhas) + 1.6)
    fig, ax = plt.subplots(figsize=(TAM["unico"][0], altura))

    for posicao, linha in zip(posicoes, linhas, strict=True):
        cor = MODEL_COLORS.get(linha.model, COR_NAO_DETECTADO)
        menor, maior = linha.e3_recall_min, linha.e3_recall_max
        if pd.notna(menor) and pd.notna(maior):
            ax.plot(
                [menor, maior],
                [posicao, posicao],
                color=COR_NAO_DETECTADO,
                linewidth=7.0,
                solid_capstyle="round",
                zorder=1,
            )
            ax.plot(
                [menor, maior],
                [posicao, posicao],
                color=COR_TEXTO_SEC,
                linewidth=1.4,
                solid_capstyle="round",
                zorder=2,
            )
        # Conector até a borda mais próxima da faixa: a divergência é o que a
        # figura existe para mostrar, e sem ele o leitor precisa medir no eixo.
        if pd.notna(menor) and pd.notna(maior):
            borda = menor if linha.pod_a1 < menor else maior
            if not menor <= linha.pod_a1 <= maior:
                ax.plot(
                    [borda, linha.pod_a1],
                    [posicao, posicao],
                    color=COR_TEXTO_SEC,
                    linestyle=":",
                    linewidth=1.0,
                    zorder=0,
                )
        ax.plot(
            linha.pod_a1,
            posicao,
            marker="o",
            markersize=9,
            color=cor,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
        # Rótulo direto do valor: a aqua fica abaixo de 3:1 no branco, então
        # nenhuma leitura pode depender só da cor.
        ax.annotate(
            f"{linha.pod_a1:.2f}",
            (linha.pod_a1, posicao),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=8,
            color=COR_TEXTO_SEC,
        )

    ax.set_yticks(posicoes)
    ax.set_yticklabels(
        [
            f"{_rotulo_injecao(linha.injection_id)} · {MODEL_LABELS.get(linha.model, linha.model)}"
            for linha in linhas
        ]
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("fração detectada — POD na E2, recall na E3")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.8, len(linhas) - 0.2)

    legenda = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=9,
            color=MODEL_COLORS["ae_denso"], label="POD(a=1) — Autoencoder Denso",
        ),
        Line2D(
            [], [], marker="o", linestyle="none", markersize=9,
            color=MODEL_COLORS["ae_lstm"], label="POD(a=1) — AE-LSTM",
        ),
        Line2D(
            [], [], color=COR_TEXTO_SEC, linewidth=1.4,
            label="faixa de recall da E3 nos ensaios reais",
        ),
    ]
    ax.legend(handles=legenda, loc="lower right", frameon=False)

    return _save_pair(
        fig,
        output,
        "Ponto e faixa compartilham o eixo por serem frações, e não por serem a "
        "mesma grandeza: escalas de normalização distintas. Distância grande "
        "sinaliza infidelidade da injeção, não erro de nenhuma das duas etapas.",
    )


def generate_all(
    output_dir: Path,
    *,
    pod: pd.DataFrame,
    fidelidade: pd.DataFrame,
) -> list[Path]:
    """Gera as figuras da E2 e devolve os caminhos, para o manifesto hasheá-los."""
    output_dir = Path(output_dir)
    caminhos: list[Path] = []
    caminhos.extend(plot_pod(pod, output_dir / "e2_pod_curvas.png"))
    caminhos.extend(plot_fidelidade(fidelidade, output_dir / "e2_fidelidade.png"))
    return caminhos


__all__ = [
    "INJECTION_LABELS",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "generate_all",
    "plot_fidelidade",
    "plot_pod",
]
