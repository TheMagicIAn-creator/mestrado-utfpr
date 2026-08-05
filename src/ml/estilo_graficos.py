"""
estilo_graficos.py - Al IAdo PV
Estilo único para TODOS os gráficos matplotlib do projeto.

Antes, cada módulo escolhia figsize (de 6.2x5.2 a 15x8), dpi (120/140/150)
e cores por conta própria — os gráficos chegavam ao chat com proporção,
nitidez e paleta diferentes. Aqui fica a fonte única de verdade:

- DPI fixo (150) e savefig com bbox "tight" via rcParams — os módulos
  chamam fig.savefig(caminho) SEM passar dpi/bbox;
- tamanhos nomeados (TAM) por tipo de gráfico, em polegadas;
- helpers para tamanhos dinâmicos (barras/matrizes que crescem com N);
- PALETA categórica validada (CVD ΔE adjacente >= 21; ver validação no
  commit) + papéis de cor fixos: o MÉTODO PROPOSTO é sempre destacado em
  azul, baselines/concorrentes em cinza neutro. Cores de baixo contraste
  no fundo branco (aqua/amarelo) exigem rótulo direto nas barras — use
  rotular_barras().

Todo módulo de plot deve chamar aplicar_estilo() antes de criar figuras
(uma vez basta; é idempotente).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

DPI = 150

# ── Paleta categórica (ordem FIXA — nunca ciclar/reordenar) ─────────────
# Validada em fundo branco: pior ΔE adjacente 24.2 (protan), banda de
# luminosidade e croma OK. Aqua (#1baf7a) e amarelo (#eda100) ficam abaixo
# de 3:1 de contraste no branco → sempre acompanhar de rótulo direto.
PALETA = [
    "#2a78d6",  # 1 azul    — reservado ao METODO PROPOSTO quando presente
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 amarelo
    "#008300",  # 4 verde
    "#4a3aa7",  # 5 violeta
    "#e34948",  # 6 vermelho
]

# Papéis fixos (cor segue a entidade, nunca o rank)
COR_METODO = PALETA[0]        # metodo proposto (Autoencoder do pipeline)
COR_NEUTRA = "#898781"        # baselines/concorrentes em comparações
COR_REFERENCIA = "#c3c2b7"    # linhas de referência (ex.: acaso 0,5)
COR_NAO_DETECTADO = "#d5d4cd" # marcas "apagadas" (ex.: severidade não detectada)

# Tinta e cromo (recessivos — o dado é o protagonista)
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SEC = "#52514e"
COR_GRADE = "#e1e0d9"
COR_EIXO = "#c3c2b7"
COR_ALERTA = "#c43d3d"
COR_SUCESSO = "#147a3d"
COR_INFO = "#2a78d6"

# Cores canônicas por família de falha FMECA (ordem fixa da paleta;
# consumidas por injecao_falhas.FALHAS e por qualquer gráfico por família)
CORES_FALHAS = {
    "contator_ac": PALETA[0],
    "igbt": PALETA[1],
    "fusivel_ac": PALETA[2],
}

# Tamanhos canônicos (polegadas).
# Qualquer gráfico deve usar um destes ou um
# helper dinâmico abaixo — nunca um figsize avulso.
TAM = {
    "unico":    (12, 5),   # painel único: séries, curvas, distribuições
    "painel_2": (14, 5.5), # 1 linha x 2 colunas
    "quadrado": (7, 6),    # matriz de confusão / heatmap pequeno
    "painel_3": (15, 5),   # 1 linha x 3 colunas
    "painel_4": (14, 9),   # 2 linhas x 2 colunas
    "painel_6": (15, 8),   # 2 linhas x 3 colunas
    "painel_9": (15, 13),  # 3 linhas x 3 colunas
}


def aplicar_estilo() -> None:
    """rcParams uniformes: paleta, fonte, grade, bordas e salvamento."""
    plt.rcParams.update({
        "figure.figsize": TAM["unico"],
        "figure.dpi": 100,             # exibição interativa; savefig usa DPI
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.prop_cycle": cycler(color=PALETA),
        "font.size": 10,
        "text.color": COR_TEXTO,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": COR_TEXTO,
        "axes.labelsize": 10,
        "axes.labelcolor": COR_TEXTO_SEC,
        "axes.edgecolor": COR_EIXO,
        "xtick.color": COR_TEXTO_SEC,
        "ytick.color": COR_TEXTO_SEC,
        "axes.grid": True,
        "grid.color": COR_GRADE,
        "grid.alpha": 1.0,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,        # grade atrás das marcas, sempre
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": COR_GRADE,
        "lines.linewidth": 2.0,
        "axes.titlepad": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })


def adicionar_nota(fig, texto: str) -> None:
    """Inclui uma nota metodológica visível dentro da própria figura."""
    fig.text(
        0.5,
        -0.055,
        texto,
        ha="center",
        va="bottom",
        fontsize=8,
        color=COR_TEXTO_SEC,
    )


def salvar_figura(fig, caminho, nota: str | None = None) -> None:
    """Salva com acabamento consistente e libera recursos do Matplotlib."""
    if nota:
        adicionar_nota(fig, nota)
    fig.savefig(caminho, facecolor="white")
    plt.close(fig)


def rotular_barras(ax, barras, fmt: str = "{:.3f}",
                   horizontal: bool = False, dx: float = 0.01,
                   fontsize: int = 8) -> None:
    """
    Rótulo direto no fim de cada barra (regra de "relevo": cores de baixo
    contraste só são válidas com o valor visível). Texto em tinta, nunca
    na cor da série.
    """
    for barra in barras:
        if horizontal:
            v = barra.get_width()
            ax.text(v + dx, barra.get_y() + barra.get_height() / 2,
                    fmt.format(v), va="center", ha="left",
                    fontsize=fontsize, color=COR_TEXTO_SEC)
        else:
            v = barra.get_height()
            ax.text(barra.get_x() + barra.get_width() / 2, v + dx,
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=fontsize, color=COR_TEXTO_SEC)


def tam_barras_h(n_itens: int) -> tuple[float, float]:
    """Barras horizontais: largura fixa, altura cresce com o nº de itens."""
    return (12, max(4.5, 0.4 * n_itens))


def tam_barras_v(n_grupos: int) -> tuple[float, float]:
    """Barras verticais agrupadas: altura fixa, largura cresce com grupos."""
    return (max(12.0, 1.1 * n_grupos), 5)


def tam_matriz(n_classes: int) -> tuple[float, float]:
    """Matriz de confusão/heatmap: quadrado que cresce com o nº de classes."""
    lado_l = max(7.0, 0.9 * n_classes)
    lado_a = max(6.0, 0.8 * n_classes)
    return (lado_l, lado_a)
