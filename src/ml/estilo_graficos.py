"""
estilo_graficos.py - Al IAdo PV
Estilo único para TODOS os gráficos matplotlib do projeto.

Antes, cada módulo escolhia figsize (de 6.2x5.2 a 15x8) e dpi (120/140/150)
por conta própria — os gráficos chegavam ao chat com proporção e nitidez
diferentes. Aqui fica a fonte única de verdade:

- DPI fixo (150) e savefig com bbox "tight" via rcParams — os módulos
  chamam fig.savefig(caminho) SEM passar dpi/bbox;
- tamanhos nomeados (TAM) por tipo de gráfico, em polegadas;
- helpers para tamanhos dinâmicos (barras/matrizes que crescem com N),
  com mínimos e passos padronizados.

Todo módulo de plot deve chamar aplicar_estilo() antes de criar figuras
(uma vez basta; é idempotente).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

DPI = 150

# Tamanhos canônicos (polegadas). Todo gráfico deve usar um destes ou um
# helper dinâmico abaixo — nunca um figsize avulso.
TAM = {
    "unico":    (12, 5),   # painel único: séries, curvas, distribuições
    "quadrado": (7, 6),    # matriz de confusão / heatmap pequeno
    "painel_3": (15, 5),   # 1 linha x 3 colunas
    "painel_6": (15, 8),   # 2 linhas x 3 colunas
}


def aplicar_estilo() -> None:
    """rcParams uniformes: fonte, grade, bordas e política de salvamento."""
    plt.rcParams.update({
        "figure.figsize": TAM["unico"],
        "figure.dpi": 100,             # exibição interativa; savefig usa DPI
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
    })


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
