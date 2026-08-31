"""
Padronização de tabelas (src/core/formatacao.py) e gráficos
(src/ml/estilo_graficos.py) — garante que a política única de formatação
não regrida e que todo módulo de plot usa os tamanhos canônicos.
"""

import re
from pathlib import Path

import pytest

from src.core.formatacao import (
    fmt_fisico,
    fmt_limiar,
    fmt_metrica,
    fmt_num,
    fmt_pct,
    fmt_pvalor,
    tabela_markdown,
)

RAIZ = Path(__file__).resolve().parent.parent
MODULOS_PLOT = [
    "graficos_comparacao", "graficos_confiabilidade",
]


# ── números ──────────────────────────────────────────────────────────────────

def test_fmt_num_politica_de_casas():
    assert fmt_metrica(0.86833) == "0.868"
    assert fmt_limiar(2.07845258) == "2.0785"
    assert fmt_fisico(49.456) == "49.5"
    assert fmt_pct(4.3478) == "4.3%"


@pytest.mark.parametrize("ruim", [None, "abc", True, [1]])
def test_fmt_num_nao_numerico_vira_traco(ruim):
    assert fmt_num(ruim) == "-"


def test_fmt_pvalor_convencao_academica():
    assert fmt_pvalor(0.00135) == "p=0.0014"
    assert fmt_pvalor(1.8e-07) == "p<0.0001"
    assert fmt_pvalor(None) == "-"


# ── tabelas ──────────────────────────────────────────────────────────────────

def test_tabela_markdown_alinhamento_padrao():
    t = tabela_markdown(["Falha", "AUC"], [["LCL", "0.935"]])
    linhas = t.strip().splitlines()
    assert linhas[0] == "| Falha | AUC |"
    assert linhas[1] == "|---|---:|"   # 1ª à esquerda, demais à direita
    assert linhas[2] == "| LCL | 0.935 |"


def test_tabela_markdown_none_e_linha_curta():
    t = tabela_markdown(["A", "B", "C"], [["x", None]])
    assert "| x | - | - |" in t


# ── gráficos ─────────────────────────────────────────────────────────────────

def test_estilo_tamanhos_canonicos():
    from src.ml.estilo_graficos import TAM, tam_barras_h, tam_barras_v, tam_matriz

    assert set(TAM) == {
        "unico", "painel_2", "quadrado", "painel_3", "painel_4",
        "painel_6", "painel_8", "painel_9",
    }
    assert tam_barras_h(3)[0] == 12          # largura fixa
    assert tam_barras_h(30)[1] > tam_barras_h(3)[1]
    assert tam_barras_v(20)[1] == 5          # altura fixa
    assert tam_matriz(4) == (7.0, 6.0)       # mínimo quadrado


def test_aplicar_estilo_define_salvamento_unico():
    import matplotlib.pyplot as plt

    from src.ml.estilo_graficos import DPI, aplicar_estilo

    aplicar_estilo()
    assert plt.rcParams["savefig.dpi"] == DPI
    assert plt.rcParams["savefig.bbox"] == "tight"


def test_modulos_de_plot_sem_figsize_ou_dpi_avulsos():
    """Nenhum módulo de plot pode voltar a fixar figsize numérico ou dpi."""
    fig_avulso = re.compile(r"figsize=\(\s*\d")
    dpi_avulso = re.compile(r"savefig\([^)\n]*dpi=")
    for nome in MODULOS_PLOT:
        codigo = (RAIZ / "src" / "ml" / f"{nome}.py").read_text(encoding="utf-8")
        assert not fig_avulso.search(codigo), f"{nome}.py: figsize numérico avulso"
        assert not dpi_avulso.search(codigo), f"{nome}.py: dpi avulso no savefig"
        assert "aplicar_estilo()" in codigo, f"{nome}.py: não aplica o estilo único"
