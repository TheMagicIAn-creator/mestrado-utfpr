"""
eda.py — Al IAdo PV / Fase 5
Análise Exploratória dos dados do inversor.

Gera:
  - Estatísticas descritivas
  - Gráficos de distribuição
  - Correlações
  - Visualização das séries temporais
  - Análise de desequilíbrio de fase

Uso:
  python src/ml/eda.py

Autor: Rodolfo Torres (UTFPR)
"""

from src.core.logs import get_logger as _get_logger

_logger = _get_logger("eda")


def _log(*args, sep=" ", end="\n", flush=None):
    """Progresso/sumário de ML vai para o ARQUIVO de log — o terminal
    fica silencioso quando rodando pelo app. Scripts manuais reativam o
    eco chamando habilitar_console() no bloco __main__. Linhas de
    progresso com \\r são rebaixadas a DEBUG."""
    texto = sep.join(str(a) for a in args)
    if not texto.strip():
        return
    if texto.startswith("\r"):
        _logger.debug(texto.strip())
        return
    _logger.info(texto.rstrip("\n"))



import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMINHO_CSV = Path(__file__).parent.parent.parent / "dados" / "brutos" / "Inverter_Data_Set.csv"
PASTA_GRAFICOS= Path(__file__).parent.parent.parent / "resultados" / "eda"
TAXA_AMOSTRAGEM = 10_000  # Hz


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados() -> pd.DataFrame:
    _log("📂 Carregando dataset...")
    df = pd.read_csv(CAMINHO_CSV)
    _log(f"   ✅ {len(df):,} amostras | {len(df.columns)} colunas")
    return df


# ============================================================
# ESTATÍSTICAS BÁSICAS
# ============================================================

def analise_basica(df: pd.DataFrame):
    _log("\n📊 ESTATÍSTICAS DESCRITIVAS")
    _log("=" * 60)

    stats = df.describe().T
    stats["cv%"] = (stats["std"] / stats["mean"].abs() * 100).round(2)

    _log(stats[["mean", "std", "min", "max", "cv%"]].to_string())

    # Verifica valores nulos
    nulos = df.isnull().sum()
    if nulos.any():
        _log(f"\n⚠️  Valores nulos encontrados:")
        _log(nulos[nulos > 0])
    else:
        _log(f"\n✅ Nenhum valor nulo encontrado!")

    # Velocidade do motor
    _log(f"\n🔄 Velocidade do motor (n_k):")
    _log(f"   Mín : {df['n_k'].min():.1f} RPM")
    _log(f"   Máx : {df['n_k'].max():.1f} RPM")
    _log(f"   Média: {df['n_k'].mean():.1f} RPM")


# ============================================================
# ANÁLISE DE DESEQUILÍBRIO DE FASE
# ============================================================

def analise_fases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de desequilíbrio entre as fases A, B, C.
    Desequilíbrio é um dos principais indicadores de falha.
    """
    _log("\n⚡ ANÁLISE DE DESEQUILÍBRIO DE FASE")
    _log("=" * 60)

    # RMS das correntes por fase (janela de 100 amostras = 10ms)
    janela = 100

    df["rms_ia"] = df["i_a_k"].rolling(janela).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    df["rms_ib"] = df["i_b_k"].rolling(janela).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    df["rms_ic"] = df["i_c_k"].rolling(janela).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )

    # Desequilíbrio (diferença máxima entre fases em %)
    df["rms_media"] = (df["rms_ia"] + df["rms_ib"] + df["rms_ic"]) / 3
    df["desequilibrio"] = (
        (df[["rms_ia","rms_ib","rms_ic"]].max(axis=1) -
         df[["rms_ia","rms_ib","rms_ic"]].min(axis=1)) /
        df["rms_media"] * 100
    )

    df_valido = df.dropna(subset=["desequilibrio"])
    _log(f"   Desequilíbrio médio : {df_valido['desequilibrio'].mean():.2f}%")
    _log(f"   Desequilíbrio máximo: {df_valido['desequilibrio'].max():.2f}%")
    _log(f"   Desequilíbrio > 5%  : {(df_valido['desequilibrio'] > 5).sum():,} amostras")

    return df


# ============================================================
# GRÁFICOS
# ============================================================

def plotar_series_temporais(df: pd.DataFrame):
    """Plota as correntes trifásicas e tensão CC."""

    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    # Usa apenas as primeiras 2000 amostras para visualização
    amostra = df.head(2000).copy()
    amostra["tempo_ms"] = amostra.index * 0.1  # cada amostra = 0.1ms

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Correntes Trifásicas (i_a, i_b, i_c)",
            "Tensão CC (u_dc)",
            "Velocidade do Motor (n_k)"
        ],
        vertical_spacing=0.08
    )

    # Correntes
    for fase, cor in [("i_a_k","#E63946"), ("i_b_k","#2196F3"), ("i_c_k","#4CAF50")]:
        fig.add_trace(
            go.Scatter(x=amostra["tempo_ms"], y=amostra[fase],
                      name=fase, line=dict(color=cor, width=1)),
            row=1, col=1
        )

    # Tensão CC
    fig.add_trace(
        go.Scatter(x=amostra["tempo_ms"], y=amostra["u_dc_k"],
                  name="u_dc", line=dict(color="#FF9800", width=1)),
        row=2, col=1
    )

    # Velocidade
    fig.add_trace(
        go.Scatter(x=amostra["tempo_ms"], y=amostra["n_k"],
                  name="n_k", line=dict(color="#9C27B0", width=1)),
        row=3, col=1
    )

    fig.update_xaxes(title_text="Tempo (ms)")
    fig.update_yaxes(title_text="Corrente (A)", row=1, col=1)
    fig.update_yaxes(title_text="Tensão (V)", row=2, col=1)
    fig.update_yaxes(title_text="RPM", row=3, col=1)

    fig.update_layout(
        title="Análise Exploratória — Inversor Fotovoltaico",
        height=700,
        template="plotly_dark"
    )

    caminho = PASTA_GRAFICOS / "series_temporais.html"
    fig.write_html(str(caminho))
    _log(f"\n✅ Gráfico salvo: {caminho}")


def plotar_distribuicoes(df: pd.DataFrame):
    """Distribuição das correntes por fase."""

    fig = go.Figure()

    for fase, cor in [("i_a_k","#E63946"), ("i_b_k","#2196F3"), ("i_c_k","#4CAF50")]:
        fig.add_trace(go.Histogram(
            x=df[fase], name=fase, opacity=0.7,
            marker_color=cor, nbinsx=100
        ))

    fig.update_layout(
        title="Distribuição das Correntes Trifásicas",
        xaxis_title="Corrente (A)",
        yaxis_title="Frequência",
        barmode="overlay",
        template="plotly_dark"
    )

    caminho = PASTA_GRAFICOS / "distribuicao_correntes.html"
    fig.write_html(str(caminho))
    _log(f"✅ Gráfico salvo: {caminho}")


def plotar_correlacoes(df: pd.DataFrame):
    """Matriz de correlação das variáveis principais."""

    colunas = ["n_k", "u_dc_k", "i_a_k", "i_b_k", "i_c_k",
               "u_a_k-1", "u_b_k-1", "u_c_k-1",
               "d_a_k-2", "d_b_k-2", "d_c_k-2"]

    corr = df[colunas].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Matriz de Correlação — Variáveis Principais",
        template="plotly_dark"
    )

    fig.update_layout(height=600)

    caminho = PASTA_GRAFICOS / "correlacoes.html"
    fig.write_html(str(caminho))
    _log(f"✅ Gráfico salvo: {caminho}")


def plotar_desequilibrio(df: pd.DataFrame):
    """Plota o desequilíbrio de fase ao longo do tempo."""

    if "desequilibrio" not in df.columns:
        return

    df_valido = df.dropna(subset=["desequilibrio"]).copy()
    df_valido["indice"] = range(len(df_valido))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_valido["indice"],
        y=df_valido["desequilibrio"],
        mode="lines",
        name="Desequilíbrio (%)",
        line=dict(color="#FF5722", width=1)
    ))

    # Linha de referência em 5%
    fig.add_hline(
        y=5, line_dash="dash", line_color="yellow",
        annotation_text="Limite 5%"
    )

    fig.update_layout(
        title="Desequilíbrio de Fase ao Longo do Tempo",
        xaxis_title="Amostra",
        yaxis_title="Desequilíbrio (%)",
        template="plotly_dark"
    )

    caminho = PASTA_GRAFICOS / "desequilibrio_fase.html"
    fig.write_html(str(caminho))
    _log(f"✅ Gráfico salvo: {caminho}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_eda() -> bool:
    _log("=" * 60)
    _log("  AL IADO PV — ANÁLISE EXPLORATÓRIA (EDA)")
    _log("=" * 60)

    # 1. Carrega dados
    df = carregar_dados()

    # 2. Estatísticas básicas
    analise_basica(df)

    # 3. Análise de fases
    df = analise_fases(df)

    # 4. Gráficos
    _log("\n📈 GERANDO GRÁFICOS...")
    plotar_series_temporais(df)
    plotar_distribuicoes(df)
    plotar_correlacoes(df)
    plotar_desequilibrio(df)

    _log("\n" + "=" * 60)
    _log("  EDA CONCLUÍDA!")
    _log(f"  Gráficos salvos em: resultados/eda/")
    _log("  Abra os arquivos .html no navegador")
    _log("=" * 60)

    return True


if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    executar_eda()