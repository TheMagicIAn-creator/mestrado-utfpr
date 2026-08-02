"""
graficos_autoencoder.py - Al IAdo PV

As três figuras que documentam a CALIBRAÇÃO do Autoencoder: convergência do
treino, distribuição do erro de reconstrução e erro ao longo do tempo.

Vive separado de `autoencoder.py` por um motivo concreto: aquele módulo importa
`torch` no topo, então regenerar uma figura a partir de artefatos já salvos
exigia a stack de ML inteira — inclusive no Streamlit Cloud em modo consulta,
onde `torch` não existe. Estas funções precisam apenas de numpy e matplotlib.

Todas plotam **MSE**, não o escore localizado. A comparação MSE × localizado
vive em `src/ml/diagnostico_escore.py`.

⚠️ EDITOU UM PLOT AQUI? REGENERE AS FIGURAS.
============================================
O manifesto de proveniência da etapa `autoencoder` hasheia **um** arquivo
(`src/ml/pipeline.py::_code_path`), e esse arquivo é `autoencoder.py`. Como os
plots saíram de lá, editar uma função deste módulo **não** marca a etapa como
`stale` — os PNGs em `resultados/autoencoder/` ficariam de código antigo sem
aviso.

Optou-se por NÃO estender o hash a múltiplos arquivos: isso marcaria a etapa
como stale hoje e pediria um retreino, arriscando os números por um problema
cosmético (os PNGs são renderização; nenhum resultado depende deles).

O conserto é barato e não precisa de dataset nem de torch:

    python -c "from src.ml.graficos_autoencoder import regenerar_graficos_autoencoder as r; r()"

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   # sem display — salva direto em arquivo
import matplotlib.pyplot as plt  # noqa: E402

from src.core.logs import get_logger  # noqa: E402
from src.ml.estilo_graficos import (  # noqa: E402
    COR_ALERTA,
    COR_SUCESSO,
    PALETA,
    TAM,
    aplicar_estilo,
    salvar_figura,
)

aplicar_estilo()

RAIZ = Path(__file__).parent.parent.parent
PASTA_SAIDA = RAIZ / "resultados" / "autoencoder"

_logger = get_logger("autoencoder")


def _log(*args, sep=" ", end="\n", flush=None):
    """Mesmo contrato do _log de autoencoder.py: progresso vai para o log."""
    texto = sep.join(str(a) for a in args)
    if not texto.strip():
        return
    if texto.startswith("\r"):
        _logger.debug(texto.strip())
        return
    _logger.info(texto.rstrip("\n"))


def plotar_curvas(hist_treino: list, hist_val: list,
                  epoca_melhor: int, pasta: Path):
    """Curvas de loss por época."""
    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    epocas = range(1, len(hist_treino) + 1)
    ax.plot(epocas, hist_treino, label="Treino", color=PALETA[0], alpha=0.5)
    ax.plot(epocas, hist_val, label="Calibração", color=PALETA[1], alpha=0.55)
    if len(hist_treino) >= 7:
        treino_suave = pd.Series(hist_treino).rolling(7, center=True, min_periods=1).median()
        val_suave = pd.Series(hist_val).rolling(7, center=True, min_periods=1).median()
        ax.plot(epocas, treino_suave, color=PALETA[0], label="Treino (mediana móvel)")
        ax.plot(epocas, val_suave, color=PALETA[1], label="Calibração (mediana móvel)")
    ax.axvline(epoca_melhor, color=COR_SUCESSO, linestyle="--",
               alpha=0.85, label=f"Melhor época ({epoca_melhor})")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder — convergência do treinamento\n"
                 "A calibração orienta o early stopping; o teste permanece isolado")
    ax.legend(ncol=2)
    caminho = pasta / "curva_treino.png"
    salvar_figura(
        fig,
        caminho,
        "Loss de treino pode superar a calibração porque o dropout atua somente durante o treino.",
    )
    _log(f"   📊 {caminho.name}")


def plotar_distribuicao(erros_treino: np.ndarray,
                        erros_calibracao: np.ndarray,
                        erros_teste: np.ndarray,
                        info_limiar: dict, pasta: Path):
    """
    CALIBRAÇÃO DO DETECTOR (não é análise de falha): histograma do erro de
    reconstrução (MSE) do Autoencoder em dados SAUDÁVEIS (treino + validação),
    usado para fixar o limiar operacional p99. Uma anomalia real cairia à
    DIREITA do limiar; a fração da validação saudável acima do limiar é a taxa
    de falsos positivos. Não representa nenhum componente/modo da FMECA.
    """
    fig, (ax_hist, ax_ecdf) = plt.subplots(
        1, 2, figsize=TAM["painel_2"], layout="constrained"
    )
    conjuntos = [
        ("Treino", np.asarray(erros_treino), PALETA[0]),
        ("Calibração", np.asarray(erros_calibracao), PALETA[1]),
        ("Teste isolado", np.asarray(erros_teste), PALETA[2]),
    ]
    positivos = np.concatenate([v[v > 0] for _, v, _ in conjuntos])
    minimo = max(float(positivos.min()), 1e-8)
    maximo = float(max(v.max() for _, v, _ in conjuntos))
    # 14 bins (não 28): com poucas amostras, bins log demais criavam degraus
    # finos e vazados ("quadradão"). stepfilled com alpha suaviza a leitura.
    bins = np.geomspace(minimo, max(maximo, minimo * 10), 14)
    for nome, valores, cor in conjuntos:
        ax_hist.hist(
            np.clip(valores, minimo, None), bins=bins, density=True,
            histtype="stepfilled", alpha=0.35, linewidth=1.6,
            edgecolor=cor, color=cor, label=f"{nome} (n={len(valores)})",
        )
        ordenados = np.sort(valores)
        ecdf = np.arange(1, len(ordenados) + 1) / len(ordenados)
        ax_ecdf.step(ordenados, ecdf, where="post", color=cor, label=nome)

    limiar = info_limiar["limiar"]
    for ax in (ax_hist, ax_ecdf):
        ax.axvline(limiar, color=COR_ALERTA, linewidth=2, linestyle="--",
                   label=f"p99 calibração = {limiar:.4f}")

    # μ+kσ entra apenas como REFERÊNCIA comparativa (não é o limiar em uso).
    mu3s = info_limiar.get("limiar_mu3sigma", info_limiar.get("limiar_mu3s"))
    if mu3s is not None:
        ax_hist.axvline(mu3s, color="#898781", linewidth=1.5, linestyle=":",
                        label=f"μ+{info_limiar['k']:.0f}σ = {mu3s:.4f}")

    fp = info_limiar.get("fp_test_pct")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Erro de reconstrução (MSE, escala log)")
    ax_hist.set_ylabel("Densidade")
    ax_hist.set_title("Distribuição do erro saudável")
    ax_hist.legend(fontsize=8)
    ax_ecdf.set_xscale("log")
    ax_ecdf.set_ylim(0.80, 1.005)
    ax_ecdf.set_xlabel("Erro de reconstrução (MSE, escala log)")
    ax_ecdf.set_ylabel("Probabilidade acumulada")
    ax_ecdf.set_title("Cauda superior — ECDF")
    ax_ecdf.legend(fontsize=8)
    fig.suptitle(
        "Calibração do detector em dados saudáveis"
        + (f" — falsos positivos no teste: {fp:.2f}%" if isinstance(fp, (int, float)) else "")
    )
    caminho = pasta / "distribuicao_erro.png"
    salvar_figura(
        fig,
        caminho,
        "O limiar é estimado somente na calibração; o bloco de teste não participa do ajuste.",
    )
    _log(f"   📊 {caminho.name}")


def plotar_erro_temporal(erros: np.ndarray,
                         tempos: np.ndarray,
                         info_limiar: dict, pasta: Path,
                         indices_teste: np.ndarray | None = None):
    """Erro de reconstrução ao longo do tempo."""
    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    ax.plot(tempos, erros, color=PALETA[0], alpha=0.8, linewidth=0.8)
    ax.axhline(info_limiar["limiar"], color=COR_ALERTA, linestyle="--",
               linewidth=1.5, label=f"Limiar = {info_limiar['limiar']:.4f}")
    alarmes = erros > info_limiar["limiar"]
    ax.scatter(tempos[alarmes], erros[alarmes], color=COR_ALERTA, s=22,
               zorder=3, label="Alarme em dado saudável")
    if indices_teste is not None and len(indices_teste):
        inicio_teste = float(tempos[int(indices_teste[0])])
        ax.axvspan(inicio_teste, float(tempos[-1]), color=PALETA[2], alpha=0.08,
                   label="Bloco de teste isolado")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Erro de Reconstrução (MSE)")
    ax.set_yscale("log")
    ax.set_title("Erro temporal — Paderborn saudável\n"
                 "Pontos vermelhos são falsos alarmes, não falhas confirmadas")
    ax.legend()
    caminho = pasta / "erro_temporal.png"
    salvar_figura(fig, caminho)
    _log(f"   📊 {caminho.name}")


def _info_em_escala_mse(info: dict, erros_teste=None) -> dict:
    """Vista de `limiar.json` na escala do MSE, para os gráficos de MSE.

    O campo `limiar` salvo em limiar.json é o limiar OPERACIONAL do escore
    localizado (~7,8; ver a sobrescrita em `executar_autoencoder`). Já os três
    gráficos deste módulo plotam **MSE**, cujo p99 é ~2,5.

    Passar o dicionário cru para eles desenha a linha de limiar muito acima do
    eixo e reporta "zero alarmes" no erro temporal — figura ERRADA, não figura
    desatualizada. O caminho do pipeline escapava disso porque monta seu
    próprio `info_mse`; a regeneração a partir do disco não escapava.

    `fp_test_pct` é recalculado contra o limiar de MSE pelo mesmo motivo: o
    valor salvo se refere ao limiar operacional.
    """
    escala = dict(info)
    limiar_mse = info.get("limiar_mse")
    if limiar_mse is None:
        # Artefato anterior ao escore localizado: `limiar` já era o de MSE.
        return escala
    escala["limiar"] = float(limiar_mse)
    if erros_teste is not None and len(erros_teste):
        escala["fp_test_pct"] = float(
            (np.asarray(erros_teste) > float(limiar_mse)).mean() * 100
        )
    return escala


def regenerar_graficos_autoencoder(pasta: Path = PASTA_SAIDA) -> bool:
    """Refaz somente as figuras a partir dos vetores diagnósticos persistidos."""
    arq_diag = pasta / "diagnostico_autoencoder.npz"
    arq_limiar = pasta / "limiar.json"
    if not arq_diag.exists() or not arq_limiar.exists():
        return False
    with np.load(arq_diag) as diag:
        info = json.loads(arq_limiar.read_text(encoding="utf-8"))
        info_mse = _info_em_escala_mse(info, diag["erros_teste"])
        plotar_curvas(
            diag["historico_treino"].tolist(),
            diag["historico_calibracao"].tolist(),
            int(diag["epoca_melhor"][0]),
            pasta,
        )
        plotar_distribuicao(
            diag["erros_treino"], diag["erros_calibracao"],
            diag["erros_teste"], info_mse, pasta,
        )
        if len(diag["tempos"]):
            plotar_erro_temporal(
                diag["erros_todos"], diag["tempos"], info_mse, pasta,
                indices_teste=diag["indices_teste"],
            )
    return True
