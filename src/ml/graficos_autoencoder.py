"""
graficos_autoencoder.py - Al IAdo PV

As três figuras e a tabela que documentam a CALIBRAÇÃO do Autoencoder:
convergência do treino, distribuição do erro de reconstrução, erro ao longo do
tempo e resumo tabular com excedências/IC95%.

Vive separado de `autoencoder.py` por um motivo concreto: aquele módulo importa
`torch` no topo, então regenerar uma figura a partir de artefatos já salvos
exigia a stack de ML inteira — inclusive no Streamlit Cloud em modo consulta,
onde `torch` não existe. Estas funções precisam apenas de numpy e matplotlib.

Todas plotam **MSE**, não o escore localizado. A comparação MSE × localizado
vive em `src/ml/diagnostico_escore.py`.

EDITOU UM PLOT AQUI? REGENERE AS FIGURAS.
=========================================
Desde os manifestos v2, este módulo entra em `code_dependencies` da etapa
`autoencoder`; portanto mudanças aqui marcam a etapa como `stale`. Se a mudança
for apenas de apresentação, a regeneração barata a partir dos vetores salvos
continua disponível e não precisa de dataset nem de torch:

    python -c "from src.ml.graficos_autoencoder import regenerar_graficos_autoencoder as r; r()"

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # sem display — salva direto em arquivo
import matplotlib.pyplot as plt  # noqa: E402

from src.core.logs import adaptar_logger_como_print, get_logger  # noqa: E402
from src.ml.estatistica import intervalo_wilson  # noqa: E402
from src.ml.estilo_graficos import (  # noqa: E402
    COR_ALERTA,
    COR_NAO_DETECTADO,
    COR_REFERENCIA,
    COR_SUCESSO,
    COR_TEXTO_SEC,
    PALETA,
    TAM,
    aplicar_estilo,
    salvar_figura,
)

aplicar_estilo()

RAIZ = Path(__file__).parent.parent.parent
PASTA_SAIDA = RAIZ / "resultados" / "autoencoder"

_logger = get_logger("autoencoder")
_log = adaptar_logger_como_print(_logger)


def resumo_excedencia(valores: np.ndarray, limiar: float) -> dict:
    """Conta excedências finitas e adiciona IC95% de Wilson descritivo."""
    bruto = np.asarray(valores, dtype=float).ravel()
    validos = np.isfinite(bruto)
    arr = bruto[validos]
    n = int(len(arr))
    k = int((arr > float(limiar)).sum()) if n else 0
    taxa = float(100.0 * k / n) if n else float("nan")
    ci_low, ci_high = intervalo_wilson(k, n)
    return {
        "count": k,
        "n": n,
        "rate_pct": taxa,
        "ci95_low_pct": float(ci_low * 100.0),
        "ci95_high_pct": float(ci_high * 100.0),
        "n_invalid": int((~validos).sum()),
    }


def _fmt(valor: float | int | None, casas: int = 4) -> str:
    if valor is None:
        return "-"
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{v:.{casas}f}"


def _fmt_excedencia(resumo: dict) -> str:
    if not resumo or not resumo.get("n"):
        return "-"
    return (
        f"{resumo['count']}/{resumo['n']} = {resumo['rate_pct']:.2f}% "
        f"[{resumo['ci95_low_pct']:.2f}; {resumo['ci95_high_pct']:.2f}]"
    )


def _normalizar_intervalos(split: dict | None, chave: str) -> list[tuple[int, int]]:
    """Aceita tanto o split histórico único quanto a lista intercalada atual."""
    if not split:
        return []
    bruto = (split.get("limites") or {}).get(chave) or []
    if (
        len(bruto) == 2
        and all(isinstance(v, (int, np.integer)) for v in bruto)
    ):
        bruto = [bruto]

    intervalos = []
    for item in bruto:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        ini, fim = int(item[0]), int(item[1])
        if 0 <= ini < fim:
            intervalos.append((ini, fim))
    return intervalos


def _posicoes_sem_compartilhamento(
    split: dict | None,
    chave: str,
    n_valores: int,
    *,
    distancia_minima: int = 2,
) -> np.ndarray:
    """Seleciona janelas que não compartilham amostras no protocolo de 50%.

    Distância dois corresponde a reter uma janela a cada duas no conjunto de
    features vigente. Isso remove o compartilhamento direto de amostras, mas
    não autoriza chamar as observações de estatisticamente independentes.
    """
    if distancia_minima < 1:
        raise ValueError("distancia_minima deve ser positiva")
    globais = [
        indice
        for ini, fim in _normalizar_intervalos(split, chave)
        for indice in range(ini, fim)
    ]
    if len(globais) != int(n_valores):
        return np.asarray([], dtype=int)

    selecionadas = []
    ultimo_global = None
    for posicao, indice_global in enumerate(globais):
        if ultimo_global is None or indice_global - ultimo_global >= distancia_minima:
            selecionadas.append(posicao)
            ultimo_global = indice_global
    return np.asarray(selecionadas, dtype=int)


def _resumo_sem_compartilhamento(
    valores: np.ndarray,
    limiar: float,
    split: dict | None,
    chave: str,
) -> dict:
    posicoes = _posicoes_sem_compartilhamento(split, chave, len(valores))
    if not len(posicoes):
        return {}
    return resumo_excedencia(np.asarray(valores)[posicoes], limiar)


def _resumo_vetor(valores: np.ndarray) -> dict:
    arr = np.asarray(valores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {
            "n_janelas": 0,
            "mediana": float("nan"),
            "iqr": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "n_janelas": int(len(arr)),
        "mediana": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _linha_calibracao(
    nome: str,
    chave_split: str,
    mse: np.ndarray,
    score: np.ndarray | None,
    limiar_mse: float,
    limiar_score: float,
    split: dict | None,
) -> dict:
    resumo_mse = _resumo_vetor(mse)
    exc_mse = resumo_excedencia(mse, limiar_mse)
    resumo_score = _resumo_vetor(score) if score is not None else {}
    exc_score = (
        resumo_excedencia(score, limiar_score) if score is not None else {}
    )
    exc_mse_sem_comp = _resumo_sem_compartilhamento(
        mse, limiar_mse, split, chave_split
    )
    exc_score_sem_comp = (
        _resumo_sem_compartilhamento(score, limiar_score, split, chave_split)
        if score is not None else {}
    )
    return {
        "bloco": nome,
        **{f"mse_{k}": v for k, v in resumo_mse.items()},
        "mse_ref_p99_count": exc_mse.get("count"),
        "mse_ref_p99_rate_pct": exc_mse.get("rate_pct"),
        "mse_ref_p99_ci95_low_pct": exc_mse.get("ci95_low_pct"),
        "mse_ref_p99_ci95_high_pct": exc_mse.get("ci95_high_pct"),
        "mse_sem_compartilhamento_count": exc_mse_sem_comp.get("count"),
        "mse_sem_compartilhamento_n": exc_mse_sem_comp.get("n"),
        "mse_sem_compartilhamento_rate_pct": exc_mse_sem_comp.get("rate_pct"),
        "mse_sem_compartilhamento_ci95_low_pct": exc_mse_sem_comp.get("ci95_low_pct"),
        "mse_sem_compartilhamento_ci95_high_pct": exc_mse_sem_comp.get("ci95_high_pct"),
        **{f"score_operacional_{k}": v for k, v in resumo_score.items()},
        "score_operacional_count": exc_score.get("count"),
        "score_operacional_rate_pct": exc_score.get("rate_pct"),
        "score_operacional_ci95_low_pct": exc_score.get("ci95_low_pct"),
        "score_operacional_ci95_high_pct": exc_score.get("ci95_high_pct"),
        "score_sem_compartilhamento_count": exc_score_sem_comp.get("count"),
        "score_sem_compartilhamento_n": exc_score_sem_comp.get("n"),
        "score_sem_compartilhamento_rate_pct": exc_score_sem_comp.get("rate_pct"),
        "score_sem_compartilhamento_ci95_low_pct": exc_score_sem_comp.get("ci95_low_pct"),
        "score_sem_compartilhamento_ci95_high_pct": exc_score_sem_comp.get("ci95_high_pct"),
    }


def salvar_resumo_calibracao(
    erros_treino: np.ndarray,
    erros_calibracao: np.ndarray,
    erros_teste: np.ndarray,
    info_limiar: dict,
    pasta: Path,
    *,
    scores_treino: np.ndarray | None = None,
    scores_calibracao: np.ndarray | None = None,
    scores_teste: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Grava tabela CSV/Markdown de calibração com contagens e IC95%."""
    limiar_mse = float(info_limiar.get(
        "mse_p99", info_limiar.get("limiar_p99", info_limiar["limiar"])
    ))
    limiar_score = float(info_limiar.get(
        "score_threshold", info_limiar.get("limiar_operacional", info_limiar["limiar"])
    ))
    metodo = info_limiar.get("score_method") or info_limiar.get("metodo_escore") or "mse"
    percentil = info_limiar.get(
        "threshold_effective_percentile", info_limiar.get("percentil_limiar")
    )
    split = info_limiar.get("split_temporal")
    linhas = [
        _linha_calibracao(
            "treino", "treino", erros_treino, scores_treino,
            limiar_mse, limiar_score, split,
        ),
        _linha_calibracao(
            "calibracao", "val", erros_calibracao, scores_calibracao,
            limiar_mse, limiar_score, split,
        ),
        _linha_calibracao(
            "teste_isolado", "teste", erros_teste, scores_teste,
            limiar_mse, limiar_score, split,
        ),
    ]
    df = pd.DataFrame(linhas)
    csv_path = pasta / "calibracao_autoencoder.csv"
    md_path = pasta / "calibracao_autoencoder.md"
    df.to_csv(csv_path, index=False)

    ponto = str(metodo)
    if percentil is not None:
        ponto += f" / percentil efetivo {_fmt(percentil, 1)}"
    n_calibracao = int(np.isfinite(np.asarray(erros_calibracao)).sum())
    resolucao_calibracao = 100.0 / n_calibracao if n_calibracao else float("nan")
    texto = [
        "# Calibração acadêmica do Autoencoder",
        "",
        (
            "> Split temporal com purga. O bloco de teste isolado não participa "
            "do scaler, do early stopping nem da escolha de limiar."
        ),
        "",
        f"- Escore operacional: `{ponto}`.",
        f"- Limiar operacional (`score_threshold`): `{_fmt(limiar_score, 6)}`.",
        f"- Referência MSE p99 para gráficos de reconstrução: `{_fmt(limiar_mse, 6)}`.",
        f"- O p99 é um quantil nominal interpolado (`method=linear`); com "
        f"{n_calibracao} janelas, a resolução empírica da cauda é "
        f"1/{n_calibracao} = {_fmt(resolucao_calibracao, 2)}%, não 1%.",
        "- Intervalos entre colchetes são IC95% de Wilson, usados como referências "
        "binomiais por janela. "
        "A sobreposição e a dependência serial limitam a interpretação inferencial.",
        "",
        "| Bloco | n janelas | n sem compartilhamento | MSE mediana | MSE IQR | "
        "MSE p99 | > ref. MSE p99 por janela | > ref. sem compartilhamento |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in linhas:
        exc_mse = {
            "count": row.get("mse_ref_p99_count"),
            "n": row.get("mse_n_janelas"),
            "rate_pct": row.get("mse_ref_p99_rate_pct"),
            "ci95_low_pct": row.get("mse_ref_p99_ci95_low_pct"),
            "ci95_high_pct": row.get("mse_ref_p99_ci95_high_pct"),
        }
        exc_mse_sem_comp = {
            "count": row.get("mse_sem_compartilhamento_count"),
            "n": row.get("mse_sem_compartilhamento_n"),
            "rate_pct": row.get("mse_sem_compartilhamento_rate_pct"),
            "ci95_low_pct": row.get("mse_sem_compartilhamento_ci95_low_pct"),
            "ci95_high_pct": row.get("mse_sem_compartilhamento_ci95_high_pct"),
        }
        texto.append(
            f"| {row['bloco']} | {int(row['mse_n_janelas'])} | "
            f"{row.get('mse_sem_compartilhamento_n') or '-'} | "
            f"{_fmt(row['mse_mediana'])} | {_fmt(row['mse_iqr'])} | "
            f"{_fmt(row['mse_p99'])} | {_fmt_excedencia(exc_mse)} | "
            f"{_fmt_excedencia(exc_mse_sem_comp)} |"
        )
    if str(metodo) != "mse":
        texto.extend([
            "",
            "| Bloco | Escore mediana | > limiar operacional por janela |",
            "|---|---:|---:|",
        ])
        for row in linhas:
            exc_score = {
                "count": row.get("score_operacional_count"),
                "n": row.get("score_operacional_n_janelas"),
                "rate_pct": row.get("score_operacional_rate_pct"),
                "ci95_low_pct": row.get("score_operacional_ci95_low_pct"),
                "ci95_high_pct": row.get("score_operacional_ci95_high_pct"),
            }
            texto.append(
                f"| {row['bloco']} | {_fmt(row.get('score_operacional_mediana'))} | "
                f"{_fmt_excedencia(exc_score)} |"
            )
    texto.extend([
        "",
        (
            "Observação metodológica: 'sem compartilhamento' retém uma janela "
            "a cada duas dentro de cada bloco, coerente com 50% de sobreposição. "
            "Isso remove amostras brutas compartilhadas, mas não garante "
            "independência estatística ou temporal. Os gráficos estão na escala "
            "MSE; a decisão operacional usa o escore registrado em `limiar.json`."
        ),
        "",
    ])
    md_path.write_text("\n".join(texto), encoding="utf-8")
    return csv_path, md_path


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
    if hist_val:
        melhor_val = min(hist_val)
        final_val = hist_val[-1]
        ax.text(
            0.985, 0.06,
            (
                f"épocas: {len(hist_treino)}\n"
                f"melhor época: {epoca_melhor}\n"
                f"loss calib. mín.: {melhor_val:.4f}\n"
                f"loss calib. final: {final_val:.4f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=COR_TEXTO_SEC,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#e1e0d9"},
        )
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
    Calibração do detector em dados saudáveis: ECDF completa e cauda empírica.

    A figura separa o quantil nominal p99 da frequência realmente observada.
    Não representa nenhum componente ou modo de falha da FMECA.
    """
    fig, (ax_ecdf, ax_cauda) = plt.subplots(
        1, 2, figsize=TAM["painel_2"], layout="constrained"
    )
    conjuntos = [
        ("Treino", np.asarray(erros_treino), PALETA[0], "-"),
        ("Calibração", np.asarray(erros_calibracao), PALETA[1], "--"),
        ("Teste isolado", np.asarray(erros_teste), PALETA[2], "-."),
    ]
    limiar = float(info_limiar["limiar"])
    for nome, valores, cor, estilo in conjuntos:
        ordenados = np.sort(valores[np.isfinite(valores) & (valores > 0)])
        if not len(ordenados):
            continue
        ecdf = np.arange(1, len(ordenados) + 1) / len(ordenados)
        sobrevivencia = (
            len(ordenados) - np.arange(1, len(ordenados) + 1)
        ) / len(ordenados)
        rotulo = f"{nome} (n={len(ordenados)})"
        ax_ecdf.step(
            ordenados, ecdf, where="post", color=cor,
            linestyle=estilo, label=rotulo,
        )
        ax_cauda.step(
            ordenados[sobrevivencia > 0], sobrevivencia[sobrevivencia > 0],
            where="post", color=cor,
            linestyle=estilo, label=rotulo,
        )
        exc = resumo_excedencia(ordenados, limiar)
        if exc["count"]:
            ax_cauda.scatter(
                [limiar], [exc["rate_pct"] / 100.0], color=cor,
                edgecolor="white", linewidth=0.7, s=42, zorder=4,
            )

    for ax in (ax_ecdf, ax_cauda):
        ax.axvline(limiar, color=COR_ALERTA, linewidth=2, linestyle="--",
                   label=f"Referência MSE p99 = {limiar:.4f}")

    # μ+kσ entra apenas como referência comparativa, nunca como corte em uso.
    mu3s = info_limiar.get("limiar_mu3sigma", info_limiar.get("limiar_mu3s"))
    if mu3s is not None:
        ax_ecdf.axvline(
            mu3s, color="#898781", linewidth=1.4, linestyle=":",
            label=f"Referência μ+{info_limiar['k']:.0f}σ = {mu3s:.4f}",
        )

    fp_teste = resumo_excedencia(erros_teste, limiar)
    fp_calib = resumo_excedencia(erros_calibracao, limiar)
    ax_ecdf.axhline(
        0.99, color=COR_REFERENCIA, linewidth=1.1, linestyle=":",
        label="Quantil nominal 0,99",
    )
    ax_cauda.axhline(
        0.01, color=COR_REFERENCIA, linewidth=1.1, linestyle=":",
        label="Cauda nominal do p99 (1%)",
    )
    fp_teste_sem_comp = _resumo_sem_compartilhamento(
        erros_teste, limiar, info_limiar.get("split_temporal"), "teste"
    )
    resolucao = 100.0 / fp_calib["n"] if fp_calib["n"] else float("nan")
    ax_cauda.text(
        0.03,
        0.05,
        (
            f"Calibração: p99 interpolado; resolução 1/{fp_calib['n']} "
            f"= {resolucao:.2f}%\n"
            f"Teste por janela: {_fmt_excedencia(fp_teste)}\n"
            "Teste sem compartilhamento: "
            f"{_fmt_excedencia(fp_teste_sem_comp)}"
        ),
        transform=ax_cauda.transAxes,
        fontsize=8.5,
        color=COR_TEXTO_SEC,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#e1e0d9"},
    )
    ax_ecdf.set_xscale("log")
    ax_ecdf.set_ylim(0.0, 1.02)
    ax_ecdf.set_xlabel("Erro de reconstrução (MSE, escala log)")
    ax_ecdf.set_ylabel("Probabilidade acumulada")
    ax_ecdf.set_title("Distribuição acumulada empírica (ECDF)")
    ax_ecdf.legend(fontsize=8)
    ax_cauda.set_xscale("log")
    ax_cauda.set_yscale("log")
    ax_cauda.set_ylim(0.005, 1.05)
    ax_cauda.set_xlabel("Erro de reconstrução (MSE, escala log)")
    ax_cauda.set_ylabel("Probabilidade empírica P(MSE > x)")
    ax_cauda.set_title("Cauda superior (escala log)")
    ax_cauda.legend(fontsize=8, loc="upper right")
    fig.suptitle("Erro de reconstrução do autoencoder em dados saudáveis")
    caminho = pasta / "distribuicao_erro.png"
    salvar_figura(
        fig,
        caminho,
        (
            "p99 nominal interpolado na calibração; teste separado. IC95% de "
            "Wilson é descritivo por janela; sobreposição e dependência serial "
            "limitam a interpretação inferencial."
        ),
    )
    _log(f"   📊 {caminho.name}")


def _sombrear_split_temporal(ax, tempos: np.ndarray, split: dict | None) -> None:
    if not split:
        return
    blocos = [
        ("treino", "Treino", PALETA[0]),
        ("val", "Calibração", PALETA[1]),
        ("teste", "Teste isolado", PALETA[2]),
    ]
    n = len(tempos)
    for chave, rotulo, cor in blocos:
        for numero, (ini, fim) in enumerate(_normalizar_intervalos(split, chave)):
            if not (0 <= ini < fim <= n):
                continue
            ax.axvspan(
                float(tempos[ini]),
                float(tempos[fim - 1]),
                color=cor,
                alpha=0.075,
                label=rotulo if numero == 0 else "_nolegend_",
                zorder=0,
            )


def plotar_erro_temporal(erros: np.ndarray,
                         tempos: np.ndarray,
                         info_limiar: dict, pasta: Path,
                         indices_teste: np.ndarray | None = None):
    """Erro de reconstrução ao longo do tempo."""
    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    _sombrear_split_temporal(ax, tempos, info_limiar.get("split_temporal"))
    ax.plot(tempos, erros, color=PALETA[0], alpha=0.8, linewidth=0.8)
    ax.axhline(info_limiar["limiar"], color=COR_ALERTA, linestyle="--",
               linewidth=1.5, label=f"Referência MSE p99 = {info_limiar['limiar']:.4f}")
    alarmes = erros > info_limiar["limiar"]
    ax.scatter(tempos[alarmes], erros[alarmes], color=COR_ALERTA, s=22,
               zorder=3, label="Acima da referência MSE")
    if (
        not info_limiar.get("split_temporal")
        and indices_teste is not None
        and len(indices_teste)
    ):
        inicio_teste = float(tempos[int(indices_teste[0])])
        ax.axvspan(inicio_teste, float(tempos[-1]), color=COR_NAO_DETECTADO, alpha=0.08,
                   label="Bloco de teste isolado")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Erro de Reconstrução (MSE)")
    ax.set_yscale("log")
    ax.set_title("Erro temporal de reconstrução em dados saudáveis\n"
                 "Referência MSE p99 e split intercalado com purga")
    ax.legend()
    caminho = pasta / "erro_temporal.png"
    salvar_figura(
        fig,
        caminho,
        "As faixas mostram o split temporal com purga; pontos acima da referência não são falhas confirmadas.",
    )
    _log(f"   📊 {caminho.name}")


def _info_em_escala_mse(info: dict, erros_teste=None) -> dict:
    """Vista de `limiar.json` na escala do MSE, para os gráficos de MSE.

    O contrato canônico atual usa MSE p99 também como escore operacional. Em
    artefatos históricos ou execuções experimentais, porém, `limiar` pode ser
    o limiar de outro escore. Os três gráficos deste módulo sempre plotam MSE,
    portanto usam explicitamente `limiar_mse` quando esse campo está presente.

    Passar o dicionário cru para eles desenha a linha de limiar muito acima do
    eixo e reporta "zero alarmes" no erro temporal — figura ERRADA, não figura
    desatualizada. O caminho do pipeline escapava disso porque monta seu
    próprio `info_mse`; a regeneração a partir do disco não escapava.

    `fp_test_pct` é recalculado contra o limiar de MSE pelo mesmo motivo: em um
    artefato não canônico, o valor salvo pode se referir a outro escore.
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
        salvar_resumo_calibracao(
            diag["erros_treino"], diag["erros_calibracao"],
            diag["erros_teste"], info, pasta,
            scores_treino=diag["scores_operacionais_treino"]
            if "scores_operacionais_treino" in diag.files else None,
            scores_calibracao=diag["scores_operacionais_calibracao"]
            if "scores_operacionais_calibracao" in diag.files else None,
            scores_teste=diag["scores_operacionais_teste"]
            if "scores_operacionais_teste" in diag.files else None,
        )
    return True
