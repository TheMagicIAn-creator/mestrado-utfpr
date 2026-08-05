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

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   # sem display — salva direto em arquivo
import matplotlib.pyplot as plt  # noqa: E402

from src.core.logs import adaptar_logger_como_print, get_logger  # noqa: E402
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
from src.ml.estatistica import intervalo_wilson  # noqa: E402

aplicar_estilo()

RAIZ = Path(__file__).parent.parent.parent
PASTA_SAIDA = RAIZ / "resultados" / "autoencoder"

_logger = get_logger("autoencoder")
_log = adaptar_logger_como_print(_logger)


def resumo_excedencia(valores: np.ndarray, limiar: float) -> dict:
    """Conta excedências e adiciona IC95% de Wilson para a proporção."""
    arr = np.asarray(valores, dtype=float)
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


def _linha_calibracao(nome: str, mse: np.ndarray, score: np.ndarray | None,
                      limiar_mse: float, limiar_score: float) -> dict:
    resumo_mse = _resumo_vetor(mse)
    exc_mse = resumo_excedencia(mse, limiar_mse)
    resumo_score = _resumo_vetor(score) if score is not None else {}
    exc_score = (
        resumo_excedencia(score, limiar_score) if score is not None else {}
    )
    return {
        "bloco": nome,
        **{f"mse_{k}": v for k, v in resumo_mse.items()},
        "mse_ref_p99_count": exc_mse.get("count"),
        "mse_ref_p99_rate_pct": exc_mse.get("rate_pct"),
        "mse_ref_p99_ci95_low_pct": exc_mse.get("ci95_low_pct"),
        "mse_ref_p99_ci95_high_pct": exc_mse.get("ci95_high_pct"),
        **{f"score_operacional_{k}": v for k, v in resumo_score.items()},
        "score_operacional_count": exc_score.get("count"),
        "score_operacional_rate_pct": exc_score.get("rate_pct"),
        "score_operacional_ci95_low_pct": exc_score.get("ci95_low_pct"),
        "score_operacional_ci95_high_pct": exc_score.get("ci95_high_pct"),
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
    linhas = [
        _linha_calibracao("treino", erros_treino, scores_treino, limiar_mse, limiar_score),
        _linha_calibracao(
            "calibracao", erros_calibracao, scores_calibracao,
            limiar_mse, limiar_score,
        ),
        _linha_calibracao(
            "teste_isolado", erros_teste, scores_teste,
            limiar_mse, limiar_score,
        ),
    ]
    df = pd.DataFrame(linhas)
    csv_path = pasta / "calibracao_autoencoder.csv"
    md_path = pasta / "calibracao_autoencoder.md"
    df.to_csv(csv_path, index=False)

    ponto = str(metodo)
    if percentil is not None:
        ponto += f" / percentil efetivo {_fmt(percentil, 1)}"
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
        "- Intervalos entre colchetes são IC95% de Wilson para a proporção de excedências.",
        "",
        "| Bloco | n | MSE mediana | MSE IQR | MSE p99 | > ref. MSE p99 | "
        "Escore mediana | > limiar operacional |",
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
        exc_score = {
            "count": row.get("score_operacional_count"),
            "n": row.get("score_operacional_n_janelas"),
            "rate_pct": row.get("score_operacional_rate_pct"),
            "ci95_low_pct": row.get("score_operacional_ci95_low_pct"),
            "ci95_high_pct": row.get("score_operacional_ci95_high_pct"),
        }
        texto.append(
            f"| {row['bloco']} | {int(row['mse_n_janelas'])} | "
            f"{_fmt(row['mse_mediana'])} | {_fmt(row['mse_iqr'])} | "
            f"{_fmt(row['mse_p99'])} | {_fmt_excedencia(exc_mse)} | "
            f"{_fmt(row.get('score_operacional_mediana'))} | "
            f"{_fmt_excedencia(exc_score)} |"
        )
    texto.extend([
        "",
        (
            "Observação metodológica: os gráficos `distribuicao_erro.png` e "
            "`erro_temporal.png` estão na escala MSE; a decisão operacional do "
            "pipeline usa o escore canônico registrado em `limiar.json`."
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
                   label=f"Referência MSE p99 = {limiar:.4f}")

    # μ+kσ entra apenas como REFERÊNCIA comparativa (não é o limiar em uso).
    mu3s = info_limiar.get("limiar_mu3sigma", info_limiar.get("limiar_mu3s"))
    if mu3s is not None:
        ax_hist.axvline(mu3s, color="#898781", linewidth=1.5, linestyle=":",
                        label=f"μ+{info_limiar['k']:.0f}σ = {mu3s:.4f}")

    fp_teste = resumo_excedencia(erros_teste, limiar)
    fp_calib = resumo_excedencia(erros_calibracao, limiar)
    ax_ecdf.axhline(
        0.99, color=COR_REFERENCIA, linewidth=1.1, linestyle=":",
        label="Probabilidade 0,99",
    )
    ax_ecdf.text(
        0.03,
        0.08,
        (
            "Excedência acima da referência MSE p99\n"
            f"calibração: {_fmt_excedencia(fp_calib)}\n"
            f"teste isolado: {_fmt_excedencia(fp_teste)}"
        ),
        transform=ax_ecdf.transAxes,
        fontsize=8.5,
        color=COR_TEXTO_SEC,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#e1e0d9"},
    )
    fp = fp_teste.get("rate_pct")
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
        "Referência MSE do autoencoder em dados saudáveis"
        + (f" — excedência MSE no teste: {fp:.2f}%" if isinstance(fp, (int, float)) else "")
    )
    caminho = pasta / "distribuicao_erro.png"
    salvar_figura(
        fig,
        caminho,
        (
            "A linha pontilhada vermelha é referência de MSE; a decisão "
            "operacional localizada está registrada em limiar.json."
        ),
    )
    _log(f"   📊 {caminho.name}")


def _sombrear_split_temporal(ax, tempos: np.ndarray, split: dict | None) -> None:
    if not split:
        return
    limites = split.get("limites") or {}
    blocos = [
        ("treino", "Treino", PALETA[0]),
        ("val", "Calibração", PALETA[1]),
        ("teste", "Teste isolado", PALETA[2]),
    ]
    n = len(tempos)
    for chave, rotulo, cor in blocos:
        intervalo = limites.get(chave) or []
        if len(intervalo) != 2:
            continue
        ini, fim = int(intervalo[0]), int(intervalo[1])
        if not (0 <= ini < fim <= n):
            continue
        ax.axvspan(
            float(tempos[ini]),
            float(tempos[fim - 1]),
            color=cor,
            alpha=0.055,
            label=rotulo,
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
                 "Referência MSE p99; decisão operacional localizada em limiar.json")
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
