"""
validacao.py — Al IAdo PV / Fase 5
Validação sintética interna E2 do detector com métricas quantitativas.

Fundamentação:
  Avalia o Autoencoder treinado com métricas padronizadas de classificação
  binária (anomalia / saudável), conectando os resultados da injeção de
  falhas com métricas interpretáveis para a banca de defesa.

Protocolo de validação:
  1. Classe NEGATIVA (saudável): janelas de teste dos ensaios GPVS F0L/F0M
  2. Classe POSITIVA (falha): janelas com falhas sintéticas injetadas
     nas 3 severidades mais representativas (0.30, 0.50, 1.00)
  3. Limiar variado de 0 a max_erro para construir a curva ROC
  4. Métricas no limiar operacional carregado de limiar.json:
     Precision, Recall, F1-Score, Accuracy, AUC-ROC

Métricas geradas:
  - Curva ROC com AUC para cada tipo de falha
  - Matriz de Confusão no limiar operacional
  - Heatmap/tabela comparativa por falha e severidade

Conexão com FMECA:
  As métricas são reportadas por falha, conectando com os índices
  NPR da FMECA do TCC (Torres, 2024) para cada componente crítico.

Saída:
  resultados/autoencoder/validacao_roc.png
  resultados/autoencoder/validacao_matriz.png
  resultados/autoencoder/validacao_metricas.png
  resultados/autoencoder/validacao_report.json
  resultados/autoencoder/validacao_tabela.csv

Uso:
  python src/ml/validacao.py

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta: python src/ml/<arquivo>.py
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("validacao")
_log = _adaptar_log(_logger)


import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ml.estilo_graficos import (
    COR_ALERTA, TAM, aplicar_estilo, salvar_figura,
)

aplicar_estilo()
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)

from src.ml.gpvs_principal import (
    carregar_normalizacao_baseline,
    extrair_janela,
    JANELA,
    FS,
    normalizar_vetores_f0,
    preparar_janelas_holdout,
)
from src.ml.estatistica import bootstrap_auc_ci, intervalo_wilson
from src.ml.injecao_falhas   import (
    FUNCOES_FALHA, FALHAS,
)

if TYPE_CHECKING:
    import torch

    from src.ml.autoencoder import Autoencoder

# ── Caminhos ─────────────────────────────────────────────────
RAIZ        = Path(__file__).parent.parent.parent
PASTA_AE    = RAIZ / "resultados" / "autoencoder"

# Severidades avaliadas na validação sintética interna E2
SEVS_VALIDACAO = [0.30, 0.50, 1.00]

# Número de janelas por classe
N_JANELAS_SAUDAVEL = 40
N_JANELAS_FALHA    = 40

# Prevalência REALISTA de falha em operação (falhas CA são eventos raros).
# O teste é balanceado (50/50) para estimar TPR/FPR de forma estável, mas
# precision/F1 a 50% NÃO refletem o campo. Reportamos também precision/F1 na
# prevalência rara abaixo — recall (TPR), specificity e AUC independem dela.
PREVALENCIA_RARA = 0.05


# ============================================================
# COLETA DE ERROS (saudável e falhas)
# ============================================================

def coletar_erros(janelas_holdout: list[pd.DataFrame],
                  modelo: Autoencoder,
                  scaler,
                  device: torch.device,
                  colunas_feat: list,
                  tipo_falha: str,
                  severidade: float,
                  n_janelas: int,
                  estat_residuo: dict | None = None,
                  metodo: str = "mse",
                  normalizacao_baseline: dict | None = None) -> np.ndarray:
    """
    Coleta o ESCORE de anomalia em janelas não sobrepostas do holdout isolado.

    Escore via src/ml/escore_anomalia.py: MSE médio (padrão) ou localizado
    (`metodo="localizado"` + régua `estat_residuo`). Mesmo escore do limiar.
    """
    from src.ml import escore_anomalia as ea

    fn = FUNCOES_FALHA.get(tipo_falha)  # None se for "saudavel"
    erros = []

    for i, janela_base in enumerate(janelas_holdout[:n_janelas]):
        janela = janela_base.copy()

        if fn is not None:
            if tipo_falha == "contator_ac":
                janela = fn(janela, severidade, seed=20_000 + i)
            else:
                janela = fn(janela, severidade)

        feats  = extrair_janela(janela)
        vetor  = np.array([feats.get(c, 0.0) for c in colunas_feat],
                          dtype=np.float32)
        if normalizacao_baseline is not None:
            vetor = normalizar_vetores_f0(
                vetor.reshape(1, -1), [janela.attrs.get("ensaio")],
                normalizacao_baseline,
            )[0]
        vnorm  = scaler.transform(vetor.reshape(1, -1)).astype(np.float32)
        residuo = ea.residuo_de_vetor(modelo, vnorm, device)
        erros.append(float(ea.pontuar(residuo, estat_residuo, metodo)[0]))

    return np.array(erros)


# ============================================================
# MÉTRICAS NO LIMIAR OPERACIONAL
# ============================================================

def metricas_no_limiar(erros_neg: np.ndarray,
                        erros_pos: np.ndarray,
                        limiar: float,
                        seed: int = 42) -> dict:
    """
    Calcula métricas de classificação binária no limiar fixo.
    Negativo=saudável (0), Positivo=falha (1).
    """
    y_true = np.concatenate([np.zeros(len(erros_neg)),
                              np.ones(len(erros_pos))])
    y_pred = np.concatenate([
        (erros_neg > limiar).astype(int),
        (erros_pos > limiar).astype(int)
    ])
    y_score = np.concatenate([erros_neg, erros_pos])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
    pr_auc               = auc(rec_arr, prec_arr)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # — Ponto de operação (limiar congelado): TPR/FPR independem da prevalência —
    tpr_op = float((erros_pos > limiar).mean())   # = recall
    fpr_op = float((erros_neg > limiar).mean())   # = 1 - specificity

    # — Precision/F1 reprojetados para a prevalência RARA (operação real) —
    # Bayes no ponto de operação: prec = π·TPR / (π·TPR + (1−π)·FPR).
    # AUC e recall NÃO mudam; só precision/F1 (que dependem da base rate).
    pi = PREVALENCIA_RARA
    denom = pi * tpr_op + (1.0 - pi) * fpr_op
    precision_raro = float(pi * tpr_op / denom) if denom > 0 else 0.0
    f1_raro = (float(2 * precision_raro * tpr_op / (precision_raro + tpr_op))
               if (precision_raro + tpr_op) > 0 else 0.0)
    recall_ci = intervalo_wilson(int(tp), int(tp + fn))
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    specificity_ci = intervalo_wilson(int(tn), int(tn + fp))
    precision_ci = intervalo_wilson(int(tp), int(tp + fp))
    auc_ci = bootstrap_auc_ci(erros_neg, erros_pos, n_boot=500, seed=seed)

    return {
        "precision"  : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"     : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"         : float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy"   : float(accuracy_score(y_true, y_pred)),
        "auc_roc"    : float(roc_auc),
        "auc_pr"     : float(pr_auc),
        "specificity": specificity,
        "fnr"        : float(1.0 - tpr_op),
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
        "specificity_ci_low": specificity_ci[0],
        "specificity_ci_high": specificity_ci[1],
        "precision_ci_low": precision_ci[0],
        "precision_ci_high": precision_ci[1],
        **auc_ci,
        # Regime raro (prevalência realista de falha CA) — precision/F1 honestos:
        "prevalencia_raro" : pi,
        "tpr_op"           : tpr_op,
        "fpr_op"           : fpr_op,
        "precision_raro"   : precision_raro,
        "f1_raro"          : f1_raro,
        "confusion"  : cm.tolist(),
        "fpr"        : fpr.tolist(),
        "tpr"        : tpr.tolist(),
        "prec_arr"   : prec_arr.tolist(),
        "rec_arr"    : rec_arr.tolist(),
        "n_neg"      : len(erros_neg),
        "n_pos"      : len(erros_pos),
    }


# ============================================================
# VISUALIZAÇÕES
# ============================================================

def plotar_roc(resultados: dict, limiar: float, pasta: Path):
    """Curvas ROC para cada combinação falha × severidade."""
    fig, axes = plt.subplots(
        1, 3, figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle("Curvas ROC — Detecção de Anomalias por Tipo de Falha",
                 fontsize=13, fontweight="bold")

    cores_sev = {0.30: "#FFB300", 0.50: "#FB8C00", 1.00: "#E53935"}

    for ax, falha in zip(axes, FALHAS):
        fid  = falha["id"]
        nome = falha["nome"]
        npr  = falha["npr"]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

        for sev in SEVS_VALIDACAO:
            chave = f"{fid}_sev{sev}"
            if chave not in resultados:
                continue
            res = resultados[chave]
            ax.plot(res["fpr"], res["tpr"],
                    color=cores_sev[sev], linewidth=2,
                    label=(f"sev={sev} · AUC={res['auc_roc']:.3f} "
                           f"[{res['auc_roc_ci_low']:.3f}; {res['auc_roc_ci_high']:.3f}]"))
            ax.scatter(
                [res["fpr_op"]], [res["tpr_op"]], color=cores_sev[sev],
                edgecolors="black", linewidths=0.6, s=42, zorder=4,
            )

        npm_str = f"NPR={npr}"
        ax.set_title(f"{nome}\n({npm_str})", fontsize=10)
        ax.set_xlabel("Taxa de Falso Positivo")
        ax.set_ylabel("Taxa de Verdadeiro Positivo (Recall)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    arq = pasta / "validacao_roc.png"
    salvar_figura(
        fig,
        arq,
        "Os círculos marcam o limiar operacional congelado; faixas na legenda são IC95% bootstrap da AUC.",
    )
    _log(f"   📊 {arq.name}")


def plotar_pr(resultados: dict, pasta: Path):
    """Curvas Precision-Recall por falha × severidade (complementa a ROC,
    importante quando há desbalanceamento entre saudável e falha)."""
    fig, axes = plt.subplots(
        1, 3, figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle("Curvas Precision-Recall — Detecção de Anomalias por Tipo de Falha",
                 fontsize=13, fontweight="bold")

    cores_sev = {0.30: "#FFB300", 0.50: "#FB8C00", 1.00: "#E53935"}

    for ax, falha in zip(axes, FALHAS):
        fid, nome, npr = falha["id"], falha["nome"], falha["npr"]
        for sev in SEVS_VALIDACAO:
            chave = f"{fid}_sev{sev}"
            if chave not in resultados:
                continue
            res = resultados[chave]
            ax.plot(res["rec_arr"], res["prec_arr"],
                    color=cores_sev[sev], linewidth=2,
                    label=f"sev={sev} (AUC-PR={res['auc_pr']:.3f})")
            ax.scatter(
                [res["recall"]], [res["precision"]], color=cores_sev[sev],
                edgecolors="black", linewidths=0.6, s=42, zorder=4,
            )
        npm_str = f"NPR={npr}"
        ax.set_title(f"{nome}\n({npm_str})", fontsize=10)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    arq = pasta / "validacao_pr.png"
    salvar_figura(
        fig,
        arq,
        "Os círculos marcam o ponto operacional; teste interno balanceado, evidência sintética E2.",
    )
    _log(f"   📊 {arq.name}")


def plotar_matrizes(resultados: dict, pasta: Path):
    """Matrizes de confusão para severidade=1.0 de cada falha."""
    fig, axes = plt.subplots(
        1, 3, figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle("Matrizes de Confusão — Severidade 1.0 (limiar operacional)",
                 fontsize=12, fontweight="bold")

    for ax, falha in zip(axes, FALHAS):
        fid   = falha["id"]
        chave = f"{fid}_sev1.0"
        if chave not in resultados:
            ax.axis("off")
            continue

        cm = np.array(resultados[chave]["confusion"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

        classes = ["Saudável", "Falha"]
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight="bold")

        f1  = resultados[chave]["f1"]
        auc = resultados[chave]["auc_roc"]
        rec = resultados[chave]["recall"]
        npm_str = f"NPR={falha['npr']}"
        ax.set_title(f"{falha['nome']}\n({npm_str}) · Recall={rec:.2f} · F1={f1:.2f} · AUC={auc:.2f}",
                     fontsize=9)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predito")

    arq = pasta / "validacao_matriz.png"
    salvar_figura(
        fig, arq,
        "Matriz no ponto operacional; linhas = classe real, colunas = predição.",
    )
    _log(f"   📊 {arq.name}")


def plotar_matrizes_todas_severidades(resultados: dict, pasta: Path):
    """Matrizes de confusão para toda combinação falha x severidade."""
    fig, axes = plt.subplots(
        len(FALHAS), len(SEVS_VALIDACAO),
        figsize=TAM["painel_9"], layout="constrained",
    )
    fig.suptitle("Matrizes de confusão — todas as severidades no limiar operacional")

    for linha, falha in enumerate(FALHAS):
        for coluna, sev in enumerate(SEVS_VALIDACAO):
            ax = axes[linha][coluna]
            chave = f"{falha['id']}_sev{sev}"
            res = resultados[chave]
            cm = np.asarray(res["confusion"])
            ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=cm.max())
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            ax.set_xticks([0, 1], ["Saudável", "Falha"], fontsize=8)
            ax.set_yticks([0, 1], ["Saudável", "Falha"], fontsize=8)
            ax.set_title(
                f"{falha['nome']} · sev={sev:.2f}\n"
                f"Recall={res['recall']:.2f} · FNR={res['fnr']:.2f}",
                fontsize=9,
            )
            if coluna == 0:
                ax.set_ylabel(f"Real\nNPR={falha['npr']}")
            if linha == len(FALHAS) - 1:
                ax.set_xlabel("Predito")

    arq = pasta / "validacao_matrizes_severidades.png"
    salvar_figura(
        fig, arq,
        "E2 sintético em janelas não sobrepostas do holdout temporal; n por classe indicado no CSV.",
    )
    _log(f"   📊 {arq.name}")


def plotar_tabela_metricas(tabela_df: pd.DataFrame, pasta: Path):
    """Heatmap das métricas F1, AUC, Recall por falha × severidade."""
    fig, axes = plt.subplots(2, 2, figsize=TAM["painel_4"], layout="constrained")
    fig.suptitle("Métricas por Tipo de Falha e Severidade",
                 fontsize=13, fontweight="bold")

    metricas_plot = ["f1", "auc_roc", "recall", "fnr"]
    titulos = ["F1-Score", "AUC-ROC", "Recall (Sensibilidade)", "FNR (Falhas Perdidas)"]
    cmaps = ["YlOrRd", "YlGnBu", "PuRd", "Reds"]
    ordem_falhas = [falha["nome"] for falha in FALHAS]

    for ax, metrica, titulo, cmap in zip(axes.ravel(), metricas_plot, titulos, cmaps):
        pivot = tabela_df.pivot(index="falha", columns="severidade",
                                values=metrica)
        pivot = pivot.reindex(ordem_falhas)
        im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1,
                       aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([str(s) for s in pivot.columns], fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Severidade")
        ax.set_title(titulo, fontsize=10)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if val > 0.6 else "black",
                        fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    arq = pasta / "validacao_metricas.png"
    salvar_figura(
        fig, arq,
        "AUC mede ranking; Recall/FNR mostram o comportamento efetivo no limiar operacional.",
    )
    _log(f"   📊 {arq.name}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_validacao() -> bool:
    _log("=" * 60)
    _log("  AL IADO PV — VALIDAÇÃO SINTÉTICA INTERNA E2")
    _log("=" * 60)

    # ── 1. Carrega artefatos ─────────────────────────────────
    _log(f"\n📂 Carregando artefatos...")
    arq_modelo = PASTA_AE / "modelo_autoencoder.pt"
    arq_scaler = PASTA_AE / "scaler.pkl"
    arq_limiar = PASTA_AE / "limiar.json"

    for arq in [arq_modelo, arq_scaler, arq_limiar]:
        if not arq.exists():
            _log(f"   ❌ {arq.name} não encontrado")
            return False

    import torch
    from src.ml.autoencoder import Autoencoder

    checkpoint = torch.load(arq_modelo, map_location="cpu",
                            weights_only=False)
    from src.core.seguranca import carregar_pickle_com_sidecar

    scaler = carregar_pickle_com_sidecar(arq_scaler)
    with open(arq_limiar, "r") as f:
        info_limiar = json.load(f)

    n_features   = checkpoint["n_features"]
    latente_dim  = checkpoint["latente_dim"]
    colunas_feat = checkpoint["colunas_feat"]
    limiar       = info_limiar["limiar"]   # OPERACIONAL (método escolhido)

    # Escore operacional (mesmo do limiar): método + régua por-feature dos
    # artefatos do autoencoder. Sem eles (artefato antigo), cai para MSE.
    from src.ml import escore_anomalia as ea

    metodo_escore = info_limiar.get("metodo_escore", "mse")
    estat_residuo = ea.carregar_estatistica(PASTA_AE)
    normalizacao_baseline = carregar_normalizacao_baseline(PASTA_AE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    _log(f"   ✅ Modelo carregado | limiar={limiar:.4f} | "
          f"escore={ea.descricao_metodo(metodo_escore, info_limiar.get('k_localizado', 5))}")

    # ── 2. Holdout temporal isolado ───────────────────────────
    _log(f"\n📂 Carregando dataset...")
    janelas_holdout, meta_holdout = preparar_janelas_holdout(
        n_max=max(N_JANELAS_SAUDAVEL, N_JANELAS_FALHA)
    )
    _log(f"   ✅ {len(janelas_holdout)} janelas não sobrepostas do teste")

    # ── 3. Erros classe negativa (saudável) ──────────────────
    _log(f"\n⚕️  Coletando erros — classe SAUDÁVEL ({N_JANELAS_SAUDAVEL} janelas)...")
    erros_neg = coletar_erros(
        janelas_holdout, modelo, scaler, device, colunas_feat,
        "saudavel", 0.0, N_JANELAS_SAUDAVEL, estat_residuo, metodo_escore,
        normalizacao_baseline,
    )
    _log(f"   μ={erros_neg.mean():.4f} ± {erros_neg.std():.4f} | "
          f"FP={( erros_neg > limiar).mean()*100:.1f}%")

    # ── 4. Erros por falha e severidade ──────────────────────
    _log(f"\n💉 Coletando erros — classes FALHA...")
    resultados = {}
    linhas_tabela = []

    for falha in FALHAS:
        fid  = falha["id"]
        nome = falha["nome"]
        _log(f"\n   🔴 {nome}")

        for sev in SEVS_VALIDACAO:
            erros_pos = coletar_erros(
                janelas_holdout, modelo, scaler, device, colunas_feat,
                fid, sev, N_JANELAS_FALHA, estat_residuo, metodo_escore,
                normalizacao_baseline,
            )

            seed_boot = 42 + 100 * len(resultados) + int(sev * 100)
            res = metricas_no_limiar(
                erros_neg, erros_pos, limiar, seed=seed_boot
            )
            chave = f"{fid}_sev{sev}"
            resultados[chave] = res

            _log(f"      sev={sev:.2f} | "
                  f"F1={res['f1']:.3f} | "
                  f"AUC={res['auc_roc']:.3f} | "
                  f"Recall={res['recall']:.3f} | "
                  f"Precision={res['precision']:.3f}")

            linhas_tabela.append({
                "falha"    : nome,
                "falha_id" : fid,
                "npr"      : falha["npr"],
                "severidade": sev,
                "f1"       : res["f1"],
                "auc_roc"  : res["auc_roc"],
                "recall"   : res["recall"],
                "precision": res["precision"],
                "accuracy" : res["accuracy"],
                "auc_pr"   : res["auc_pr"],
                "specificity": res["specificity"],
                "fpr"      : res["fpr_op"],
                "fnr"      : res["fnr"],
                "n_neg"    : res["n_neg"],
                "n_pos"    : res["n_pos"],
                "recall_ci_low": res["recall_ci_low"],
                "recall_ci_high": res["recall_ci_high"],
                "specificity_ci_low": res["specificity_ci_low"],
                "specificity_ci_high": res["specificity_ci_high"],
                "auc_roc_ci_low": res["auc_roc_ci_low"],
                "auc_roc_ci_high": res["auc_roc_ci_high"],
                "threshold_method": info_limiar.get("threshold_method", "p99"),
                "score_method": info_limiar.get(
                    "score_method", info_limiar.get("metodo_escore")
                ),
                "score_threshold": info_limiar.get(
                    "score_threshold", info_limiar.get("limiar")
                ),
                "threshold_effective_percentile": info_limiar.get(
                    "threshold_effective_percentile",
                    info_limiar.get("percentil_limiar"),
                ),
                "threshold_source": info_limiar.get(
                    "threshold_source", "bloco_calibracao_temporal"
                ),
                "evidence_level": "E2",
                # regime raro (prevalência realista)
                "precision_raro": res["precision_raro"],
                "f1_raro"       : res["f1_raro"],
            })

    # ── 5. Visualizações ─────────────────────────────────────
    _log(f"\n📊 Gerando gráficos...")
    plotar_roc(resultados, limiar, PASTA_AE)
    plotar_pr(resultados, PASTA_AE)
    plotar_matrizes(resultados, PASTA_AE)
    plotar_matrizes_todas_severidades(resultados, PASTA_AE)

    tabela_df = pd.DataFrame(linhas_tabela)
    plotar_tabela_metricas(tabela_df, PASTA_AE)

    # ── 6. Salva resultados ──────────────────────────────────
    arq_csv = PASTA_AE / "validacao_tabela.csv"
    tabela_df.to_csv(arq_csv, index=False)
    _log(f"   📋 {arq_csv.name}")

    arq_md = PASTA_AE / "validacao_tabela.md"
    cabecalho = (
        "| Falha | Sev. | AUC-ROC (IC95%) | Recall (IC95%) | FNR | "
        "Especificidade | n/classe |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    linhas_md = []
    for row in linhas_tabela:
        linhas_md.append(
            f"| {row['falha']} | {row['severidade']:.2f} | "
            f"{row['auc_roc']:.3f} [{row['auc_roc_ci_low']:.3f}; "
            f"{row['auc_roc_ci_high']:.3f}] | {row['recall']:.3f} "
            f"[{row['recall_ci_low']:.3f}; {row['recall_ci_high']:.3f}] | "
            f"{row['fnr']:.3f} | {row['specificity']:.3f} | "
            f"{row['n_pos']} |"
        )
    arq_md.write_text(
        "# Validação sintética interna E2\n\n"
        "> Holdout temporal não sobreposto; limiar operacional congelado na calibração.\n\n"
        + cabecalho + "\n".join(linhas_md) + "\n",
        encoding="utf-8",
    )
    _log(f"   📋 {arq_md.name}")

    arq_json = PASTA_AE / "validacao_report.json"
    referencia_negativa = next(iter(resultados.values()))
    fpr_holdout = float(referencia_negativa["fpr_op"])
    fp_holdout = int(referencia_negativa["confusion"][0][1])
    n_neg_holdout = int(referencia_negativa["n_neg"])
    rotulo_fp = "falso positivo" if fp_holdout == 1 else "falsos positivos"
    report_serializavel = {
        "__meta__": {
            "evidence_level": "E2",
            "evidence_note": (
                "Validação sintética orientada pela FMECA: classe negativa = "
                "janelas saudáveis; classe positiva = falhas injetadas (ground "
                "truth). Limiar CONGELADO, carregado de limiar.json — NÃO "
                "otimizado no teste. Não é prova de desempenho industrial (E3)."
            ),
            "threshold_method": info_limiar.get("threshold_method", "p99"),
            "score_method": info_limiar.get(
                "score_method", info_limiar.get("metodo_escore")
            ),
            "score_threshold": info_limiar.get(
                "score_threshold", info_limiar.get("limiar")
            ),
            "mse_p99": info_limiar.get("mse_p99", info_limiar.get("limiar_p99")),
            "sigma_multiplier": info_limiar.get(
                "sigma_multiplier", info_limiar.get("k")
            ),
            "top_k": info_limiar.get("top_k", info_limiar.get("k_localizado")),
            "threshold_fallback_percentile": info_limiar.get(
                "threshold_fallback_percentile"
            ),
            "threshold_effective_percentile": info_limiar.get(
                "threshold_effective_percentile",
                info_limiar.get("percentil_limiar"),
            ),
            "threshold_source": info_limiar.get(
                "threshold_source", "bloco_calibracao_temporal"
            ),
            "limiar_operacional": float(limiar),
            "protocolo_avaliacao": meta_holdout,
            "prevalencia_teste": 0.5,
            "prevalencia_raro": PREVALENCIA_RARA,
            "falsos_positivos_holdout": fp_holdout,
            "n_saudaveis_holdout": n_neg_holdout,
            "fpr_holdout": fpr_holdout,
            "nota_regime_raro": (
                "O teste é balanceado (50% falha) para estimar TPR/FPR com "
                "estabilidade, mas falhas CA são RARAS em operação. Por isso "
                "reportamos também precision_raro/f1_raro reprojetados para "
                f"prevalência de {PREVALENCIA_RARA:.0%} (regra de Bayes no ponto "
                "de operação). AUC, recall (TPR) e specificity independem da "
                "prevalência; só precision/F1 mudam. Nesta execução, o "
                f"limiar operacional produziu {fp_holdout} {rotulo_fp} em "
                f"{n_neg_holdout} janelas saudáveis (FPR={fpr_holdout:.2%}). "
                "Esse FPR observado reduz de forma relevante a precisão projetada "
                "para o regime raro. Como o holdout é pequeno, o valor tem resolução "
                "amostral limitada e não deve ser interpretado como taxa de campo."
            ),
        },
    }
    for chave, res in resultados.items():
        report_serializavel[chave] = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in res.items()
        }
    with open(arq_json, "w", encoding="utf-8") as f:
        json.dump(report_serializavel, f, indent=2, ensure_ascii=False)
    _log(f"   ✅ {arq_json.name}")

    # ── 7. Resumo final ──────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"  VALIDAÇÃO CONCLUÍDA!")
    _log(f"\n  {'Falha':<30} {'Sev':>5} {'AUC':>7} {'Recall':>8} "
          f"{'F1@50%':>8} {'F1@5%':>8}")
    _log(f"  {'-'*72}")
    for row in linhas_tabela:
        _log(f"  {row['falha']:<30} {row['severidade']:>5.2f} "
              f"{row['auc_roc']:>7.3f} {row['recall']:>8.3f} "
              f"{row['f1']:>8.3f} {row['f1_raro']:>8.3f}")
    _log("\n  AUC/Recall independem da prevalência; F1@50% é o teste balanceado "
         f"e F1@{PREVALENCIA_RARA:.0%} reflete a raridade real das falhas CA.")

    # Melhor AUC geral
    melhor = max(linhas_tabela, key=lambda x: x["auc_roc"])
    _log(f"\n  Melhor resultado: {melhor['falha']} "
          f"(sev={melhor['severidade']}) — AUC={melhor['auc_roc']:.3f}")
    _log(f"\n  Próximo passo: dissertação — Capítulo 4 (Resultados)")
    _log(f"{'='*60}")

    return True


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    executar_validacao()
