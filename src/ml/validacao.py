"""
validacao.py — Al IAdo PV / Fase 5
Validação formal do detector de anomalias com métricas quantitativas.

Fundamentação:
  Avalia o Autoencoder treinado com métricas padronizadas de classificação
  binária (anomalia / saudável), conectando os resultados da injeção de
  falhas com métricas interpretáveis para a banca de defesa.

Protocolo de validação:
  1. Classe NEGATIVA (saudável): janelas do período estável de Paderborn
     (t=10-20s, pós-transiente) — ground truth de operação normal
  2. Classe POSITIVA (falha): janelas com falhas sintéticas injetadas
     nas 3 severidades mais representativas (0.30, 0.50, 1.00)
  3. Limiar variado de 0 a max_erro para construir a curva ROC
  4. Métricas no limiar operacional carregado de limiar.json:
     Precision, Recall, F1-Score, Accuracy, AUC-ROC

Métricas geradas:
  - Curva ROC com AUC para cada tipo de falha
  - Matriz de Confusão no limiar operacional
  - Heatmap/tabela comparativa por falha e severidade

Conexão com FMEA:
  As métricas são reportadas por falha, conectando com os índices
  NPR do FMEA do TCC (Torres, 2024) para cada componente crítico.

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

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import torch
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)

from src.ml.features_ca     import extrair_janela, JANELA, FS
from src.ml.autoencoder      import Autoencoder
from src.ml.injecao_falhas   import (
    FUNCOES_FALHA, FALHAS,
    T_INICIO_ESTAVEL, T_FIM_ESTAVEL
)

# ── Caminhos ─────────────────────────────────────────────────
RAIZ        = Path(__file__).parent.parent.parent
ARQUIVO_CSV = RAIZ / "dados" / "brutos" / "Inverter_Data_Set.csv"
PASTA_AE    = RAIZ / "resultados" / "autoencoder"

# Severidades avaliadas na validação formal
SEVS_VALIDACAO = [0.30, 0.50, 1.00]

# Número de janelas por classe
N_JANELAS_SAUDAVEL = 50
N_JANELAS_FALHA    = 50


# ============================================================
# COLETA DE ERROS (saudável e falhas)
# ============================================================

def coletar_erros(df_estavel: pd.DataFrame,
                  modelo: Autoencoder,
                  scaler,
                  device: torch.device,
                  colunas_feat: list,
                  tipo_falha: str,
                  severidade: float,
                  n_janelas: int) -> np.ndarray:
    """
    Coleta erros de reconstrução para janelas com falha injetada.
    Usa offsets variados para amostrar diferentes posições do sinal.
    """
    fn = FUNCOES_FALHA.get(tipo_falha)  # None se for "saudavel"
    erros = []

    passo_max = max(1, (len(df_estavel) - JANELA) // n_janelas)

    for i in range(n_janelas):
        inicio = (i * passo_max) % (len(df_estavel) - JANELA)
        janela = df_estavel.iloc[inicio:inicio + JANELA].copy()

        if fn is not None:
            janela = fn(janela, severidade)

        feats  = extrair_janela(janela)
        vetor  = np.array([feats.get(c, 0.0) for c in colunas_feat],
                          dtype=np.float32)
        vnorm  = scaler.transform(vetor.reshape(1, -1)).astype(np.float32)

        modelo.eval()
        with torch.no_grad():
            x     = torch.from_numpy(vnorm).to(device)
            x_rec = modelo(x)
            erro  = float(((x - x_rec) ** 2).mean().cpu())
        erros.append(erro)

    return np.array(erros)


# ============================================================
# MÉTRICAS NO LIMIAR OPERACIONAL
# ============================================================

def metricas_no_limiar(erros_neg: np.ndarray,
                        erros_pos: np.ndarray,
                        limiar: float) -> dict:
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

    return {
        "precision"  : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"     : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"         : float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy"   : float(accuracy_score(y_true, y_pred)),
        "auc_roc"    : float(roc_auc),
        "auc_pr"     : float(pr_auc),
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
                    label=f"sev={sev} (AUC={res['auc_roc']:.3f})")

        npm_str = f"NPR={npr}" if npr else "D=10"
        ax.set_title(f"{nome}\n({npm_str})", fontsize=10)
        ax.set_xlabel("Taxa de Falso Positivo")
        ax.set_ylabel("Taxa de Verdadeiro Positivo (Recall)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    plt.tight_layout()
    arq = pasta / "validacao_roc.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 {arq.name}")


def plotar_matrizes(resultados: dict, pasta: Path):
    """Matrizes de confusão para severidade=1.0 de cada falha."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Matrizes de Confusão — Severidade 1.0 (Limiar Operacional p99)",
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
        npm_str = f"NPR={falha['npr']}" if falha['npr'] else "D=10"
        ax.set_title(f"{falha['nome']}\n({npm_str}) | F1={f1:.3f} | AUC={auc:.3f}",
                     fontsize=9)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predito")

    plt.tight_layout()
    arq = pasta / "validacao_matriz.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 {arq.name}")


def plotar_tabela_metricas(tabela_df: pd.DataFrame, pasta: Path):
    """Heatmap das métricas F1, AUC, Recall por falha × severidade."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Métricas por Tipo de Falha e Severidade",
                 fontsize=13, fontweight="bold")

    metricas_plot = ["f1", "auc_roc", "recall"]
    titulos       = ["F1-Score", "AUC-ROC", "Recall (Sensibilidade)"]
    cmaps         = ["YlOrRd", "YlGnBu", "PuRd"]

    for ax, metrica, titulo, cmap in zip(axes, metricas_plot, titulos, cmaps):
        pivot = tabela_df.pivot(index="falha", columns="severidade",
                                values=metrica)
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

    plt.tight_layout()
    arq = pasta / "validacao_metricas.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 {arq.name}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_validacao() -> bool:
    print("=" * 60)
    print("  AL IADO PV — VALIDAÇÃO FORMAL DO DETECTOR")
    print("=" * 60)

    # ── 1. Carrega artefatos ─────────────────────────────────
    print(f"\n📂 Carregando artefatos...")
    arq_modelo = PASTA_AE / "modelo_autoencoder.pt"
    arq_scaler = PASTA_AE / "scaler.pkl"
    arq_limiar = PASTA_AE / "limiar.json"

    for arq in [arq_modelo, arq_scaler, arq_limiar]:
        if not arq.exists():
            print(f"   ❌ {arq.name} não encontrado")
            return False

    checkpoint = torch.load(arq_modelo, map_location="cpu",
                            weights_only=False)
    with open(arq_scaler, "rb") as f:
        scaler = pickle.load(f)
    with open(arq_limiar, "r") as f:
        info_limiar = json.load(f)

    n_features   = checkpoint["n_features"]
    latente_dim  = checkpoint["latente_dim"]
    colunas_feat = checkpoint["colunas_feat"]
    limiar       = info_limiar["limiar"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    print(f"   ✅ Modelo carregado | limiar={limiar:.4f}")

    # ── 2. Dataset estável ───────────────────────────────────
    print(f"\n📂 Carregando dataset...")
    df = pd.read_csv(ARQUIVO_CSV)
    idx_i = int(T_INICIO_ESTAVEL * FS)
    idx_f = int(T_FIM_ESTAVEL * FS)
    df_estavel = df.iloc[idx_i:idx_f].reset_index(drop=True)
    print(f"   ✅ {len(df_estavel):,} amostras estáveis disponíveis")

    # ── 3. Erros classe negativa (saudável) ──────────────────
    print(f"\n⚕️  Coletando erros — classe SAUDÁVEL ({N_JANELAS_SAUDAVEL} janelas)...")
    erros_neg = coletar_erros(
        df_estavel, modelo, scaler, device, colunas_feat,
        "saudavel", 0.0, N_JANELAS_SAUDAVEL
    )
    print(f"   μ={erros_neg.mean():.4f} ± {erros_neg.std():.4f} | "
          f"FP={( erros_neg > limiar).mean()*100:.1f}%")

    # ── 4. Erros por falha e severidade ──────────────────────
    print(f"\n💉 Coletando erros — classes FALHA...")
    resultados = {}
    linhas_tabela = []

    for falha in FALHAS:
        fid  = falha["id"]
        nome = falha["nome"]
        print(f"\n   🔴 {nome}")

        for sev in SEVS_VALIDACAO:
            erros_pos = coletar_erros(
                df_estavel, modelo, scaler, device, colunas_feat,
                fid, sev, N_JANELAS_FALHA
            )

            res   = metricas_no_limiar(erros_neg, erros_pos, limiar)
            chave = f"{fid}_sev{sev}"
            resultados[chave] = res

            print(f"      sev={sev:.2f} | "
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
            })

    # ── 5. Visualizações ─────────────────────────────────────
    print(f"\n📊 Gerando gráficos...")
    plotar_roc(resultados, limiar, PASTA_AE)
    plotar_matrizes(resultados, PASTA_AE)

    tabela_df = pd.DataFrame(linhas_tabela)
    plotar_tabela_metricas(tabela_df, PASTA_AE)

    # ── 6. Salva resultados ──────────────────────────────────
    arq_csv = PASTA_AE / "validacao_tabela.csv"
    tabela_df.to_csv(arq_csv, index=False)
    print(f"   📋 {arq_csv.name}")

    arq_json = PASTA_AE / "validacao_report.json"
    report_serializavel = {}
    for chave, res in resultados.items():
        report_serializavel[chave] = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in res.items()
        }
    with open(arq_json, "w", encoding="utf-8") as f:
        json.dump(report_serializavel, f, indent=2, ensure_ascii=False)
    print(f"   ✅ {arq_json.name}")

    # ── 7. Resumo final ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VALIDAÇÃO CONCLUÍDA!")
    print(f"\n  {'Falha':<30} {'Sev':>5} {'F1':>7} {'AUC':>7} {'Recall':>8}")
    print(f"  {'-'*60}")
    for row in linhas_tabela:
        npm_str = f"NPR={row['npr']}" if row["npr"] else "  D=10"
        print(f"  {row['falha']:<30} {row['severidade']:>5.2f} "
              f"{row['f1']:>7.3f} {row['auc_roc']:>7.3f} "
              f"{row['recall']:>8.3f}")

    # Melhor AUC geral
    melhor = max(linhas_tabela, key=lambda x: x["auc_roc"])
    print(f"\n  Melhor resultado: {melhor['falha']} "
          f"(sev={melhor['severidade']}) — AUC={melhor['auc_roc']:.3f}")
    print(f"\n  Próximo passo: dissertação — Capítulo 4 (Resultados)")
    print(f"{'='*60}")

    return True


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    executar_validacao()
