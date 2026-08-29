"""Figuras acadêmicas da comparação Denso versus AE-LSTM.

Cada figura recebe dados tabulares já calculados. O módulo não treina modelos,
não escolhe limiar e não recalcula métricas, o que mantém gráficos e tabelas
presos à mesma fonte numérica.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.ml.estilo_graficos import (
    COR_EIXO,
    COR_TEXTO,
    PALETA,
    TAM,
    adicionar_nota,
    aplicar_estilo,
)


aplicar_estilo()

MODEL_COLORS = {"ae_denso": PALETA[0], "ae_lstm": PALETA[1]}
MODEL_LABELS = {"ae_denso": "Autoencoder Denso", "ae_lstm": "AE-LSTM"}


def _save_pair(fig, base_path: Path, note: str) -> tuple[Path, Path]:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    adicionar_nota(fig, note)
    fig.savefig(
        png,
        metadata={"Software": "ALIAdo - Matplotlib", "Evidence": "E3"},
    )
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


def plot_e3_metric_summary(summary: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """Estimativas macro por ensaio conforme a hierarquia do pesquisador."""

    order = [
        "recall",
        "f1",
        "precision",
        "auc_roc",
        "auc_pr",
        "false_positive_rate",
    ]
    labels = {
        "recall": "Recall",
        "f1": "F1",
        "precision": "Precision",
        "auc_roc": "ROC-AUC (complementar)",
        "auc_pr": "PR-AUC (complementar)",
        "false_positive_rate": "Taxa de falso positivo",
    }
    frame = summary[summary["metric"].isin(order)].copy()
    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    y = np.arange(len(order), dtype=float)
    offsets = {"ae_denso": -0.15, "ae_lstm": 0.15}
    for model in ("ae_denso", "ae_lstm"):
        block = frame[frame["model"].eq(model)].set_index("metric").reindex(order)
        values = block["estimate"].to_numpy(dtype=float)
        lows = values - block["ci95_low"].to_numpy(dtype=float)
        highs = block["ci95_high"].to_numpy(dtype=float) - values
        ax.errorbar(
            values,
            y + offsets[model],
            xerr=np.vstack([lows, highs]),
            fmt="o",
            color=MODEL_COLORS[model],
            capsize=3,
            markersize=6,
            label=MODEL_LABELS[model],
        )
    ax.set_xlim(0.0, 1.05)
    ax.set_yticks(y, [labels[item] for item in order])
    ax.invert_yaxis()
    ax.set_xlabel("Estimativa macro por ensaio (IC95%)")
    ax.set_title("Desempenho experimental nos 14 ensaios de falha")
    ax.legend(loc="lower right")
    fig.suptitle("Autoencoder Denso versus AE-LSTM — validação experimental E3")
    return _save_pair(
        fig,
        output,
        "Recall, F1 e Precision são principais; AUCs são complementares. Bootstrap no nível do ensaio.",
    )


def plot_e3_discrimination_curves(
    scores: pd.DataFrame, output: Path
) -> tuple[Path, Path]:
    """Curvas ROC e precisão-revocação da execução de referência pré-fixada."""

    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")
    for model in ("ae_denso", "ae_lstm"):
        block = scores[scores["model"].eq(model)]
        y_true = block["y_true"].to_numpy(dtype=int)
        values = block["anomaly_index"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, values)
        precision, recall, _ = precision_recall_curve(y_true, values)
        auc_roc_pooled = float(roc_auc_score(y_true, values))
        auc_pr_pooled = float(auc(recall, precision))
        auc_roc_macro = float(block["auc_roc_macro"].iloc[0])
        auc_pr_macro = float(block["auc_pr_macro"].iloc[0])
        axes[0].plot(
            fpr,
            tpr,
            color=MODEL_COLORS[model],
            label=(
                f"{MODEL_LABELS[model]} · AUC agregada={auc_roc_pooled:.3f}"
                f" · macro={auc_roc_macro:.3f}"
            ),
        )
        axes[1].plot(
            recall,
            precision,
            color=MODEL_COLORS[model],
            label=(
                f"{MODEL_LABELS[model]} · PR-AUC agregada={auc_pr_pooled:.3f}"
                f" · macro={auc_pr_macro:.3f}"
            ),
        )

    axes[0].plot([0, 1], [0, 1], color=COR_EIXO, linestyle=":", label="Acaso")
    prevalence = float(scores["y_true"].mean())
    axes[1].axhline(
        prevalence,
        color=COR_EIXO,
        linestyle=":",
        label=f"Prevalência agregada={prevalence:.2f}",
    )
    for ax, title, xlabel, ylabel in (
        (axes[0], "Curva ROC", "Taxa de falso positivo", "Taxa de verdadeiro positivo"),
        (axes[1], "Curva precisão-revocação", "Revocação", "Precisão"),
    ):
        ax.set(xlim=(0, 1), ylim=(0, 1.02), title=title, xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Curvas complementares de discriminação pré-falha versus pós-falha"
    )
    return _save_pair(
        fig,
        output,
        "Curvas e AUC agregadas usam janelas apenas para descrição; valores macro e IC95% usam o ensaio como unidade.",
    )


def plot_e3_confusion_matrices(
    confusion: pd.DataFrame, output: Path
) -> tuple[Path, Path]:
    """Matrizes normalizadas por classe com contagens absolutas explícitas."""

    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")
    image = None
    for ax, model in zip(axes, ("ae_denso", "ae_lstm"), strict=True):
        row = confusion[confusion["model"].eq(model)].iloc[0]
        counts = np.asarray(
            [[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float
        )
        normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks([0, 1], ["Saudável", "Falha"])
        ax.set_yticks([0, 1], ["Saudável", "Falha"])
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        ax.set_title(MODEL_LABELS[model])
        for i in range(2):
            for j in range(2):
                value = normalized[i, j]
                ax.text(
                    j,
                    i,
                    f"{int(counts[i, j])}\n{value:.1%}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else COR_TEXTO,
                    fontsize=10,
                )
    fig.colorbar(image, ax=axes, label="Proporção dentro da classe real", shrink=0.8)
    requested = f"{float(confusion['threshold_requested_percentile'].iloc[0]):g}".replace(
        ".", ","
    )
    fig.suptitle(
        f"Matrizes de confusão no limiar saudável p{requested} — validação E3"
    )
    return _save_pair(
        fig,
        output,
        "Contagens agregadas de janelas, apenas descritivas; execução de referência com pesos e limiar congelados.",
    )


def plot_e3_scenarios(scenarios: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """Recall e F1 por ensaio, preservando resultados negativos."""

    reference = scenarios[scenarios["is_reference"]].copy()
    experiments = [f"F{i}{mode}" for i in range(1, 8) for mode in "LM"]
    fig, axes = plt.subplots(
        1, 2, figsize=TAM["painel_4"], layout="constrained", sharey=True
    )
    for ax, metric, title in (
        (axes[0], "recall", "Recall por ensaio no limiar calibrado"),
        (axes[1], "f1", "F1 por ensaio no limiar calibrado"),
    ):
        matrix = (
            reference.pivot(index="experiment", columns="model", values=metric)
            .reindex(experiments)
            .loc[:, ["ae_denso", "ae_lstm"]]
        )
        image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks([0, 1], ["Denso", "AE-LSTM"])
        ax.set_yticks(range(len(experiments)), experiments)
        ax.set_title(title)
        for i in range(len(experiments)):
            for j in range(2):
                value = float(matrix.iloc[i, j])
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=axes, label="Proporção", shrink=0.78)
    fig.suptitle("Resposta dos dois detectores aos 14 ensaios experimentais")
    return _save_pair(
        fig,
        output,
        "F1-F7 não participam do treino, da seleção de arquitetura nem da calibração do limiar.",
    )


def generate_all(
    output_dir: Path,
    *,
    e3_summary: pd.DataFrame,
    e3_scores: pd.DataFrame,
    e3_confusion: pd.DataFrame,
    e3_scenarios: pd.DataFrame,
) -> list[Path]:
    """Gera somente as figuras publicáveis da comparação dos modelos."""

    output_dir = Path(output_dir)
    paths: list[Path] = []
    pairs = (
        plot_e3_metric_summary(e3_summary, output_dir / "e3_metricas_macro"),
        plot_e3_discrimination_curves(e3_scores, output_dir / "e3_curvas_discriminacao"),
        plot_e3_confusion_matrices(e3_confusion, output_dir / "e3_matrizes_confusao"),
        plot_e3_scenarios(e3_scenarios, output_dir / "e3_resultados_por_ensaio"),
    )
    for pair in pairs:
        paths.extend(pair)
    return paths


__all__ = [
    "MODEL_COLORS",
    "MODEL_LABELS",
    "generate_all",
    "plot_e3_confusion_matrices",
    "plot_e3_discrimination_curves",
    "plot_e3_metric_summary",
    "plot_e3_scenarios",
]
