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
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.ml.estilo_graficos import (
    COR_EIXO,
    COR_GRADE,
    COR_TEXTO,
    COR_TEXTO_SEC,
    PALETA,
    TAM,
    adicionar_nota,
    aplicar_estilo,
)


aplicar_estilo()

MODEL_COLORS = {"ae_denso": PALETA[0], "ae_lstm": PALETA[1]}
MODEL_LABELS = {"ae_denso": "Autoencoder Denso", "ae_lstm": "AE-LSTM"}
COMPONENT_LABELS = {
    "contator_ac": "Contator AC",
    "igbt": "IGBT",
    "fusivel_ac": "Fusível AC",
}


def _save_pair(fig, base_path: Path, note: str) -> tuple[Path, Path]:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    adicionar_nota(fig, note)
    fig.savefig(
        png,
        metadata={"Software": "ALIAdo - Matplotlib", "Evidence": "E2/E3"},
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
    """Estimativas macro por ensaio e IC95%, com AUC-PR em primeiro plano."""

    order = [
        "auc_pr",
        "auc_roc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "f1",
    ]
    labels = {
        "auc_pr": "AUC-PR",
        "auc_roc": "AUC-ROC",
        "sensitivity": "Sensibilidade",
        "specificity": "Especificidade",
        "balanced_accuracy": "Acurácia balanceada",
        "mcc": "MCC",
        "f1": "F1",
    }
    frame = summary[summary["metric"].isin(order)].copy()
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
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
    lower_limit = min(-0.05, float(frame["ci95_low"].min()) - 0.03)
    ax.set_xlim(max(-1.0, lower_limit), 1.05)
    ax.set_yticks(y, [labels[item] for item in order])
    ax.invert_yaxis()
    ax.set_xlabel("Estimativa macro por ensaio (IC95%)")
    ax.set_title("Desempenho experimental nos 14 ensaios de falha")
    ax.legend(loc="lower right")
    fig.suptitle("Autoencoder Denso versus AE-LSTM — validação experimental E3")
    return _save_pair(
        fig,
        output,
        "GPVS-Faults F1L-F7M; bootstrap no nível do ensaio. AUC-PR é a métrica primária.",
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
        auc_pr_pooled = float(average_precision_score(y_true, values))
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
                f"{MODEL_LABELS[model]} · AP agregada={auc_pr_pooled:.3f}"
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
    fig.suptitle("Discriminação pré-falha versus pós-falha — execução de referência")
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
    fig.suptitle("Classificação no limiar p99 congelado — validação E3")
    return _save_pair(
        fig,
        output,
        "Contagens agregadas de janelas, apenas descritivas; execução de referência com pesos e limiar congelados.",
    )


def plot_e3_scenarios(scenarios: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """AUC-PR e sensibilidade por ensaio, preservando resultados negativos."""

    reference = scenarios[scenarios["is_reference"]].copy()
    experiments = [f"F{i}{mode}" for i in range(1, 8) for mode in "LM"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout="constrained", sharey=True)
    for ax, metric, title in (
        (axes[0], "auc_pr", "AUC-PR por ensaio"),
        (axes[1], "sensitivity", "Sensibilidade no limiar p99"),
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
    fig.suptitle("Resposta dos dois detectores aos 14 ensaios GPVS-Faults")
    return _save_pair(
        fig,
        output,
        "F1-F7 não participam do treino, da seleção de arquitetura nem da calibração do limiar.",
    )


def plot_e2_detection_curves(curves: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """Probabilidade empírica de detecção por magnitude e componente."""

    fig, axes = plt.subplots(1, 3, figsize=TAM["painel_3"], layout="constrained", sharey=True)
    for ax, component in zip(
        axes, ("contator_ac", "igbt", "fusivel_ac"), strict=True
    ):
        for model in ("ae_denso", "ae_lstm"):
            block = curves[
                curves["component"].eq(component) & curves["model"].eq(model)
            ].sort_values("magnitude")
            x = block["magnitude"].to_numpy(dtype=float)
            y = block["detection_probability"].to_numpy(dtype=float)
            low = block["ci95_low"].to_numpy(dtype=float)
            high = block["ci95_high"].to_numpy(dtype=float)
            ax.plot(x, y, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            ax.fill_between(x, low, high, color=MODEL_COLORS[model], alpha=0.13)
        ax.axhline(0.95, color=COR_EIXO, linestyle=":", linewidth=1.1)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), title=COMPONENT_LABELS[component])
        ax.set_xlabel("Magnitude injetada, $a_{det}$")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Probabilidade de detecção (IC95% Wilson)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Detecção de assinaturas FMECA por magnitude — validação sintética E2")
    return _save_pair(
        fig,
        output,
        "Mesmas janelas, magnitudes e perturbações para os dois modelos; eixo é magnitude, não tempo.",
    )


def plot_e2_smd(summary: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """Compara o SMD95 sem converter ausência de cruzamento em número."""

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    components = ["contator_ac", "igbt", "fusivel_ac"]
    y = np.arange(len(components), dtype=float)
    offsets = {"ae_denso": -0.14, "ae_lstm": 0.14}
    for model in ("ae_denso", "ae_lstm"):
        block = summary[summary["model"].eq(model)].set_index("component")
        for i, component in enumerate(components):
            value = block.loc[component, "smd95"]
            if pd.isna(value):
                ax.scatter(
                    [1.02],
                    [y[i] + offsets[model]],
                    marker="x",
                    color=MODEL_COLORS[model],
                    s=65,
                )
                ax.text(
                    1.04,
                    y[i] + offsets[model],
                    "não atingido",
                    va="center",
                    fontsize=8,
                    color=MODEL_COLORS[model],
                )
            else:
                ax.scatter(
                    [float(value)],
                    [y[i] + offsets[model]],
                    color=MODEL_COLORS[model],
                    s=55,
                    label=MODEL_LABELS[model] if i == 0 else None,
                )
    ax.set_xlim(0, 1.2)
    ax.set_yticks(y, [COMPONENT_LABELS[item] for item in components])
    ax.invert_yaxis()
    ax.set_xlabel("SMD95 — menor magnitude com limite inferior do IC95% ≥ 95%")
    ax.set_title("Magnitude mínima de detecção sustentada")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save_pair(
        fig,
        output,
        "Marcadores à direita indicam que o alvo não foi atingido até a assinatura nominal a=1.",
    )


def plot_e2_empirical_functions(
    empirical: pd.DataFrame, output: Path
) -> tuple[Path, Path]:
    """S_D(a), F_D(a) e risco discreto, sem semântica de confiabilidade física."""

    components = ["contator_ac", "igbt", "fusivel_ac"]
    functions = [
        ("survival", "$S_D(a)$", "Ainda não detectado"),
        ("cumulative_detection", "$F_D(a)$", "Detecção acumulada"),
        ("discrete_hazard", "$h_D(a)$", "Risco discreto de primeiro cruzamento"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=TAM["painel_9"], layout="constrained")
    for row, component in enumerate(components):
        for column, (field, symbol, title) in enumerate(functions):
            ax = axes[row, column]
            for model in ("ae_denso", "ae_lstm"):
                block = empirical[
                    empirical["component"].eq(component)
                    & empirical["model"].eq(model)
                ].sort_values("magnitude")
                ax.step(
                    block["magnitude"],
                    block[field],
                    where="post",
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            if row == 0:
                ax.set_title(f"{symbol} — {title}", fontsize=10)
            if column == 0:
                ax.set_ylabel(COMPONENT_LABELS[component])
            if row == 2:
                ax.set_xlabel("Magnitude da assinatura, a")
    axes[0, -1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Funções empíricas do primeiro cruzamento do detector — evidência E2")
    return _save_pair(
        fig,
        output,
        "Kaplan-Meier e risco discreto no eixo de magnitude; estas funções não são R(t), F(t), f(t) ou h(t) físicos.",
    )


def plot_e2_weibull_probability(
    points: pd.DataFrame, fits: pd.DataFrame, output: Path
) -> tuple[Path, Path]:
    """Papel de probabilidade: pontos observados e reta 2P apenas diagnóstica."""

    fig, axes = plt.subplots(1, 3, figsize=TAM["painel_3"], layout="constrained", sharey=True)
    for ax, component in zip(
        axes, ("contator_ac", "igbt", "fusivel_ac"), strict=True
    ):
        for model, marker in (("ae_denso", "o"), ("ae_lstm", "s")):
            block = points[
                points["component"].eq(component) & points["model"].eq(model)
            ]
            fit = fits[
                fits["component"].eq(component) & fits["model"].eq(model)
            ].iloc[0]
            status = (
                "ajuste Weibull aceito"
                if bool(fit["parametric_recommended"])
                else "ajuste Weibull não aceito"
            )
            ax.scatter(
                block["log_magnitude"],
                block["weibull_y"],
                s=24,
                marker=marker,
                facecolors="none",
                edgecolors=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]} · {status}",
            )
            if bool(fit["fit_converged"]) and len(block):
                x = np.linspace(block["log_magnitude"].min(), block["log_magnitude"].max(), 100)
                y = float(fit["beta"]) * x - float(fit["beta"]) * np.log(float(fit["eta"]))
                ax.plot(x, y, color=MODEL_COLORS[model], linestyle="--", linewidth=1.2)
        ax.set_title(COMPONENT_LABELS[component])
        ax.set_xlabel("ln(a)")
        ax.legend(fontsize=7.5, loc="best")
    axes[0].set_ylabel("ln[-ln(1 - F_D(a))]")
    fig.suptitle("Diagnóstico Weibull 2P da magnitude de detecção")
    return _save_pair(
        fig,
        output,
        (
            "Pontos são posições empíricas; retas são diagnósticas. A não "
            "aceitação do ajuste Weibull não constitui reprovação do detector."
        ),
    )


def generate_all(
    output_dir: Path,
    *,
    e3_summary: pd.DataFrame,
    e3_scores: pd.DataFrame,
    e3_confusion: pd.DataFrame,
    e3_scenarios: pd.DataFrame,
    e2_curves: pd.DataFrame,
    e2_summary: pd.DataFrame,
    e2_empirical: pd.DataFrame,
    e2_probability_points: pd.DataFrame,
    e2_fits: pd.DataFrame,
) -> list[Path]:
    """Gera o conjunto pequeno e completo de figuras publicáveis."""

    output_dir = Path(output_dir)
    paths: list[Path] = []
    pairs = (
        plot_e3_metric_summary(e3_summary, output_dir / "e3_metricas_macro"),
        plot_e3_discrimination_curves(e3_scores, output_dir / "e3_curvas_discriminacao"),
        plot_e3_confusion_matrices(e3_confusion, output_dir / "e3_matrizes_confusao"),
        plot_e3_scenarios(e3_scenarios, output_dir / "e3_resultados_por_ensaio"),
        plot_e2_detection_curves(e2_curves, output_dir / "e2_deteccao_por_magnitude"),
        plot_e2_smd(e2_summary, output_dir / "e2_smd95"),
        plot_e2_empirical_functions(e2_empirical, output_dir / "e2_funcoes_empiricas"),
        plot_e2_weibull_probability(
            e2_probability_points,
            e2_fits,
            output_dir / "e2_diagnostico_weibull",
        ),
    )
    for pair in pairs:
        paths.extend(pair)
    return paths


__all__ = [
    "COMPONENT_LABELS",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "generate_all",
    "plot_e2_detection_curves",
    "plot_e2_empirical_functions",
    "plot_e2_smd",
    "plot_e2_weibull_probability",
    "plot_e3_confusion_matrices",
    "plot_e3_discrimination_curves",
    "plot_e3_metric_summary",
    "plot_e3_scenarios",
]
