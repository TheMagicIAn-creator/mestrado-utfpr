"""Figuras essenciais da selecao e calibracao do autoencoder V2."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.estilo_graficos import (
    COR_ALERTA,
    COR_METODO,
    COR_NEUTRA,
    COR_REFERENCIA,
    COR_TEXTO_SEC,
    PALETA,
    TAM,
    aplicar_estilo,
    salvar_figura,
    tam_barras_h,
)
from src.ml.gpvs import FALHAS_CURTAS

aplicar_estilo()


def _ecdf(valores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(valores, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def plotar_selecao_modelo(
    execucoes: pd.DataFrame,
    resumo: pd.DataFrame,
    arquitetura_escolhida: str,
    caminho: Path,
) -> None:
    """Mostra variacao entre seeds e compromisso perda-complexidade."""

    ordem = resumo.sort_values("n_parametros")["arquitetura"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")

    for indice, nome in enumerate(ordem):
        bloco = execucoes[execucoes["arquitetura"].eq(nome)]
        deslocamento = np.linspace(-0.10, 0.10, len(bloco))
        cor = COR_METODO if nome == arquitetura_escolhida else COR_NEUTRA
        axes[0].scatter(
            indice + deslocamento,
            bloco["perda_validacao"],
            color=cor,
            s=34,
            alpha=0.82,
            label="Seeds" if indice == 0 else None,
            zorder=3,
        )
        mediana = float(bloco["perda_validacao"].median())
        axes[0].plot(
            [indice - 0.20, indice + 0.20],
            [mediana, mediana],
            color=cor,
            linewidth=3,
        )
    axes[0].set_xticks(range(len(ordem)), ordem, rotation=16, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Perda balanceada na validação saudável")
    axes[0].set_title("Variação entre cinco inicializações")

    for linha in resumo.itertuples(index=False):
        escolhido = linha.arquitetura == arquitetura_escolhida
        axes[1].scatter(
            linha.n_parametros,
            linha.mediana_validacao,
            color=COR_METODO if escolhido else PALETA[1],
            marker="D" if escolhido else "o",
            s=74 if escolhido else 48,
            zorder=3,
        )
        axes[1].annotate(
            linha.arquitetura,
            (linha.n_parametros, linha.mediana_validacao),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
            color=COR_TEXTO_SEC,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Número de parâmetros treináveis")
    axes[1].set_ylabel("Mediana da perda de validação")
    axes[1].set_title("Desempenho versus complexidade")
    axes[1].axhline(
        float(resumo["mediana_validacao"].min()) * 1.02,
        color=COR_REFERENCIA,
        linestyle=":",
        label="Faixa de 2% do melhor",
    )
    axes[1].legend(loc="best")

    fig.suptitle("Seleção da arquitetura do autoencoder em dados saudáveis GPVS-Faults")
    salvar_figura(
        fig,
        caminho,
        (
            "Arquitetura selecionada somente em F0L/F0M; ensaios F1-F7 não "
            "participam da escolha. Traços horizontais no painel esquerdo: medianas."
        ),
    )


def plotar_calibracao(
    scores_calibracao: np.ndarray,
    scores_teste: np.ndarray,
    limiar: float,
    resumo_excedencia: dict,
    caminho: Path,
) -> None:
    """Documenta o quantil finito e a excedencia no teste saudavel."""

    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")
    for valores, rotulo, cor in (
        (scores_calibracao, f"Calibracao (n={len(scores_calibracao)})", COR_METODO),
        (scores_teste, f"Teste saudavel (n={len(scores_teste)})", PALETA[1]),
    ):
        x, y = _ecdf(valores)
        axes[0].step(x, y, where="post", color=cor, label=rotulo)
    axes[0].axvline(
        limiar,
        color=COR_ALERTA,
        linestyle="--",
        label=f"Limiar = {limiar:.4g}",
    )
    if min(np.min(scores_calibracao), np.min(scores_teste)) > 0:
        axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.01)
    axes[0].set_xlabel("Erro de reconstrução balanceado por família")
    axes[0].set_ylabel("Probabilidade acumulada empírica")
    axes[0].set_title("Distribuição do escore saudável")
    axes[0].legend(loc="lower right")

    valores = [
        float(resumo_excedencia["calibracao"]["taxa_pct"]),
        float(resumo_excedencia["teste"]["taxa_pct"]),
    ]
    barras = axes[1].bar(
        [0, 1], valores, color=(COR_METODO, PALETA[1]), width=0.58
    )
    teste = resumo_excedencia["teste"]
    media = valores[1]
    erro = np.asarray(
        [[media - teste["ic95_low_pct"]], [teste["ic95_high_pct"] - media]]
    )
    axes[1].errorbar(
        [1], [media], yerr=erro, color=COR_TEXTO_SEC, capsize=5, fmt="none"
    )
    axes[1].axhline(1.0, color=COR_REFERENCIA, linestyle=":", label="Cauda nominal: 1%")
    axes[1].set_xticks([0, 1], ["Calibracao", "Teste saudavel"])
    axes[1].set_ylabel("Janelas acima do limiar (%)")
    axes[1].set_title("Excedência observada")
    teto = max(3.0, teste["ic95_high_pct"] * 1.18, max(valores) * 1.4)
    axes[1].set_ylim(0, teto)
    for barra, valor in zip(barras, valores, strict=True):
        axes[1].text(
            barra.get_x() + barra.get_width() / 2,
            valor + teto * 0.035,
            f"{valor:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].legend(loc="upper left")

    fig.suptitle("Calibração do limiar em operação saudável GPVS-Faults F0")
    salvar_figura(
        fig,
        caminho,
        (
            "O limiar usa ordem estatística finita na calibração; o IC95% de "
            "Wilson do teste é descritivo por janela e não remove dependência temporal."
        ),
    )


def plotar_desempenho_por_ensaio(
    cenarios: pd.DataFrame,
    caminho: Path,
) -> None:
    """Compara o detector congelado com PCA sem ocultar heterogeneidade."""

    metodos = (
        ("autoencoder_v2", "Autoencoder V2", COR_METODO, "o"),
        ("pca", "PCA", COR_NEUTRA, "s"),
    )
    metricas = (
        ("auc_roc", "AUC-ROC"),
        ("sensitivity", "Sensibilidade"),
        ("specificity", "Especificidade"),
    )
    ordem = [f"F{i}{modo}" for i in range(1, 8) for modo in "LM"]
    y = np.arange(len(ordem))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, tam_barras_h(len(ordem))[1]),
        sharey=True,
        layout="constrained",
    )
    for ax, (metrica, titulo) in zip(axes, metricas, strict=True):
        for deslocamento, (metodo, rotulo, cor, marcador) in zip(
            (-0.11, 0.11), metodos, strict=True
        ):
            bloco = (
                cenarios[cenarios["method"].eq(metodo)]
                .set_index("experiment")
                .reindex(ordem)
            )
            ax.scatter(
                bloco[metrica],
                y + deslocamento,
                color=cor,
                marker=marcador,
                s=38,
                label=rotulo if metrica == "auc_roc" else None,
                zorder=3,
            )
        ax.axvline(0.5, color=COR_REFERENCIA, linestyle=":", linewidth=1.1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Proporção")
        ax.set_title(titulo)
    axes[0].set_yticks(y, ordem)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    fig.suptitle("Desempenho do detector congelado nos 14 ensaios GPVS-Faults")
    salvar_figura(
        fig,
        caminho,
        (
            "Cada ponto representa um ensaio independente. AUC é descritiva por "
            "janela; sensibilidade e especificidade usam o limiar calibrado em F0."
        ),
    )


def plotar_mapa_ponto_operacional(
    cenarios: pd.DataFrame,
    caminho: Path,
) -> None:
    """Exibe sensibilidade e especificidade sem chamar o mapa de matriz."""

    bloco = cenarios[cenarios["method"].eq("autoencoder_v2")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), layout="constrained")
    imagem = None
    for ax, metrica, titulo in zip(
        axes,
        ("sensitivity", "specificity"),
        ("Sensibilidade pós-falha", "Especificidade pré-falha"),
        strict=True,
    ):
        matriz = bloco.pivot(index="fault", columns="mode", values=metrica)
        matriz = matriz.reindex(index=range(1, 8), columns=list("LM"))
        imagem = ax.imshow(
            matriz.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto"
        )
        ax.set_xticks((0, 1), ("IPPT (L)", "MPPT (M)"))
        ax.set_yticks(range(7), [f"F{i}" for i in range(1, 8)])
        ax.set_title(titulo)
        for i in range(7):
            for j in range(2):
                valor = float(matriz.iloc[i, j])
                ax.text(
                    j,
                    i,
                    f"{valor:.2f}",
                    ha="center",
                    va="center",
                    color="white" if valor < 0.58 else "black",
                    fontsize=9,
                )
    fig.colorbar(imagem, ax=axes, label="Proporção", fraction=0.035, pad=0.04)
    fig.suptitle("Ponto operacional do autoencoder V2 por falha e modo")
    salvar_figura(
        fig,
        caminho,
        (
            "F1-F7 são rótulos experimentais do GPVS-Faults; o mapa não é uma "
            "matriz de confusão multiclasse nem prova isolamento da causa."
        ),
    )


def plotar_matrizes_confusao(
    cenarios: pd.DataFrame,
    caminho: Path,
) -> None:
    """Matrizes 2x2 agregadas, normalizadas dentro da classe verdadeira."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, metodo, titulo in zip(
        axes,
        ("autoencoder_v2", "pca"),
        ("Autoencoder V2", "PCA"),
        strict=True,
    ):
        bloco = cenarios[cenarios["method"].eq(metodo)]
        tn = int(bloco["true_negatives"].sum())
        fp = int(bloco["false_positives"].sum())
        fn = int(bloco["false_negatives"].sum())
        tp = int(bloco["true_positives"].sum())
        contagens = np.asarray([[tn, fp], [fn, tp]], dtype=int)
        matriz = contagens / contagens.sum(axis=1, keepdims=True)
        imagem = ax.imshow(matriz, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks((0, 1), ("Saudável", "Anômala"))
        ax.set_yticks((0, 1), ("Pré-falha", "Pós-falha"))
        ax.set_xlabel("Classe indicada")
        ax.set_ylabel("Condição experimental")
        ax.set_title(titulo)
        for i in range(2):
            for j in range(2):
                valor = float(matriz[i, j])
                ax.text(
                    j,
                    i,
                    f"{valor:.1%}\n(n={contagens[i, j]})",
                    ha="center",
                    va="center",
                    color="white" if valor > 0.55 else "black",
                )
        fig.colorbar(imagem, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Classificação binária no limiar congelado")
    salvar_figura(
        fig,
        caminho,
        (
            "Normalização por linha. Contagens são agregadas apenas para descrição; "
            "a inferência macro usa os 14 ensaios, não janelas autocorrelacionadas."
        ),
    )


def _curvas_por_ensaio(
    scores: pd.DataFrame,
    metodo: str,
    *,
    curva: str,
    grade: np.ndarray,
) -> np.ndarray:
    curvas = []
    bloco_metodo = scores[scores["method"].eq(metodo)]
    for _, bloco in bloco_metodo.groupby("experiment", sort=True):
        avaliacao = bloco[bloco["evaluation_role"].isin(("negative", "positive"))]
        y = avaliacao["evaluation_role"].eq("positive").astype(int).to_numpy()
        score = avaliacao["anomaly_index"].to_numpy(dtype=float)
        if curva == "roc":
            x, valor, _ = roc_curve(y, score)
        else:
            valor, x, _ = precision_recall_curve(y, score)
            x, valor = x[::-1], valor[::-1]
        x_unico, indices = np.unique(x, return_index=True)
        curvas.append(np.interp(grade, x_unico, valor[indices]))
    return np.vstack(curvas)


def plotar_curvas_macro(
    scores: pd.DataFrame,
    macros: dict,
    caminho: Path,
) -> None:
    """ROC/PR macro: curva média e IQR entre ensaios, sem pseudo-replicação."""

    grade = np.linspace(0, 1, 301)
    metodos = (
        ("autoencoder_v2", "Autoencoder V2", COR_METODO),
        ("pca", "PCA", COR_NEUTRA),
    )
    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")
    for ax, curva, titulo, xrot, yrot in (
        (axes[0], "roc", "Curva ROC macro", "Taxa de falso positivo", "Sensibilidade"),
        (axes[1], "pr", "Curva precisão-revocação macro", "Revocação", "Precisão"),
    ):
        for metodo, rotulo, cor in metodos:
            matriz = _curvas_por_ensaio(scores, metodo, curva=curva, grade=grade)
            media = matriz.mean(axis=0)
            q25, q75 = np.percentile(matriz, (25, 75), axis=0)
            chave = "auc_roc" if curva == "roc" else "average_precision"
            valor_macro = macros[metodo][chave]["mean"]
            abreviacao = "AUC" if curva == "roc" else "AP"
            ax.plot(
                grade,
                media,
                color=cor,
                label=f"{rotulo} ({abreviacao}={valor_macro:.3f})",
            )
            ax.fill_between(grade, q25, q75, color=cor, alpha=0.14)
        if curva == "roc":
            ax.plot((0, 1), (0, 1), color=COR_REFERENCIA, linestyle=":", label="Acaso")
        else:
            avaliacao = scores[
                scores["evaluation_role"].isin(("negative", "positive"))
            ]
            prevalencias = avaliacao.groupby("experiment")["evaluation_role"].apply(
                lambda serie: float(serie.eq("positive").mean())
            )
            ax.axhline(
                float(prevalencias.mean()),
                color=COR_REFERENCIA,
                linestyle=":",
                label="Prevalência média do protocolo",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel(xrot)
        ax.set_ylabel(yrot)
        ax.set_title(titulo)
        ax.legend(loc="best")
    fig.suptitle("Discriminação por janela com agregação macro dos 14 ensaios")
    salvar_figura(
        fig,
        caminho,
        (
            "Linhas: média das curvas por ensaio; faixas: intervalo interquartil. "
            "A prevalência pós-falha é imposta pelo protocolo e não representa campo."
        ),
    )


def plotar_series_temporais(
    scores: pd.DataFrame,
    caminho: Path,
) -> None:
    """Alinha cada ensaio ao início pós-falha e usa escala local por falha."""

    bloco = scores[scores["method"].eq("autoencoder_v2")]
    fig, axes = plt.subplots(7, 1, figsize=(13, 18), layout="constrained")
    for falha, ax in enumerate(axes, start=1):
        valores_painel = []
        for modo, cor in zip("LM", (COR_METODO, PALETA[1]), strict=True):
            serie = bloco[(bloco["fault"].eq(falha)) & (bloco["mode"].eq(modo))]
            x = serie["time_from_nominal_midpoint_s"].to_numpy(dtype=float)
            y = np.maximum(serie["anomaly_index"].to_numpy(dtype=float), 1e-6)
            valores_painel.extend(y.tolist())
            ax.plot(x, y, color=cor, linewidth=1.15, label=f"Modo {modo}")
        ax.axvline(0, color=COR_ALERTA, linestyle="--", linewidth=1.0)
        ax.axhline(1, color=COR_REFERENCIA, linestyle=":", linewidth=1.1)
        positivos = np.asarray(valores_painel, dtype=float)
        baixo = max(1e-4, float(np.quantile(positivos, 0.005)) / 1.6)
        alto = max(2.0, float(np.quantile(positivos, 0.995)) * 1.6)
        ax.set_yscale("log")
        ax.set_ylim(baixo, alto)
        ax.set_ylabel("Índice")
        ax.set_title(f"F{falha}: {FALHAS_CURTAS[falha]}", loc="left", fontsize=10)
    axes[-1].set_xlabel("Tempo relativo à fronteira nominal de 50% do registro (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle("Resposta temporal do autoencoder V2 nos ensaios GPVS-Faults")
    salvar_figura(
        fig,
        caminho,
        (
            "Linha vertical: fronteira nominal informada pela fonte (50% do registro); "
            "linha horizontal: limiar. Os CSVs não têm canal de disparo instrumentado."
        ),
    )


def plotar_contribuicoes_familias(
    contribuicoes: pd.DataFrame,
    caminho: Path,
) -> None:
    """Mostra a razão pós/pré do resíduo de cada família física."""

    familias = [
        "operacao_cc",
        "corrente_ca",
        "tensao_ca",
        "potencia_ca",
    ]
    ordem = [f"F{i}{modo}" for i in range(1, 8) for modo in "LM"]
    matriz = (
        contribuicoes.set_index("experiment").reindex(ordem)[familias].astype(float)
    )
    razao = np.maximum(matriz.to_numpy(), np.finfo(float).tiny)
    log2_razao = np.log2(razao)
    limite = max(1.0, float(np.quantile(np.abs(log2_razao), 0.95)))
    limite = min(limite, 8.0)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    imagem = ax.imshow(
        np.clip(log2_razao, -limite, limite),
        cmap="RdBu_r",
        vmin=-limite,
        vmax=limite,
        aspect="auto",
    )
    ax.set_xticks(
        range(4),
        ("Operação CC", "Corrente CA", "Tensão CA", "Potência CA"),
        rotation=15,
        ha="right",
    )
    ax.set_yticks(range(len(ordem)), ordem)
    for i in range(len(ordem)):
        for j in range(4):
            valor = float(razao[i, j])
            texto = f"{valor:.1f}×" if 0.1 <= valor < 100 else f"{valor:.1e}×"
            ax.text(j, i, texto, ha="center", va="center", fontsize=8)
    fig.colorbar(imagem, ax=ax, label="log₂(mediana pós / mediana pré)")
    ax.set_title("Mudança do resíduo por família física após a falha")
    fig.suptitle("Auditabilidade física do escore do autoencoder V2")
    salvar_figura(
        fig,
        caminho,
        (
            "Valores são razões de medianas no autoencoder canônico. Azul indica "
            "redução, vermelho aumento; contribuições somam o escore balanceado."
        ),
    )
