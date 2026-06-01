"""
experimentos_artigos.py — Al IAdo PV
Banco de experimentos de ML fundamentados nos artigos-base da dissertação.

Cada artigo vira um "experimento" reproduzível: um dataset, uma tarefa
(classificação supervisionada OU detecção de anomalia) e um conjunto de
modelos. O pesquisador escolhe na barra lateral, roda, e os resultados são
salvos em resultados/experimentos/<key>/.

Princípios:
- Espelha o padrão de `src/ml/pipeline.py` (registry declarativo).
- Degradação honesta: um modelo que exige biblioteca não instalada NÃO some —
  fica registrado e é reportado como "requer <lib>" em vez de quebrar a corrida.
- Métricas e artefatos padronizados para permitir comparação entre artigos.

Artigos-base:
  1. Ghoneim, Rashed & Elkalashy (2021)  — RF, AdaBoost, LogReg, NaiveBayes, CN2
  2. Francisti et al. (2025)             — RF (reg+clf) + Z-score
  3. Ibrahim et al. (2022)               — AE-LSTM, Prophet, Isolation Forest
  4. Sharma et al. (2026)                — Isolation Forest + PPO; baselines RNN/ANN/CNN/KNN/SVM
  5. Stender, Wallscheid & Böcker (2020) — descrição do dataset (Paderborn)
  6. Ahirwar & Nandanwar (2025)          — híbrido AE-LSTM + Prophet + IForest + BayesOpt

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.core.config import RAIZ_PROJETO

PASTA_EXPERIMENTOS = RAIZ_PROJETO / "resultados" / "experimentos"
METRICAS_BASE = ("accuracy", "precision", "recall", "f1", "auc", "specificity")
METRICAS_GRAFICO = ("accuracy", "precision", "recall", "f1", "auc", "specificity")


# ============================================================
# DISPONIBILIDADE DE BIBLIOTECAS (degradação honesta)
# ============================================================

def lib_disponivel(nome: str | None) -> bool:
    """True se a biblioteca está instalada (ou se nenhuma é exigida)."""
    if not nome:
        return True
    try:
        return importlib.util.find_spec(nome) is not None
    except (ImportError, ValueError):
        return False


# ============================================================
# METRICAS PADRONIZADAS
# ============================================================

def _specificity_macro(y_true, y_pred) -> float:
    """Specificity macro no esquema one-vs-rest."""
    import numpy as np
    from sklearn.metrics import confusion_matrix

    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    valores = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        valores.append(float(tn / denom) if denom else 0.0)
    return float(np.mean(valores)) if valores else 0.0


def _auc_seguro(y_true, y_score) -> float | None:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if y_score is None:
        return None
    try:
        y_true_arr = np.asarray(y_true)
        score_arr = np.asarray(y_score)
        classes = np.unique(y_true_arr)
        if len(classes) < 2:
            return None
        if score_arr.ndim == 1:
            if len(classes) != 2:
                return None
            return float(roc_auc_score(y_true_arr, score_arr))
        return float(
            roc_auc_score(
                y_true_arr,
                score_arr,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        return None


def _metricas_classificacao(y_true, y_pred, y_score=None) -> dict:
    """Schema unico para classificacao e deteccao por ponto de operacao."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    media = "binary" if len(labels) == 2 and set(labels).issubset({0, 1}) else "macro"

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average=media, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average=media, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average=media, zero_division=0)),
        "auc": _auc_seguro(y_true_arr, y_score),
        "specificity": _specificity_macro(y_true_arr, y_pred_arr),
        "amostras": int(len(y_true_arr)),
        "n_classes": int(len(labels)),
        "classes": [str(x) for x in labels.tolist()],
        "matriz_confusao": confusion_matrix(y_true_arr, y_pred_arr, labels=labels).astype(int).tolist(),
        "disponivel": True,
    }


# ============================================================
# ESPECIFICAÇÃO DE MODELO E EXPERIMENTO
# ============================================================

@dataclass(frozen=True)
class ModeloSpec:
    """Um modelo dentro de um experimento."""
    nome: str
    familia: str = ""          # "arvore", "linear", "bayes", "rede", "anomalia", "rl", "regras"
    requer: str | None = None  # nome do módulo importável exigido (None = sempre disponível)

    @property
    def disponivel(self) -> bool:
        return lib_disponivel(self.requer)


@dataclass(frozen=True)
class ExperimentoArtigo:
    """Um experimento reproduzível ancorado em um artigo-base."""
    key: str
    artigo: str            # título curto
    referencia: str        # "Autor (ano)"
    ano: int
    dataset: str           # "PV Farms" | "Paderborn" | "—"
    tarefa: str            # "classificacao" | "anomalia" | "dataset"
    modelos: tuple[ModeloSpec, ...]
    descricao: str
    runner: str = ""       # nome da função runner neste módulo ("" = sem execução)

    def pasta(self) -> Path:
        return PASTA_EXPERIMENTOS / self.key

    def modelos_disponiveis(self) -> list[ModeloSpec]:
        return [m for m in self.modelos if m.disponivel]

    def modelos_indisponiveis(self) -> list[ModeloSpec]:
        return [m for m in self.modelos if not m.disponivel]

    def executavel(self) -> bool:
        """Há ao menos um modelo rodável e um runner definido?"""
        return bool(self.runner) and bool(self.modelos_disponiveis())

    def resultado_existe(self) -> bool:
        return (self.pasta() / "resultado.json").exists()


# ============================================================
# REGISTRO DOS EXPERIMENTOS (os 6 artigos)
# ============================================================

REGISTRO: dict[str, ExperimentoArtigo] = {
    "ghoneim": ExperimentoArtigo(
        key="ghoneim",
        artigo="Fault Detection Algorithms for Service Continuity in PV Farms",
        referencia="Ghoneim, Rashed & Elkalashy (2021)",
        ano=2021,
        dataset="PV Farms",
        tarefa="classificacao",
        descricao=(
            "Classificação supervisionada de falhas CC (Normal, F1 string, "
            "F2 string-terra, F3 string-string) no dataset rotulado PV Farms."
        ),
        modelos=(
            ModeloSpec("Random Forest", "arvore"),
            ModeloSpec("AdaBoost", "arvore"),
            ModeloSpec("Regressão Logística", "linear"),
            ModeloSpec("Naive Bayes", "bayes"),
            ModeloSpec("CN2 (indução de regras)", "regras", requer="Orange"),
        ),
        runner="executar_classificacao_supervisionada",
    ),
    "francisti": ExperimentoArtigo(
        key="francisti",
        artigo="Predictive Modeling and Anomaly Detection in Solar PV Inverters",
        referencia="Francisti et al. (2025)",
        ano=2025,
        dataset="Paderborn",
        tarefa="anomalia",
        descricao=(
            "Detecção de anomalia no inversor saudável (Paderborn) com Random "
            "Forest e limiar estatístico Z-score, avaliada contra falhas "
            "sintéticas injetadas (ground truth do FMEA)."
        ),
        modelos=(
            ModeloSpec("Z-score (estatístico)", "anomalia"),
            ModeloSpec("Random Forest (anomalia)", "arvore"),
        ),
        runner="executar_anomalia",
    ),
    "ibrahim": ExperimentoArtigo(
        key="ibrahim",
        artigo="Machine Learning Schemes for Anomaly Detection in Solar Power Plants",
        referencia="Ibrahim et al. (2022)",
        ano=2022,
        dataset="Paderborn",
        tarefa="anomalia",
        descricao=(
            "Esquemas de detecção de anomalia: Isolation Forest, Autoencoder "
            "LSTM e Facebook Prophet, avaliados contra falhas injetadas."
        ),
        modelos=(
            ModeloSpec("Isolation Forest", "anomalia"),
            ModeloSpec("AE-LSTM", "rede", requer="torch"),
            ModeloSpec("Facebook Prophet", "anomalia", requer="prophet"),
        ),
        runner="executar_anomalia",
    ),
    "sharma": ExperimentoArtigo(
        key="sharma",
        artigo="Self-Tuning RL-Driven Isolation Forest for Anomaly Detection",
        referencia="Sharma et al. (2026)",
        ano=2026,
        dataset="Paderborn",
        tarefa="anomalia",
        descricao=(
            "Isolation Forest auto-ajustável por RL (PPO) frente a baselines "
            "RNN, ANN, CNN, KNN e SVM, na detecção de anomalia do inversor."
        ),
        modelos=(
            ModeloSpec("Isolation Forest", "anomalia"),
            ModeloSpec("KNN", "vizinhanca"),
            ModeloSpec("SVM", "kernel"),
            ModeloSpec("ANN (MLP)", "rede"),
            ModeloSpec("RNN", "rede", requer="torch"),
            ModeloSpec("CNN", "rede", requer="torch"),
            ModeloSpec("Isolation Forest + PPO", "rl", requer="stable_baselines3"),
        ),
        runner="executar_anomalia",
    ),
    "stender": ExperimentoArtigo(
        key="stender",
        artigo="Data Set Description: Three-Phase IGBT Two-Level Inverter",
        referencia="Stender, Wallscheid & Böcker (2020)",
        ano=2020,
        dataset="Paderborn",
        tarefa="dataset",
        descricao=(
            "Artigo de descrição do dataset de Paderborn (inversor IGBT "
            "trifásico saudável, 10 kHz). É a referência de normalidade — "
            "não propõe modelo, fornece os dados de treino do detector."
        ),
        modelos=(),
        runner="",
    ),
    "ahirwar": ExperimentoArtigo(
        key="ahirwar",
        artigo="Enhanced Anomaly Detection Using Hybrid ML Techniques",
        referencia="Ahirwar & Nandanwar (2025)",
        ano=2025,
        dataset="Paderborn",
        tarefa="anomalia",
        descricao=(
            "Abordagem híbrida: combina Autoencoder-LSTM, Facebook Prophet e "
            "Isolation Forest, com otimização bayesiana de hiperparâmetros."
        ),
        modelos=(
            ModeloSpec("Isolation Forest", "anomalia"),
            ModeloSpec("AE-LSTM", "rede", requer="torch"),
            ModeloSpec("Facebook Prophet", "anomalia", requer="prophet"),
            ModeloSpec("Híbrido (voto)", "ensemble"),
        ),
        runner="executar_anomalia",
    ),
}

ORDEM_EXPERIMENTOS = list(REGISTRO.keys())


# ============================================================
# CONSULTAS AO REGISTRO
# ============================================================

def get_experimento(key: str) -> ExperimentoArtigo:
    try:
        return REGISTRO[key]
    except KeyError as exc:
        raise ValueError(f"Experimento desconhecido: {key}") from exc


def listar_experimentos() -> list[ExperimentoArtigo]:
    return [REGISTRO[k] for k in ORDEM_EXPERIMENTOS]


def catalogo_experimentos_md() -> str:
    """Markdown legível com os experimentos e o status de cada modelo."""
    linhas = ["## Experimentos por artigo-base\n"]
    for exp in listar_experimentos():
        linhas.append(f"\n### {exp.referencia} — {exp.dataset}")
        linhas.append(f"_{exp.descricao}_\n")
        if not exp.modelos:
            linhas.append("- (sem modelo — cartão de dataset)")
            continue
        for m in exp.modelos:
            if m.disponivel:
                linhas.append(f"- ✅ {m.nome}")
            else:
                linhas.append(f"- ⛔ {m.nome} — requer `{m.requer}` (não instalado)")
    return "\n".join(linhas)


# ============================================================
# SALVAMENTO DE ARTEFATOS (padrão comum a todos os experimentos)
# ============================================================

def _resultado_serializavel(resultado: dict) -> dict:
    """Remove campos privados e converte objetos numericos para JSON/CSV."""
    import json

    def limpar(valor):
        if isinstance(valor, dict):
            return {k: limpar(v) for k, v in valor.items() if not str(k).startswith("_")}
        if isinstance(valor, (list, tuple)):
            return [limpar(v) for v in valor]
        try:
            import numpy as np

            if isinstance(valor, np.generic):
                return valor.item()
            if isinstance(valor, np.ndarray):
                return valor.tolist()
        except Exception:
            pass
        try:
            json.dumps(valor)
            return valor
        except TypeError:
            return str(valor)

    return limpar(resultado)


def _salvar_metricas_csv(exp: ExperimentoArtigo, resultado: dict) -> None:
    try:
        import pandas as pd
    except Exception:
        return

    linhas = []
    for nome, m in resultado.get("modelos", {}).items():
        linha = {
            "experimento": exp.key,
            "referencia": exp.referencia,
            "tarefa": exp.tarefa,
            "modelo": nome,
            "disponivel": m.get("disponivel", True),
            "motivo": m.get("motivo", ""),
        }
        for k, v in m.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                linha[k] = v
        linhas.append(linha)

    if linhas:
        pd.DataFrame(linhas).to_csv(exp.pasta() / "metricas.csv", index=False)


def _origem_dados(exp: ExperimentoArtigo) -> dict:
    """Documenta se o experimento usa dados locais ou so referencia do artigo."""
    if exp.key == "ghoneim":
        return {
            "tipo": "dataset_local_rotulado",
            "descricao": (
                "Usa os arquivos locais train_data.csv e test_data.csv em dados/brutos. "
                "O artigo define a metodologia/base PV Farms; os numeros sao recalculados "
                "no repositorio, nao copiados do paper."
            ),
            "arquivos": [
                "dados/brutos/train_data.csv",
                "dados/brutos/test_data.csv",
            ],
        }
    if exp.tarefa == "anomalia":
        return {
            "tipo": "dataset_local_paderborn_com_falhas_sinteticas",
            "descricao": (
                "Usa features locais do Paderborn extraidas de Inverter_Data_Set.csv. "
                "Como o Paderborn e saudavel, as anomalias avaliadas sao sinteticas, "
                "geradas no pipeline para criar ground truth. O artigo inspira os modelos "
                "e a metodologia; os dados avaliados sao os do repositorio."
            ),
            "arquivos": [
                "dados/brutos/Inverter_Data_Set.csv",
                "dados/processados/features_paderborn.parquet",
            ],
        }
    return {
        "tipo": "referencia_metodologica",
        "descricao": "Cartao de referencia; nao executa treinamento.",
        "arquivos": [],
    }


def _salvar_resultado(exp: ExperimentoArtigo, resultado: dict) -> Path:
    """Grava resultado.json, metricas.csv e relatorio.txt."""
    import json

    pasta = exp.pasta()
    pasta.mkdir(parents=True, exist_ok=True)
    resultado = _resultado_serializavel(resultado)

    (pasta / "resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _salvar_metricas_csv(exp, resultado)

    linhas = [
        "=" * 64,
        f"  EXPERIMENTO: {exp.referencia}",
        f"  {exp.artigo}",
        f"  Dataset: {exp.dataset} | Tarefa: {exp.tarefa}",
        "=" * 64,
        "",
        f"Origem dos dados: {resultado.get('origem_dados', {}).get('descricao', '-')}",
        "",
        f"Métrica principal: {resultado.get('metrica_principal', '-')}",
        "",
    ]
    for nome, m in resultado.get("modelos", {}).items():
        if not m.get("disponivel", True):
            linhas.append(f"- {nome}: INDISPONÍVEL ({m.get('motivo', 'requer biblioteca')})")
            continue
        partes = [
            f"{k}={v:.4f}"
            for k, v in m.items()
            if k in METRICAS_BASE and isinstance(v, (int, float))
        ]
        if isinstance(m.get("anomalias_detectadas"), int):
            partes.append(f"anomalias_detectadas={m['anomalias_detectadas']}")
        linhas.append(f"- {nome}: " + ", ".join(partes))
    linhas += [
        "",
        f"MELHOR MODELO: {resultado.get('melhor_modelo', '-')} "
        f"({resultado.get('metrica_principal', '')}="
        f"{resultado.get('melhor_valor', float('nan')):.4f})",
        "=" * 64,
    ]
    (pasta / "relatorio.txt").write_text("\n".join(linhas), encoding="utf-8")
    return pasta


def _grafico_comparacao_legacy(exp: ExperimentoArtigo, resultado: dict) -> Path | None:
    """Barras comparando os modelos pela métrica principal (PNG via matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    metrica = resultado.get("metrica_principal", "f1")
    itens = [
        (nome, m.get(metrica))
        for nome, m in resultado.get("modelos", {}).items()
        if m.get("disponivel", True) and isinstance(m.get(metrica), (int, float))
    ]
    if not itens:
        return None
    itens.sort(key=lambda x: x[1], reverse=True)
    nomes = [i[0] for i in itens]
    valores = [i[1] for i in itens]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cores = ["#2E7D32" if v == max(valores) else "#4C72B0" for v in valores]
    ax.barh(nomes[::-1], valores[::-1], color=cores[::-1])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metrica)
    ax.set_title(f"{exp.referencia} — comparação de modelos")
    for i, v in enumerate(valores[::-1]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()

    caminho = exp.pasta() / "comparacao.png"
    fig.savefig(caminho, dpi=110)
    plt.close(fig)
    return caminho


def _slug_modelo(nome: str) -> str:
    """Nome estavel para arquivos de artefatos por modelo."""
    import re
    import unicodedata

    texto = unicodedata.normalize("NFD", nome.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    return texto or "modelo"


def _registrar_grafico_modelo(modelo: dict, chave: str, caminho: Path) -> None:
    modelo.setdefault("graficos", [])
    caminho_abs = str(caminho.resolve())
    if caminho_abs not in modelo["graficos"]:
        modelo["graficos"].append(caminho_abs)
    modelo[chave] = caminho_abs


def _grafico_metricas_modelo(exp: ExperimentoArtigo, nome: str, modelo: dict, plt, np) -> Path | None:
    metricas = [
        met for met in METRICAS_GRAFICO
        if isinstance(modelo.get(met), (int, float))
    ]
    if not metricas:
        return None

    valores = [float(modelo[met]) for met in metricas]
    cores = ["#2F80ED", "#27AE60", "#F2994A", "#9B51E0", "#EB5757", "#56CCF2"][:len(metricas)]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    barras = ax.bar(metricas, valores, color=cores)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("valor")
    ax.set_title(f"{exp.referencia} - {nome}")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            min(1.03, valor + 0.02),
            f"{valor:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    linhas = []
    if isinstance(modelo.get("anomalias_detectadas"), int):
        linhas.append(f"Anomalias detectadas: {modelo['anomalias_detectadas']}")
    if isinstance(modelo.get("anomalias_reais"), int):
        linhas.append(f"Anomalias reais: {modelo['anomalias_reais']}")
    if modelo.get("ponto_operacao"):
        rotulos_ponto = {
            "limiar_otimo_score": "limiar otimizado",
            "decisao_nativa_modelo": "decisao nativa",
        }
        linhas.append(f"Ponto: {rotulos_ponto.get(modelo['ponto_operacao'], modelo['ponto_operacao'])}")
    if linhas:
        fig.text(
            0.02,
            0.03,
            "\n".join(linhas),
            ha="left",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0.15 if linhas else 0, 1, 1))
    caminho = exp.pasta() / f"modelo_{_slug_modelo(nome)}_metricas.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    _registrar_grafico_modelo(modelo, "grafico_metricas", caminho)
    return caminho


def _grafico_matriz_modelo(exp: ExperimentoArtigo, nome: str, modelo: dict, plt, np) -> Path | None:
    if not modelo.get("matriz_confusao"):
        return None

    cm = np.asarray(modelo["matriz_confusao"], dtype=int)
    if cm.ndim != 2 or cm.size == 0:
        return None

    labels = modelo.get("classes") or [str(i) for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=(max(5.2, len(labels) * 0.9), max(4.8, len(labels) * 0.8)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matriz de confusao - {nome}")
    ax.set_xlabel("predito")
    ax.set_ylabel("real")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    limite = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cor = "white" if cm[i, j] > limite else "#111111"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=cor)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    caminho = exp.pasta() / f"modelo_{_slug_modelo(nome)}_matriz_confusao.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    _registrar_grafico_modelo(modelo, "grafico_matriz_confusao", caminho)
    return caminho


def _grafico_comparacao(exp: ExperimentoArtigo, resultado: dict) -> list[Path]:
    """Gera PNGs comparativos e artefatos individuais por modelo."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    exp.pasta().mkdir(parents=True, exist_ok=True)
    graficos: list[Path] = []
    modelos = [
        (nome, m)
        for nome, m in resultado.get("modelos", {}).items()
        if m.get("disponivel", True)
    ]

    for nome, modelo in modelos:
        graf = _grafico_metricas_modelo(exp, nome, modelo, plt, np)
        if graf:
            graficos.append(graf)
        graf = _grafico_matriz_modelo(exp, nome, modelo, plt, np)
        if graf:
            graficos.append(graf)

    metricas = [
        met for met in METRICAS_GRAFICO
        if any(isinstance(m.get(met), (int, float)) for _, m in modelos)
    ]
    if modelos and metricas:
        nomes = [n for n, _ in modelos]
        x = np.arange(len(nomes))
        largura = min(0.16, 0.78 / max(1, len(metricas)))
        fig, ax = plt.subplots(figsize=(max(9, len(nomes) * 1.25), 5.2))
        for i, met in enumerate(metricas):
            vals = [
                float(m.get(met)) if isinstance(m.get(met), (int, float)) else np.nan
                for _, m in modelos
            ]
            ax.bar(x + (i - (len(metricas) - 1) / 2) * largura, vals, largura, label=met)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("valor")
        ax.set_title(f"{exp.referencia} - comparacao multi-metrica")
        ax.set_xticks(x)
        ax.set_xticklabels(nomes, rotation=25, ha="right")
        ax.legend(ncol=min(3, len(metricas)), fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        caminho = exp.pasta() / "comparacao_metricas.png"
        fig.savefig(caminho, dpi=120)
        plt.close(fig)
        graficos.append(caminho)

    itens_anomalia = [
        (nome, int(m["anomalias_detectadas"]))
        for nome, m in modelos
        if isinstance(m.get("anomalias_detectadas"), int)
    ]
    if itens_anomalia:
        nomes = [n for n, _ in itens_anomalia]
        valores = [v for _, v in itens_anomalia]
        fig, ax = plt.subplots(figsize=(max(8, len(nomes) * 1.1), 4.5))
        ax.bar(nomes, valores, color="#7B4CC2")
        ax.set_ylabel("anomalias detectadas")
        ax.set_title(f"{exp.referencia} - anomalias no ponto de operacao")
        ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(valores):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        caminho = exp.pasta() / "anomalias_detectadas.png"
        fig.savefig(caminho, dpi=120)
        plt.close(fig)
        graficos.append(caminho)

    melhor = resultado.get("melhor_modelo")
    modelo_cm = resultado.get("modelos", {}).get(melhor, {})
    if not modelo_cm.get("matriz_confusao"):
        for _, m in modelos:
            if m.get("matriz_confusao"):
                modelo_cm = m
                break
    if modelo_cm.get("matriz_confusao"):
        caminho_individual = modelo_cm.get("grafico_matriz_confusao")
        if caminho_individual and Path(caminho_individual).exists():
            caminho = exp.pasta() / "matriz_confusao.png"
            import shutil
            shutil.copyfile(caminho_individual, caminho)
            graficos.append(caminho)

    return graficos


# ============================================================
# RUNNER 1 — CLASSIFICAÇÃO SUPERVISIONADA (Ghoneim, PV Farms)
# ============================================================

def _carregar_pv_farms():
    import pandas as pd

    base = RAIZ_PROJETO / "dados" / "brutos"
    df_tr = pd.read_csv(base / "train_data.csv", sep=";")
    df_te = pd.read_csv(base / "test_data.csv", sep=";")
    X_tr, y_tr = df_tr.drop(columns=["class"]), df_tr["class"]
    X_te, y_te = df_te.drop(columns=["class"]), df_te["class"]
    return X_tr, y_tr, X_te, y_te


def _estimador_supervisionado(nome: str):
    """Mapeia o nome do modelo para um estimador sklearn (ou None se especial)."""
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    mapa = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
    }
    return mapa.get(nome)


def _cn2_orange(X_tr, y_tr, X_te, y_te) -> dict | None:
    """CN2 (indução de regras) via Orange3, se instalado. None se indisponível."""
    if not lib_disponivel("Orange"):
        return None
    try:
        import numpy as np
        import Orange
        classes = sorted(set(map(int, y_tr)))
        dominio = Orange.data.Domain(
            [Orange.data.ContinuousVariable(c) for c in map(str, X_tr.columns)],
            Orange.data.DiscreteVariable("class", values=[str(c) for c in classes]),
        )
        tab_tr = Orange.data.Table.from_numpy(
            dominio, np.asarray(X_tr, float),
            np.array([classes.index(int(v)) for v in y_tr], float),
        )
        aprendiz = Orange.classification.CN2Learner()
        modelo = aprendiz(tab_tr)
        pred_idx = modelo(np.asarray(X_te, float))
        y_pred = np.array([classes[int(i)] for i in pred_idx])
        y_true = np.asarray(list(map(int, y_te)))
        return _metricas_classificacao(y_true, y_pred)
    except Exception as exc:  # noqa: BLE001
        return {"disponivel": False, "motivo": f"erro no Orange/CN2: {exc}"}


def executar_classificacao_supervisionada(exp: ExperimentoArtigo, progresso=None) -> dict:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    if progresso:
        progresso("Carregando dataset PV Farms...")
    X_tr, y_tr, X_te, y_te = _carregar_pv_farms()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelos_out: dict[str, dict] = {}
    for spec in exp.modelos:
        if not spec.disponivel:
            modelos_out[spec.nome] = {
                "disponivel": False,
                "motivo": f"requer {spec.requer}",
            }
            continue

        if spec.nome.startswith("CN2"):
            if progresso:
                progresso("Treinando CN2 (Orange)...")
            res_cn2 = _cn2_orange(X_tr, y_tr, X_te, y_te)
            modelos_out[spec.nome] = res_cn2 or {
                "disponivel": False, "motivo": "Orange indisponível",
            }
            continue

        est = _estimador_supervisionado(spec.nome)
        if est is None:
            modelos_out[spec.nome] = {"disponivel": False, "motivo": "não implementado"}
            continue

        if progresso:
            progresso(f"Treinando {spec.nome}...")
        scores = cross_val_score(est, X_tr_s, y_tr, cv=cv, scoring="accuracy")
        est.fit(X_tr_s, y_tr)
        y_pred = est.predict(X_te_s)
        if hasattr(est, "predict_proba"):
            y_score = est.predict_proba(X_te_s)
        elif hasattr(est, "decision_function"):
            y_score = est.decision_function(X_te_s)
        else:
            y_score = None
        metricas = _metricas_classificacao(y_te, y_pred, y_score=y_score)
        metricas["cv_accuracy_media"] = float(scores.mean())
        metricas["cv_accuracy_desvio"] = float(scores.std())
        modelos_out[spec.nome] = metricas

    return _consolidar(exp, modelos_out, metrica_principal="f1")


# ============================================================
# CONSOLIDAÇÃO + DISPATCH
# ============================================================

def _consolidar(exp: ExperimentoArtigo, modelos_out: dict, metrica_principal: str) -> dict:
    from datetime import datetime

    validos = {
        n: m for n, m in modelos_out.items()
        if m.get("disponivel", True) and isinstance(m.get(metrica_principal), (int, float))
    }
    if validos:
        melhor = max(validos, key=lambda n: validos[n][metrica_principal])
        melhor_valor = validos[melhor][metrica_principal]
    else:
        melhor, melhor_valor = "-", float("nan")

    resultado = {
        "experimento": exp.key,
        "referencia": exp.referencia,
        "artigo": exp.artigo,
        "dataset": exp.dataset,
        "tarefa": exp.tarefa,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrica_principal": metrica_principal,
        "origem_dados": _origem_dados(exp),
        "modelos": modelos_out,
        "melhor_modelo": melhor,
        "melhor_valor": melhor_valor,
    }
    graficos = _grafico_comparacao(exp, resultado)
    if graficos:
        resultado["graficos"] = [str(g.resolve()) for g in graficos]
        resultado["grafico"] = str(graficos[0].resolve())
    _salvar_resultado(exp, resultado)
    return resultado


def executar_experimento(key: str, progresso=None) -> dict:
    """Executa um experimento por chave. Dispatcher por runner declarado."""
    # Blinda os prints internos (features_ca etc.) contra emoji no Windows,
    # independentemente de quem chamou (app, terminal, orquestrador, teste).
    from src.core.utils import configurar_saida_utf8
    configurar_saida_utf8()

    exp = get_experimento(key)
    if not exp.runner:
        return {
            "experimento": key,
            "ok": False,
            "mensagem": f"{exp.referencia} é um cartão de dataset — não há modelo a treinar.",
        }
    if not exp.modelos_disponiveis():
        faltantes = ", ".join(f"{m.nome} (requer {m.requer})"
                              for m in exp.modelos_indisponiveis())
        return {
            "experimento": key,
            "ok": False,
            "mensagem": f"Nenhum modelo disponível para {exp.referencia}. Faltam: {faltantes}.",
        }

    runner = _DISPATCH_RUNNERS.get(exp.runner)
    if runner is None:
        return {
            "experimento": key,
            "ok": False,
            "mensagem": f"Runner '{exp.runner}' não implementado.",
        }
    resultado = runner(exp, progresso=progresso)
    resultado["ok"] = True
    return resultado


# ============================================================
# RUNNER 2 — DETECÇÃO DE ANOMALIA (Paderborn + falhas sintéticas)
# ============================================================

_META_FEATURES = ("janela_idx", "amostra_inicio", "tempo_s")


def _carregar_features_paderborn(progresso=None):
    """Carrega a matriz de features do Paderborn; extrai se ainda não existir."""
    import pandas as pd

    from src.ml.features_ca import PASTA_SAIDA, executar_features_ca

    parquet = PASTA_SAIDA / "features_paderborn.parquet"
    if not parquet.exists():
        if progresso:
            progresso("Extraindo features do Paderborn (primeira vez)...")
        if not executar_features_ca():
            raise RuntimeError("Falha ao extrair features do Paderborn.")

    df = pd.read_parquet(parquet)
    cols = [c for c in df.columns if c not in _META_FEATURES]
    return df[cols].to_numpy(dtype=float), cols


def _gerar_anomalias(X_base, rng, severidade: float = 3.0):
    """
    Gera anomalias sintéticas perturbando um subconjunto de features de cada
    janela por `severidade × desvio-padrão` — emula assinaturas de falha
    (desvios de harmônicos/RMS/desbalanceamento) e dá ground truth para AUC.
    """
    import numpy as np

    std = X_base.std(axis=0) + 1e-9
    n, n_feat = X_base.shape
    X_anom = X_base.copy()
    for i in range(n):
        k = int(rng.integers(3, max(4, n_feat // 4)))
        cols = rng.choice(n_feat, size=k, replace=False)
        sinal = rng.choice([-1.0, 1.0], size=k)
        X_anom[i, cols] += severidade * sinal * std[cols]
    return X_anom


def _melhor_limiar(y_true, score):
    """Limiar que maximiza F1 (varre os scores)."""
    import numpy as np
    from sklearn.metrics import f1_score

    cand = np.unique(score)
    if len(cand) > 200:
        cand = np.quantile(score, np.linspace(0, 1, 200))
    melhor_thr, melhor_f1 = cand[0], -1.0
    for thr in cand:
        f1 = f1_score(y_true, (score >= thr).astype(int), zero_division=0)
        if f1 > melhor_f1:
            melhor_f1, melhor_thr = f1, thr
    return float(melhor_thr)


def _metricas_anomalia_legacy(y_true, score) -> dict:
    import numpy as np
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    score = np.asarray(score, dtype=float)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    thr = _melhor_limiar(y_true, score)
    y_pred = (score >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precisao": float(precision_score(y_true, y_pred, zero_division=0)),
        "disponivel": True,
    }


def _score_anomalia_legacy(nome, dados, progresso=None):
    """
    Retorna o vetor de score de anomalia para o conjunto de teste, ou None se
    o modelo não estiver implementado neste runner. `dados` é o pacote comum.
    """
    import numpy as np

    Xn_tr = dados["Xn_tr"]          # normal de treino (modela normalidade)
    X_tr_sup = dados["X_tr_sup"]    # treino supervisionado (normal+anom)
    y_tr_sup = dados["y_tr_sup"]
    X_te = dados["X_te"]            # teste (normal+anom)

    base = nome.lower()

    # --- estatístico: Z-score (desvio médio absoluto, já padronizado) ---
    if "z-score" in base or "zscore" in base:
        return np.mean(np.abs(X_te), axis=1)

    # --- Isolation Forest (não supervisionado, fit no normal) ---
    if "isolation forest" in base and "ppo" not in base:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
        iso.fit(Xn_tr)
        return -iso.decision_function(X_te)

    # --- supervisionados (normal vs anomalia sintética) ---
    if "random forest" in base:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr_sup, y_tr_sup)
        return clf.predict_proba(X_te)[:, 1]

    if base.startswith("knn") or "knn" in base:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=15)
        clf.fit(X_tr_sup, y_tr_sup)
        return clf.predict_proba(X_te)[:, 1]

    if base == "svm" or base.startswith("svm"):
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", probability=True, random_state=42)
        clf.fit(X_tr_sup, y_tr_sup)
        return clf.predict_proba(X_te)[:, 1]

    if "ann" in base or "mlp" in base:
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_tr_sup, y_tr_sup)
        return clf.predict_proba(X_te)[:, 1]

    # --- redes neurais (PyTorch) ---
    if "ae-lstm" in base or "ae lstm" in base or "autoencoder" in base:
        return _score_ae_lstm(dados)
    if base == "rnn" or base.startswith("rnn"):
        return _score_rnn_torch(dados)
    if base == "cnn" or base.startswith("cnn"):
        return _score_cnn_torch(dados)

    # --- Facebook Prophet (série temporal univariada) ---
    if "prophet" in base:
        return _score_prophet(dados)

    # --- Isolation Forest auto-ajustado por RL (PPO) ---
    if "ppo" in base:
        return _score_ppo_iforest(dados)

    return None


def _metricas_anomalia(y_true, score, y_pred=None) -> dict:
    import numpy as np

    y_true_arr = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if y_pred is None:
        thr = _melhor_limiar(y_true_arr, score)
        y_pred = (score >= thr).astype(int)
        ponto = "limiar_otimo_score"
    else:
        thr = None
        y_pred = np.asarray(y_pred).astype(int)
        ponto = "decisao_nativa_modelo"

    metricas = _metricas_classificacao(y_true_arr, y_pred, y_score=score)
    if metricas.get("matriz_confusao") and len(metricas["matriz_confusao"]) == 2:
        metricas["classes"] = ["Normal", "Anomalia"]
    metricas.update({
        "limiar_score": thr,
        "ponto_operacao": ponto,
        "anomalias_detectadas": int(np.sum(y_pred == 1)),
        "anomalias_reais": int(np.sum(y_true_arr == 1)),
        "taxa_anomalias_detectadas": float(np.mean(y_pred == 1)),
    })
    return metricas


def _score_anomalia(nome, dados, progresso=None):
    """Retorna (score, y_pred) no ponto de operacao real quando existe."""
    import numpy as np

    Xn_tr = dados["Xn_tr"]
    X_tr_sup = dados["X_tr_sup"]
    y_tr_sup = dados["y_tr_sup"]
    X_te = dados["X_te"]
    base = nome.lower()

    if "z-score" in base or "zscore" in base:
        return np.mean(np.abs(X_te), axis=1), None

    if "isolation forest" in base and "ppo" not in base:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
        iso.fit(Xn_tr)
        score = -iso.decision_function(X_te)
        y_pred = (iso.predict(X_te) == -1).astype(int)
        return score, y_pred

    if "random forest" in base:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr_sup, y_tr_sup)
        score = clf.predict_proba(X_te)[:, 1]
        return score, (score >= 0.5).astype(int)

    if base.startswith("knn") or "knn" in base:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=15)
        clf.fit(X_tr_sup, y_tr_sup)
        score = clf.predict_proba(X_te)[:, 1]
        return score, (score >= 0.5).astype(int)

    if base == "svm" or base.startswith("svm"):
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", probability=True, random_state=42)
        clf.fit(X_tr_sup, y_tr_sup)
        score = clf.predict_proba(X_te)[:, 1]
        return score, (score >= 0.5).astype(int)

    if "ann" in base or "mlp" in base:
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf.fit(X_tr_sup, y_tr_sup)
        score = clf.predict_proba(X_te)[:, 1]
        return score, (score >= 0.5).astype(int)

    if "ae-lstm" in base or "ae lstm" in base or "autoencoder" in base:
        return _score_ae_lstm(dados), None
    if base == "rnn" or base.startswith("rnn"):
        score = _score_rnn_torch(dados)
        return score, (score >= 0.5).astype(int)
    if base == "cnn" or base.startswith("cnn"):
        score = _score_cnn_torch(dados)
        return score, (score >= 0.5).astype(int)

    if "prophet" in base:
        return _score_prophet(dados), None

    if "ppo" in base:
        return _score_ppo_iforest(dados)

    return None


# ---- Redes neurais compactas (PyTorch) -------------------------------------

def _score_ae_lstm(dados, epochs: int = 60):
    """Autoencoder-LSTM: erro de reconstrução como score (fit no normal)."""
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    Xn = torch.tensor(dados["Xn_tr"], dtype=torch.float32)
    Xte = torch.tensor(dados["X_te"], dtype=torch.float32)
    n_feat = Xn.shape[1]

    class AELSTM(nn.Module):
        def __init__(self, hid=32, lat=8):
            super().__init__()
            self.enc = nn.LSTM(1, hid, batch_first=True)
            self.to_lat = nn.Linear(hid, lat)
            self.from_lat = nn.Linear(lat, hid)
            self.dec = nn.LSTM(hid, hid, batch_first=True)
            self.out = nn.Linear(hid, 1)

        def forward(self, x):
            seq = x.unsqueeze(-1)                       # (B, F, 1)
            _, (h, _) = self.enc(seq)
            lat = self.to_lat(h[-1])                    # (B, lat)
            dec_in = self.from_lat(lat).unsqueeze(1).repeat(1, seq.size(1), 1)
            dec_out, _ = self.dec(dec_in)
            return self.out(dec_out).squeeze(-1)        # (B, F)

    model = AELSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xn), Xn)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        rec = model(Xte)
        return ((rec - Xte) ** 2).mean(dim=1).numpy()


def _treinar_clf_torch(model, dados, epochs: int = 60):
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    Xtr = torch.tensor(dados["X_tr_sup"], dtype=torch.float32)
    ytr = torch.tensor(dados["y_tr_sup"], dtype=torch.float32)
    Xte = torch.tensor(dados["X_te"], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte)).numpy()


def _score_rnn_torch(dados):
    import torch.nn as nn

    n_feat = dados["X_te"].shape[1]

    class RNNClf(nn.Module):
        def __init__(self, hid=32):
            super().__init__()
            self.rnn = nn.LSTM(1, hid, batch_first=True)
            self.fc = nn.Linear(hid, 1)

        def forward(self, x):
            _, (h, _) = self.rnn(x.unsqueeze(-1))
            return self.fc(h[-1]).squeeze(-1)

    return _treinar_clf_torch(RNNClf(), dados)


def _score_cnn_torch(dados):
    import torch.nn as nn

    class CNNClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.fc = nn.Linear(16 * 8, 1)

        def forward(self, x):
            z = self.conv(x.unsqueeze(1))
            return self.fc(z.flatten(1)).squeeze(-1)

    return _treinar_clf_torch(CNNClf(), dados)


# ---- Facebook Prophet (univariado sobre a feature mais informativa) --------

def _score_prophet(dados):
    """
    Prophet aplicado à feature de maior variância no normal: aprende o nível e
    a banda de incerteza; o score é o desvio do valor em relação à banda.
    Univariado por natureza — resultado honesto e mais modesto que o multivar.
    """
    import logging

    import numpy as np
    import pandas as pd

    logging.getLogger("prophet").setLevel(logging.CRITICAL)
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    from prophet import Prophet

    Xn = dados["Xn_tr"]
    Xte = dados["X_te"]
    j = int(np.argmax(Xn.var(axis=0)))
    yn = Xn[:, j]
    ds = pd.date_range("2020-01-01", periods=len(yn), freq="D")
    df = pd.DataFrame({"ds": ds, "y": yn})

    m = Prophet(weekly_seasonality=False, yearly_seasonality=False,
                daily_seasonality=False)
    m.fit(df)
    fc = m.predict(df)
    centro = float(np.mean(fc["yhat"].to_numpy()))
    meia_banda = float(np.mean((fc["yhat_upper"] - fc["yhat_lower"]).to_numpy()) / 2)
    meia_banda = meia_banda if meia_banda > 1e-9 else 1.0
    return np.abs(Xte[:, j] - centro) / meia_banda


# ---- Isolation Forest auto-ajustado por RL (PPO) ---------------------------

def _score_ppo_iforest(dados, timesteps: int = 600):
    """
    Reproduz a ideia de Sharma et al. (2026): um agente PPO ajusta a
    'contamination' do Isolation Forest. Ambiente de 1 passo (bandit):
    ação → contamination; recompensa → AUC numa validação interna.
    """
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from stable_baselines3 import PPO

    rng = np.random.default_rng(7)
    Xn = dados["Xn_tr"]
    Xn_fit, Xn_val = train_test_split(Xn, test_size=0.4, random_state=7)
    Xa_val = _gerar_anomalias(Xn_val, rng)
    Xval = np.vstack([Xn_val, Xa_val])
    yval = np.r_[np.zeros(len(Xn_val)), np.ones(len(Xa_val))]

    def auc_para(cont):
        cont = float(np.clip(cont, 0.01, 0.45))
        iso = IsolationForest(n_estimators=120, contamination=cont, random_state=42)
        iso.fit(Xn_fit)
        return float(roc_auc_score(yval, -iso.decision_function(Xval)))

    class EnvIForest(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            cont = 0.01 + (float(action[0]) + 1.0) / 2.0 * 0.44
            reward = auc_para(cont)
            return np.zeros(1, dtype=np.float32), reward, True, False, {"cont": cont}

    modelo = PPO("MlpPolicy", EnvIForest(), seed=42, verbose=0,
                 n_steps=64, batch_size=32)
    modelo.learn(total_timesteps=timesteps)

    obs = np.zeros(1, dtype=np.float32)
    accao, _ = modelo.predict(obs, deterministic=True)
    melhor_cont = float(np.clip(0.01 + (float(accao[0]) + 1.0) / 2.0 * 0.44, 0.01, 0.45))

    iso = IsolationForest(n_estimators=200, contamination=melhor_cont, random_state=42)
    iso.fit(Xn)
    score = -iso.decision_function(dados["X_te"])
    y_pred = (iso.predict(dados["X_te"]) == -1).astype(int)
    return score, y_pred


def executar_anomalia(exp: ExperimentoArtigo, progresso=None) -> dict:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    if progresso:
        progresso("Carregando features de normalidade (Paderborn)...")
    X_normal, _nomes = _carregar_features_paderborn(progresso)

    Xn_tr, Xn_te = train_test_split(X_normal, test_size=0.4, random_state=42)
    Xa_tr = _gerar_anomalias(Xn_tr, rng)
    Xa_te = _gerar_anomalias(Xn_te, rng)

    scaler = StandardScaler().fit(Xn_tr)
    Xn_tr_s = scaler.transform(Xn_tr)
    Xn_te_s = scaler.transform(Xn_te)
    Xa_tr_s = scaler.transform(Xa_tr)
    Xa_te_s = scaler.transform(Xa_te)

    dados = {
        "Xn_tr": Xn_tr_s,
        "X_tr_sup": np.vstack([Xn_tr_s, Xa_tr_s]),
        "y_tr_sup": np.r_[np.zeros(len(Xn_tr_s)), np.ones(len(Xa_tr_s))],
        "X_te": np.vstack([Xn_te_s, Xa_te_s]),
        "y_te": np.r_[np.zeros(len(Xn_te_s)), np.ones(len(Xa_te_s))],
    }
    y_te = dados["y_te"]

    modelos_out: dict[str, dict] = {}
    scores_individuais: dict[str, np.ndarray] = {}
    predicoes_individuais: dict[str, np.ndarray] = {}
    for spec in exp.modelos:
        if not spec.disponivel:
            modelos_out[spec.nome] = {"disponivel": False, "motivo": f"requer {spec.requer}"}
            continue
        if "híbrido" in spec.nome.lower() or "hibrido" in spec.nome.lower():
            continue  # tratado ao final, combinando os scores
        if progresso:
            progresso(f"Treinando {spec.nome}...")
        try:
            score = _score_anomalia(spec.nome, dados, progresso)
        except Exception as exc:  # noqa: BLE001
            modelos_out[spec.nome] = {"disponivel": False, "motivo": f"erro: {exc}"}
            continue
        if score is None:
            modelos_out[spec.nome] = {"disponivel": False, "motivo": "implementação pendente"}
            continue
        if isinstance(score, tuple):
            score, y_pred = score
        else:
            y_pred = None
        scores_individuais[spec.nome] = np.asarray(score, dtype=float)
        if y_pred is not None:
            predicoes_individuais[spec.nome] = np.asarray(y_pred, dtype=int)
        modelos_out[spec.nome] = _metricas_anomalia(y_te, score, y_pred)

    # Híbrido (voto): média normalizada dos scores dos componentes disponíveis.
    for spec in exp.modelos:
        if "híbrido" in spec.nome.lower() or "hibrido" in spec.nome.lower():
            if len(scores_individuais) >= 2:
                def _norm(s):
                    rng_ = s.max() - s.min()
                    return (s - s.min()) / rng_ if rng_ > 1e-12 else s * 0.0
                combo = np.mean([_norm(s) for s in scores_individuais.values()], axis=0)
                if len(predicoes_individuais) >= 2:
                    votos = np.mean(list(predicoes_individuais.values()), axis=0)
                    y_pred_combo = (votos >= 0.5).astype(int)
                else:
                    y_pred_combo = None
                modelos_out[spec.nome] = _metricas_anomalia(y_te, combo, y_pred_combo)
            else:
                modelos_out[spec.nome] = {
                    "disponivel": False,
                    "motivo": "precisa de ≥2 componentes disponíveis",
                }

    return _consolidar(exp, modelos_out, metrica_principal="f1")


_DISPATCH_RUNNERS: dict[str, Callable] = {
    "executar_classificacao_supervisionada": executar_classificacao_supervisionada,
    "executar_anomalia": executar_anomalia,
}
