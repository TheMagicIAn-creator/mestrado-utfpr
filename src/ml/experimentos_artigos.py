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

Artigos-base do NÚCLEO (curadoria "só anomalia CA por modelagem de normalidade"):
  1. Francisti et al. (2025)             — Z-score (Shewhart/SPC), não-supervisionado
  2. Ibrahim et al. (2022)               — AE-LSTM, Prophet, Isolation Forest
  3. Ahirwar & Nandanwar (2025)          — híbrido AE-LSTM + Prophet + IForest (voto)
  4. Stender, Wallscheid & Böcker (2020) — descrição do dataset (Paderborn)

Removidos da curadoria (treinavam nos rótulos da injeção sintética ou
inadequados): Ghoneim (classificação CC supervisionada — segue no
classificador_pv, não como experimento), Sharma (baselines supervisionados +
RNN/CNN + IForest+PPO degenerado) e o Random Forest supervisionado do Francisti.

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
    """
    Schema único para classificação e detecção por ponto de operação.

    Especificidade com SEMÂNTICA EXPLÍCITA (item 4.1):
    - `specificity`           = TN/(TN+FP) no caso BINÁRIO; em multiclasse cai
                                para o macro one-vs-rest (mesmo valor de
                                `specificity_macro_ovr`);
    - `specificity_macro_ovr` = média one-vs-rest (sempre presente);
    - `specificity_tipo`      = qual definição `specificity` representa.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    binario = len(labels) == 2 and set(labels.tolist()).issubset({0, 1})
    media = "binary" if binario else "macro"
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    spec_macro = _specificity_macro(y_true_arr, y_pred_arr)

    if binario:
        # labels ordenados por np.unique → [0, 1]; cm: linhas=real, col=previsto
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
        spec_tipo = "binaria_TN/(TN+FP)"
    else:
        specificity = spec_macro          # em multiclasse usa-se o macro OvR
        fpr = fnr = None                   # FPR/FNR binários não se aplicam
        spec_tipo = "macro_one_vs_rest"

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average=media, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average=media, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average=media, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "auc": _auc_seguro(y_true_arr, y_score),
        "specificity": specificity,
        "specificity_macro_ovr": spec_macro,
        "specificity_tipo": spec_tipo,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "amostras": int(len(y_true_arr)),
        "n_classes": int(len(labels)),
        "classes": [str(x) for x in labels.tolist()],
        "matriz_confusao": cm.astype(int).tolist(),
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


def comparar_anomalia_por_auc() -> dict:
    """
    Comparativo dos experimentos de anomalia pela AUC — a ÚNICA métrica
    comparável entre protocolos (independe do ponto de operação, que difere
    por artigo). Lê os resultado.json já salvos (não re-roda nada), gera um
    gráfico de barras e uma tabela Markdown.

    Retorna {"ok", "grafico", "tabela_md", "dados", "mensagem"}.
    """
    import json

    chaves = [k for k in ORDEM_EXPERIMENTOS
              if REGISTRO[k].tarefa == "anomalia"]
    linhas = []   # (referencia, modelo, auc)
    faltando = []
    for k in chaves:
        arq = PASTA_EXPERIMENTOS / k / "resultado.json"
        if not arq.exists():
            faltando.append(k)
            continue
        try:
            res = json.loads(arq.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            # JSON corrompido/ilegível não derruba a comparação: registra e pula.
            from src.core.logs import get_logger

            get_logger("experimentos_artigos").warning(
                "resultado.json inválido em %s: %s", k, exc)
            faltando.append(f"{k} (JSON inválido)")
            continue
        ref = res.get("referencia", k)
        for nome, m in res.get("modelos", {}).items():
            if isinstance(m, dict) and isinstance(m.get("auc"), (int, float)):
                linhas.append((ref, nome, float(m["auc"])))

    if not linhas:
        return {"ok": False, "grafico": None, "tabela_md": "",
                "dados": [], "mensagem": (
                    "Nenhum experimento de anomalia salvo ainda. Rode-os antes "
                    "(ex.: python scripts/rodar_experimentos.py --todos).")}

    # tabela markdown ordenada por AUC desc
    linhas_ord = sorted(linhas, key=lambda t: t[2], reverse=True)
    tab = ["| Experimento | Modelo | AUC |", "|---|---|---|"]
    for ref, nome, auc in linhas_ord:
        tab.append(f"| {ref} | {nome} | {auc:.3f} |")
    tabela_md = "\n".join(tab)

    grafico = _grafico_auc_anomalia(linhas)
    nota = ""
    if faltando:
        nota = f" (sem resultado salvo: {', '.join(faltando)})"
    return {
        "ok": True,
        "grafico": grafico,
        "tabela_md": tabela_md,
        "dados": linhas_ord,
        "mensagem": (
            "Comparação por AUC (métrica comparável entre protocolos; F1 não é, "
            "pois cada artigo opera em um ponto de decisão próprio)." + nota),
    }


def _grafico_auc_anomalia(linhas: list) -> str | None:
    """Barras horizontais de AUC por modelo, agrupadas por experimento."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.core.utils import to_project_relative_path

    fig = None
    try:
        # agrupa por experimento, preserva ordem do registro
        refs = []
        for ref, _n, _a in linhas:
            if ref not in refs:
                refs.append(ref)
        cores = plt.cm.tab10.colors

        rotulos, valores, cores_barra = [], [], []
        for i, ref in enumerate(refs):
            for r, nome, auc in linhas:
                if r == ref:
                    rotulos.append(f"{nome}")
                    valores.append(auc)
                    cores_barra.append(cores[i % len(cores)])

        fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(valores))))
        y = range(len(valores))
        ax.barh(list(y), valores, color=cores_barra)
        ax.set_yticks(list(y))
        ax.set_yticklabels(rotulos, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.5, color="grey", ls="--", lw=1, label="acaso (0,5)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUC (comparável entre protocolos)")
        ax.set_title("Detecção de anomalia — AUC por modelo e artigo (E1)")
        # legenda por experimento
        from matplotlib.patches import Patch
        leg = [Patch(color=cores[i % len(cores)], label=ref)
               for i, ref in enumerate(refs)]
        leg.append(Patch(color="grey", label="acaso (0,5)"))
        ax.legend(handles=leg, fontsize=7, loc="lower right")
        for yi, v in zip(y, valores):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=7)
        fig.tight_layout()
        destino = PASTA_EXPERIMENTOS / "comparacao_auc_anomalia.png"
        fig.savefig(destino, dpi=120)
        return to_project_relative_path(destino)
    except Exception as exc:  # noqa: BLE001
        from src.core.logs import get_logger

        get_logger("experimentos_artigos").warning(
            "Falha ao gerar gráfico AUC: %s", exc)
        return None
    finally:
        # Garante a liberação da figura mesmo se o plot levantar exceção.
        if fig is not None:
            plt.close(fig)


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


def _slug_modelo(nome: str) -> str:
    """Nome estavel para arquivos de artefatos por modelo."""
    import re
    import unicodedata

    texto = unicodedata.normalize("NFD", nome.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    return texto or "modelo"


def _registrar_grafico_modelo(modelo: dict, chave: str, caminho: Path) -> None:
    from src.core.utils import to_project_relative_path

    modelo.setdefault("graficos", [])
    rel = to_project_relative_path(caminho)  # relativo ao projeto (portável)
    if rel not in modelo["graficos"]:
        modelo["graficos"].append(rel)
    modelo[chave] = rel


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

# ============================================================
# CONSOLIDAÇÃO + DISPATCH
# ============================================================

def _consolidar(exp: ExperimentoArtigo, modelos_out: dict, metrica_principal: str,
                metodologia: dict | None = None) -> dict:
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
        # Benchmark EXPLORATÓRIO. Anomalia usa perturbação GENÉRICA das features
        # (não a injeção FMEA do pipeline principal, que é E2). Classificação usa
        # PV Farms (falhas CC). Nunca é validação formal nem prova industrial.
        "evidence_level": "E1",
        "evidence_note": (
            "E1 — benchmark exploratório (perturbação genérica / dataset rotulado "
            "CC); não é validação formal nem desempenho industrial."
        ),
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrica_principal": metrica_principal,
        "origem_dados": _origem_dados(exp),
        "modelos": modelos_out,
        "melhor_modelo": melhor,
        "melhor_valor": melhor_valor,
    }
    if metodologia:
        # Protocolo por artigo: split temporal, injeção FMEA e a regra de
        # decisão de CADA modelo — rastreabilidade completa no resultado.
        resultado["metodologia"] = metodologia
        if metodologia.get("injecao", {}).get("tipo") == "fmea_espaco_features":
            # A nota padrão fala em "perturbação genérica"; os protocolos
            # usam injeção ORIENTADA PELO FMEA — a proveniência deve refletir.
            resultado["evidence_note"] = (
                "E1 — benchmark exploratório (injeção sintética orientada "
                "pelo FMEA no espaço de features, com protocolo de decisão "
                "do próprio artigo); não é validação formal nem desempenho "
                "industrial."
            )
    graficos = _grafico_comparacao(exp, resultado)
    if graficos:
        from src.core.utils import to_project_relative_path

        resultado["graficos"] = [to_project_relative_path(g) for g in graficos]
        resultado["grafico"] = to_project_relative_path(graficos[0])
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


# Cache de módulo: o parquet de features é relido por TODO experimento de
# anomalia; com 4 experimentos em sequência isso era I/O repetido. A chave
# inclui o mtime do arquivo → recomputar features invalida o cache sozinho.
_CACHE_FEATURES: dict = {}


def _carregar_features_paderborn(progresso=None):
    """Carrega a matriz de features do Paderborn; extrai se ainda não existir.

    Retorna ``(X, cols)`` com X sempre uma CÓPIA (caller pode mutar à vontade).
    """
    import pandas as pd

    from src.ml.features_ca import PASTA_SAIDA, executar_features_ca

    parquet = PASTA_SAIDA / "features_paderborn.parquet"
    if not parquet.exists():
        if progresso:
            progresso("Extraindo features do Paderborn (primeira vez)...")
        if not executar_features_ca():
            raise RuntimeError("Falha ao extrair features do Paderborn.")

    chave = (str(parquet), parquet.stat().st_mtime_ns)
    if chave not in _CACHE_FEATURES:
        df = pd.read_parquet(parquet)
        cols = [c for c in df.columns if c not in _META_FEATURES]
        _CACHE_FEATURES.clear()  # nunca acumula versões antigas
        _CACHE_FEATURES[chave] = (df[cols].to_numpy(dtype=float), cols)
    X, cols = _CACHE_FEATURES[chave]
    return X.copy(), list(cols)


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


def _metricas_anomalia(y_true, score, y_pred=None,
                       threshold_source: str | None = None,
                       limiar: float | None = None) -> dict:
    """
    Métricas de anomalia no ponto de operação informado.

    Comportamento PADRÃO (compatível com o histórico):
    - ``y_pred=None``  → limiar ótimo no próprio conjunto (EXPLORATÓRIO, E1);
    - ``y_pred`` dado  → decisão nativa do modelo.

    Protocolos por artigo passam ``threshold_source``/``limiar`` EXPLÍCITOS
    (ex.: "shewhart_3sigma_a_priori", "p99_erro_reconstrucao_treino") — esses
    limiares são definidos SEM olhar os rótulos do conjunto avaliado, então a
    métrica é marcada como "a_priori_ou_congelada".
    """
    import numpy as np

    y_true_arr = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if y_pred is None:
        thr = _melhor_limiar(y_true_arr, score)
        y_pred = (score >= thr).astype(int)
        ponto = "limiar_otimo_score"
        # Limiar escolhido NO PRÓPRIO conjunto avaliado → métrica EXPLORATÓRIA
        # (E1). Não é estimativa de generalização (ver backlog: limiar congelado
        # em val). O AUC permanece válido por ser independente de limiar.
        threshold_source = "exploratorio_no_conjunto_avaliado"
        dependencia = "exploratoria"
    elif threshold_source is None:
        thr = None
        y_pred = np.asarray(y_pred).astype(int)
        ponto = "decisao_nativa_modelo"
        # Decisão nativa do modelo (ex.: IsolationForest.predict) — não deriva
        # dos rótulos do conjunto avaliado.
        threshold_source = "decisao_nativa_modelo"
        dependencia = "nativa"
    else:
        # Protocolo por artigo: regra de decisão definida a priori ou
        # congelada em treino/validação — nunca nos rótulos do teste.
        thr = float(limiar) if limiar is not None else None
        y_pred = np.asarray(y_pred).astype(int)
        ponto = "protocolo_do_artigo"
        dependencia = "a_priori_ou_congelada"

    metricas = _metricas_classificacao(y_true_arr, y_pred, y_score=score)
    if metricas.get("matriz_confusao") and len(metricas["matriz_confusao"]) == 2:
        metricas["classes"] = ["Normal", "Anomalia"]

    # Regime RARO: precision/F1 reprojetados p/ a prevalência realista de falha
    # CA (eventos raros). O teste é ~50/50 para estimar TPR/FPR; precision/F1 a
    # 50% inflam. Regra de Bayes no ponto de operação: AUC/recall/specificity
    # NÃO mudam; precision/F1 sim. Revela o custo de FPR>0 em operação real.
    y_pred_arr = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    tpr_op = tp / (tp + fn) if (tp + fn) else 0.0
    fpr_op = fp / (fp + tn) if (fp + tn) else 0.0
    pi = 0.05
    denom = pi * tpr_op + (1.0 - pi) * fpr_op
    prec_raro = (pi * tpr_op / denom) if denom > 0 else 0.0
    f1_raro = (2 * prec_raro * tpr_op / (prec_raro + tpr_op)
               if (prec_raro + tpr_op) > 0 else 0.0)

    metricas.update({
        "limiar_score": thr,
        "ponto_operacao": ponto,
        "threshold_source": threshold_source,
        "metrica_dependente_de_limiar": dependencia,
        "anomalias_detectadas": int(np.sum(y_pred_arr == 1)),
        "anomalias_reais": int(np.sum(y_true_arr == 1)),
        "taxa_anomalias_detectadas": float(np.mean(y_pred_arr == 1)),
        "prevalencia_raro": pi,
        "fpr_operacao": float(fpr_op),
        "precision_raro": float(prec_raro),
        "f1_raro": float(f1_raro),
    })
    return metricas


def executar_anomalia(exp: ExperimentoArtigo, progresso=None) -> dict:
    # ── PROTOCOLO POR ARTIGO (caminho principal) ─────────────────────────
    # Cada artigo tem o próprio protocolo de decisão (limiar a priori,
    # p99 de treino, banda do Prophet, PPO em validação temporal, voto) —
    # ver src/ml/protocolos_artigos.py. Evita o "erro de simulação" de
    # avaliar todos os métodos sob um harness único com limiar-oráculo.
    from src.ml.protocolos_artigos import executar_protocolo

    saida_protocolo = executar_protocolo(exp.key, progresso=progresso)
    if saida_protocolo is not None:
        modelos_proto, metodologia = saida_protocolo
        # Specs indisponíveis mantêm a degradação honesta ("requer <lib>").
        for spec in exp.modelos:
            if spec.nome not in modelos_proto:
                motivo = (f"requer {spec.requer}" if not spec.disponivel
                          else "fora do protocolo deste artigo")
                modelos_proto[spec.nome] = {"disponivel": False, "motivo": motivo}
        return _consolidar(exp, modelos_proto, metrica_principal="f1",
                           metodologia=metodologia)

    # Sem protocolo registrado para esta chave. O nucleo curado (anomalia
    # CA por modelagem de normalidade) define um protocolo por artigo para
    # cada experimento; o harness generico legado foi removido na curadoria.
    raise ValueError(
        f"Experimento '{exp.key}' nao tem protocolo de anomalia registrado "
        f"(nucleo: francisti, ibrahim, ahirwar)."
    )
_DISPATCH_RUNNERS: dict[str, Callable] = {
    "executar_anomalia": executar_anomalia,
}
