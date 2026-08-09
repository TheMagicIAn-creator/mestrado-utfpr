"""Resumo acadêmico dos artefatos da validação externa GPVS-Faults."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.formatacao import fmt_num


def _fmt(valor, casas: int = 3) -> str:
    return fmt_num(valor, casas)


def resumir_gpvs(pasta: Path) -> str | None:
    caminho = pasta / "validacao_gpvs_e3.json"
    if not caminho.exists():
        return None
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        macro = dados["macro_summary"]
        estrito = macro["strict_ae"]["all"]
        adaptativo = macro["adaptive_ae"]["all"]
        protocolos = [
            ("Transferência direta AE", estrito),
            ("AE adaptativo", adaptativo),
            ("PCA adaptativo", macro["adaptive_pca"]["all"]),
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None

    linhas = [
        "## Validação externa GPVS-Faults - E3 de bancada\n\n",
        "| Protocolo | AUC macro (IC95%) | Sensibilidade pós-falha | "
        "Especificidade | Acurácia balanceada | n ensaios |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for nome, metricas in protocolos:
        auc = metricas["auc"]
        linhas.append(
            f"| {nome} | {_fmt(auc['mean'])} "
            f"[{_fmt(auc['ci95_low'])}; {_fmt(auc['ci95_high'])}] | "
            f"{_fmt(metricas['post_tpr']['mean'])} | "
            f"{_fmt(metricas['specificity']['mean'])} | "
            f"{_fmt(metricas['balanced_accuracy']['mean'])} | "
            f"{auc.get('n_experiments', '-')} |\n"
        )
    linhas.append(
        "\n**Leitura honesta:** a transferência direta do limiar do ensaio F0 "
        f"é rejeitada: sua especificidade macro é {_fmt(estrito['specificity']['mean'])}. "
        "Com adaptação usando somente o início saudável de cada ensaio, o AE "
        f"alcança AUC macro {_fmt(adaptativo['auc']['mean'])} e especificidade "
        f"{_fmt(adaptativo['specificity']['mean'])}, mas sensibilidade pós-falha "
        f"de {_fmt(adaptativo['post_tpr']['mean'])}. Os IC95% são bootstrap de "
        "14 ensaios, não de janelas. E3 aqui significa bancada experimental "
        "externa; não é campo, não identifica causa automaticamente e não "
        "calibra Weibull/RUL físico."
    )
    return "".join(linhas)
