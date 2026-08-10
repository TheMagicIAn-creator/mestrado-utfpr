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
        if "canonical_ae" in macro:
            canonico = macro["canonical_ae"]["all"]
            protocolos = [("Autoencoder canônico congelado", canonico)]
        else:
            # Compatibilidade de leitura com o schema v1 já publicado.
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
            f"{_fmt(metricas.get('sensitivity', metricas.get('post_tpr'))['mean'])} | "
            f"{_fmt(metricas['specificity']['mean'])} | "
            f"{_fmt(metricas['balanced_accuracy']['mean'])} | "
            f"{auc.get('n_experiments', '-')} |\n"
        )
    if "canonical_ae" in macro:
        linhas.append(
            "\n**Leitura honesta:** este é o mesmo detector ajustado somente em "
            "F0L/F0M e aplicado a F1-F7 sem retreino nem recalibração do limiar. "
            "A primeira metade pré-falha fornece o baseline de comissionamento; "
            "a segunda mede a especificidade. Os IC95% macro reamostram 14 "
            "ensaios, não janelas. E3 significa bancada experimental, não é campo; o "
            "detector não identifica causa automaticamente nem calibra "
            "Weibull/RUL físico."
        )
    else:
        linhas.append(
            "\n**Artefato legado (schema v1):** compara transferência direta e "
            "adaptação local. Reexecute o pipeline para publicar o detector "
            "canônico único. É bancada experimental, não é campo."
        )
    return "".join(linhas)
