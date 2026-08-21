"""Leitura acadêmica dos contratos científicos publicados."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.core.texto import normalizar_sem_acentos


ROOT = Path(RAIZ_PROJETO)
COMPARISON_DIR = ROOT / "resultados" / "comparacao"
RELIABILITY_DIR = ROOT / "resultados" / "confiabilidade"
COMPARISON_JSON = COMPARISON_DIR / "comparacao_autoencoders.json"
RELIABILITY_JSON = RELIABILITY_DIR / "metodologia.json"

MODEL_LABELS = {"ae_denso": "Autoencoder Denso", "ae_lstm": "AE-LSTM"}
COMPONENT_LABELS = {
    "contator_ac": "Contator AC",
    "igbt": "IGBT",
    "fusivel_ac": "Fusível AC",
}


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _fmt(value, digits: int = 3) -> str:
    if not isinstance(value, (int, float)):
        return "não disponível"
    return f"{float(value):.{digits}f}".replace(".", ",")


def _quer_imagens(pergunta: str) -> bool:
    text = normalizar_sem_acentos(pergunta).lower()
    return any(
        term in text
        for term in (
            "grafico",
            "graficos",
            "figura",
            "figuras",
            "imagem",
            "imagens",
            "mostre",
            "veja",
            "exiba",
            "visualize",
        )
    )


def _focus(pergunta: str) -> set[str]:
    text = normalizar_sem_acentos(pergunta).lower()
    selected: set[str] = set()
    if any(term in text for term in ("e3", "auc", "roc", "lstm", "denso", "experimental")):
        selected.add("e3")
    if any(term in text for term in ("e2", "fmeca", "smd", "detectabilidade", "weibull")):
        selected.add("e2")
    if any(
        term in text
        for term in ("confiabilidade", "taxa de falha", "h(t)", "r(t)", "f(t)", "fisica")
    ):
        selected.add("reliability")
    return selected or {"e3", "e2", "reliability"}


def _metric_rows(payload: dict) -> dict[tuple[str, str], dict]:
    rows = payload.get("e3", {}).get("macro", [])
    return {
        (row.get("model"), row.get("metric")): row
        for row in rows
        if isinstance(row, dict)
    }


def _e3_summary(payload: dict) -> str:
    rows = _metric_rows(payload)
    if not rows:
        return "## Evidência E3\n\nResultado experimental não publicado."
    lines = [
        "## Evidência E3: 14 ensaios reais GPVS-Faults",
        "",
        "| Modelo | AUC-PR (IC95%) | ROC-AUC | Sensibilidade | Especificidade | MCC | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_LABELS:
        auc = rows.get((model_id, "auc_pr"), {})
        values = [
            rows.get((model_id, metric), {}).get("estimate")
            for metric in ("auc_roc", "sensitivity", "specificity", "mcc", "f1")
        ]
        interval = (
            f"{_fmt(auc.get('estimate'))} "
            f"({_fmt(auc.get('ci95_low'))}-{_fmt(auc.get('ci95_high'))})"
        )
        lines.append(
            f"| {MODEL_LABELS[model_id]} | {interval} | "
            + " | ".join(_fmt(value) for value in values)
            + " |"
        )

    paired = next(
        (
            row
            for row in payload.get("e3", {}).get("paired_differences", [])
            if row.get("metric") == "auc_pr"
        ),
        {},
    )
    lines.extend(
        [
            "",
            "AUC-PR é a métrica principal. A diferença pareada Denso menos AE-LSTM foi "
            f"{_fmt(paired.get('difference_dense_minus_lstm'))} "
            f"(IC95% {_fmt(paired.get('ci95_low'))}-{_fmt(paired.get('ci95_high'))}).",
            "Os intervalos usam o ensaio como unidade de bootstrap; pesos, scaler e "
            "limiares permaneceram congelados nos 14 ensaios de falha.",
        ]
    )
    return "\n".join(lines)


def _e2_summary(payload: dict) -> str:
    rows = payload.get("e2", {}).get("summary", [])
    if not rows:
        return "## Evidência E2\n\nResultado sintético FMECA não publicado."
    lines = [
        "## Evidência E2: detectabilidade orientada pela FMECA",
        "",
        "| Modelo | Componente | NPR | SMD95 | Detecção em a_det=1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        reached = row.get("smd95_status") == "reached"
        smd = _fmt(row.get("smd95"), 2) if reached else "não atingido"
        detection = row.get("detection_at_max")
        detection_text = (
            f"{100 * float(detection):.1f}%".replace(".", ",")
            if isinstance(detection, (int, float))
            else "não disponível"
        )
        lines.append(
            f"| {MODEL_LABELS.get(row.get('model'), row.get('model'))} | "
            f"{COMPONENT_LABELS.get(row.get('component'), row.get('component'))} | "
            f"{row.get('npr', '-')} | {smd} | {detection_text} |"
        )
    lines.extend(
        [
            "",
            "SMD95 é a menor magnitude cujo limite inferior do IC95% Wilson alcança "
            "95%. O eixo `a_det` é magnitude de perturbação, não tempo.",
            "Os ajustes Weibull 2P desta execução não foram recomendados para síntese "
            "paramétrica. Isso rejeita o ajuste, não os detectores.",
        ]
    )
    return "\n".join(lines)


def _reliability_summary(payload: dict) -> str:
    scenarios = payload.get("scenarios", [])
    if not scenarios:
        return "## Confiabilidade física\n\nCenários bibliográficos não publicados."
    lines = [
        "## Confiabilidade física: cenários bibliográficos",
        "",
        "| Componente/cenário | Origem | λ (falha/h) | 1/λ (anos) |",
        "|---|---|---:|---:|",
    ]
    for row in scenarios:
        origin = "direta" if row.get("evidence_type") == "direct_bibliographic" else "derivada"
        lines.append(
            f"| {row.get('plot_label', row.get('component_name'))} | {origin} | "
            f"{float(row['lambda_per_hour']):.2e} | {_fmt(row.get('reciprocal_time_years'), 2)} |"
        )
    lines.extend(
        [
            "",
            "Modelo exponencial: `R(t)=exp(-λt)`, `F(t)=1-R(t)`, "
            "`f(t)=λexp(-λt)` e `h(t)=λ`.",
            "As três taxas derivadas são análises de sensibilidade baseadas nas "
            "participações de chamados; não são medições por componente. A taxa direta "
            "existe somente para o fusível. O GPVS-Faults não estima vida física, "
            "Weibull físico ou RUL.",
        ]
    )
    return "\n".join(lines)


def _images(focus: set[str], *, inline: bool) -> list[dict]:
    candidates = {
        "e3": (
            (COMPARISON_DIR / "e3_metricas_macro.png", "Métricas macro E3"),
            (COMPARISON_DIR / "e3_curvas_discriminacao.png", "Curvas ROC e precisão-revocação E3"),
            (COMPARISON_DIR / "e3_matrizes_confusao.png", "Matrizes de confusão E3"),
        ),
        "e2": (
            (COMPARISON_DIR / "e2_deteccao_por_magnitude.png", "Detectabilidade E2 por magnitude"),
            (COMPARISON_DIR / "e2_funcoes_empiricas.png", "Funções empíricas de detectabilidade"),
            (COMPARISON_DIR / "e2_diagnostico_weibull.png", "Diagnóstico Weibull no papel de probabilidade"),
        ),
        "reliability": (
            (RELIABILITY_DIR / "confiabilidade_probabilidade_falha.png", "Confiabilidade e probabilidade de falha"),
            (RELIABILITY_DIR / "densidade_taxa_falha.png", "Densidade e taxa de falha"),
            (RELIABILITY_DIR / "taxas_componentes.png", "Taxas bibliográficas por componente"),
        ),
    }
    images = []
    for section in ("e3", "e2", "reliability"):
        if section not in focus:
            continue
        for path, caption in candidates[section]:
            if path.is_file():
                images.append({"path": str(path), "caption": caption, "inline": inline})
    return images


def resumir_resultados(pergunta: str = "", *, incluir_imagens: bool = True) -> dict:
    focus = _focus(pergunta)
    comparison = _json(COMPARISON_JSON)
    reliability = _json(RELIABILITY_JSON)
    sections = []
    if "e3" in focus:
        sections.append(_e3_summary(comparison))
    if "e2" in focus:
        sections.append(_e2_summary(comparison))
    if "reliability" in focus:
        sections.append(_reliability_summary(reliability))

    inline = _quer_imagens(pergunta)
    images = _images(focus, inline=inline) if incluir_imagens else []
    message = "\n\n".join(sections)
    if images:
        message += (
            "\n\nAs figuras selecionadas estão disponíveis abaixo."
            if inline
            else f"\n\n{len(images)} figura(s) acadêmica(s) disponível(is) para visualização."
        )
    return {
        "ok": True,
        "etapa": "Resultados científicos",
        "mensagem": message,
        "imagens": images,
        "resposta_pronta": False,
    }


def salvar_resumo_resultados_ml() -> Path:
    """Grava uma nota indexável a partir dos contratos canônicos."""

    output = ROOT / "notas" / "memorias" / "resultados-canonicos-ml.md"
    summary = resumir_resultados("", incluir_imagens=False)["mensagem"]
    content = (
        "# Resultados canônicos de ML e confiabilidade\n\n"
        f"> Gerado em {agora_local().strftime('%d/%m/%Y %H:%M %Z')}\n\n"
        f"{summary}\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    return output


__all__ = ["_quer_imagens", "resumir_resultados", "salvar_resumo_resultados_ml"]
