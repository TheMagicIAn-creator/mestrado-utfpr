"""Leitura acadêmica dos contratos científicos publicados."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.core.texto import normalizar_sem_acentos


ROOT = Path(RAIZ_PROJETO)
COMPARISON_DIR = ROOT / "resultados" / "comparacao"
RELIABILITY_DIR = ROOT / "resultados" / "confiabilidade"
COMPARISON_JSON = COMPARISON_DIR / "comparacao_autoencoders.json"
RELIABILITY_JSON = RELIABILITY_DIR / "metodologia.json"
MANIFEST_DIR = ROOT / "resultados" / "manifestos"

MODEL_LABELS = {"ae_denso": "Autoencoder Denso", "ae_lstm": "AE-LSTM"}
def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _fmt(value, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
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
    if any(
        term in text
        for term in ("e3", "auc", "roc", "lstm", "denso", "experimental", "comparacao")
    ):
        selected.add("e3")
    if any(
        term in text
        for term in (
            "confiabilidade",
            "taxa de falha",
            "h(t)",
            "r(t)",
            "f(t)",
            "fisica",
            "fmeca",
            "manutencao",
            "manutenção",
        )
    ):
        selected.add("reliability")
    return selected or {"e3", "reliability"}


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
        "## Comparação Denso versus AE-LSTM: 14 ensaios experimentais",
        "",
        "| Modelo | Recall (IC95%) | F1 | Precision | ROC-AUC | PR-AUC | FP saudável |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_LABELS:
        recall = rows.get((model_id, "recall"), {})
        values = [
            rows.get((model_id, metric), {}).get("estimate")
            for metric in ("f1", "precision", "auc_roc", "auc_pr", "false_positive_rate")
        ]
        interval = (
            f"{_fmt(recall.get('estimate'))} "
            f"({_fmt(recall.get('ci95_low'))}-{_fmt(recall.get('ci95_high'))})"
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
            if row.get("metric") == "recall"
        ),
        {},
    )
    lines.extend(
        [
            "",
            "Recall, F1 e Precision são as métricas principais. Para Recall, a diferença "
            "pareada Denso menos AE-LSTM foi "
            f"{_fmt(paired.get('difference_dense_minus_lstm'))} "
            f"(IC95% {_fmt(paired.get('ci95_low'))}-{_fmt(paired.get('ci95_high'))}).",
            "Os intervalos usam o ensaio como unidade de bootstrap; pesos, scaler e "
            "limiares permaneceram congelados nos 14 ensaios de falha. ROC-AUC e "
            "PR-AUC são complementares; Precision sem alarmes positivos é N/A.",
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
            "existe somente para o fusível. Sem tempos individuais de falha e censura, "
            "não se estimam distribuição normal, Weibull físico ou RUL.",
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
        "reliability": (
            (RELIABILITY_DIR / "curva_confiabilidade.png", "Curva de confiabilidade R(t)"),
            (RELIABILITY_DIR / "curva_probabilidade_falha.png", "Probabilidade acumulada de falha F(t)"),
            (RELIABILITY_DIR / "curva_densidade_falha.png", "Densidade de probabilidade de falha f(t)"),
            (RELIABILITY_DIR / "curva_taxa_falha.png", "Taxa de falha h(t)"),
            (RELIABILITY_DIR / "taxas_componentes.png", "Taxas bibliográficas por componente"),
        ),
    }
    images = []
    for section in ("e3", "reliability"):
        if section not in focus:
            continue
        for path, caption in candidates[section]:
            if path.is_file():
                images.append({"path": str(path), "caption": caption, "inline": inline})
    return images


def _provenance_summary(focus: set[str], operation: str) -> str:
    from src.ml.pipeline import estado_publicacao

    stages = {
        "e3": ("comparacao", "comparacao_autoencoders", "Comparação Denso versus AE-LSTM"),
        "reliability": (
            "confiabilidade",
            "confiabilidade_componentes",
            "Confiabilidade física",
        ),
    }
    labels = {"ready": "ready", "stale": "stale", "pending": "pending"}
    lines = ["## Proveniência da resposta", "", f"Operação: **{operation}**."]
    for section in ("e3", "reliability"):
        if section not in focus:
            continue
        key, manifest_name, label = stages[section]
        state = estado_publicacao(key)
        path = MANIFEST_DIR / f"{manifest_name}.json"
        manifest = _json(path)
        parameters = manifest.get("parameters") or {}
        generated = manifest.get("created_at") or "não disponível"
        relative = path.relative_to(ROOT).as_posix()
        if section == "e3":
            configuration = (
                f"seed de referência={parameters.get('reference_seed', 'não disponível')}; "
                f"top-k={parameters.get('score_top_k', 'não disponível')}; "
                f"percentil={parameters.get('threshold_percentile', 'não disponível')}"
            )
        else:
            configuration = (
                f"modelo={parameters.get('model', 'não disponível')}; "
                f"horizonte={parameters.get('horizon_years', 'não disponível')} anos; "
                f"FMECA={parameters.get('fmeca_status', 'não disponível')}"
            )
        reasons = "; ".join(state.get("motivos", []))
        suffix = f"; motivos: {reasons}" if reasons else ""
        lines.append(
            f"- **{label}:** {labels.get(state['estado'], state['estado'])}{suffix}; "
            f"manifesto `{relative}`; gerado em `{generated}`; {configuration}."
        )
    lines.append(
        "O estado acima valida os artefatos publicados contra seus hashes; "
        "a auditoria profunda de código e entradas permanece no status do pipeline."
    )
    return "\n".join(lines)


def resumir_resultados(
    pergunta: str = "",
    *,
    incluir_imagens: bool = True,
    operacao: str = "consultado",
) -> dict:
    focus = _focus(pergunta)
    comparison = _json(COMPARISON_JSON)
    reliability = _json(RELIABILITY_JSON)
    sections = []
    if "e3" in focus:
        sections.append(_e3_summary(comparison))
    if "reliability" in focus:
        sections.append(_reliability_summary(reliability))
    sections.append(_provenance_summary(focus, operacao))

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
