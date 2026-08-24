"""Contexto autoritativo e seletivo dos resultados científicos publicados."""

from __future__ import annotations

_RELIABILITY_TERMS = (
    "confiabilidade",
    "reliability",
    "taxa de falha",
    "densidade de falha",
    "probabilidade de falha",
    "r(t)",
    "f(t)",
    "h(t)",
    "mttf",
    "mtbf",
    "manutenção",
    "manutencao",
    "fmeca",
    "npr",
    "contator",
    "fusível",
    "fusivel",
)

_COMPARISON_TERMS = (
    "autoencoder",
    "ae-lstm",
    "ae lstm",
    "lstm",
    "auc-pr",
    "auc pr",
    "auc-roc",
    "auc roc",
    "curva roc",
    "precisão-revocação",
    "precisao-revocacao",
    "matriz de confusão",
    "matriz de confusao",
    "denso versus",
    "denso vs",
    "comparação dos modelos",
    "comparacao dos modelos",
)

def _metric(metrics: dict, name: str) -> str:
    item = metrics[name]
    return (
        f"{item['estimate']:.6f} "
        f"(IC95% {item['ci95_low']:.6f} a {item['ci95_high']:.6f}; "
        f"n={item['n_experiments']})"
    )


def _comparison_context() -> str:
    from src.webapp.contracts import e3_contract

    e3 = e3_contract()
    dense = e3["metrics"]["ae_denso"]
    lstm = e3["metrics"]["ae_lstm"]
    difference = next(
        item for item in e3["paired_differences"] if item["metric"] == "auc_pr"
    )
    return "\n".join(
        (
            "CONTRATO AUTORITATIVO - COMPARACAO DENSO VERSUS AE-LSTM",
            "Dataset experimental único: "
            f"{e3['dataset']['name']} (DOI {e3['dataset']['doi']}). O nome do "
            "dataset identifica a proveniência; os resultados pertencem aos modelos.",
            "Modelos congelados antes dos ensaios: Autoencoder Denso "
            "24-16-8-16-24 e AE-LSTM temporal L=8, hidden=32, latent=8.",
            f"AUC-PR macro do Denso: {_metric(dense, 'auc_pr')}.",
            f"AUC-PR macro do AE-LSTM: {_metric(lstm, 'auc_pr')}.",
            "Diferença pareada Denso menos LSTM em AUC-PR: "
            f"{difference['difference_dense_minus_lstm']:.6f} "
            f"(IC95% {difference['ci95_low']:.6f} a "
            f"{difference['ci95_high']:.6f}; n=14 ensaios).",
            f"AUC-ROC do Denso: {_metric(dense, 'auc_roc')}; "
            f"AE-LSTM: {_metric(lstm, 'auc_roc')}.",
            f"Sensibilidade do Denso: {_metric(dense, 'sensitivity')}; "
            f"AE-LSTM: {_metric(lstm, 'sensitivity')}.",
            f"Especificidade do Denso: {_metric(dense, 'specificity')}; "
            f"AE-LSTM: {_metric(lstm, 'specificity')}.",
            "Não apresente métricas como resultados autônomos do dataset. "
            "Elas comparam exclusivamente os dois detectores.",
        )
    )


def _reliability_context() -> str:
    from src.webapp.contracts import reliability_contract

    reliability = reliability_contract()
    rates = "; ".join(
        f"{item['plot_label']}: lambda={item['lambda_per_hour']:.3e} h^-1 "
        f"({item['evidence_type']})"
        for item in reliability["scenarios"]
    )
    return "\n".join(
        (
            "CONTRATO AUTORITATIVO - CONFIABILIDADE E MANUTENCAO",
            "Use apenas taxas bibliográficas diretas ou cenários derivados "
            "explicitamente rotulados; não as trate como medições de campo.",
            "Modelo exponencial: R(t)=exp(-lambda*t), F(t)=1-R(t), "
            "f(t)=lambda*exp(-lambda*t) e h(t)=lambda.",
            f"Cenários por componente: {rates}.",
            "As curvas publicadas usam escalas lineares e tempo em horas/anos.",
            "Não há amostra homogênea de tempos de falha ou censura por ativo. "
            "Portanto, distribuição normal, Weibull físico e curva de banheira "
            "não são estimáveis sem fabricar evidência.",
            "Participação de chamados auxilia o planejamento de manutenção, mas "
            "não substitui severidade, ocorrência e detecção da FMECA.",
        )
    )


def scientific_context_for(question: str) -> str | None:
    """Carrega somente os contratos diretamente pertinentes à pergunta."""
    normalized = str(question or "").casefold()
    blocks: list[str] = []

    if any(term in normalized for term in _RELIABILITY_TERMS):
        blocks.append(_reliability_context())
    if any(term in normalized for term in _COMPARISON_TERMS):
        blocks.append(_comparison_context())
    if not blocks:
        return None
    header = (
        "CONTEXTO CIENTIFICO AUTORITATIVO DA EXECUCAO ATUAL\n"
        "Este conteúdo prevalece sobre memórias e artefatos legados. Não invente "
        "valores nem atribua taxas bibliográficas à base experimental."
    )
    return "\n\n".join((header, *blocks))


__all__ = ["scientific_context_for"]
