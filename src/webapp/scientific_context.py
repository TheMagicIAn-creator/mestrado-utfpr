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
    if item["estimate"] is None:
        return f"N/A (n válido={item['n_valid_experiments']})"
    return (
        f"{item['estimate']:.6f} "
        f"(IC95% {item['ci95_low']:.6f} a {item['ci95_high']:.6f}; "
        f"n válido={item['n_valid_experiments']}/{item['n_experiments']})"
    )


def _comparison_context() -> str:
    from src.webapp.contracts import e3_contract

    e3 = e3_contract()
    dense = e3["metrics"]["ae_denso"]
    lstm = e3["metrics"]["ae_lstm"]
    difference = next(
        item for item in e3["paired_differences"] if item["metric"] == "recall"
    )
    return "\n".join(
        (
            "CONTRATO AUTORITATIVO - COMPARACAO DENSO VERSUS AE-LSTM",
            "Dataset experimental único: "
            f"{e3['dataset']['name']} (DOI {e3['dataset']['doi']}). O nome do "
            "dataset identifica a proveniência; os resultados pertencem aos modelos.",
            "Modelos congelados antes dos ensaios: Autoencoder Denso "
            "24-16-8-16-24 e AE-LSTM temporal L=8, hidden=32, latent=8.",
            "Métricas principais na ordem definida pelo pesquisador: Recall, F1 e "
            "Precision. Precision é N/A quando não há alarmes positivos.",
            f"Recall macro do Denso: {_metric(dense, 'recall')}.",
            f"Recall macro do AE-LSTM: {_metric(lstm, 'recall')}.",
            f"F1 macro do Denso: {_metric(dense, 'f1')}; "
            f"AE-LSTM: {_metric(lstm, 'f1')}.",
            f"Precision macro do Denso: {_metric(dense, 'precision')}; "
            f"AE-LSTM: {_metric(lstm, 'precision')}.",
            "Diferença pareada Denso menos LSTM em Recall: "
            f"{difference['difference_dense_minus_lstm']:.6f} "
            f"(IC95% {difference['ci95_low']:.6f} a "
            f"{difference['ci95_high']:.6f}; n=14 ensaios).",
            "Métricas complementares de discriminação: "
            f"ROC-AUC do Denso: {_metric(dense, 'auc_roc')}; "
            f"AE-LSTM: {_metric(lstm, 'auc_roc')}.",
            f"PR-AUC do Denso: {_metric(dense, 'auc_pr')}; "
            f"AE-LSTM: {_metric(lstm, 'auc_pr')}.",
            f"Especificidade do Denso: {_metric(dense, 'specificity')}; "
            f"AE-LSTM: {_metric(lstm, 'specificity')}.",
            "Escore: média dos cinco maiores erros quadráticos por feature; no "
            "AE-LSTM, somente no último passo. Limiar saudável p99,9 solicitado, "
            "com order statistic e percentil efetivo registrados por modelo.",
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
