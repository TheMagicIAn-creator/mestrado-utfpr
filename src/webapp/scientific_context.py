"""Contexto autoritativo que reconcilia o agente com os resultados publicados."""

from __future__ import annotations

_RESULT_TERMS = (
    "resultado",
    "autoencoder",
    "denso",
    "lstm",
    "auc",
    "sensibilidade",
    "especificidade",
    "limiar",
    "matriz",
    "roc",
    "precision",
    "confiabilidade",
    "weibull",
    "taxa de falha",
    "fmeca",
    "npr",
    "smd95",
    "gpvs",
    "detector",
    "dissertação",
    "dissertacao",
    "orientadora",
)


def _metric(metrics: dict, name: str) -> str:
    item = metrics[name]
    return (
        f"{item['estimate']:.6f} "
        f"(IC95% {item['ci95_low']:.6f} a {item['ci95_high']:.6f}; "
        f"n={item['n_experiments']})"
    )


def scientific_context_for(question: str) -> str | None:
    """Resume contratos publicados quando a pergunta toca os resultados."""
    normalized = str(question or "").casefold()
    if not any(term in normalized for term in _RESULT_TERMS):
        return None

    from src.webapp.contracts import e2_contract, e3_contract, reliability_contract

    e3 = e3_contract()
    e2 = e2_contract()
    reliability = reliability_contract()
    dense = e3["metrics"]["ae_denso"]
    lstm = e3["metrics"]["ae_lstm"]
    difference = next(
        item for item in e3["paired_differences"] if item["metric"] == "auc_pr"
    )
    smd = "; ".join(
        (
            f"{item['model_name']}/{item['component_name']}: "
            + (
                f"{item['smd95']:.2f}"
                if item["smd95"] is not None
                else "não atingido até a_det=1"
            )
        )
        for item in e2["summary"]
    )
    rates = "; ".join(
        f"{item['plot_label']}: lambda={item['lambda_per_hour']:.3e} h^-1 "
        f"({item['evidence_type']})"
        for item in reliability["scenarios"]
    )

    return "\n".join(
        (
            "CONTRATO CIENTIFICO AUTORITATIVO DA EXECUCAO ATUAL",
            "Este bloco prevalece sobre memórias de sessões e artefatos legados. "
            "Não invente valores, não misture datasets e não chame detectabilidade "
            "sintética de confiabilidade física.",
            f"Dataset experimental único: {e3['dataset']['name']} "
            f"(DOI {e3['dataset']['doi']}); 16 ensaios, dos quais 14 E3 com falha.",
            "Modelos comparados sob o mesmo protocolo: Autoencoder Denso "
            "24-16-8-16-24 e AE-LSTM temporal L=8, hidden=32, latent=8. "
            "Semente de referência 42; estabilidade em 13, 29, 42, 71 e 101.",
            f"AUC-PR macro do Denso: {_metric(dense, 'auc_pr')}.",
            f"AUC-PR macro do AE-LSTM: {_metric(lstm, 'auc_pr')}.",
            f"Diferença pareada Denso menos LSTM em AUC-PR: "
            f"{difference['difference_dense_minus_lstm']:.6f} "
            f"(IC95% {difference['ci95_low']:.6f} a "
            f"{difference['ci95_high']:.6f}; n=14 ensaios).",
            f"AUC-ROC do Denso: {_metric(dense, 'auc_roc')}; "
            f"AE-LSTM: {_metric(lstm, 'auc_roc')}.",
            f"Sensibilidade do Denso: {_metric(dense, 'sensitivity')}; "
            f"AE-LSTM: {_metric(lstm, 'sensitivity')}.",
            f"Especificidade do Denso: {_metric(dense, 'specificity')}; "
            f"AE-LSTM: {_metric(lstm, 'specificity')}.",
            "Pesos, scaler e limiares foram congelados antes dos 14 ensaios de "
            "falha. A fronteira de 50% do registro é nominal porque não há canal "
            "instrumentado de disparo nos CSVs.",
            f"E2 SMD95: {smd}.",
            "O eixo E2 a_det é fração adimensional da assinatura sintética, não "
            "tempo. Os seis ajustes Weibull foram recusados apenas para síntese "
            "paramétrica; isso não reprova os detectores.",
            f"Confiabilidade física usa R(t)=exp(-lambda*t), F(t)=1-R(t), "
            f"f(t)=lambda*exp(-lambda*t) e h(t)=lambda. Cenários: {rates}.",
            "Weibull físico não é estimável no GPVS-Faults por ausência de tempos "
            "de vida, exposição e censura por ativo.",
        )
    )


__all__ = ["scientific_context_for"]
