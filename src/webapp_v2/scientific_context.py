"""Contexto autoritativo que mantem agente e figuras V2 reconciliados."""

from __future__ import annotations

_RESULT_TERMS = (
    "resultado",
    "autoencoder",
    "pca",
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
    "gpvs",
    "detector",
    "dissertacao",
    "dissertação",
    "orientadora",
)


def _metric(method: dict, name: str) -> str:
    item = method[name]
    return (
        f"{item['mean']:.6f} "
        f"(IC95% {item['ci95_low']:.6f} a {item['ci95_high']:.6f}; "
        f"n={item['n_experiments']})"
    )


def scientific_context_for(question: str) -> str | None:
    """Resume os contratos publicados quando a pergunta toca resultados."""
    normalized = str(question or "").casefold()
    if not any(term in normalized for term in _RESULT_TERMS):
        return None

    from src.webapp_v2.contracts import dashboard_contract

    contract = dashboard_contract()
    comparison = {
        item["method_id"]: item for item in contract["overview"]["method_comparison"]
    }
    ae = comparison["autoencoder_v2"]
    pca = comparison["pca"]
    threshold = contract["autoencoder"]["threshold"]
    physical = contract["reliability"]["physical_weibull"]
    components = contract["fmeca"]["components"]

    return "\n".join(
        (
            "CONTRATO CIENTIFICO AUTORITATIVO DA APLICACAO V2",
            "Este bloco prevalece sobre memoria de sessoes, generalizacoes teoricas e "
            "artefatos legados. Nao invente valores nem declare superioridade global. "
            "Nao ha superioridade global entre Autoencoder V2 e PCA.",
            "As diferencas observadas nao demonstram causalidade da nao linearidade. "
            "Nao use sobreposicao de IC95% como prova de equivalencia, nem chame ganhos "
            "de estatisticamente significativos sem o teste pareado publicado.",
            "A taxa de falso positivo no teste saudavel canonico e uma verificacao de "
            "calibracao distinta da especificidade macro nos 14 ensaios E3.",
            f"Dataset experimental unico: {contract['project']['dataset']} "
            f"(DOI {contract['project']['dataset_doi']}); evidencia E3 de bancada.",
            f"Veredito publicado: {contract['overview']['verdict']}",
            f"Autoencoder V2, AUC ROC: {_metric(ae, 'auc_roc')}.",
            f"PCA, AUC ROC: {_metric(pca, 'auc_roc')}.",
            f"Autoencoder V2, sensibilidade: {_metric(ae, 'sensitivity')}.",
            f"PCA, sensibilidade: {_metric(pca, 'sensitivity')}.",
            f"Autoencoder V2, especificidade: {_metric(ae, 'specificity')}.",
            f"PCA, especificidade: {_metric(pca, 'specificity')}.",
            f"Autoencoder V2, acuracia balanceada: {_metric(ae, 'balanced_accuracy')}.",
            f"PCA, acuracia balanceada: {_metric(pca, 'balanced_accuracy')}.",
            f"Arquitetura canonica: {contract['autoencoder']['architecture']['display']}; "
            f"seed {contract['autoencoder']['architecture']['canonical_seed']}; "
            f"{contract['autoencoder']['architecture']['trainable_parameters']} parametros.",
            f"Limiar canonico: {threshold['value']:.12f}; "
            f"falso positivo no teste saudavel: "
            f"{threshold['healthy_test']['taxa_pct']:.6f}%.",
            "Confiabilidade fisica: os cinco cenarios sao bibliograficos e usam "
            "hipotese exponencial; nao sao vidas observadas no GPVS-Faults.",
            f"Weibull fisico: beta={physical.get('beta')}, eta={physical.get('eta')}; "
            "nao estimavel sem tempos de vida, exposicao e censura.",
            "FMECA oficial: "
            + "; ".join(f"{item['component']} NPR={item['npr']}" for item in components)
            + ".",
            "F1-F7 sao classes de ensaio do dataset, nao componentes FMECA. "
            "Detectabilidade experimental nao equivale a tempo, RUL ou taxa fisica de falha.",
        )
    )
