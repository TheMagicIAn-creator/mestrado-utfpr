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

# Sem `"pod"` isolado: a busca é `in` sobre o texto casefolded e "pod" casa
# dentro de "pode", "podem", "poderia" — o bloco E2 dispararia em quase tudo.
_DETECTABILITY_TERMS = (
    "detectabilidade",
    "detectabilidade por magnitude",
    "magnitude",
    "severidade",
    "a_det",
    "curva pod",
    "curvas pod",
    "probability of detection",
    "injeção",
    "injecao",
    "fidelidade da injeção",
    "fidelidade da injecao",
    "e2",
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
            "Ponto operacional canônico: média dos cinco maiores erros "
            "quadráticos por feature e p99 solicitado; não é ótimo universal. "
            "p99,9 é referência histórica apenas: com as 210 janelas de "
            "calibração ele caía no máximo amostral. A sensibilidade "
            "usa k={5,10,20} por p={99;99,5;99,9}. No AE-LSTM, a decisão em W_t "
            "usa contexto causal contínuo.",
            "Não apresente métricas como resultados autônomos do dataset. "
            "Elas comparam exclusivamente os dois detectores.",
        )
    )


def _detectability_context() -> str:
    from src.webapp.contracts import e2_contract

    e2 = e2_contract()
    linhas = []
    for injecao in e2["injections"]:
        for model_id, resumo in injecao["models"].items():
            weibull = (
                "Weibull adotada"
                if resumo["weibull_2p_adotada"]
                else f"Weibull REJEITADA ({resumo['weibull_2p_motivo_da_rejeicao']})"
            )
            linhas.append(
                f"- {injecao['injection_id']}/{model_id}: "
                f"fração detectada={resumo['fracao_detectada']:.3f}; "
                f"a10 empírico={resumo['a10_empirico']}; "
                f"a50 empírico={resumo['a50_empirico']}; {weibull}."
            )
    fidelidade = "; ".join(
        f"{item['injection_id']}/{item['model']}: POD(a=1)={item['pod_a1']:.3f} "
        f"contra recall E3 {item['e3_recall_min']}-{item['e3_recall_max']}"
        for item in e2["injection_fidelity"]["comparisons"]
        if item["e3_recall_min"] is not None
    )
    metodos = "; ".join(
        f"{item['injection_id']}: {item['injection_method']} "
        f"(simulação física: {'sim' if item['physical_simulation'] else 'não'})"
        for item in e2["injections"]
    )
    return "\n".join(
        (
            "CONTRATO AUTORITATIVO - DETECTABILIDADE POR MAGNITUDE (E2)",
            "A E2 responde a partir de que MAGNITUDE cada modelo congelado passa "
            "a detectar. Ela NÃO é evidência de bancada e NÃO é degrau "
            "intermediário entre E1 e E3: é outra pergunta, sobre falha "
            "construída por injeção em janela saudável.",
            "O eixo é `a`, fração da assinatura nominal em [0, 1]. `a` não é "
            "tempo, ciclo, vida consumida, taxa de falha nem RUL. A Weibull "
            "ajustada aqui descreve dispersão de MAGNITUDE, nunca "
            "confiabilidade física.",
            f"Métodos de injeção por item: {metodos}. "
            "`measured_state_interpolation` percorre o caminho entre dois "
            "estados medidos e não simula física.",
            "Resultados publicados por item e modelo:",
            *linhas,
            "Percentil paramétrico de ajuste rejeitado NÃO se cita; o empírico "
            "vem primeiro e sempre. Trajetória que não cruza até a=1 é "
            "censurada, não `a_det = 1`.",
            f"Fidelidade da injeção em a=1: {fidelidade}.",
            e2["separation"]["caveat"],
            "Nunca apresente número de E2 como se fosse evidência de bancada, e "
            "nunca chame POD de recall.",
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
    # Lido do contrato, não escrito à mão: foi exatamente uma frase fixa sobre a
    # FMECA que ficou falsa quando o escopo passou de 3 para 6 itens validados.
    fmeca = reliability["fmeca"]
    escopo = ", ".join(
        f"{item['component_id']} (NPR={item['npr']})"
        for item in fmeca["components"]
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
            f"Escopo vigente da FMECA ({fmeca['status']}, "
            f"{len(fmeca['components'])} itens, NPR = S*O*D): {escopo}.",
            "Só IGBT, sensor/realimentação e controle têm contrapartida de "
            "ensaio no GPVS. Métricas de detector não recalculam S, O, D nem "
            "NPR, e a ponte POD_mon -> D_mon -> NPR_proj continua bloqueada.",
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
    if any(term in normalized for term in _DETECTABILITY_TERMS):
        blocks.append(_detectability_context())
    if not blocks:
        return None
    header = (
        "CONTEXTO CIENTIFICO AUTORITATIVO DA EXECUCAO ATUAL\n"
        "Este conteúdo prevalece sobre memórias e artefatos legados. Não invente "
        "valores nem atribua taxas bibliográficas à base experimental."
    )
    return "\n\n".join((header, *blocks))


__all__ = ["scientific_context_for"]
