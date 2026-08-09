"""
relatorio_weibull.py — Al IAdo PV

Montagem do artefato de saída da etapa de Weibull/RUL: o dicionário que vira
`weibull_results.json` e as linhas que viram `weibull_tabela.csv`.

POR QUE ESTE MÓDULO EXISTE
==========================
`rul_weibull.py` passou de mil linhas quando o eixo foi renomeado para `a_det`
e os desfechos passaram a distinguir indetectabilidade de censura. A costura
escolhida é esta: a MATEMÁTICA (ajuste censurado, Kaplan-Meier, curvas) fica
lá; a MONTAGEM DO ARTEFATO vem para cá.

Não importa `rul_weibull` — receberia um ciclo de import. Tudo que precisa
chega por parâmetro, o que também torna a montagem testável sem rodar o
pipeline inteiro.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations


def montar_relatorio(
    *,
    params: dict,
    a_dets_dict: dict,
    eventos_dict: dict,
    falhas: list,
    meta_holdout: dict,
    metadados_tempo: dict,
    limiar: float,
    n_traj_max: int,
    n_traj_real: int,
    n_steps: int,
    a_det_unidade: str,
    ttf_unidade: str,
    tempo_fisico_calibrado: bool,
    tempo_fisico_nota: str,
    min_eventos_weibull: int,
    max_censura_rul_pct: float,
    min_r2_papel_weibull: float,
    persistencia_cruzamento: int,
    json_seguro,
) -> tuple[dict, list[dict]]:
    """Devolve `(relatorio, linhas_da_tabela)`. Não escreve em disco."""
    relatorio = {
        "__meta__": {
            "evidence_level": "E2",
            "evidence_note": (
                "DETECTABILIDADE E2 — duplamente sintética: (1) os a_det vêm de "
                "trajetórias de magnitude crescente SIMULADAS cruzando o limiar "
                "do Autoencoder, não de dados run-to-failure reais; (2) a "
                "própria falha que define o cruzamento é injeção sintética "
                "orientada pela FMECA. Demonstra a METODOLOGIA "
                "(a_det→Weibull), NÃO é confiabilidade nem vida útil de campo. "
                "ATENÇÃO AO EIXO: é magnitude de assinatura em [0; 1], não "
                "tempo. Os nomes canônicos são média de a_det, a10 e margem "
                "residual; MTTF/B10/RUL permanecem apenas como aliases legados. "
                "As não detecções entram no MLE "
                "como censura à direita sob hipótese declarada (ver 'desfechos'); "
                "os intervalos vêm de bootstrap de trajetórias."
            ),
            "a_det_origem": "trajetorias_de_magnitude_crescente_cruzando_limiar_AE",
            "ttf_origem": "trajetorias_de_magnitude_crescente_cruzando_limiar_AE",
            "tempo": metadados_tempo,
            "adequacy_note": (
                "RMSE-KM e R2 no papel censura-aware são descritivos. A triagem "
                "R2 não é teste formal e não substitui validação externa."
            ),
            "physical_claims": {
                "rul": False,
                "reliability": False,
                "failure_rate": False,
                "wear_regime_from_beta": False,
            },
            "protocolo_avaliacao": meta_holdout,
        },
        "parametros_simulacao": {
            "n_trajetorias_max": n_traj_max,
            "n_trajetorias_efetivas": n_traj_real,
            "n_steps"      : n_steps,
            "a_det_unidade": a_det_unidade,
            "a_det_por_passo": 1.0 / (n_steps - 1),
            "ttf_unidade": ttf_unidade,
            "rul_unidade": ttf_unidade,
            "tempo_fisico_calibrado": tempo_fisico_calibrado,
            "tempo_fisico_nota": tempo_fisico_nota,
            "limiar"       : float(limiar),
            "min_eventos_weibull": min_eventos_weibull,
            "max_censura_rul_pct": max_censura_rul_pct,
            "min_r2_papel_weibull": min_r2_papel_weibull,
            "persistencia_cruzamento": persistencia_cruzamento,
        },
        "falhas": {}
    }
    for falha in falhas:
        fid = falha["id"]
        relatorio["falhas"][fid] = {
            "nome"  : falha["nome"],
            "npr"   : falha["npr"],
            "weibull": json_seguro(params[fid]),
            "ajuste_weibull_adequado": None,
            "sintese_parametrica_recomendada": params[fid].get(
                "resumo_parametrico_recomendado", False
            ),
            "status_ajuste": (
                "nao_estimavel_parametrico_rul_restrita"
                if not params[fid]["fit_converged"]
                else "nao_recomendado_alta_indetectabilidade"
                if params[fid]["rul_parametrica_alta_incerteza"]
                else "nao_recomendado_desvio_papel_weibull"
                if not params[fid].get("triagem_papel_compativel", False)
                else "exploratorio_detectabilidade"
            ),
            "ressalva_ajuste": (
                "Ajuste do primeiro cruzamento do detector sob hipótese de "
                "censura analítica; não equivale a falha, vida ou desgaste."
            ),
            "a_dets": a_dets_dict[fid].tolist(),
            # Alias: mesma lista, nome antigo, para leitores já escritos.
            "ttfs"  : a_dets_dict[fid].tolist(),
            "eventos_observados": eventos_dict[fid].tolist(),
            "desfechos": params[fid]["desfechos"],
        }
    linhas_weibull = []
    for falha in falhas:
        fid = falha["id"]
        p = params[fid]
        linhas_weibull.append({
            "falha": falha["nome"],
            "npr": falha["npr"],
            "n_traj": p["n_traj"],
            "n_eventos": p["n_eventos"],
            "n_censurados": p["n_censurados"],
            "censura_pct": p["censura_pct"],
            "n_indetectaveis_no_teto": p["desfechos"]["n_indetectaveis_no_teto"],
            "n_censura_genuina": p["desfechos"]["n_censura_genuina"],
            "pod_mon_no_teto": p["desfechos"]["pod_mon_no_teto"],
            "a_det_unidade": p["a_det_unidade"],
            "ttf_unidade": p["ttf_unidade"],
            "rul_unidade": p["rul_unidade"],
            "tempo_fisico_calibrado": p["tempo_fisico_calibrado"],
            "beta": p["beta"],
            "beta_ci_low": p["beta_ci95"][0],
            "beta_ci_high": p["beta_ci95"][1],
            "eta": p["eta"],
            "eta_ci_low": p["eta_ci95"][0],
            "eta_ci_high": p["eta_ci95"][1],
            "mttf": p["mttf"],
            "mttf_ci_low": p["mttf_ci95"][0],
            "mttf_ci_high": p["mttf_ci95"][1],
            "b10": p["b10"],
            "b10_ci_low": p["b10_ci95"][0],
            "b10_ci_high": p["b10_ci95"][1],
            "km_rmse": p["km_rmse"],
            "media_a_det_parametrica": p["media_a_det_parametrica"],
            "media_a_det_parametrica_ci_low": p["media_a_det_parametrica_ci95"][0],
            "media_a_det_parametrica_ci_high": p["media_a_det_parametrica_ci95"][1],
            "a10_parametrico": p["a10_parametrico"],
            "a10_parametrico_ci_low": p["a10_parametrico_ci95"][0],
            "a10_parametrico_ci_high": p["a10_parametrico_ci95"][1],
            "papel_weibull_r2": p["diagnostico_papel_weibull"]["r2"],
            "papel_weibull_rmse": p["diagnostico_papel_weibull"]["rmse"],
            "sintese_parametrica_recomendada": p["resumo_parametrico_recomendado"],
            "bootstrap_solicitados": p["bootstrap_solicitados"],
            "bootstrap_validos": p["bootstrap_validos"],
            "bootstrap_taxa_validos": p["bootstrap_taxa_validos"],
            "fit_converged": p["fit_converged"],
            "rul_reportavel": p["rul_reportavel"],
            "rul_parametrica_disponivel": p["rul_parametrica_disponivel"],
            "rul_parametrica_alta_incerteza": p["rul_parametrica_alta_incerteza"],
            "rul_restrita_disponivel": p["rul_restrita_disponivel"],
            "rul_restrita_horizonte": p["rul_restrita_horizonte"],
            "rul_restrita_inicial": p["rul_restrita_inicial"],
            "margem_restrita_disponivel": p["margem_restrita_disponivel"],
            "margem_restrita_horizonte": p["margem_restrita_horizonte"],
            "margem_restrita_inicial": p["margem_restrita_inicial"],
            "status_ajuste": relatorio["falhas"][fid]["status_ajuste"],
            "evidence_level": "E2",
        })
    return relatorio, linhas_weibull
