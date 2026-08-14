"""Confiabilidade física V2 sob cenários bibliográficos explícitos.

Este módulo não estima vida útil a partir do GPVS-Faults. O dataset experimental
contém séries temporais de bancada, mas não contém tempos de vida por ativo,
censura, exposição de frota ou histórico de reparo. As curvas aqui produzidas
são análises de sensibilidade de referências publicadas sob uma hipótese comum
de taxa de falha constante.

O contrato dimensional é deliberadamente rígido: todas as taxas são primeiro
normalizadas para falhas por hora e por ano; apenas depois são avaliadas as
funções exponenciais R(t), F(t), f(t) e h(t).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8_760.0
MODEL_NAME = "exponential_constant_hazard"


@dataclass(frozen=True)
class CenarioConfiabilidade:
    """Quantidade bibliográfica e contexto necessário para interpretá-la."""

    scenario_id: str
    plot_label: str
    source: str
    doi: str | None
    source_location: str
    source_type: str
    scope: str
    original_value: float
    original_unit: str
    mean_semantics: str
    caveat: str
    reported_mean_hours: float | None = None

    @property
    def lambda_per_hour(self) -> float:
        """Converte a quantidade original para uma taxa horária."""

        value = float(self.original_value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("A quantidade bibliográfica deve ser positiva e finita")
        if self.original_unit == "failures_per_hour":
            return value
        if self.original_unit == "failures_per_million_hours":
            return value / 1_000_000.0
        if self.original_unit == "failures_per_year":
            return value / HOURS_PER_YEAR
        if self.original_unit == "mean_time_to_failure_years":
            return 1.0 / (value * HOURS_PER_YEAR)
        raise ValueError(f"Unidade bibliográfica não suportada: {self.original_unit}")

    @property
    def lambda_per_year(self) -> float:
        return self.lambda_per_hour * HOURS_PER_YEAR

    @property
    def reciprocal_time_hours(self) -> float:
        return 1.0 / self.lambda_per_hour

    @property
    def reciprocal_time_years(self) -> float:
        return 1.0 / self.lambda_per_year

    def as_record(self) -> dict:
        record = asdict(self)
        record.update(
            {
                "model": MODEL_NAME,
                "hours_per_year": HOURS_PER_YEAR,
                "lambda_per_hour": self.lambda_per_hour,
                "lambda_per_year": self.lambda_per_year,
                "reciprocal_time_hours": self.reciprocal_time_hours,
                "reciprocal_time_years": self.reciprocal_time_years,
                "reported_mean_relative_difference_pct": (
                    None
                    if self.reported_mean_hours is None
                    else 100.0
                    * (self.reciprocal_time_hours - self.reported_mean_hours)
                    / self.reported_mean_hours
                ),
            }
        )
        return record


CENARIOS: tuple[CenarioConfiabilidade, ...] = (
    CenarioConfiabilidade(
        scenario_id="torres_colli_rate",
        plot_label="Torres/Colli: taxa transcrita",
        source="Torres (2024), adaptado de Colli (2015)",
        doi=None,
        source_location="TCC, Tabela 3.4, PDF p. 35 (página impressa 34)",
        source_type="secondary_bibliographic_rate",
        scope="inversor fotovoltaico genérico",
        original_value=1.75e-4,
        original_unit="failures_per_hour",
        mean_semantics="mttf_under_nonrepairable_exponential_assumption",
        caveat=(
            "Cenário de sensibilidade obtido por transcrição secundária. O cálculo "
            "posterior do TCC rotula o recíproco de uma taxa horária como anos; "
            "a V2 preserva a taxa e corrige apenas a conversão dimensional."
        ),
    ),
    CenarioConfiabilidade(
        scenario_id="cristaldi_inverter_rate",
        plot_label="Cristaldi: 1 falha em 8 anos",
        source="Cristaldi, Khalil e Soulatiantork (2017)",
        doi="10.21014/acta_imeko.v6i4.425",
        source_location="PDF p. 5, Tabela 3 e texto do modelo de Markov",
        source_type="literature_assumption",
        scope="inversor de uma string no Balance of System",
        original_value=0.125,
        original_unit="failures_per_year",
        mean_semantics="mttf_under_nonrepairable_exponential_assumption",
        caveat=(
            "A taxa do inversor é uma hipótese da literatura. O MTTF próximo de "
            "seis anos citado no artigo pertence ao conjunto string-BoS, não ao "
            "inversor isolado, cujo recíproco matemático é oito anos."
        ),
    ),
    CenarioConfiabilidade(
        scenario_id="obeidat_high_quality",
        plot_label="Obeidat: alta qualidade",
        source="Obeidat e Shuttleworth (2015)",
        doi="10.1109/PVSC.2015.7356277",
        source_location="PDF p. 5, Tabela III(a)",
        source_type="mil_hdbk_217f_prediction",
        scope="microinversor fotovoltaico de 250 W, fator de qualidade alto",
        original_value=8.069,
        original_unit="failures_per_million_hours",
        mean_semantics="mtbf_reliability_prediction",
        caveat=(
            "Predição de confiabilidade por composição e temperatura, não taxa "
            "observada em frota. O próprio artigo declara ausência de evidência "
            "de campo para confirmar essa frequência de falhas."
        ),
        reported_mean_hours=123_938.39,
    ),
    CenarioConfiabilidade(
        scenario_id="obeidat_low_quality",
        plot_label="Obeidat: baixa qualidade",
        source="Obeidat e Shuttleworth (2015)",
        doi="10.1109/PVSC.2015.7356277",
        source_location="PDF p. 5, Tabela III(b)",
        source_type="mil_hdbk_217f_prediction",
        scope="microinversor fotovoltaico de 250 W, fator de qualidade baixo",
        original_value=50.76,
        original_unit="failures_per_million_hours",
        mean_semantics="mtbf_reliability_prediction",
        caveat=(
            "Predição MIL-HDBK-217F N2 dependente da qualidade e da distribuição "
            "de temperatura; não representa observação do GPVS-Faults."
        ),
        reported_mean_hours=19_699.69,
    ),
    CenarioConfiabilidade(
        scenario_id="dhople_markov_example",
        plot_label="Dhople: exemplo Markov",
        source="Dhople e Dominguez-Garcia (2012)",
        doi="10.1109/TPWRS.2011.2165088",
        source_location="PDF p. 6, estudo de caso residencial",
        source_type="illustrative_markov_parameter",
        scope="sistema residencial com dois inversores reparáveis",
        original_value=10.0,
        original_unit="mean_time_to_failure_years",
        mean_semantics="illustrative_mttf_input_in_repairable_markov_model",
        caveat=(
            "Parâmetro ilustrativo de um estudo de caso de Markov, acompanhado de "
            "tempo médio de reparo de dez dias; não é estimativa do dataset atual."
        ),
    ),
)


def _validar_tempo_anos(t) -> np.ndarray:
    valores = np.asarray(t, dtype=float)
    if np.any(~np.isfinite(valores)) or np.any(valores < 0):
        raise ValueError("O tempo deve ser finito e não negativo")
    return valores


def _validar_taxa_ano(lambda_per_year: float) -> float:
    taxa = float(lambda_per_year)
    if not math.isfinite(taxa) or taxa <= 0:
        raise ValueError("A taxa anual deve ser positiva e finita")
    return taxa


def confiabilidade_exponencial(t_anos, lambda_per_year: float) -> np.ndarray:
    """R(t) = exp(-lambda*t), probabilidade de sobreviver além de t."""

    t = _validar_tempo_anos(t_anos)
    taxa = _validar_taxa_ano(lambda_per_year)
    return np.exp(-(taxa * t))


def probabilidade_falha_exponencial(t_anos, lambda_per_year: float) -> np.ndarray:
    """F(t) = 1 - R(t), calculada de forma estável para tempos pequenos."""

    t = _validar_tempo_anos(t_anos)
    taxa = _validar_taxa_ano(lambda_per_year)
    return -np.expm1(-(taxa * t))


def densidade_falha_exponencial(t_anos, lambda_per_year: float) -> np.ndarray:
    """f(t) = lambda*exp(-lambda*t), em probabilidade por ano."""

    taxa = _validar_taxa_ano(lambda_per_year)
    return taxa * confiabilidade_exponencial(t_anos, taxa)


def taxa_risco_exponencial(t_anos, lambda_per_year: float) -> np.ndarray:
    """h(t) = lambda, taxa de falha constante em falhas por ano."""

    t = _validar_tempo_anos(t_anos)
    taxa = _validar_taxa_ano(lambda_per_year)
    return np.full_like(t, taxa, dtype=float)


def quantil_falha_exponencial(probabilidade: float, lambda_per_year: float) -> float:
    """Tempo em que a probabilidade acumulada de falha alcança p."""

    p = float(probabilidade)
    if not 0.0 < p < 1.0:
        raise ValueError("A probabilidade deve pertencer ao intervalo aberto (0, 1)")
    taxa = _validar_taxa_ano(lambda_per_year)
    return -math.log1p(-p) / taxa


def tabela_cenarios() -> pd.DataFrame:
    return pd.DataFrame([cenario.as_record() for cenario in CENARIOS])


def curvas_cenarios(horizonte_anos: float = 20.0, n_points: int = 401) -> pd.DataFrame:
    horizonte = float(horizonte_anos)
    if not math.isfinite(horizonte) or horizonte <= 0:
        raise ValueError("O horizonte deve ser positivo e finito")
    if int(n_points) < 2:
        raise ValueError("A grade deve conter ao menos dois pontos")

    t = np.linspace(0.0, horizonte, int(n_points))
    blocos = []
    for cenario in CENARIOS:
        taxa = cenario.lambda_per_year
        blocos.append(
            pd.DataFrame(
                {
                    "scenario_id": cenario.scenario_id,
                    "time_years": t,
                    "reliability": confiabilidade_exponencial(t, taxa),
                    "cumulative_failure_probability": (
                        probabilidade_falha_exponencial(t, taxa)
                    ),
                    "failure_density_per_year": densidade_falha_exponencial(t, taxa),
                    "hazard_per_year": taxa_risco_exponencial(t, taxa),
                }
            )
        )
    return pd.concat(blocos, ignore_index=True)


def marcos_cenarios() -> pd.DataFrame:
    linhas = []
    for cenario in CENARIOS:
        taxa = cenario.lambda_per_year
        linhas.append(
            {
                "scenario_id": cenario.scenario_id,
                "b1_years": quantil_falha_exponencial(0.01, taxa),
                "b10_years": quantil_falha_exponencial(0.10, taxa),
                "median_years": quantil_falha_exponencial(0.50, taxa),
                "reciprocal_time_years": 1.0 / taxa,
                "reliability_1_year": float(confiabilidade_exponencial(1.0, taxa)),
                "reliability_5_years": float(confiabilidade_exponencial(5.0, taxa)),
                "reliability_10_years": float(confiabilidade_exponencial(10.0, taxa)),
                "failure_probability_1_year": float(
                    probabilidade_falha_exponencial(1.0, taxa)
                ),
                "failure_probability_5_years": float(
                    probabilidade_falha_exponencial(5.0, taxa)
                ),
                "failure_probability_10_years": float(
                    probabilidade_falha_exponencial(10.0, taxa)
                ),
            }
        )
    return pd.DataFrame(linhas)


def auditoria_dimensional() -> list[dict]:
    """Registra divergências sem alterar silenciosamente as fontes."""

    return [
        {
            "audit_id": "torres_rate_reciprocal_unit",
            "source_location": "Torres (2024), PDF p. 61 (página impressa 60)",
            "source_expression": "1 / (1,8 x 10^-4 falha/h) = 5.555,55 anos",
            "dimensional_result": "5.555,55 horas, aproximadamente 0,634 ano",
            "status": "source_unit_inconsistency",
            "treatment": (
                "A V2 usa a taxa exata 1,75 x 10^-4 falha/h da Tabela 3.4 e "
                "mantém esta divergência como ressalva bibliográfica."
            ),
        },
        {
            "audit_id": "torres_repair_rate_reciprocal_unit",
            "source_location": "Torres (2024), PDF p. 61 (página impressa 60)",
            "source_expression": "1 / (0,0833 reparo/h) = 12 anos",
            "dimensional_result": "aproximadamente 12 horas",
            "status": "source_unit_inconsistency",
            "treatment": "A taxa de reparo não é usada nas curvas V2.",
        },
        {
            "audit_id": "cristaldi_scope_distinction",
            "source_location": "Cristaldi et al. (2017), PDF pp. 5 e 7",
            "source_expression": "lambda_inversor = 0,125/ano; MTTF string-BoS ~ 6 anos",
            "dimensional_result": "1/lambda_inversor = 8 anos",
            "status": "different_system_scopes",
            "treatment": "Não identificar o MTTF do string-BoS como MTTF do inversor.",
        },
        {
            "audit_id": "obeidat_prediction_not_field_observation",
            "source_location": "Obeidat e Shuttleworth (2015), PDF pp. 5-6",
            "source_expression": "MIL-HDBK-217F N2 com qualidade e temperatura",
            "dimensional_result": "taxas e MTBFs preditos, com arredondamento coerente",
            "status": "prediction_only",
            "treatment": "Rotular como predição; não atribuir ao GPVS-Faults.",
        },
        {
            "audit_id": "dhople_illustrative_markov_input",
            "source_location": "Dhople e Dominguez-Garcia (2012), PDF p. 6",
            "source_expression": "MTTF = 10 anos; MTTR = 10 dias",
            "dimensional_result": "lambda = 0,1/ano sob hipótese exponencial",
            "status": "illustrative_parameter",
            "treatment": "Usar apenas como cenário ilustrativo de sensibilidade.",
        },
    ]

