"""Confiabilidade física bibliográfica e contrato atual da FMECA.

O GPVS-Faults não contém tempos de vida. Este módulo implementa somente
cenários históricos de sensibilidade exponenciais rastreáveis ao TCC de Torres
(2024), sem estimar parâmetros físicos a partir dos detectores ou da validação
E3.

A FMECA atual possui escopo próprio. Os valores S/O/D foram definidos pelo
pesquisador para IGBT, sistema de sensor/realimentação e sistema/circuito de
controle do inversor. A rastreabilidade documental específica desses valores
deve ser registrada separadamente.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


HOURS_PER_YEAR = 8_760.0
INVERTER_RATE_PER_HOUR = 1.75e-4

SOURCE_PDF = (
    "literatura/inversores-pv/"
    "torres_aplicacao-da-metodologia-reliability-centred-maintenance-a-s_2024.pdf"
)


@dataclass(frozen=True)
class ReliabilityScenario:
    scenario_id: str
    component_id: str
    component_name: str
    plot_label: str
    evidence_type: str
    lambda_per_hour: float
    source_pdf: str
    pdf_page: int
    printed_page: int
    source_table: str
    original_expression: str
    conversion_formula: str
    ticket_share: float | None
    caveat: str

    @property
    def lambda_per_year(self) -> float:
        return float(self.lambda_per_hour * HOURS_PER_YEAR)

    @property
    def reciprocal_time_hours(self) -> float:
        return float(1.0 / self.lambda_per_hour)

    @property
    def reciprocal_time_years(self) -> float:
        return float(self.reciprocal_time_hours / HOURS_PER_YEAR)

    def as_record(self) -> dict:
        return {
            **asdict(self),
            "lambda_per_year": self.lambda_per_year,
            "reciprocal_time_hours": self.reciprocal_time_hours,
            "reciprocal_time_years": self.reciprocal_time_years,
            "time_model": "exponential_constant_hazard",
            "time_unit_primary": "hour",
            "hours_per_year": HOURS_PER_YEAR,
            "scientific_role": "historical_tcc_reliability_sensitivity",
        }


@dataclass(frozen=True)
class FmecaComponent:
    component_id: str
    component_name: str
    function: str
    failure_mode: str
    native_experiments: tuple[str, ...]
    severity: int | None = None
    occurrence: int | None = None
    detectability: int | None = None
    npr: int | None = None
    status: str = "awaiting_user_fmeca"

    def __post_init__(self) -> None:
        """Valida a consistência interna dos valores da FMECA."""

        scores = (
            self.severity,
            self.occurrence,
            self.detectability,
        )

        provided_scores = [value is not None for value in scores]

        # Não permite FMECA parcialmente preenchida.
        if any(provided_scores) and not all(provided_scores):
            raise ValueError(
                f"FMECA parcialmente preenchida para {self.component_name}. "
                "Severity, occurrence e detectability devem ser informados em conjunto."
            )

        # Se S/O/D não existem, NPR também não deve existir.
        if not any(provided_scores):
            if self.npr is not None:
                raise ValueError(
                    f"NPR informado sem S/O/D para {self.component_name}."
                )
            return

        # Garante que os escores sejam inteiros positivos.
        for field_name, value in (
            ("severity", self.severity),
            ("occurrence", self.occurrence),
            ("detectability", self.detectability),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{field_name} de {self.component_name} deve ser inteiro."
                )

            if value <= 0:
                raise ValueError(
                    f"{field_name} de {self.component_name} deve ser positivo."
                )

        expected_npr = (
            int(self.severity)
            * int(self.occurrence)
            * int(self.detectability)
        )

        # Se o pesquisador não informar NPR, calcula automaticamente.
        if self.npr is None:
            object.__setattr__(self, "npr", expected_npr)

        # Se informar, verifica obrigatoriamente a consistência.
        elif self.npr != expected_npr:
            raise ValueError(
                f"NPR inconsistente para {self.component_name}: "
                f"esperado {expected_npr}, recebido {self.npr}."
            )

    @property
    def calculation_enabled(self) -> bool:
        """Indica se S/O/D/NPR estão integralmente definidos."""

        return all(
            value is not None
            for value in (
                self.severity,
                self.occurrence,
                self.detectability,
                self.npr,
            )
        )

    def as_record(self) -> dict:
        return {
            **asdict(self),
            "native_experiments": list(self.native_experiments),
            "calculation_enabled": self.calculation_enabled,
        }


FMECA_COMPONENTS = (
    FmecaComponent(
        component_id="igbt",
        component_name="IGBT",
        function="Realizar o chaveamento da conversão CC-CA",
        failure_mode="Falha completa de um IGBT",
        native_experiments=("F1L", "F1M"),
        severity=5,
        occurrence=6,
        detectability=5,
        npr=150,
        status="validated",
    ),
    FmecaComponent(
        component_id="sensor_feedback_system",
        component_name="Sistema de sensor/realimentação",
        function="Medir e realimentar as variáveis elétricas usadas pelo controle",
        failure_mode="Erro de 20% no sistema de sensor/realimentação",
        native_experiments=("F2L", "F2M"),
        severity=5,
        occurrence=8,
        detectability=7,
        npr=280,
        status="validated",
    ),
    FmecaComponent(
        component_id="inverter_control_system",
        component_name="Sistema/circuito de controle do inversor",
        function="Regular a operação MPPT/IPPT por meio do controlador PI",
        failure_mode=(
            "Anomalia funcional no ganho ou na constante de tempo "
            "do controlador PI"
        ),
        native_experiments=("F6L", "F6M", "F7L", "F7M"),
        severity=5,
        occurrence=6,
        detectability=8,
        npr=240,
        status="validated",
    ),
)


SCENARIOS = (
    ReliabilityScenario(
        scenario_id="contator_ac_derived",
        component_id="contator_ac",
        component_name="Contator AC",
        plot_label="Contator AC - cenário derivado",
        evidence_type="derived_sensitivity",
        lambda_per_hour=2.10e-5,
        source_pdf=SOURCE_PDF,
        pdf_page=35,
        printed_page=34,
        source_table="Tabelas 3.3 e 3.4",
        original_expression="1,75e-4 falha/h x 12% dos chamados",
        conversion_formula="lambda_contator = lambda_inversor * 0.12",
        ticket_share=0.12,
        caveat=(
            "Alocação proporcional de uma taxa agregada de inversor; não é uma "
            "taxa de falha observada diretamente para contatores."
        ),
    ),
    ReliabilityScenario(
        scenario_id="igbt_derived",
        component_id="igbt",
        component_name="IGBT",
        plot_label="IGBT - cenário derivado",
        evidence_type="derived_sensitivity",
        lambda_per_hour=1.05e-5,
        source_pdf=SOURCE_PDF,
        pdf_page=35,
        printed_page=34,
        source_table="Tabelas 3.3 e 3.4",
        original_expression="1,75e-4 falha/h x 6% dos chamados",
        conversion_formula="lambda_igbt = lambda_inversor * 0.06",
        ticket_share=0.06,
        caveat=(
            "Alocação proporcional de uma taxa agregada de inversor; não é uma "
            "taxa de falha observada diretamente para IGBTs."
        ),
    ),
    ReliabilityScenario(
        scenario_id="fusivel_ac_derived",
        component_id="fusivel_ac",
        component_name="Fusível AC",
        plot_label="Fusível AC - cenário derivado",
        evidence_type="derived_sensitivity",
        lambda_per_hour=7.00e-6,
        source_pdf=SOURCE_PDF,
        pdf_page=35,
        printed_page=34,
        source_table="Tabelas 3.3 e 3.4",
        original_expression="1,75e-4 falha/h x 4% dos chamados",
        conversion_formula="lambda_fusivel_derivada = lambda_inversor * 0.04",
        ticket_share=0.04,
        caveat=(
            "Alocação proporcional de uma taxa agregada de inversor. É mantida "
            "separada da taxa bibliográfica direta do fusível."
        ),
    ),
    ReliabilityScenario(
        scenario_id="fusivel_ac_direct",
        component_id="fusivel_ac",
        component_name="Fusível AC",
        plot_label="Fusível - taxa direta da Tabela 3.4",
        evidence_type="direct_bibliographic",
        lambda_per_hour=2.17e-6,
        source_pdf=SOURCE_PDF,
        pdf_page=35,
        printed_page=34,
        source_table="Tabela 3.4",
        original_expression="2,17e-6 falha/h",
        conversion_formula="lambda_ano = lambda_hora * 8760",
        ticket_share=None,
        caveat=(
            "Taxa transcrita para o subcomponente genérico fusível, adaptada de "
            "Colli (2015); não representa uma medição de campo desta pesquisa."
        ),
    ),
)


def _validate_time_hours(time_hours) -> np.ndarray:
    values = np.asarray(time_hours, dtype=float)

    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            "O tempo em horas deve ser finito e não negativo"
        )

    return values


def _validate_rate(lambda_per_hour: float) -> float:
    rate = float(lambda_per_hour)

    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(
            "A taxa por hora deve ser positiva e finita"
        )

    return rate


def reliability(
    time_hours,
    lambda_per_hour: float,
) -> np.ndarray:
    """R(t)=exp(-lambda*t), com t em horas e lambda em falhas por hora."""

    time = _validate_time_hours(time_hours)
    rate = _validate_rate(lambda_per_hour)

    return np.exp(-(rate * time))


def cumulative_failure(
    time_hours,
    lambda_per_hour: float,
) -> np.ndarray:
    """F(t)=1-R(t), calculada de forma estável para tempos pequenos."""

    time = _validate_time_hours(time_hours)
    rate = _validate_rate(lambda_per_hour)

    return -np.expm1(-(rate * time))


def failure_density(
    time_hours,
    lambda_per_hour: float,
) -> np.ndarray:
    """f(t)=lambda*exp(-lambda*t), em probabilidade por hora."""

    rate = _validate_rate(lambda_per_hour)

    return rate * reliability(time_hours, rate)


def hazard_rate(
    time_hours,
    lambda_per_hour: float,
) -> np.ndarray:
    """h(t)=lambda, constante no cenário exponencial."""

    time = _validate_time_hours(time_hours)
    rate = _validate_rate(lambda_per_hour)

    return np.full_like(
        time,
        rate,
        dtype=float,
    )


def distribution_model_contracts() -> dict:
    """Disponibilidade e parâmetros mínimos dos modelos físicos discutidos."""

    missing_lifetime_evidence = [
        "tempos individuais ate falha por ativo",
        "indicador e tempo de censura por ativo",
        "exposicao observada e identificacao da populacao",
    ]

    return {
        "exponential": {
            "status": "published_bibliographic_sensitivity",
            "parameters": {
                "lambda_per_hour": "available_by_scenario"
            },
            "required_evidence": [
                "taxa bibliografica rastreada"
            ],
            "outputs": [
                "R(t)",
                "F(t)",
                "f(t)",
                "h(t)",
            ],
        },
        "weibull_2p": {
            "status": (
                "blocked_no_traceable_igbt_parameters_in_current_corpus"
            ),
            "parameters": {
                "beta": None,
                "eta_hours": None,
            },
            "required_evidence": missing_lifetime_evidence,
            "outputs": [],
            "corpus_audit": {
                "date": "2026-09-01",
                "igbt_chunks_reviewed": 22,
                "weibull_chunks_found": 74,
                "joint_sources": [SOURCE_PDF],
                "joint_source_pages_reviewed": [35, 48],
                "finding": (
                    "A fonte comum discute IGBT e Weibull em contextos "
                    "separados, sem fornecer beta ou eta para IGBT."
                ),
            },
        },
        "normal": {
            "status": "blocked_missing_lifetime_evidence",
            "parameters": {
                "mean_hours": None,
                "std_hours": None,
            },
            "required_evidence": missing_lifetime_evidence,
            "outputs": [],
            "additional_check": (
                "suporte negativo e adequacao empirica devem ser avaliados"
            ),
        },
        "lognormal": {
            "status": "blocked_missing_lifetime_evidence",
            "parameters": {
                "mu_log_hours": None,
                "sigma_log_hours": None,
            },
            "required_evidence": missing_lifetime_evidence,
            "outputs": [],
        },
        "lifetime_histogram": {
            "status": "blocked_missing_lifetime_sample",
            "parameters": {
                "observed_lifetimes_hours": None
            },
            "required_evidence": missing_lifetime_evidence,
            "outputs": [],
        },
    }


def scenario_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            scenario.as_record()
            for scenario in SCENARIOS
        ]
    )


def component_curves(
    horizon_years: float = 20.0,
    n_points: int = 401,
) -> pd.DataFrame:
    if (
        not math.isfinite(float(horizon_years))
        or float(horizon_years) <= 0
    ):
        raise ValueError(
            "O horizonte em anos deve ser positivo e finito"
        )

    if int(n_points) < 2:
        raise ValueError(
            "A grade temporal deve ter ao menos dois pontos"
        )

    time_years = np.linspace(
        0.0,
        float(horizon_years),
        int(n_points),
    )

    time_hours = time_years * HOURS_PER_YEAR

    frames = []

    for scenario in SCENARIOS:
        rate = scenario.lambda_per_hour

        density_hour = failure_density(
            time_hours,
            rate,
        )

        hazard_hour = hazard_rate(
            time_hours,
            rate,
        )

        frames.append(
            pd.DataFrame(
                {
                    "scenario_id": scenario.scenario_id,
                    "component_id": scenario.component_id,
                    "evidence_type": scenario.evidence_type,
                    "time_hours": time_hours,
                    "time_years": time_years,
                    "reliability": reliability(
                        time_hours,
                        rate,
                    ),
                    "cumulative_failure_probability": cumulative_failure(
                        time_hours,
                        rate,
                    ),
                    "failure_density_per_hour": density_hour,
                    "hazard_per_hour": hazard_hour,
                    "failure_density_per_year": (
                        density_hour * HOURS_PER_YEAR
                    ),
                    "hazard_per_year": (
                        hazard_hour * HOURS_PER_YEAR
                    ),
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def methodology() -> dict:
    fmeca_calculation_enabled = all(
        component.calculation_enabled
        for component in FMECA_COMPONENTS
    )

    return {
        "schema_version": 7,
        "status": "bibliographic_component_sensitivity",
        "evidence_scope": "bibliographic_reliability_only",
        "time_unit_primary": "hour",
        "hours_per_year": HOURS_PER_YEAR,
        "formulas": {
            "reliability": "R(t) = exp(-lambda*t)",
            "cumulative_failure": "F(t) = 1 - R(t)",
            "failure_density": (
                "f(t) = lambda*exp(-lambda*t)"
            ),
            "hazard": "h(t) = lambda",
        },
        "fmeca": {
            "status": "validated",
            "calculation_enabled": fmeca_calculation_enabled,
            "formula": "NPR = S * O * D",

            # Estes campos continuam nulos até que a rastreabilidade
            # bibliográfica específica dos escores atuais seja documentada.
            "traceability_status": "pending_source_documentation",
            "source_pdf": None,
            "source_page": None,
            "source_table": None,

            "components": [
                component.as_record()
                for component in FMECA_COMPONENTS
            ],

            "priority_order": [
                "sensor_feedback_system",
                "inverter_control_system",
                "igbt",
            ],

            "boundary": (
                "A validação E3 mede detecção de anomalias e não fornece "
                "valores ordinais S/O/D nem recalcula NPR."
            ),

            "legacy_tcc_scope": {
                "status": "historical_not_canonical",
                "components": [
                    "Contator AC",
                    "IGBT",
                    "Fusível AC",
                ],
                "source_pdf": SOURCE_PDF,
                "pdf_page": 35,
                "printed_page": 34,
                "source_table": "Tabela 3.3",
            },
        },

        "distribution_models": distribution_model_contracts(),

        "physical_weibull": {
            "status": (
                "blocked_no_traceable_igbt_parameters_in_current_corpus"
            ),
            "beta": None,
            "eta": None,
            "reason": (
                "O corpus não fornece beta/eta de IGBT nem tempos "
                "individuais de falha, exposição e censura por ativo."
            ),
        },

        "scenarios": [
            scenario.as_record()
            for scenario in SCENARIOS
        ],
    }


__all__ = [
    "FMECA_COMPONENTS",
    "FmecaComponent",
    "HOURS_PER_YEAR",
    "INVERTER_RATE_PER_HOUR",
    "ReliabilityScenario",
    "SCENARIOS",
    "SOURCE_PDF",
    "component_curves",
    "cumulative_failure",
    "failure_density",
    "hazard_rate",
    "methodology",
    "reliability",
    "scenario_table",
    "distribution_model_contracts",
]