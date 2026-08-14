"""Gera o pacote acadêmico de confiabilidade física bibliográfica V2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.core.config import RAIZ_PROJETO
from src.ml.confiabilidade_fisica_v2 import (
    CENARIOS,
    HOURS_PER_YEAR,
    MODEL_NAME,
    auditoria_dimensional,
    curvas_cenarios,
    marcos_cenarios,
    tabela_cenarios,
)
from src.ml.graficos_confiabilidade_fisica_v2 import (
    plotar_confiabilidade,
    plotar_densidade_e_taxa,
    plotar_marcos,
    plotar_probabilidade_falha,
)
from src.ml.proveniencia import gerar_manifesto, sha256_arquivo

RAIZ = Path(RAIZ_PROJETO)
PASTA_SAIDA = RAIZ / "resultados" / "v2" / "confiabilidade"
ANALYSIS_DATE = "2026-08-13"
HORIZON_YEARS = 20.0
N_CURVE_POINTS = 401

SOURCE_FILES = {
    "torres_colli_rate": RAIZ
    / "literatura"
    / "inversores-pv"
    / "torres_aplicacao-da-metodologia-reliability-centred-maintenance-a-s_2024.pdf",
    "cristaldi_inverter_rate": RAIZ
    / "literatura"
    / "inversores-pv"
    / "cristaldi_a-root-cause-analysis-and-a-risk-evaluation-of-pv-balance-of_2017.pdf",
    "obeidat_high_quality": RAIZ
    / "literatura"
    / "inversores-pv"
    / "shuttleworth_reliability-prediction-of-pv-inverters-based-on-mil-hdbk-217_2015.pdf",
    "obeidat_low_quality": RAIZ
    / "literatura"
    / "inversores-pv"
    / "shuttleworth_reliability-prediction-of-pv-inverters-based-on-mil-hdbk-217_2015.pdf",
    "dhople_markov_example": RAIZ
    / "literatura"
    / "inversores-pv"
    / "dhople_estimation-of-photovoltaic-system-reliability-and-performanc_2012.pdf",
}


def _relativo(path: Path) -> str:
    return path.relative_to(RAIZ).as_posix()


def _cenario_json(cenario) -> dict:
    fonte = SOURCE_FILES[cenario.scenario_id]
    return {
        **cenario.as_record(),
        "source_artifact": _relativo(fonte),
        "source_sha256": sha256_arquivo(fonte),
    }


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{float(value):.{decimals}f}".replace(".", ",")


def _fmt_scientific(value: float) -> str:
    mantissa, exponent = f"{float(value):.3e}".split("e")
    return f"{mantissa.replace('.', ',')} × 10^{int(exponent)}"


def _gerar_relatorio(cenarios: pd.DataFrame, marcos: pd.DataFrame) -> str:
    combinado = cenarios.merge(
        marcos[["scenario_id", "b10_years"]],
        on="scenario_id",
        validate="one_to_one",
    )
    linhas = [
        "# Confiabilidade física V2 — cenários bibliográficos",
        "",
        "## Veredito",
        "",
        (
            "O GPVS-Faults sustenta a avaliação experimental do detector, mas não "
            "contém tempos de vida por ativo, censura, exposição de frota ou histórico "
            "de reparos. Portanto, ele **não estima confiabilidade física, taxa de falha, "
            "Weibull temporal, MTTF, MTBF ou RUL**."
        ),
        "",
        (
            "As curvas deste pacote são análises de sensibilidade: quantidades de "
            "fontes identificadas foram normalizadas dimensionalmente e avaliadas sob "
            "o mesmo modelo exponencial de taxa constante. Os cenários diferem em "
            "escopo e natureza da evidência e não devem ser tratados como réplicas."
        ),
        "",
        "## Cenários normalizados",
        "",
        "| Cenário | Quantidade original | λ (ano⁻¹) | 1/λ (anos) | B10 (anos) | Natureza |",
        "|---|---:|---:|---:|---:|---|",
    ]
    unidades = {
        "failures_per_hour": "falha/h",
        "failures_per_year": "falha/ano",
        "failures_per_million_hours": "falhas/10⁶ h",
        "mean_time_to_failure_years": "anos (MTTF de entrada)",
    }
    for row in combinado.itertuples(index=False):
        linhas.append(
            "| "
            + " | ".join(
                (
                    row.plot_label,
                    f"{_fmt(row.original_value, 6)} {unidades[row.original_unit]}",
                    _fmt(row.lambda_per_year, 6),
                    _fmt(row.reciprocal_time_years, 3),
                    _fmt(row.b10_years, 3),
                    row.source_type.replace("_", " "),
                )
            )
            + " |"
        )

    linhas.extend(
        [
            "",
            "A conversão adota `1 ano = 8.760 h`. O valor `1/λ` é MTTF somente "
            "sob o modelo não reparável exponencial; quando a fonte usa MTBF ou um "
            "modelo reparável, a semântica original permanece registrada no JSON.",
            "",
            "## Funções publicadas",
            "",
            "Para `t` em anos e `λ` em falhas por ano:",
            "",
            "- `R(t) = exp(-λt)`: confiabilidade ou sobrevivência;",
            "- `F(t) = 1 - exp(-λt)`: probabilidade acumulada de falha;",
            "- `f(t) = λ exp(-λt)`: densidade de probabilidade, em ano⁻¹;",
            "- `h(t) = λ`: taxa instantânea de falha constante, em ano⁻¹;",
            "- `B_p = -ln(1-p)/λ`: tempo em que a fração acumulada `p` falhou.",
            "",
            "A densidade `f(t)` é uma **curva analítica suave**. Pontos dispersos "
            "pertencem a um gráfico de probabilidade construído com tempos de falha "
            "observados. Como essa amostra de vida não existe no GPVS, nenhum papel "
            "de Weibull físico é produzido nesta etapa.",
            "",
            "## Auditoria dimensional",
            "",
            (
                "A Tabela 3.4 de Torres (2024) transcreve `λ = "
                f"{_fmt_scientific(1.75e-4)} falha/h` para o inversor. Na seção "
                "posterior de disponibilidade, `1/(1,8 × 10^-4)` é apresentado como "
                "`5.555,55 anos`; dimensionalmente, o resultado é `5.555,55 horas`, "
                "aproximadamente `0,634 ano`. A V2 não altera a fonte: usa a taxa "
                "exata da tabela e registra a inconsistência."
            ),
            "",
            (
                "Cristaldi et al. (2017) informam `0,125 falha/ano` para o inversor, "
                "mas o MTTF próximo de seis anos citado no artigo é do sistema "
                "string-BoS completo. O recíproco da taxa isolada do inversor é oito "
                "anos; são escopos diferentes."
            ),
            "",
            (
                "Obeidat e Shuttleworth (2015) publicam predições MIL-HDBK-217F N2, "
                "não observações de frota. Dhople e Dominguez-Garcia (2012) usam dez "
                "anos como parâmetro ilustrativo em um caso Markov reparável."
            ),
            "",
            "## Limites de inferência",
            "",
            "- Nenhum cenário foi ajustado aos ensaios GPVS-Faults.",
            "- Não há intervalos de confiança porque as fontes não fornecem a amostra "
            "primária necessária para reamostragem.",
            "- As curvas não estimam taxas específicas de contator AC, IGBT ou fusível AC.",
            "- A priorização FMECA permanece julgamento de risco separado do detector e "
            "dos cenários de confiabilidade.",
            "- Um Weibull físico exigirá tempos de vida/exposição, modo de falha, censura "
            "e unidade observacional definidos antes do ajuste.",
            "",
            "## Artefatos",
            "",
            "- `cenarios.csv`: valores originais, conversões e ressalvas;",
            "- `curvas.csv`: funções amostradas em uma grade temporal comum;",
            "- `marcos.csv`: B1, B10, mediana, 1/λ e probabilidades em 1, 5 e 10 anos;",
            "- `resultado.json`: contrato consolidado para a aplicação web;",
            "- figuras em PNG de 300 dpi e PDF vetorial;",
            "- `manifesto_v2.json`: hashes das fontes, código e saídas.",
            "",
        ]
    )
    return "\n".join(linhas)


def gerar() -> list[Path]:
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    cenarios = tabela_cenarios()
    curvas = curvas_cenarios(HORIZON_YEARS, N_CURVE_POINTS)
    marcos = marcos_cenarios()

    outputs: list[Path] = []
    for nome, frame in (
        ("cenarios.csv", cenarios),
        ("curvas.csv", curvas),
        ("marcos.csv", marcos),
    ):
        path = PASTA_SAIDA / nome
        frame.to_csv(path, index=False, lineterminator="\n")
        outputs.append(path)

    figure_paths = []
    figure_paths.extend(
        plotar_confiabilidade(curvas, PASTA_SAIDA / "confiabilidade_cenarios")
    )
    figure_paths.extend(
        plotar_probabilidade_falha(curvas, PASTA_SAIDA / "probabilidade_falha_cenarios")
    )
    figure_paths.extend(
        plotar_densidade_e_taxa(curvas, PASTA_SAIDA / "densidade_taxa_falha")
    )
    figure_paths.extend(plotar_marcos(marcos, PASTA_SAIDA / "marcos_confiabilidade"))
    outputs.extend(figure_paths)

    resultado = {
        "schema_version": 2,
        "analysis_date": ANALYSIS_DATE,
        "status": "bibliographic_sensitivity_not_dataset_estimate",
        "experimental_dataset": "GPVS-Faults",
        "dataset_role": "detector_evaluation_only_not_physical_reliability",
        "model": {
            "name": MODEL_NAME,
            "assumption": "constant_hazard_within_each_scenario",
            "time_unit": "year",
            "rate_unit": "failures_per_year",
            "hours_per_year": HOURS_PER_YEAR,
            "formulas": {
                "reliability": "R(t) = exp(-lambda*t)",
                "cumulative_failure_probability": "F(t) = 1 - exp(-lambda*t)",
                "failure_density": "f(t) = lambda*exp(-lambda*t)",
                "hazard": "h(t) = lambda",
                "failure_quantile": "B_p = -ln(1-p)/lambda",
            },
        },
        "scenarios": [_cenario_json(cenario) for cenario in CENARIOS],
        "milestones": marcos.to_dict(orient="records"),
        "dimensional_audit": auditoria_dimensional(),
        "physical_weibull": {
            "status": "not_estimable_from_current_dataset",
            "beta": None,
            "eta": None,
            "reason": (
                "GPVS-Faults has no asset-level lifetime, exposure and censoring contract"
            ),
        },
        "data_files": {
            "scenarios": "cenarios.csv",
            "curves": "curvas.csv",
            "milestones": "marcos.csv",
        },
        "figures": [path.name for path in figure_paths],
    }
    resultado_path = PASTA_SAIDA / "resultado.json"
    resultado_path.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs.append(resultado_path)

    relatorio_path = PASTA_SAIDA / "relatorio.md"
    relatorio_path.write_text(
        _gerar_relatorio(cenarios, marcos), encoding="utf-8", newline="\n"
    )
    outputs.append(relatorio_path)

    source_inputs = {
        _relativo(path): path for path in sorted(set(SOURCE_FILES.values()))
    }
    manifesto = gerar_manifesto(
        stage="confiabilidade_fisica_v2",
        code_path=Path(__file__),
        parameters={
            "model": MODEL_NAME,
            "hours_per_year": HOURS_PER_YEAR,
            "horizon_years": HORIZON_YEARS,
            "n_curve_points": N_CURVE_POINTS,
            "scenario_ids": [cenario.scenario_id for cenario in CENARIOS],
        },
        input_artifacts=source_inputs,
        outputs=outputs,
        code_dependencies={
            "src/ml/confiabilidade_fisica_v2.py": (
                RAIZ / "src" / "ml" / "confiabilidade_fisica_v2.py"
            ),
            "src/ml/graficos_confiabilidade_fisica_v2.py": (
                RAIZ / "src" / "ml" / "graficos_confiabilidade_fisica_v2.py"
            ),
            "src/ml/estilo_graficos.py": RAIZ / "src" / "ml" / "estilo_graficos.py",
        },
        evidence_level="bibliographic_sensitivity",
    )
    manifesto_path = PASTA_SAIDA / "manifesto_v2.json"
    manifesto_path.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs.append(manifesto_path)
    return outputs


def main() -> int:
    outputs = gerar()
    print(f"Confiabilidade física V2: {len(outputs)} artefatos em {PASTA_SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
