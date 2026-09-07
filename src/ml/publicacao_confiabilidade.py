"""Publicação rastreável dos cenários canônicos de confiabilidade física."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config import RAIZ_PROJETO
from src.ml.confiabilidade_componentes import (
    HOURS_PER_YEAR,
    SCENARIOS,
    SOURCE_PDF,
    component_curves,
    methodology,
    scenario_table,
)
from src.ml.graficos_confiabilidade import generate_all
from src.ml.proveniencia import gerar_manifesto, salvar_manifesto, sha256_arquivo

ROOT = Path(RAIZ_PROJETO)
OUTPUT_DIR = ROOT / "resultados" / "confiabilidade"
HORIZON_YEARS = 20.0
N_POINTS = 401


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _report() -> str:
    return """# Confiabilidade física por componente

## Escopo

As curvas são cenários bibliográficos históricos de sensibilidade do TCC,
independentes da base experimental usada na comparação dos detectores e do
escopo atual da FMECA. As fontes disponíveis não fornecem uma amostra de tempos
individuais de falha, exposição de frota e censura por ativo.

## Taxas

Cenários históricos do TCC (alocação proporcional por participação de chamados):

- Contator AC: 2,10e-5 falha/h, derivada de 1,75e-4 x 12%.
- IGBT: 1,05e-5 falha/h, derivada de 1,75e-4 x 6%.
- Fusível AC: 7,00e-6 falha/h, derivada de 1,75e-4 x 4%.
- Fusível: 2,17e-6 falha/h, transcrita diretamente da Tabela 3.4.

Taxas bibliográficas externas do escopo de 5 itens (datasheet/norma/campo,
faixas como par lower/upper, FIT = falha por 1e9 h):

- IGBT: 10-100 FIT (datasheet, Mitsubishi) e 685-2740 FIT (campo FIDES,
  Abunima & Teh 2020).
- Contatores CA/CC: ~100 FIT (ABB; consultado em cópia não oficial, verificar).
- Ventiladores: 1,4e-5 a 2,0e-5 falha/h (Sanyo Denki 9RA).

Nenhuma é medição de campo desta pesquisa. **PCB** e **Control Software** ficam
sem λ: nenhuma fonte decompõe a taxa por componente dentro do inversor —
registrado em `lambda_not_found`, sem zero, estimativa ou proxy.

## FMECA vigente (6 itens)

Escopo de 6 itens, todos validados por `NPR = S * O * D`. Proveniência mista:
os cinco itens de survey de campo levam os escores de Cristaldi et al. (2017),
Tabela 6; o sensor/realimentação mantém o valor definido pelo pesquisador, a ser
revisado.

- Sistema de sensor/realimentação: S=5, O=8, D=7, NPR=280 (pesquisador).
- PCB: S=7, O=4, D=6, NPR=168 (Cristaldi).
- Contatores CA/CC: S=6, O=5, D=5, NPR=150 (Cristaldi).
- Sistema/circuito de controle: S=6, O=7, D=3, NPR=126 (Cristaldi).
- IGBT: S=3, O=3, D=7, NPR=63 (Cristaldi).
- Ventiladores de refrigeração: S=4, O=3, D=4, NPR=48 (Cristaldi).

Só IGBT, sensor/realimentação e controle têm ensaio no GPVS — são os que a
injeção E2 cobre. Os demais entram como criticidade/manutenção, sem injeção.

A validação E3 mede detecção de anomalias. Ela não produz escalas ordinais da
FMECA e não recalcula NPR.

## Modelo

Adota-se o cenário exponencial de taxa constante: R(t)=exp(-lambda*t),
F(t)=1-R(t), f(t)=lambda*exp(-lambda*t) e h(t)=lambda. A conversão usa
1 ano=8.760 horas. As figuras usam escalas lineares. A busca no corpus local
encontrou 22 trechos sobre IGBT e 74 sobre Weibull; a única fonte comum discute
os assuntos separadamente e não fornece beta ou eta para IGBT. Por isso Weibull
2P, distribuição normal, Lognormal, histograma de vidas, curva de banheira e RUL
físico permanecem bloqueados, sem parâmetros fabricados.
"""


def generate() -> dict:
    """Gera tabelas, figuras, relatório e manifesto da confiabilidade física."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = scenario_table()
    curves = component_curves(HORIZON_YEARS, N_POINTS)
    methodology_payload = methodology()
    fmeca_contract = methodology_payload["fmeca"]
    outputs: list[Path] = []

    scenarios_path = OUTPUT_DIR / "cenarios.csv"
    curves_path = OUTPUT_DIR / "curvas.csv"
    scenarios.to_csv(scenarios_path, index=False, lineterminator="\n")
    curves.to_csv(curves_path, index=False, lineterminator="\n")
    outputs.extend([scenarios_path, curves_path])
    outputs.extend(generate_all(OUTPUT_DIR, curves=curves, scenarios=scenarios))

    source_path = ROOT / SOURCE_PDF
    payload = {
        **methodology_payload,
        "source": {
            "artifact": SOURCE_PDF,
            "sha256": sha256_arquivo(source_path),
            "pdf_page": 35,
            "printed_page": 34,
            "tables": ["Tabela 3.3", "Tabela 3.4"],
        },
        "data_files": {"scenarios": "cenarios.csv", "curves": "curvas.csv"},
        "figures": [
            path.name for path in outputs if path.suffix in {".png", ".pdf"}
        ],
    }
    outputs.append(_write_json(OUTPUT_DIR / "metodologia.json", payload))
    report_path = OUTPUT_DIR / "relatorio.md"
    report_path.write_text(_report(), encoding="utf-8", newline="\n")
    outputs.append(report_path)

    manifest = gerar_manifesto(
        stage="confiabilidade_componentes",
        code_path=Path(__file__),
        parameters={
            "model": "exponential_constant_hazard",
            "hours_per_year": HOURS_PER_YEAR,
            "horizon_years": HORIZON_YEARS,
            "n_points": N_POINTS,
            "scenarios": [scenario.scenario_id for scenario in SCENARIOS],
            "fmeca_status": fmeca_contract["status"],
            "fmeca_calculation_enabled": fmeca_contract["calculation_enabled"],
            "fmeca_score_origin": "mixed_cristaldi_2017_and_researcher",
            "fmeca_scale_basis": "cristaldi_2017_table_6_and_researcher_sensor",
            "fmeca_scope_item_count": fmeca_contract["scope_item_count"],
            "fmeca_traceability_status": fmeca_contract["traceability_status"],
            "fmeca_traceability_pending_scope": (
                "exact_source_location_and_scale_criteria"
            ),
            "fmeca_scores": {
                component["component_id"]: {
                    "severity": component["severity"],
                    "occurrence": component["occurrence"],
                    "detectability": component["detectability"],
                    "npr": component["npr"],
                }
                for component in fmeca_contract["components"]
            },
            "weibull_2p_status": (
                "blocked_no_traceable_igbt_parameters_in_current_corpus"
            ),
        },
        input_artifacts={"torres_tcc": source_path},
        outputs=outputs,
        code_dependencies={
            "reliability": ROOT / "src" / "ml" / "confiabilidade_componentes.py",
            "plots": ROOT / "src" / "ml" / "graficos_confiabilidade.py",
            "style": ROOT / "src" / "ml" / "estilo_graficos.py",
        },
        evidence_level="bibliographic_sensitivity",
    )
    manifest_path = salvar_manifesto(manifest)
    return {"outputs": outputs, "manifest": manifest_path, "payload": payload}


def main() -> int:
    resultado = generate()
    print(
        json.dumps(
            {
                "outputs": [str(path.relative_to(ROOT)) for path in resultado["outputs"]],
                "manifest": str(resultado["manifest"].relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


__all__ = ["HORIZON_YEARS", "N_POINTS", "OUTPUT_DIR", "generate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
