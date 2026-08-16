"""Publica os cenários canônicos de confiabilidade física por componente."""

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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _report() -> str:
    return """# Confiabilidade física por componente

## Escopo

As curvas são cenários bibliográficos de sensibilidade. O GPVS-Faults avalia os
detectores Denso e AE-LSTM, mas não contém tempos de vida, exposição de frota ou
censura por ativo e, portanto, não estima confiabilidade física.

## Taxas

- Contator AC: 2,10e-5 falha/h, derivada de 1,75e-4 x 12%.
- IGBT: 1,05e-5 falha/h, derivada de 1,75e-4 x 6%.
- Fusível AC: 7,00e-6 falha/h, derivada de 1,75e-4 x 4%.
- Fusível: 2,17e-6 falha/h, transcrita diretamente da Tabela 3.4.

Os percentuais são participações de chamados, não frações demonstradas da taxa
de falha do inversor. Por isso, as três alocações são rotuladas como derivadas.
A ausência de taxas diretas equivalentes para Contator AC e IGBT é preservada.

## Modelo

Adota-se o cenário exponencial de taxa constante: R(t)=exp(-lambda*t),
F(t)=1-R(t), f(t)=lambda*exp(-lambda*t) e h(t)=lambda. A conversão usa
1 ano=8.760 horas. Não são estimados beta, eta, curva de banheira ou RUL físico.
"""


def generate() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = scenario_table()
    curves = component_curves(HORIZON_YEARS, N_POINTS)
    outputs: list[Path] = []

    scenarios_path = OUTPUT_DIR / "cenarios.csv"
    curves_path = OUTPUT_DIR / "curvas.csv"
    scenarios.to_csv(scenarios_path, index=False, lineterminator="\n")
    curves.to_csv(curves_path, index=False, lineterminator="\n")
    outputs.extend([scenarios_path, curves_path])

    outputs.extend(generate_all(OUTPUT_DIR, curves=curves, scenarios=scenarios))
    source_path = ROOT / SOURCE_PDF
    payload = {
        **methodology(),
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
    result = generate()
    print(f"Confiabilidade: {len(result['outputs'])} artefatos em {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
