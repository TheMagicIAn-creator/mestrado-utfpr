"""Avaliação offline das rotas e salvaguardas científicas do ALIAdo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.conhecimento.roteamento_ferramentas import decidir_acao
from src.ml.resultados import resumir_resultados


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    expected: str | None
    observed: str | None


ROUTE_CASES = (
    ("greeting", "oi", None),
    ("concept", "explique FMECA", None),
    ("comparison_query", "compare o Denso e o AE-LSTM", "consultar_comparacao_autoencoders"),
    ("result_query", "mostre os gráficos E3", "consultar_resultados"),
    ("dataset", "qual dataset está ativo?", "consultar_datasets"),
    ("comparison_run", "recalcule a comparação dos autoencoders", "executar_comparacao_autoencoders"),
    ("reliability_run", "regenere a confiabilidade física", "gerar_confiabilidade"),
    ("full_run", "rode o pipeline completo", "executar_pipeline_cientifico"),
    ("catalog", "liste toda a base bibliográfica", "listar_base_bibliografica"),
    ("memory", "registre esse resultado no cérebro", "registrar_no_cerebro"),
)


def evaluate_routes() -> list[Check]:
    checks = []
    for name, question, expected in ROUTE_CASES:
        decision = decidir_acao(question)
        observed = decision["ferramenta"] if decision["usar_ferramenta"] else None
        checks.append(Check(name, observed == expected, expected, observed))
    return checks


def evaluate_scientific_contract() -> list[Check]:
    text = resumir_resultados("resumo completo", incluir_imagens=False)["mensagem"]
    requirements = {
        "models": ("Autoencoder Denso" in text and "AE-LSTM" in text),
        "e3_unit": "14 ensaios reais" in text,
        "e2_axis": "não tempo" in text,
        "smd95": "SMD95" in text,
        "physical_separation": "não são medições" in text,
    }
    return [Check(name, passed, "presente", "presente" if passed else "ausente") for name, passed in requirements.items()]


def run() -> dict:
    checks = [*evaluate_routes(), *evaluate_scientific_contract()]
    return {
        "ok": all(item.passed for item in checks),
        "passed": sum(item.passed for item in checks),
        "total": len(checks),
        "checks": [asdict(item) for item in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Arquivo opcional para o relatório JSON")
    args = parser.parse_args()
    report = run()
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
