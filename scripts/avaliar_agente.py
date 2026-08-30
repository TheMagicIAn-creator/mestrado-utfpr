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

from src.conhecimento.benchmark_retrieval import (  # noqa: E402
    resolver_caminho_no_projeto,
)
from src.conhecimento.roteamento_ferramentas import decidir_acao  # noqa: E402
from src.ml.resultados import resumir_resultados  # noqa: E402


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
        "e3_unit": "14 ensaios experimentais" in text,
        "active_scope": "detectabilidade sintética" not in text,
        "reliability_functions": all(term in text for term in ("R(t)", "F(t)", "f(t)", "h(t)")),
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
    parser.add_argument(
        "--benchmark-retrieval",
        action="store_true",
        help="Mede o Evidence RAG vigente com o gold set versionado.",
    )
    parser.add_argument(
        "--gold-set",
        type=Path,
        default=RAIZ / "literatura" / "gold_set_retrieval_v1.json",
        help="Contrato provisório de perguntas e evidências.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=RAIZ / "artefatos" / "literatura_indexada.jsonl.gz",
        help="Snapshot portátil da literatura usado na validação.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Relatório Markdown opcional do benchmark de retrieval.",
    )
    parser.add_argument(
        "--git-revision",
        help="Revisão do algoritmo baseline avaliado.",
    )
    parser.add_argument(
        "--baseline-result",
        type=Path,
        help=(
            "Resultado de referência; por padrão usa R0-R1 para R2 e R2 para R3."
        ),
    )
    parser.add_argument(
        "--candidate-runtime",
        type=Path,
        help="Diretório local isolado para os índices candidatos não promovidos.",
    )
    args = parser.parse_args()

    if args.benchmark_retrieval:
        from src.conhecimento.benchmark_retrieval import (
            comparar_benchmarks,
            executar_baseline_local,
            relatorio_markdown,
        )

        gold_set = resolver_caminho_no_projeto(args.gold_set, deve_existir=True)
        snapshot = resolver_caminho_no_projeto(args.snapshot, deve_existir=True)
        output_json = (
            resolver_caminho_no_projeto(args.json) if args.json is not None else None
        )
        output_markdown = (
            resolver_caminho_no_projeto(args.markdown)
            if args.markdown is not None
            else None
        )
        opcoes_execucao = {"git_revision": args.git_revision}
        if args.candidate_runtime is not None:
            opcoes_execucao["candidate_runtime"] = resolver_caminho_no_projeto(
                args.candidate_runtime
            )
        report = executar_baseline_local(gold_set, snapshot, **opcoes_execucao)
        comparison_ok = True
        if report.get("stage") in {"R2", "R3"}:
            baseline_default = (
                RAIZ
                / "resultados"
                / "manifestos"
                / (
                    "evidence_rag_schema_v2_r2.json"
                    if report["stage"] == "R3"
                    else "evidence_rag_baseline_v1.json"
                )
            )
            baseline_path = resolver_caminho_no_projeto(
                args.baseline_result or baseline_default,
                deve_existir=True,
            )
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            comparacao = comparar_benchmarks(baseline, report)
            report["comparison_to_baseline"] = comparacao
            campos_gate = [
                "corpus_identity_preserved",
                "ranking_contract_preserved",
            ]
            if report["stage"] == "R2":
                campos_gate.append("scientific_metrics_identical")
            comparison_ok = all(comparacao[campo] for campo in campos_gate)
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(serialized, encoding="utf-8")
        if output_markdown:
            output_markdown.parent.mkdir(parents=True, exist_ok=True)
            output_markdown.write_text(relatorio_markdown(report), encoding="utf-8")
        print(
            json.dumps(
                {
                    "benchmark_id": report["benchmark_id"],
                    "gold_set": report["gold_set"],
                    "summary": report["summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if comparison_ok else 1

    report = run()
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        output_json = resolver_caminho_no_projeto(args.json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
