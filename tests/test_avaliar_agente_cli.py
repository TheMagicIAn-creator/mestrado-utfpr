"""Contratos do CLI de avaliação offline do ALIAdo."""

from __future__ import annotations

import json
import sys

from scripts import avaliar_agente
from src.conhecimento import benchmark_retrieval


def test_cli_publica_benchmark_retrieval(monkeypatch, tmp_path, capsys):
    gold_set = tmp_path / "gold.json"
    snapshot = tmp_path / "snapshot.jsonl.gz"
    output_json = tmp_path / "reports" / "baseline.json"
    output_markdown = tmp_path / "reports" / "baseline.md"
    report = {
        "benchmark_id": "evidence-rag-baseline-v1",
        "gold_set": {"n_queries": 40},
        "summary": {"page": {"recall_at_5": 0.25}},
    }
    observed = {}

    def fake_execute(gold_path, snapshot_path, *, git_revision):
        observed.update(
            gold_path=gold_path,
            snapshot_path=snapshot_path,
            git_revision=git_revision,
        )
        return report

    monkeypatch.setattr(benchmark_retrieval, "executar_baseline_local", fake_execute)
    monkeypatch.setattr(benchmark_retrieval, "relatorio_markdown", lambda _: "# Baseline\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "avaliar_agente.py",
            "--benchmark-retrieval",
            "--gold-set",
            str(gold_set),
            "--snapshot",
            str(snapshot),
            "--json",
            str(output_json),
            "--markdown",
            str(output_markdown),
            "--git-revision",
            "abc123",
        ],
    )

    assert avaliar_agente.main() == 0
    assert observed == {
        "gold_path": gold_set,
        "snapshot_path": snapshot,
        "git_revision": "abc123",
    }
    assert json.loads(output_json.read_text(encoding="utf-8")) == report
    assert output_markdown.read_text(encoding="utf-8") == "# Baseline\n"
    assert json.loads(capsys.readouterr().out)["benchmark_id"] == report["benchmark_id"]
