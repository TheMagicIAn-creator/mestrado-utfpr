"""Execução independente do auditor da publicação científica."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.auditar_resultados import _audit_retrieval_baseline


def test_auditor_executa_fora_da_raiz(tmp_path):
    raiz = Path(__file__).resolve().parents[1]
    processo = subprocess.run(
        [sys.executable, str(raiz / "scripts" / "auditar_resultados.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert processo.returncode == 0, processo.stdout + processo.stderr
    assert "APROVADO" in processo.stdout
    assert "30 artefatos" in processo.stdout


def test_auditor_rejeita_contrato_retrieval_divergente(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stage": "R2",
                "variant": "experimental",
                "gold_set": {
                    "status": "final",
                    "researcher_review_required_at": "R7",
                },
                "corpus": {
                    "n_documents": 1,
                    "n_chunks": 2,
                    "collection_count": 3,
                },
                "metric_definition": {"primary_level": "document"},
                "summary": {
                    "n_queries": 1,
                    "n_retrieval_queries": 1,
                    "n_future_abstention_queries": 0,
                },
                "queries": [{"query_id": "duplicada"}] * 40,
            }
        ),
        encoding="utf-8",
    )
    errors = []

    _audit_retrieval_baseline(path, errors)

    assert len(errors) == 9
    assert any("schema_version" in error for error in errors)
    assert any("query_id duplicado" in error for error in errors)


def test_auditor_rejeita_resultados_incompletos_e_json_invalido(tmp_path):
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"queries": []}), encoding="utf-8")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    errors = []

    _audit_retrieval_baseline(incomplete, errors)
    _audit_retrieval_baseline(invalid, errors)

    assert any("resultados por pergunta incompletos" in error for error in errors)
    assert any("manifesto inválido" in error for error in errors)
