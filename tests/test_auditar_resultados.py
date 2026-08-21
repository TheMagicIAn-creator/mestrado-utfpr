"""Execução independente do auditor da publicação científica."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
    assert "40 artefatos" in processo.stdout
