"""Lock de indexação entre threads e processos."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src.conhecimento.index_lock import indexacao_ocupada, lock_indexacao


def test_lock_serializa_threads(tmp_path):
    eventos = []
    caminho = tmp_path / "index.lock"

    def trabalho(nome):
        with lock_indexacao(timeout=5, caminho_lock=caminho):
            eventos.append(f"{nome}-inicio")
            time.sleep(0.05)
            eventos.append(f"{nome}-fim")

    t1 = threading.Thread(target=trabalho, args=("A",))
    t2 = threading.Thread(target=trabalho, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert eventos[1] == eventos[0].replace("-inicio", "-fim")
    assert eventos[3] == eventos[2].replace("-inicio", "-fim")


def test_indexacao_ocupada_e_lock_reentrante(tmp_path):
    caminho = tmp_path / "index.lock"
    assert not indexacao_ocupada(caminho)
    with lock_indexacao(caminho_lock=caminho):
        assert indexacao_ocupada(caminho)
        with lock_indexacao(caminho_lock=caminho):
            assert indexacao_ocupada(caminho)
    assert not indexacao_ocupada(caminho)


def test_lock_bloqueia_outro_processo(tmp_path):
    caminho = tmp_path / "index.lock"
    codigo = """
from pathlib import Path
from src.conhecimento.index_lock import lock_indexacao
try:
    with lock_indexacao(timeout=0.2, caminho_lock=Path(r'%s')):
        raise SystemExit(7)
except TimeoutError:
    raise SystemExit(0)
""" % caminho

    env = os.environ.copy()
    raiz = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        parte for parte in (raiz, env.get("PYTHONPATH", "")) if parte
    )

    with lock_indexacao(timeout=2, caminho_lock=caminho):
        filho = subprocess.run(
            [sys.executable, "-c", codigo],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )

    assert filho.returncode == 0, filho.stderr


def test_lock_e_liberado_apos_excecao(tmp_path):
    caminho = tmp_path / "index.lock"
    with pytest.raises(RuntimeError):
        with lock_indexacao(caminho_lock=caminho):
            raise RuntimeError("falha simulada")
    with lock_indexacao(timeout=0.2, caminho_lock=caminho):
        pass
