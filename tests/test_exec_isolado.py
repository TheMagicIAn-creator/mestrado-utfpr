"""
Sprint 5 — 10.4: isolamento de experimentos pesados em subprocesso.

Testes COMPORTAMENTAIS e torch-free: usam um ``Popen`` fake (sem spawnar
processo real, sem importar Orange/RL/torch), cobrindo os caminhos:
  - opt-out (AL_IADO_SEM_ISOLAMENTO) → in-process;
  - sucesso isolado (lê o JSON do filho + encaminha progresso);
  - returncode != 0 / arquivo ausente → dict ok=False (graceful);
  - timeout → dict ok=False (graceful), processo morto;
  - falha ao lançar o subprocesso → fallback in-process.
"""

from __future__ import annotations

import json
import subprocess as real_sub
from pathlib import Path

import src.ml.exec_experimento_isolado as ei


class _FakeProc:
    def __init__(self, linhas, returncode=0, raise_on_wait=None):
        self.stdout = iter(linhas)
        self.returncode = returncode
        self._raise = raise_on_wait
        self.killed = False

    def wait(self, timeout=None):
        if self._raise:
            raise self._raise
        return self.returncode

    def kill(self):
        self.killed = True


def _limpa_envs(monkeypatch):
    monkeypatch.delenv("AL_IADO_SEM_ISOLAMENTO", raising=False)
    monkeypatch.delenv("AL_IADO_EXP_CHILD", raising=False)


def test_slug():
    assert ei._slug("Ghoneim 2021!") == "ghoneim_2021"
    assert ei._slug("") == "exp"


def test_optout_roda_inproc(monkeypatch):
    monkeypatch.setenv("AL_IADO_SEM_ISOLAMENTO", "1")
    monkeypatch.setattr(
        ei, "_rodar_inproc",
        lambda key, progresso=None: {"ok": True, "via": "inproc", "experimento": key},
    )
    out = ei.executar_experimento_isolado("ghoneim")
    assert out == {"ok": True, "via": "inproc", "experimento": "ghoneim"}


def test_child_marker_roda_inproc(monkeypatch):
    _limpa_envs(monkeypatch)
    monkeypatch.setenv("AL_IADO_EXP_CHILD", "1")
    monkeypatch.setattr(
        ei, "_rodar_inproc",
        lambda key, progresso=None: {"ok": True, "via": "inproc"},
    )
    assert ei.executar_experimento_isolado("x")["via"] == "inproc"


def test_sucesso_isolado_le_json_e_encaminha_progresso(monkeypatch):
    _limpa_envs(monkeypatch)

    def fake_popen(cmd, **kw):
        # o pai passa o caminho de saída como último argumento
        Path(cmd[-1]).write_text(
            json.dumps({"experimento": "x", "ok": True, "mensagem": "feito"}),
            encoding="utf-8",
        )
        return _FakeProc(["progresso 1", "progresso 2"], returncode=0)

    monkeypatch.setattr(ei.subprocess, "Popen", fake_popen)
    recebidas = []
    out = ei.executar_experimento_isolado("x", progresso=recebidas.append)
    assert out["ok"] is True and out["experimento"] == "x"
    assert "progresso 1" in recebidas and "progresso 2" in recebidas


def test_ok_false_legitimo_passa_mesmo_com_returncode_nao_zero(monkeypatch):
    # "cartão de dataset"/"lib faltando": resultado VÁLIDO ok=False; mesmo que o
    # processo encerre != 0, o JSON gravado deve prevalecer (não vira "crash").
    _limpa_envs(monkeypatch)

    def fake_popen(cmd, **kw):
        Path(cmd[-1]).write_text(
            json.dumps({"experimento": "stender", "ok": False,
                        "mensagem": "cartão de dataset — sem modelo"}),
            encoding="utf-8",
        )
        return _FakeProc(["aviso"], returncode=1)

    monkeypatch.setattr(ei.subprocess, "Popen", fake_popen)
    out = ei.executar_experimento_isolado("stender")
    assert out["ok"] is False
    assert out["mensagem"] == "cartão de dataset — sem modelo"


def test_returncode_nao_zero_graceful(monkeypatch):
    _limpa_envs(monkeypatch)

    def fake_popen(cmd, **kw):
        # NÃO escreve o JSON → simula crash do filho
        return _FakeProc(["traceback boom"], returncode=1)

    monkeypatch.setattr(ei.subprocess, "Popen", fake_popen)
    out = ei.executar_experimento_isolado("x")
    assert out["ok"] is False
    assert "subprocesso" in out["mensagem"].lower()


def test_timeout_graceful_mata_processo(monkeypatch):
    _limpa_envs(monkeypatch)
    proc_box = {}

    def fake_popen(cmd, **kw):
        p = _FakeProc(["linha"], raise_on_wait=real_sub.TimeoutExpired(cmd, 1))
        proc_box["p"] = p
        return p

    monkeypatch.setattr(ei.subprocess, "Popen", fake_popen)
    out = ei.executar_experimento_isolado("x", timeout_s=1)
    assert out["ok"] is False
    assert "excedeu" in out["mensagem"].lower()
    assert proc_box["p"].killed is True


def test_falha_ao_lancar_cai_inproc(monkeypatch):
    _limpa_envs(monkeypatch)

    def boom(*a, **k):
        raise OSError("sem subprocess neste ambiente")

    monkeypatch.setattr(ei.subprocess, "Popen", boom)
    monkeypatch.setattr(
        ei, "_rodar_inproc",
        lambda key, progresso=None: {"ok": True, "via": "fallback", "experimento": key},
    )
    out = ei.executar_experimento_isolado("x")
    assert out == {"ok": True, "via": "fallback", "experimento": "x"}
