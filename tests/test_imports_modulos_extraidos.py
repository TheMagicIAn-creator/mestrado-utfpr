"""Contratos de importacao dos modulos extraidos pelos PRs #99 e #102."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


RAIZ = Path(__file__).resolve().parents[1]

MODULOS_EXTRAIDOS = (
    "src.conhecimento.agente_contexto",
    "src.conhecimento.agente_interacao",
    "src.conhecimento.agente_recuperacao",
    "src.conhecimento.consultas_obsidian",
    "src.conhecimento.ferramentas_academicas",
    "src.conhecimento.roteamento_ferramentas",
    "src.ml.graficos_experimentos",
    "src.ml.graficos_rul",
)


@pytest.mark.parametrize("modulo", MODULOS_EXTRAIDOS)
def test_modulo_extraido_importa_em_processo_limpo(modulo):
    codigo = f"""
import builtins

_importar = builtins.__import__

def _sem_streamlit(nome, globals=None, locals=None, fromlist=(), level=0):
    if nome == "streamlit" or nome.startswith("streamlit."):
        raise ModuleNotFoundError("streamlit bloqueado pelo contrato de importacao leve")
    return _importar(nome, globals, locals, fromlist, level)

builtins.__import__ = _sem_streamlit
import {modulo}
"""
    processo = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert processo.returncode == 0, processo.stdout + processo.stderr


@pytest.mark.parametrize(
    ("fachada", "simbolo", "origem"),
    (
        ("src.conhecimento.agente", "preparar_prompt", "src.conhecimento.agente_contexto"),
        ("src.conhecimento.obsidian", "buscar_notas_obsidian", "src.conhecimento.consultas_obsidian"),
        ("src.ml.experimentos_artigos", "_grafico_comparacao", "src.ml.graficos_experimentos"),
        ("src.ml.rul_weibull", "plotar_rul", "src.ml.graficos_rul"),
    ),
)
def test_fachada_preserva_reexportacao(fachada, simbolo, origem):
    valor = getattr(importlib.import_module(fachada), simbolo)

    assert valor.__module__ == origem
