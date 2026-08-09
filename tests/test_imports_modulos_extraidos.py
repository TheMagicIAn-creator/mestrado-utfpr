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
    "src.interface.ciclo_chat",
    "src.interface.renderizacao_imagens",
    "src.interface.sidebar",
    "src.ml.graficos_experimentos",
    "src.ml.graficos_rul",
)


@pytest.mark.parametrize("modulo", MODULOS_EXTRAIDOS)
def test_modulo_extraido_importa_em_processo_limpo(modulo):
    processo = subprocess.run(
        [sys.executable, "-c", f"import {modulo}"],
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
        ("src.interface.streamlit_app", "renderizar_sidebar", "src.interface.sidebar"),
        ("src.ml.experimentos_artigos", "_grafico_comparacao", "src.ml.graficos_experimentos"),
        ("src.ml.rul_weibull", "plotar_rul", "src.ml.graficos_rul"),
    ),
)
def test_fachada_preserva_reexportacao(fachada, simbolo, origem):
    valor = getattr(importlib.import_module(fachada), simbolo)

    assert valor.__module__ == origem
