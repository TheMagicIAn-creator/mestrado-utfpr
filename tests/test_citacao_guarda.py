"""Trava estrutural contra citação sem lastro (lógica pura)."""

from __future__ import annotations

from src.core.citacao_guarda import alerta_citacao_infundada


def test_norma_iec_nao_recuperada_e_sinalizada():
    resposta = "Na IEC 60812:2018, Cláusula 7.3.3, p. 27, o NPR = S×O×D."
    citacoes = {"a": "Sakurada (1998) — FMECA — p. 28 — trecho: ..."}
    aviso = alerta_citacao_infundada(resposta, citacoes)
    assert aviso
    assert "IEC 60812" in aviso
    assert "NÃO verificad" in aviso or "não verificad" in aviso.lower()


def test_norma_presente_nas_fontes_nao_alarma():
    resposta = "A IEC 60812 define o NPR."
    citacoes = {"a": "IEC 60812 (2018) — Failure modes — p. 27 — trecho: RPN..."}
    assert alerta_citacao_infundada(resposta, citacoes) == ""


def test_sem_fontes_mas_cita_pagina_e_sinalizado():
    resposta = "Conforme Sakurada (1998, p. 7), FMECA = FMEA + Criticidade."
    assert alerta_citacao_infundada(resposta, {}) != ""


def test_resposta_ancorada_sem_norma_nao_alarma():
    resposta = "Segundo Sakurada (1998, p. 28), FMECA estende a FMEA com criticidade."
    citacoes = {"a": "Sakurada (1998) — FMECA — p. 28 — trecho: extensão..."}
    assert alerta_citacao_infundada(resposta, citacoes) == ""


def test_resposta_vazia_nao_alarma():
    assert alerta_citacao_infundada("", {}) == ""
    assert alerta_citacao_infundada("   ", {"a": "x"}) == ""


def test_varias_normas_diferentes():
    resposta = "Ver IEC 60812 e ISO 14224 e ABNT NBR 5462."
    aviso = alerta_citacao_infundada(resposta, {"a": "Torres (2024) — p. 22"})
    assert "IEC 60812" in aviso
    # limita a 3 exemplos no texto, mas detecta as normas
    assert aviso.count(",") >= 1
