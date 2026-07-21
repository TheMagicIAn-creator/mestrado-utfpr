"""Trava estrutural contra citação sem lastro (lógica pura)."""

from __future__ import annotations

from src.core.citacao_guarda import (
    alerta_citacao_infundada,
    montar_restricao_fontes,
)


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


def test_restricao_lista_apenas_fontes_do_rodape():
    citacoes = {
        "a": "Sakurada (1998) — As Tecnicas... — p. 10 — trecho: 'abc'",
        "b": "Torres (2024) — Aplicacao RCM... — p. 59 — trecho: 'def'",
    }
    bloco = montar_restricao_fontes(citacoes)
    assert "Sakurada (1998)" in bloco and "p. 10" in bloco
    assert "Torres (2024)" in bloco and "p. 59" in bloco
    assert "trecho" not in bloco  # o trecho e removido do rotulo
    assert "EXATAMENTE estas" in bloco


def test_restricao_sem_fontes_proibe_citacao():
    bloco = montar_restricao_fontes({})
    assert "NENHUMA" in bloco
    assert "NAO cite" in bloco
    assert "invencao" in bloco.lower()


def test_restricao_fonte_nao_listada_fica_de_fora():
    # NASA nao foi aprovada pelo auditor -> nao entra na lista -> nao pode citar.
    citacoes = {"a": "Sakurada (1998) — ... — p. 10 — trecho: 'x'"}
    bloco = montar_restricao_fontes(citacoes)
    assert "Sakurada" in bloco
    assert "NASA" not in bloco and "Administration" not in bloco


def test_iec_inventada_sem_retrieval_e_sinalizada():
    """Caso real: literatura NAO consultada (citacoes vazio) e o LLM inventa
    'IEC 60812:2018 Clause 8.3.4 p.40'. O guard, rodando sempre, deve avisar."""
    resposta = (
        "Na IEC 60812:2018 (Clause 8.3.4, p. 40-41) o NPR = S×O×D; "
        "e na IEC 60812:2006 (p. 27) consta a definicao."
    )
    aviso = alerta_citacao_infundada(resposta, {})  # {} = literatura nao consultada
    assert aviso
    assert "IEC 60812" in aviso
