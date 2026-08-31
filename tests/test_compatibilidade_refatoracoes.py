"""Regressoes das refatoracoes historicas das PRs #99 e #102."""

from src.conhecimento import agente, agente_contexto, agente_interacao, agente_recuperacao
from src.conhecimento import consultas_obsidian, ferramentas, ferramentas_academicas, obsidian
from src.conhecimento import intencoes_ferramentas


def test_fachada_agente_preserva_exportacoes_extraidas_na_pr102():
    contratos = {
        "pedido_sem_literatura": agente_interacao.pedido_sem_literatura,
        "_montar_prompt": agente_recuperacao._montar_prompt,
        "buscar_contexto": agente_contexto.buscar_contexto,
        "preparar_prompt": agente_contexto.preparar_prompt,
    }

    for nome, implementacao in contratos.items():
        assert getattr(agente, nome) is implementacao


def test_fachada_obsidian_preserva_consultas_extraidas_na_pr102():
    contratos = {
        "responder_consulta_cronologica": consultas_obsidian.responder_consulta_cronologica,
        "responder_inventario_vault": consultas_obsidian.responder_inventario_vault,
        "buscar_notas_obsidian": consultas_obsidian.buscar_notas_obsidian,
    }

    for nome, implementacao in contratos.items():
        assert getattr(obsidian, nome) is implementacao


def test_ferramentas_reexporta_contratos_extraidos_na_pr102():
    assert ferramentas.buscar_na_web is ferramentas_academicas.buscar_na_web
    assert (
        ferramentas.listar_base_bibliografica
        is ferramentas_academicas.listar_base_bibliografica
    )
    assert (
        ferramentas._quer_registrar_no_cerebro
        is intencoes_ferramentas._quer_registrar_no_cerebro
    )
