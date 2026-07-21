"""Ligações reais entre notas do vault (lógica pura, sem I/O)."""

from __future__ import annotations

from src.conhecimento.vault_links import (
    bloco_notas_relacionadas,
    notas_relacionadas,
)


def _item(id_, conteudo, status="ativo"):
    return {"id": id_, "conteudo": conteudo, "status": status}


def test_encontra_item_com_sobreposicao_lexical():
    itens = [
        _item("a1", "Prefere respostas objetivas em portugues sobre o pipeline"),
        _item("a2", "Decidiu injetar primeiro o Contator AC por causa do NPR"),
    ]
    relacionados = notas_relacionadas(
        "Qual falha o Rodolfo decidiu injetar primeiro no pipeline?", itens
    )
    ids = [i["id"] for i in relacionados]
    assert "a2" in ids


def test_ignora_itens_superados():
    itens = [_item("a1", "Contator AC e a primeira falha a injetar", status="superado")]
    assert notas_relacionadas("primeira falha contator ac", itens) == []


def test_exclui_o_proprio_item():
    itens = [
        _item("a1", "Contator AC e a primeira falha prioritaria a injetar"),
        _item("a2", "Contator AC tem o maior NPR entre os componentes CA"),
    ]
    relacionados = notas_relacionadas(
        "Contator AC e a primeira falha prioritaria a injetar",
        itens,
        excluir_id="a1",
    )
    ids = [i["id"] for i in relacionados]
    assert "a1" not in ids


def test_exige_overlap_minimo_sem_falso_positivo():
    itens = [_item("a1", "Prefere respostas curtas em ingles sobre estatistica")]
    assert notas_relacionadas("gere um grafico da distribuicao de erro", itens) == []


def test_respeita_max_links():
    itens = [_item(f"a{i}", "pipeline autoencoder weibull falha contator") for i in range(10)]
    relacionados = notas_relacionadas(
        "pipeline autoencoder weibull falha contator", itens, max_links=2
    )
    assert len(relacionados) == 2


def test_bloco_vazio_sem_itens():
    assert bloco_notas_relacionadas([]) == ""


def test_bloco_formata_wikilinks():
    itens = [_item("abc123", "Decisao sobre a ordem de injecao das falhas FMECA")]
    bloco = bloco_notas_relacionadas(itens)
    assert "[[Memoria validada - abc123]]" in bloco
    assert "## Notas relacionadas" in bloco
