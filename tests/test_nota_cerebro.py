"""Ferramenta de REGISTRO no cérebro — o agente escrevendo no vault."""

from __future__ import annotations

import src.conhecimento.ferramentas as fr
from src.conhecimento.nota_cerebro import registrar_nota_cerebro


def test_cria_nota_com_frontmatter_e_tags_validas(tmp_path):
    r = registrar_nota_cerebro(
        titulo="Resultado da comparação", conteudo="O método proposto venceu.",
        tipo="resultado", tags=["comparacao-literatura", "igbt"],
        nivel_evidencia="E2", fonte="resultados/macro/comparacao.json",
        pasta_base=tmp_path,
    )
    assert r["ok"]
    txt = (tmp_path / "Resultados" / "Resultado da comparação.md").read_text(encoding="utf-8")
    assert txt.startswith("---")
    assert "tipo: resultado" in txt and "nivel_evidencia: E2" in txt
    assert "comparacao-literatura" in txt and "igbt" in txt
    assert "cerebro" in txt                      # tag estrutural sempre presente
    assert "[[00 - Painel do cerebro]]" in txt   # conectada ao hub
    assert "resultados/macro/comparacao.json" in txt


def test_descarta_tag_fora_da_taxonomia(tmp_path):
    r = registrar_nota_cerebro("T", "corpo", tags=["igbt", "tag-inventada"],
                               pasta_base=tmp_path)
    assert "igbt" in r["tags"]
    assert "tag-inventada" in r["descartadas"]
    assert "tag-inventada" not in r["tags"]


def test_tipo_define_a_subpasta(tmp_path):
    registrar_nota_cerebro("Conceito X", "c", tipo="conceito", pasta_base=tmp_path)
    registrar_nota_cerebro("Decisao Y", "c", tipo="decisao", pasta_base=tmp_path)
    assert (tmp_path / "Conceitos" / "Conceito X.md").exists()
    assert (tmp_path / "Decisoes" / "Decisao Y.md").exists()


def test_titulo_vira_h1_quando_o_corpo_nao_tem(tmp_path):
    registrar_nota_cerebro("Meu titulo", "só o corpo", pasta_base=tmp_path)
    txt = (tmp_path / "Meu titulo.md").read_text(encoding="utf-8")
    assert "# Meu titulo" in txt


def test_recusa_titulo_ou_conteudo_vazio(tmp_path):
    assert not registrar_nota_cerebro("", "corpo", pasta_base=tmp_path)["ok"]
    assert not registrar_nota_cerebro("t", "  ", pasta_base=tmp_path)["ok"]


def test_nome_de_arquivo_seguro(tmp_path):
    from pathlib import Path
    r = registrar_nota_cerebro('Titulo com / e : proibidos', "c", pasta_base=tmp_path)
    nome = Path(r["caminho"]).name
    assert r["ok"] and "/" not in nome and ":" not in nome


# ── roteamento: o agente escolhe a ferramenta certa ──────────────────────────

def test_roteia_pedido_de_registro():
    assert fr._quer_registrar_no_cerebro("guarde esse resultado no cérebro")
    assert fr._quer_registrar_no_cerebro("registre essa decisão no vault")
    assert fr._quer_registrar_no_cerebro("anote isso no obsidian")


def test_nao_confunde_com_memoria_nem_com_consulta():
    # memória validada (preferência) NÃO é registro de nota curada
    assert not fr._quer_registrar_no_cerebro("lembre que prefiro gráficos escuros")
    # consulta ao vault é RAG, não escrita
    assert not fr._quer_registrar_no_cerebro("o que tem no cérebro sobre FMECA?")
    # limpeza continua sendo limpeza
    assert not fr._quer_registrar_no_cerebro("apague os resultados do weibull")


def test_ferramenta_registrada_no_roteador():
    assert "registrar_no_cerebro" in fr._DESPACHO
    nomes = {e["name"] for e in fr.ESPEC_FERRAMENTAS}
    assert "registrar_no_cerebro" in nomes


def test_pede_titulo_e_conteudo_quando_faltam():
    r = fr.registrar_no_cerebro(pergunta="guarde no cérebro")
    assert not r["ok"] and "título" in r["mensagem"].lower()
