"""
Sprint 3 — roteamento dos tools do classificador PV Farms (6.4).

Rotear certo não basta: `avaliar_classificador_pv` chamava `Path()` sem
importar `pathlib` e foi para o main, porque aqui só se verificava para onde
a pergunta era roteada — a função nunca era EXECUTADA. Os testes de execução
abaixo fecham esse furo.
"""

import src.conhecimento.ferramentas as fr
from src.conhecimento.ferramentas import _decisao_rapida, classificar_amostra_pv


def test_roteamento_classificador():
    casos = [
        ("treine o classificador PV Farms", "treinar_classificador_pv"),
        ('classifique a amostra {"f1": 1}', "classificar_amostra_pv"),
        ("mostre as metricas do classificador", "avaliar_classificador_pv"),
    ]
    for p, esp in casos:
        assert (_decisao_rapida(p) or {}).get("ferramenta") == esp, p


def test_nao_colide_com_experimento():
    d = _decisao_rapida("rode o experimento do ghoneim") or {}
    assert d.get("ferramenta") == "rodar_experimento_artigo"


def test_classificar_sem_json_pede_json():
    r = classificar_amostra_pv(pergunta="classifique a amostra")
    assert r["resposta_pronta"] and "json" in r["mensagem"].lower()


# ── execução: rotear certo não prova que a ferramenta roda ───────────────────

def test_avaliar_classificador_executa_sem_nameerror():
    """Era NameError: Path() usado sem `from pathlib import Path` na função."""
    r = fr.avaliar_classificador_pv(pergunta="mostre as metricas do classificador")
    assert isinstance(r, dict) and r["ok"]
    assert r["mensagem"].strip()


def test_ferramentas_de_leitura_executam():
    """Varre as ferramentas seguras (só leem artefatos) atrás do mesmo defeito.

    Não inclui as que treinam, apagam ou chamam rede — essas seguem cobertas
    pelo lint de nomes indefinidos (ruff F821, bloqueante no CI).
    """
    seguras = [
        "consultar_status_pipeline", "consultar_resultados", "consultar_datasets",
        "listar_base_bibliografica", "listar_experimentos_artigos",
        "avaliar_classificador_pv",
    ]
    for nome in seguras:
        fn = fr._DESPACHO.get(nome)
        assert fn is not None, f"{nome} saiu do despacho"
        r = fr.executar_ferramenta(nome, pergunta="consulta de rotina")
        assert isinstance(r, dict), nome
        assert "mensagem" in r, nome
