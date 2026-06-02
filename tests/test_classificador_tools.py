"""
Sprint 3 — roteamento dos tools do classificador PV Farms (6.4).
"""

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
