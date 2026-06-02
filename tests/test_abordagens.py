"""
Sprint 3 — comparar_abordagens_ml (6.4): supervisionado x não supervisionado
x sintético, sem misturar domínios CC/CA.
"""

from src.conhecimento.ferramentas import (
    _decisao_rapida,
    comparar_abordagens_ml,
)


def test_roteamento_comparar_abordagens():
    for p in (
        "compare as abordagens de ML",
        "qual a diferenca entre supervisionado e nao supervisionado?",
    ):
        d = _decisao_rapida(p) or {}
        assert d.get("ferramenta") == "comparar_abordagens_ml", p


def test_conteudo_separa_dominios_e_evidencia():
    res = comparar_abordagens_ml()
    assert res["resposta_pronta"] and res["ok"]
    msg = res["mensagem"].lower()
    assert "supervisionada" in msg
    assert "pv farms" in msg and "paderborn" in msg
    assert "e2" in msg                      # validação sintética = E2
    assert "diagnostica" in msg             # ressalva: PV Farms não diagnostica CA
