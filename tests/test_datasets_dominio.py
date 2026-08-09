"""
Sprint 3 — integração PV Farms + separação de domínio (critérios #13/#14/#15).

- o perfil do agente conhece a regra: PV Farms (CC) NUNCA diagnostica CA;
- 'quais os datasets', 'diferença entre paderborn e pv farms' → consultar_datasets;
- a ferramenta separa explicitamente os domínios CC e CA.
"""

from src.conhecimento.agente import PERFIL_COMPACTO
from src.conhecimento.ferramentas import _decisao_rapida, consultar_datasets


def test_perfil_tem_regra_separacao_dominio():
    txt = " ".join(PERFIL_COMPACTO.lower().split())  # normaliza espaços/quebras
    assert "pv farms" in txt and "paderborn" in txt
    assert "separacao de dominio" in txt or "separação de domínio" in txt
    assert "pv farms diagnostica falhas ca" in txt  # nunca diagnostica CA


def test_roteamento_consultar_datasets():
    for p in (
        "quais os datasets do projeto?",
        "me explique o dataset de paderborn",
        "qual a diferenca entre paderborn e pv farms?",
    ):
        d = _decisao_rapida(p) or {}
        assert d.get("ferramenta") == "consultar_datasets", p
    # não confundir com rodar experimento
    d = _decisao_rapida("rode o experimento do ghoneim") or {}
    assert d.get("ferramenta") != "consultar_datasets"


def test_consultar_datasets_separa_dominio():
    res = consultar_datasets()
    assert res["resposta_pronta"] and res["ok"]
    msg = res["mensagem"].lower()
    assert "pv farms" in msg and "paderborn" in msg
    assert "stender" in msg and "bearing dataset" in msg
    assert "simulado" in msg and "gpvs-faults" in msg
    assert "cc" in msg and "ca" in msg
    assert "separação de domínio" in msg or "separacao de dominio" in msg
    assert "weibull físico" in msg and "a_det" in msg
