"""O GPVS-Faults é o único domínio experimental ativo."""

from src.conhecimento.agente import PERFIL_COMPACTO
from src.conhecimento.ferramentas import _decisao_rapida, consultar_datasets


def test_agent_profile_declares_domain_separation():
    text = " ".join(PERFIL_COMPACTO.lower().split())
    assert "gpvs-faults é o único dataset ativo" in text
    assert "separação de domínio" in text or "separacao de dominio" in text
    assert "paderborn" in text and "pv farms" in text and "não fornecem amostras" in text


def test_dataset_questions_route_to_the_canonical_inventory():
    for question in (
        "quais os datasets do projeto?",
        "me explique o dataset de Paderborn",
        "qual a diferença entre Paderborn e PV Farms?",
        "qual é o papel do GPVS?",
    ):
        decision = _decisao_rapida(question) or {}
        assert decision.get("ferramenta") == "consultar_datasets", question


def test_dataset_tool_exposes_active_and_inactive_roles():
    response = consultar_datasets()
    message = response["mensagem"].lower()
    assert response["resposta_pronta"] and response["ok"]
    assert "único conjunto de dados ativo" in message
    assert "f0l/f0m" in message and "f1l-f7m" in message
    assert "14 ensaios reais de bancada" in message
    assert "paderborn" in message and "pmsm" in message and "pv farms" in message
    assert "não fornecem amostras" in message
    assert "weibull físico" in message and "rul" in message
