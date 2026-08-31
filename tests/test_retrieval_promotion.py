from src.conhecimento.agente_contexto import perfil_retrieval_ativo


def test_r4_e_o_perfil_canonico_promovido(monkeypatch):
    monkeypatch.delenv("AL_IADO_RETRIEVAL_PROFILE", raising=False)
    assert perfil_retrieval_ativo() == "r4_hybrid"


def test_rollback_para_baseline_e_explicito(monkeypatch):
    monkeypatch.setenv("AL_IADO_RETRIEVAL_PROFILE", "baseline")
    assert perfil_retrieval_ativo() == "baseline"


def test_perfil_desconhecido_falha_fechado_no_canonico(monkeypatch):
    monkeypatch.setenv("AL_IADO_RETRIEVAL_PROFILE", "inventado")
    assert perfil_retrieval_ativo() == "r4_hybrid"
