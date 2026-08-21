"""
Sprint 4 — robustez: confirmação em duas etapas para ações destrutivas (10.2).

Nenhuma limpeza ocorre sem o token explícito na mensagem do usuário.
"""

from src.conhecimento import ferramentas as F


def test_limpeza_pede_confirmacao_sem_token():
    res = F.limpar_resultados_ml(
        pergunta="apague os resultados da comparação"
    )
    msg = res["mensagem"]
    assert "CONFIRMAR LIMPEZA COMPARACAO" in msg
    assert "irrevers" in msg.lower()
    # é pedido de confirmação, NÃO execução
    assert "resultados apagados" not in msg.lower()


def test_limpeza_executa_somente_com_token(monkeypatch):
    chamadas = {}

    def fake_limpar(etapa):
        chamadas["etapa"] = etapa
        return []

    monkeypatch.setattr(F, "limpar_artefatos", fake_limpar)
    monkeypatch.setattr(F, "artefatos_a_partir", lambda etapa: [])

    res = F.limpar_resultados_ml(pergunta="CONFIRMAR LIMPEZA COMPARACAO")
    assert chamadas.get("etapa") == "comparacao"
    assert res["ok"]
