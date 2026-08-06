"""
Testes da detecção de "recalcule de verdade" e da mensagem de SKIP.

Por que este arquivo existe: o pesquisador relatou que os JSON e os gráficos em
`resultados/` nunca mudavam, e suspeitou que nada estivesse sendo recalculado.
A suspeita estava certa, por dois mecanismos somados:

1. `_deve_forcar` casava uma lista fechada de substrings que exigia o
   INFINITIVO ("rodar de novo"). Quem fala no imperativo — "rode o pipeline de
   novo", "retreine o autoencoder", "execute novamente" — não ativava `force`,
   e a etapa READY era pulada.
2. A mensagem do SKIP era só "já está pronto", e o chamador concatenava a
   tabela de resultados logo abaixo. Lia-se como execução fresca.

Somados ao determinismo do treino (semente fixa ⇒ números idênticos), não havia
como distinguir um SKIP de um recálculo real olhando os arquivos.

Falso-negativo aqui é o defeito grave: o pesquisador acredita ter retreinado
sem ter. Falso-positivo apenas gasta tempo de máquina.
"""

from __future__ import annotations

import pytest

from src.conhecimento.intencoes_ferramentas import _deve_forcar

# ── frases REAIS que o pesquisador usa e que antes NÃO forçavam ─────────────

FRASES_QUE_PEDEM_RECALCULO = [
    # imperativo — a flexão que a lista fechada não cobria
    "rode o pipeline de novo",
    "rode tudo de novo",
    "roda de novo o pipeline",
    "rode a validacao mais uma vez",
    "execute o pipeline novamente",
    "reexecute a validacao",
    # verbo específico de treino
    "retreine o autoencoder",
    "treine o autoencoder novamente",
    "quero retreinar tudo",
    # já funcionavam antes — regressão
    "refaça o pipeline",
    "recalcule tudo",
    "quero rodar do zero",
    "regenerar os graficos",
    "apagar os resultados e comecar de novo",
]

FRASES_DE_LEITURA = [
    "mostre a matriz de confusao",
    "qual o limiar operacional",
    "compare meu metodo com a literatura",
    "quais artigos falam de FMECA",
    "mostre os graficos",
    "qual a SMD do IGBT",
    "explique o que e o NPR projetado",
    "qual o status do pipeline",
    "me mostre a tabela de validacao",
]


@pytest.mark.parametrize("frase", FRASES_QUE_PEDEM_RECALCULO)
def test_pedido_de_recalculo_ativa_force(frase):
    assert _deve_forcar(frase) is True, (
        f"'{frase}' nao ativou force — a etapa seria PULADA e o pesquisador "
        "acreditaria ter recalculado"
    )


@pytest.mark.parametrize("frase", FRASES_DE_LEITURA)
def test_consulta_de_leitura_nao_ativa_force(frase):
    assert _deve_forcar(frase) is False, (
        f"'{frase}' e consulta, nao pedido de recalculo — forcar aqui joga "
        "fora artefatos por engano"
    )


def test_flexao_verbal_e_coberta_nao_so_o_infinitivo():
    """A regressão original: infinitivo passava, imperativo não."""
    for verbo in ("rodar", "rode", "roda", "executar", "execute", "executa",
                  "recalcular", "recalcule", "refazer", "refaça"):
        frase = f"{verbo} o pipeline de novo"
        assert _deve_forcar(frase) is True, frase


def test_verbo_sem_marcador_de_repeticao_nao_forca():
    """"rode o pipeline" é primeira execução, não recálculo."""
    assert _deve_forcar("rode o pipeline") is False
    assert _deve_forcar("execute a validacao") is False


def test_termos_que_ja_significam_recalculo_dispensam_marcador():
    """"retreine" e "recalcule" já carregam o "de novo" na própria palavra."""
    assert _deve_forcar("retreine o autoencoder") is True
    assert _deve_forcar("recalcule a validacao") is True
    assert _deve_forcar("reprocessar a literatura") is True


def test_acento_e_caixa_nao_alteram_a_deteccao():
    for variante in ("REFAÇA O PIPELINE", "refaca o pipeline",
                     "Refaça O Pipeline"):
        assert _deve_forcar(variante) is True


# ── a mensagem de SKIP precisa dizer que não recalculou ────────────────────


def test_mensagem_de_skip_declara_que_nao_recalculou(monkeypatch):
    """Número sem carimbo de origem é o que induz o erro de leitura."""
    from src.ml import pipeline

    monkeypatch.setattr(pipeline, "estado_etapa_completo",
                        lambda _k: {"estado": "ready", "motivos": []})
    monkeypatch.setattr(pipeline, "_data_do_manifesto",
                        lambda _k: "2026-08-05T01:39:59-03:00")
    monkeypatch.setattr(pipeline, "dependencias_pendentes", lambda _k: [])

    r = pipeline.executar_etapa("autoencoder", force=False)

    assert r["executou"] is False
    assert r["recalculou"] is False
    assert "NAO recalculei" in r["mensagem"]
    assert "2026-08-05T01:39:59-03:00" in r["mensagem"]
    assert r["artefatos_de"] == "2026-08-05T01:39:59-03:00"


def test_data_do_manifesto_nunca_derruba_a_etapa(monkeypatch):
    """Diagnóstico quebrado não pode impedir a execução."""
    from src.ml import pipeline

    def explode(_k):
        raise RuntimeError("manifesto corrompido")

    monkeypatch.setattr("src.ml.proveniencia.carregar_manifesto", explode)
    assert pipeline._data_do_manifesto("autoencoder") == "data desconhecida"
