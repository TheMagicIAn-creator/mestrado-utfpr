"""
A geometria do chat da Web V2 — o que fazia a tela parecer menor que a tela.

POR QUE ESTE TESTE EXISTE
=========================
A queixa foi visual: *"não vejo que a tela do chat está grande, ela aparece uma
janela um pouco menor"*. A causa não era estética, era aritmética. A superfície
do chat tinha altura fixada por conta:

    .chat-surface  { min-height: calc(100vh - 240px) }
    .chat-messages { max-height: calc(100vh - 354px) }

Os 240 e os 354 só fecham para UMA combinação de topbar, cabeçalho da vista e
fileira de sugestões. Basta o cabeçalho quebrar em duas linhas, ou uma sugestão
passar para a segunda linha, e sobra um retângulo no meio da página — com o
resto da altura desperdiçado.

A correção troca as contas por flexbox: a lista de mensagens recebe o que
sobrar. Estes testes impedem que os números mágicos voltem, porque o sintoma é
visual e não aparece em nenhum teste de rota.

Medido no Chromium com a página real, depois da correção: o chat passou a
ocupar 76% da altura num monitor de 900 px e 70% num laptop de 720 px, contra
58% antes; a página parou de rolar e a lista de mensagens virou o único
elemento rolável.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
CSS = (RAIZ / "src/webapp_v2/static/styles.css").read_text(encoding="utf-8")
JS = (RAIZ / "src/webapp_v2/static/app.js").read_text(encoding="utf-8")

# Os comentários deste arquivo CITAM os valores antigos para explicar o defeito
# — varrer o texto cru faria a guarda acusar a própria documentação da correção.
CSS_SEM_COMENTARIO = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)


def _bloco(seletor: str) -> str:
    """Primeiro corpo de regra do seletor, para asserções locais."""
    achado = re.search(
        rf"(?m)^{re.escape(seletor)}\s*\{{(.*?)\}}", CSS_SEM_COMENTARIO, re.DOTALL
    )
    assert achado, f"regra `{seletor}` sumiu do styles.css"
    return achado.group(1)


# ── os números mágicos não podem voltar ────────────────────────────────────

@pytest.mark.parametrize("seletor", [".chat-surface", ".chat-messages"])
def test_altura_do_chat_nao_e_conta_com_o_viewport(seletor):
    """Altura por `calc(100vh - N)` quebra sempre que o cromo acima muda."""
    corpo = _bloco(seletor)
    assert "calc(100vh" not in corpo, (
        f"{seletor} voltou a fixar altura por conta com o viewport; use flex"
    )
    assert "calc(100dvh" not in corpo, (
        f"{seletor} voltou a fixar altura por conta com o viewport; use flex"
    )


def test_a_lista_de_mensagens_recebe_o_que_sobrar():
    corpo = _bloco(".chat-messages")
    assert "flex: 1 1 auto" in corpo
    # Sem `min-height: 0` um filho longo estoura o contêiner flex e a rolagem
    # migra para a página inteira, levando o compositor para fora da tela.
    assert "min-height: 0" in corpo
    assert "overflow-y: auto" in corpo


def test_a_superficie_do_chat_tambem_e_flexivel():
    corpo = _bloco(".chat-surface")
    assert "flex: 1 1 auto" in corpo
    assert "min-height: 0" in corpo


def test_nenhuma_regra_do_chat_usa_vh_cru():
    """`100vh` conta a barra de endereço do celular mesmo recolhida.

    Era o que empurrava o compositor para fora da tela no telefone. O projeto
    inteiro usa `dvh`; esta guarda impede a regressão.
    """
    for regra in re.findall(r"(?m)^[^@{}]*\{[^}]*\}", CSS_SEM_COMENTARIO):
        if "100vh" in regra and "chat" in regra.split("{")[0]:
            pytest.fail(f"regra do chat com 100vh cru:\n{regra[:200]}")


# ── a vista em tela cheia ──────────────────────────────────────────────────

def test_o_modo_tela_cheia_e_ligado_pelo_javascript():
    """A classe é o contrato entre `switchView` e o CSS."""
    assert 'classList.toggle("is-agent-view", view === "agent")' in JS
    assert "body.is-agent-view" in CSS


def test_apenas_a_vista_do_agente_trava_a_rolagem_da_pagina():
    """Painéis, tabelas e figuras precisam da rolagem normal da página."""
    corpo = _bloco("body.is-agent-view")
    assert "overflow: hidden" in corpo
    # A trava é da classe, nunca do body solto.
    assert not re.search(r"(?m)^body\s*\{[^}]*overflow:\s*hidden", CSS_SEM_COMENTARIO)


def test_o_compositor_nao_fica_atras_da_barra_inferior_no_celular():
    """Abaixo de 900 px a navegação vira barra fixa de 62 px no rodapé.

    Sem folga o compositor existe, recebe foco e é invisível — foi o que a
    medição no Chromium pegou em 390 px de largura.
    """
    achado = re.search(
        r"body\.is-agent-view main\s*\{([^}]*padding-bottom[^}]*)\}",
        CSS_SEM_COMENTARIO,
    )
    assert achado, "a vista do agente perdeu a folga da barra inferior"
    assert "62px" in achado.group(1)
    assert "safe-area-inset-bottom" in achado.group(1), (
        "iPhone com barra de gestos precisa da área segura"
    )


# ── a coluna de leitura ────────────────────────────────────────────────────

def test_a_coluna_do_chat_vem_de_um_unico_token():
    """Eram 780, 680 e 900 px em três regras — as bordas não se alinhavam."""
    assert "--chat-col:" in CSS
    assert CSS.count("var(--chat-col)") >= 4
    for largura in ("900px", "780px"):
        assert f"min(calc(100% - 36px), {largura})" not in CSS_SEM_COMENTARIO


def test_a_resposta_do_agente_ocupa_a_coluna_inteira():
    """Tabela FMECA e bloco de código não cabem num balão de 78%.

    Só o turno do usuário é balão, como nas interfaces de referência.
    """
    corpo = _bloco(".message-body")
    assert "width: 100%" in corpo
    assert "border: 0" in corpo
    assert "background: transparent" in corpo

    usuario = _bloco(".user-message .message-body")
    assert "width: fit-content" in usuario
    assert "#2675d8" in usuario


def test_as_sugestoes_somem_quando_a_conversa_comeca():
    assert "body.has-conversation .prompt-row" in CSS
    assert 'if (isUser) document.body.classList.add("has-conversation")' in JS
