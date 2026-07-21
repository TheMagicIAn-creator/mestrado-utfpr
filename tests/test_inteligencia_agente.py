"""
Reforma de comportamento do agente (2026-07):
- perguntas AUTORAIS interpretam via LLM, mesmo em ferramentas de resposta
  direta (antes o forcar_resposta_direta despejava a tabela crua);
- gráficos ficam desacoplados: por padrão oferecem antevisão sob demanda e
  download; renderizam inline só sob pedido explícito.

CI-leve: stub de langchain_core; nada de LLM/torch reais.
"""

from __future__ import annotations

import sys
import types

import pytest

# stub mínimo de langchain_core.messages para o caminho de interpretação
if "langchain_core" not in sys.modules:
    _lc = types.ModuleType("langchain_core")
    _lcm = types.ModuleType("langchain_core.messages")
    _lcm.HumanMessage = lambda content=None: {"content": content}
    _lc.messages = _lcm
    sys.modules["langchain_core"] = _lc
    sys.modules["langchain_core.messages"] = _lcm

from src.conhecimento.ferramentas import (  # noqa: E402
    _quer_resposta_autoral,
    comentar_resultado,
)
from src.ml.resultados import _quer_imagens  # noqa: E402


class _FakeLLM:
    def __init__(self):
        self.chamado = False
        self.mensagens = None

    def invoke(self, msgs):
        self.chamado = True
        self.mensagens = msgs
        return types.SimpleNamespace(content="[INTERPRETAÇÃO]")


_RES_DIRETO = {
    "ok": True,
    "mensagem": "| tabela crua |",
    "forcar_resposta_direta": True,
    "resposta_pronta": True,
}


def test_pergunta_neutra_devolve_tabela_direta_sem_llm():
    llm = _FakeLLM()
    out = comentar_resultado("mostre os resultados", _RES_DIRETO, "perfil", llm)
    assert out == "| tabela crua |"
    assert llm.chamado is False          # não gastou LLM à toa


@pytest.mark.parametrize("pergunta", [
    "qual a sua opinião sobre os resultados?",
    "o que isso significa para a dissertação?",
    "interprete os resultados do pipeline",
    "esses resultados reforçam minha proposta?",
])
def test_pergunta_autoral_interpreta_via_llm_mesmo_com_resposta_direta(pergunta):
    llm = _FakeLLM()
    out = comentar_resultado(pergunta, _RES_DIRETO, "perfil do agente", llm)
    assert out == "[INTERPRETAÇÃO]"
    assert llm.chamado is True


def test_sem_llm_degrada_para_tabela():
    out = comentar_resultado("sua opinião?", _RES_DIRETO, "perfil", None)
    assert out == "| tabela crua |"


def test_deteccao_autoral_vs_neutra():
    assert _quer_resposta_autoral("na sua opinião, qual o melhor?")
    assert _quer_resposta_autoral("o que isso significa?")
    assert not _quer_resposta_autoral("mostre a matriz de confusão")
    assert not _quer_resposta_autoral("rode o pipeline")


def test_intencao_de_imagens_distingue_mostrar_de_gerar():
    assert _quer_imagens("mostre os gráficos")
    assert _quer_imagens("veja a curva ROC")
    assert not _quer_imagens("quero os resultados do pipeline")
    assert not _quer_imagens("qual a situação geral do trabalho")


def test_comentador_recebe_inventario_visual_e_proibe_descricao_inventada():
    llm = _FakeLLM()
    resultado = {
        "ok": True,
        "mensagem": "AUC e contagens recalculadas.",
        "imagens": [
            {"caption": "Francisti - comparação por pontos", "grupo": "Francisti"},
            {"caption": "Ibrahim - anomalias detectadas", "grupo": "Ibrahim"},
        ],
        "resposta_pronta": False,
    }

    comentar_resultado("compare e interprete os modelos", resultado, "perfil", llm)

    mensagem = llm.mensagens[0]
    prompt = mensagem.get("content") if isinstance(mensagem, dict) else mensagem.content
    assert "inventário autoritativo" in prompt
    assert "Francisti - comparação por pontos" in prompt
    assert "Ibrahim - anomalias detectadas" in prompt
    assert "Nenhum dos dois mostra distribuição de scores" in prompt
