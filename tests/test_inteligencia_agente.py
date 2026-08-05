"""
Regressoes de comportamento do agente.

CI leve: stub de langchain_core; nada de LLM/torch reais.
"""

from __future__ import annotations

import sys
import types
import unicodedata

import pytest

if "langchain_core" not in sys.modules:
    _lc = types.ModuleType("langchain_core")
    _lcm = types.ModuleType("langchain_core.messages")
    _lcm.HumanMessage = lambda content=None: {"content": content}
    _lc.messages = _lcm
    sys.modules["langchain_core"] = _lc
    sys.modules["langchain_core.messages"] = _lcm

from src.conhecimento.ferramentas import (  # noqa: E402
    _corrigir_descricao_visual,
    _quer_resposta_autoral,
    comentar_resultado,
)
from src.ml.resultados import _quer_imagens  # noqa: E402


def _sem_acentos(texto: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", texto)
        if unicodedata.category(ch) != "Mn"
    )


class _FakeLLM:
    def __init__(self):
        self.chamado = False
        self.mensagens = None

    def invoke(self, msgs):
        self.chamado = True
        self.mensagens = msgs
        return types.SimpleNamespace(content="[INTERPRETACAO]")


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
    assert llm.chamado is False


@pytest.mark.parametrize("pergunta", [
    "qual a sua opiniao sobre os resultados?",
    "o que isso significa para a dissertacao?",
    "interprete os resultados do pipeline",
    "esses resultados reforcam minha proposta?",
])
def test_pergunta_autoral_interpreta_via_llm_mesmo_com_resposta_direta(pergunta):
    llm = _FakeLLM()
    out = comentar_resultado(pergunta, _RES_DIRETO, "perfil do agente", llm)
    assert out == "[INTERPRETACAO]"
    assert llm.chamado is True


def test_sem_llm_degrada_para_tabela():
    out = comentar_resultado("sua opiniao?", _RES_DIRETO, "perfil", None)
    assert out == "| tabela crua |"


def test_deteccao_autoral_vs_neutra():
    assert _quer_resposta_autoral("na sua opiniao, qual o melhor?")
    assert _quer_resposta_autoral("o que isso significa?")
    assert not _quer_resposta_autoral("mostre a matriz de confusao")
    assert not _quer_resposta_autoral("rode o pipeline")


def test_intencao_de_imagens_distingue_mostrar_de_gerar():
    assert _quer_imagens("mostre os graficos")
    assert _quer_imagens("veja a curva ROC")
    assert not _quer_imagens("quero os resultados do pipeline")
    assert not _quer_imagens("qual a situacao geral do trabalho")


def test_comentador_recebe_inventario_visual_e_proibe_descricao_inventada():
    llm = _FakeLLM()
    resultado = {
        "ok": True,
        "mensagem": "AUC e contagens recalculadas.",
        "imagens": [
            {"caption": "Proposto - comparacao por pontos", "grupo": "Proposto"},
            {"caption": "Ibrahim - anomalias detectadas", "grupo": "Ibrahim"},
        ],
        "resposta_pronta": False,
    }

    comentar_resultado("compare e interprete os modelos", resultado, "perfil", llm)

    mensagem = llm.mensagens[0]
    prompt = mensagem.get("content") if isinstance(mensagem, dict) else mensagem.content
    prompt_norm = _sem_acentos(prompt)
    assert "inventario autoritativo" in prompt_norm
    assert "Proposto - comparacao por pontos" in prompt
    assert "Ibrahim - anomalias detectadas" in prompt
    assert "Nenhum dos dois mostra distribuicao de scores" in prompt_norm


def test_corretor_remove_descricao_visual_incompativel_com_as_legendas():
    resposta = (
        "O AE-LSTM lidera por AUC.\n\n"
        "Os graficos mostram distribuicoes de scores, curvas ROC e deteccoes "
        "ao longo do tempo.\n\n"
        "A comparacao deve considerar tambem o ponto de operacao."
    )
    imagens = [
        {"caption": "Proposto - comparacao por pontos", "path": "comparacao_metricas_pontos.png"},
        {"caption": "Ibrahim - anomalias detectadas", "path": "anomalias_detectadas.png"},
    ]

    corrigida = _corrigir_descricao_visual(resposta, imagens)
    corrigida_norm = _sem_acentos(corrigida)

    assert "distribuicoes de scores" not in corrigida
    assert "curvas ROC" not in corrigida
    assert "O AE-LSTM lidera por AUC." in corrigida
    assert "comparacao das metricas por pontos" in corrigida_norm
    assert "contagens de deteccoes e a cobertura percentual" in corrigida_norm
