"""
Regressoes de intencao do chat.

Esses casos protegem contra o erro mais perigoso do agente: transformar uma
pergunta discursiva/conceitual em execucao pesada de pipeline ou experimento.
"""

from src.conhecimento.agente import deve_consultar_literatura
from src.conhecimento.ferramentas import (
    _decisao_rapida,
    decidir_acao,
    limpar_experimentos_artigos,
)


def _ferramenta(pergunta: str):
    decisao = _decisao_rapida(pergunta) or {}
    return decisao.get("ferramenta")


def test_revisao_bibliografica_de_rul_nao_roda_weibull():
    decisao = _decisao_rapida(
        "Faça uma revisão bibliográfica curta sobre RUL em eletrônica de potência."
    )
    assert decisao == {"usar_ferramenta": False, "ferramenta": None}


def test_citar_artigos_sobre_anomalias_vai_para_rag_nao_experimentos():
    pergunta = "Cite artigos sobre detecção de anomalias em inversores fotovoltaicos."
    esperado = {"usar_ferramenta": False, "ferramenta": None}

    assert _decisao_rapida(pergunta) == esperado
    assert deve_consultar_literatura(pergunta)

    class RoteadorRuim:
        def invoke(self, *_args, **_kwargs):  # pragma: no cover - nao deve chamar
            raise AssertionError("o LLM roteador nao deve ser chamado")

    assert decidir_acao(pergunta, RoteadorRuim()) == esperado


def test_explicacao_de_fmea_com_base_no_projeto_nao_aciona_pipeline():
    decisao = _decisao_rapida("Não use literatura. Explique FMEA com base no projeto.")
    assert decisao == {"usar_ferramenta": False, "ferramenta": None}
    assert not deve_consultar_literatura(
        "Não use literatura. Explique FMEA com base no projeto."
    )


def test_stender_sobre_paderborn_vai_para_rag_nao_catalogo_dataset():
    decisao = _decisao_rapida("O que o Stender diz sobre o dataset de Paderborn?")
    assert decisao == {"usar_ferramenta": False, "ferramenta": None}
    assert deve_consultar_literatura(
        "O que o Stender diz sobre o dataset de Paderborn?"
    )


def test_compare_autores_com_artefatos_consulta_resultados_sem_treinar():
    pergunta = (
        "Compare Sharma, Ibrahim e Ahirwar usando somente os artefatos "
        "recalculados do repositório. Mostre gráficos e matrizes."
    )
    assert _ferramenta(pergunta) == "consultar_resultados"


def test_resultados_locais_vs_replicacoes_consulta_proveniencia():
    pergunta = (
        "Quais resultados são dos datasets locais e quais são apenas "
        "replicações dos artigos?"
    )
    assert _ferramenta(pergunta) == "consultar_resultados"


def test_apagar_experimentos_tem_rota_propria_e_confirmacao():
    assert _ferramenta("Quero que apague os experimentos.") == "limpar_experimentos_artigos"
    res = limpar_experimentos_artigos(pergunta="Quero que apague os experimentos.")
    assert res["ok"]
    assert "CONFIRMAR LIMPEZA EXPERIMENTOS" in res["mensagem"]
    assert "dados brutos" in res["mensagem"].lower()
