from __future__ import annotations

from src.conhecimento.ferramentas import ESPEC_FERRAMENTAS
from src.conhecimento.roteamento_ferramentas import _decisao_rapida, decidir_acao


def _tool(question: str):
    decision = decidir_acao(question)
    return decision["ferramenta"] if decision["usar_ferramenta"] else None


def test_greeting_and_conceptual_question_stay_with_the_agent():
    assert _tool("oi") is None
    assert _tool("explique a diferença conceitual entre FMEA e FMECA") is None


def test_explicit_execution_routes_to_the_two_real_publications():
    assert _tool("recalcule a comparação entre Denso e LSTM") == "executar_comparacao_autoencoders"
    assert _tool("regenere as curvas de confiabilidade física") == "gerar_confiabilidade"
    assert _tool("rode o pipeline completo") == "executar_pipeline_cientifico"


def test_result_queries_never_trigger_training():
    assert _tool("compare o Denso e o AE-LSTM nos resultados") == "consultar_comparacao_autoencoders"
    assert _tool("mostre os gráficos ROC e as matrizes") == "consultar_resultados"
    assert _tool("qual foi o SMD95 do fusível?") is None


def test_dataset_catalog_and_memory_have_dedicated_routes():
    assert _tool("qual dataset está ativo?") == "consultar_datasets"
    assert _tool("liste toda a base bibliográfica") == "listar_base_bibliografica"
    assert _tool("registre esse resultado no cérebro") == "registrar_no_cerebro"


def test_code_authoring_is_not_confused_with_pipeline_execution():
    decision = _decisao_rapida("escreva um código Python para plotar R(t)")
    assert decision == {"usar_ferramenta": False, "ferramenta": None}


def test_tool_contract_contains_no_retired_experiment_or_classifier():
    names = {item["name"] for item in ESPEC_FERRAMENTAS}
    assert "consultar_comparacao_autoencoders" in names
    assert "executar_comparacao_autoencoders" in names
    assert not any("macro" in name or "classificador" in name or "experimento" in name for name in names)
