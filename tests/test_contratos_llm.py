from __future__ import annotations

import pytest

from src.conhecimento.contratos_llm import LLMRequest, LLMResult, TaskType


def test_request_valida_campos_criticos():
    request = LLMRequest(
        task_type=TaskType.SCIENTIFIC_REASONING,
        messages=[{"role": "user", "content": "analise"}],
        methodological_risk="high",
        max_output_tokens=500,
    )
    assert request.task_type == TaskType.SCIENTIFIC_REASONING


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_type": "", "messages": [{"content": "x"}]},
        {"task_type": "chat", "messages": []},
        {"task_type": "chat", "messages": [{"content": "x"}], "max_cost": -1},
        {
            "task_type": "chat",
            "messages": [{"content": "x"}],
            "methodological_risk": "inventado",
        },
    ],
)
def test_request_rejeita_contrato_invalido(kwargs):
    with pytest.raises(ValueError):
        LLMRequest(**kwargs)


def test_result_exige_identidade_da_execucao():
    with pytest.raises(ValueError):
        LLMResult(content="ok", provider="", model="x", task_type="chat")
