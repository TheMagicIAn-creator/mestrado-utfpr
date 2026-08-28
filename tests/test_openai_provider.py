from __future__ import annotations

from types import SimpleNamespace

from src.conhecimento.contratos_llm import LLMRequest
from src.conhecimento.provedores.openai import OpenAIProvider


class Responses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(type="response.created"),
                    SimpleNamespace(type="response.output_text.delta", delta="res"),
                    SimpleNamespace(type="response.output_text.delta", delta="posta"),
                ]
            )
        return SimpleNamespace(
            output_text='{"status":"ok"}',
            usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
        )


def test_openai_structured_output_e_usage_sem_chamada_real():
    responses = Responses()
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    request = LLMRequest(
        task_type="evidence_audit",
        messages=[{"role": "user", "content": "audite"}],
        structured_output={"type": "object", "properties": {"status": {"type": "string"}}},
        reasoning_level="low",
        max_output_tokens=300,
    )
    result = provider.generate(request, model_id="modelo-configurado")
    assert result.structured_data == {"status": "ok"}
    assert result.usage.total_tokens == 14
    assert responses.calls[0]["reasoning"] == {"effort": "low"}
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"


def test_openai_stream_sem_chamada_real():
    responses = Responses()
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    request = LLMRequest(
        task_type="simple_chat",
        messages=[{"role": "user", "content": "oi"}],
    )
    chunks = list(provider.stream(request, model_id="modelo-configurado"))
    assert "".join(item.content for item in chunks) == "resposta"
