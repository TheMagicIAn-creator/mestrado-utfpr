from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.conhecimento.contratos_llm import LLMRequest
from src.conhecimento.provedores.base import ProviderError, ProviderNotConfiguredError
from src.conhecimento.provedores.gemini import GeminiProvider
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


def test_openai_le_blocos_e_calcula_total_quando_sdk_nao_fornece():
    content = [SimpleNamespace(text="parte 1"), SimpleNamespace(text=" parte 2")]
    response = SimpleNamespace(
        output=[SimpleNamespace(content=content)],
        usage=SimpleNamespace(input_tokens=8, output_tokens=2),
    )
    responses = SimpleNamespace(create=lambda **_kwargs: response)
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    result = provider.generate(
        LLMRequest(task_type="factual_short", messages=[{"content": "x"}]),
        model_id="modelo",
    )
    assert result.content == "parte 1 parte 2"
    assert result.usage.total_tokens == 10


def test_openai_rejeita_json_estruturado_que_nao_seja_objeto():
    responses = SimpleNamespace(
        create=lambda **_kwargs: SimpleNamespace(output_text="[1, 2]", usage=None)
    )
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    request = LLMRequest(
        task_type="evidence_audit",
        messages=[{"content": "x"}],
        structured_output={"type": "object"},
    )
    with pytest.raises(ValueError):
        provider.generate(request, model_id="modelo")


def test_openai_sem_configuracao_falha_sem_tentar_rede(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider(api_key=None)
    with pytest.raises(ProviderNotConfiguredError):
        provider.generate(
            LLMRequest(task_type="simple_chat", messages=[{"content": "oi"}]),
            model_id="modelo",
        )


def test_openai_classifica_erro_do_sdk():
    class RateLimitError(RuntimeError):
        status_code = 429

    responses = SimpleNamespace(
        create=lambda **_kwargs: (_ for _ in ()).throw(RateLimitError("limite"))
    )
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses))
    request = LLMRequest(task_type="simple_chat", messages=[{"content": "oi"}])
    with pytest.raises(ProviderError) as captured:
        provider.generate(request, model_id="modelo")
    assert captured.value.transient is True


def test_gemini_provider_texto_json_e_stream_sem_chamada_real():
    class Models:
        def generate_content(self, **_kwargs):
            return SimpleNamespace(text="resposta")

        def generate_content_stream(self, **_kwargs):
            return iter([SimpleNamespace(text="res"), SimpleNamespace(text="posta")])

    provider = GeminiProvider(client=SimpleNamespace(models=Models()))
    text_request = LLMRequest(
        task_type="scientific_reasoning",
        messages=[{"content": "analise"}],
    )
    result = provider.generate(text_request, model_id="modelo")
    assert result.content == "resposta"
    assert "".join(
        chunk.content for chunk in provider.stream(text_request, model_id="modelo")
    ) == "resposta"


def test_gemini_provider_saida_estruturada_e_stream_atomico():
    class Models:
        def generate_content(self, **_kwargs):
            return SimpleNamespace(text='{"status":"aprovado"}')

    provider = GeminiProvider(client=SimpleNamespace(models=Models()))
    request = LLMRequest(
        task_type="evidence_audit",
        messages=[{"content": "audite"}],
        structured_output={"type": "object"},
        max_output_tokens=200,
    )
    result = provider.generate(request, model_id="modelo")
    assert result.structured_data == {"status": "aprovado"}
    chunks = list(provider.stream(request, model_id="modelo"))
    assert len(chunks) == 1
    assert chunks[0].content == result.content


def test_gemini_provider_sem_configuracao(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiProvider(api_key=None)
    request = LLMRequest(task_type="simple_chat", messages=[{"content": "oi"}])
    with pytest.raises(ProviderNotConfiguredError):
        provider.generate(request, model_id="modelo")
