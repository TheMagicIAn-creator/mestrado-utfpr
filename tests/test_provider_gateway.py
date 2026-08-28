from __future__ import annotations

import pytest

from src.conhecimento.contratos_llm import LLMRequest, LLMResult, LLMStreamChunk
from src.conhecimento.provedores import (
    ModelRegistration,
    ModelStatus,
    ProviderGateway,
    ProviderRegistry,
)
from src.conhecimento.provedores.base import ProviderError


class FakeProvider:
    name = "fake"
    configured = True

    def generate(self, request, *, model_id):
        return LLMResult(
            content="ok",
            provider=self.name,
            model=model_id,
            task_type=request.task_type,
        )

    def stream(self, request, *, model_id):
        yield LLMStreamChunk("o", self.name, model_id, request.task_type)
        yield LLMStreamChunk("k", self.name, model_id, request.task_type)


def _request(multimodal=False):
    return LLMRequest(
        task_type="scientific_reasoning",
        messages=[{"role": "user", "content": "teste"}],
        multimodal=multimodal,
    )


def test_gateway_executa_modelo_operacional_e_mede_latencia():
    registry = ProviderRegistry()
    registry.register_provider("fake", FakeProvider())
    registry.register_model(
        ModelRegistration("fake", "terra", "modelo-real", capabilities=frozenset({"text"}))
    )
    result = ProviderGateway(registry).execute(
        _request(), provider="fake", model_alias="terra"
    )
    assert result.content == "ok"
    assert result.model == "modelo-real"
    assert result.latency_ms is not None and result.latency_ms >= 0


def test_gateway_stream_preserva_metadados():
    registry = ProviderRegistry()
    registry.register_provider("fake", FakeProvider())
    registry.register_model(ModelRegistration("fake", "luna", "modelo"))
    chunks = list(
        ProviderGateway(registry).stream(
            _request(), provider="fake", model_alias="luna"
        )
    )
    assert "".join(item.content for item in chunks) == "ok"
    assert {item.provider for item in chunks} == {"fake"}


def test_modelo_experimental_nao_entra_no_caminho_critico():
    registry = ProviderRegistry()
    registry.register_provider("fake", FakeProvider())
    registry.register_model(
        ModelRegistration("fake", "novo", "modelo", status=ModelStatus.EXPERIMENTAL)
    )
    gateway = ProviderGateway(registry)
    with pytest.raises(ProviderError):
        gateway.execute(_request(), provider="fake", model_alias="novo")
    assert gateway.execute(
        _request(), provider="fake", model_alias="novo", allow_experimental=True
    ).content == "ok"


def test_capacidade_multimodal_e_obrigatoria():
    registry = ProviderRegistry()
    registry.register_provider("fake", FakeProvider())
    registry.register_model(ModelRegistration("fake", "texto", "modelo"))
    with pytest.raises(ProviderError):
        ProviderGateway(registry).execute(
            _request(multimodal=True), provider="fake", model_alias="texto"
        )


def test_registry_status_nao_expoe_chaves():
    registry = ProviderRegistry()
    registry.register_provider("fake", FakeProvider())
    registry.register_model(ModelRegistration("fake", "luna", "modelo"))
    payload = registry.status()
    assert payload["providers"]["fake"] == {"configured": True}
    assert "api_key" not in str(payload).lower()
