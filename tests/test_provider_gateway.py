from __future__ import annotations

import pytest

from src.conhecimento.contratos_llm import LLMRequest, LLMResult, LLMStreamChunk
from src.conhecimento.provedores import (
    ModelRegistration,
    ModelStatus,
    ProviderGateway,
    ProviderRegistry,
)
from src.conhecimento.provedores.base import (
    ProviderError,
    ProviderNotConfiguredError,
    classify_provider_exception,
    normalized_messages,
)
from src.conhecimento.provedores.gateway import build_default_gateway


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


class UnconfiguredProvider(FakeProvider):
    configured = False


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


def test_capacidade_de_saida_estruturada_e_obrigatoria():
    registry = ProviderRegistry()
    registry.register_provider("fake", FakeProvider())
    registry.register_model(ModelRegistration("fake", "texto", "modelo"))
    request = LLMRequest(
        task_type="scientific_reasoning",
        messages=[{"role": "user", "content": "teste"}],
        structured_output={"type": "object"},
    )
    with pytest.raises(ProviderError):
        ProviderGateway(registry).execute(
            request, provider="fake", model_alias="texto"
        )


def test_registry_status_nao_expoe_chaves():
    registry = ProviderRegistry()
    registry.register_provider("fake", FakeProvider())
    registry.register_model(ModelRegistration("fake", "luna", "modelo"))
    payload = registry.status()
    assert payload["providers"]["fake"] == {"configured": True}
    assert "api_key" not in str(payload).lower()


def test_gateway_oferece_registro_publico_e_rejeita_provedor_sem_chave():
    gateway = ProviderGateway()
    gateway.register_provider("fake", UnconfiguredProvider())
    gateway.register_model(ModelRegistration("fake", "luna", "modelo"))
    with pytest.raises(ProviderNotConfiguredError):
        gateway.execute(_request(), provider="fake", model_alias="luna")


def test_registry_rejeita_identidades_invalidas_e_modelo_desabilitado():
    registry = ProviderRegistry()
    with pytest.raises(ValueError):
        registry.register_provider("", FakeProvider())
    with pytest.raises(ValueError):
        registry.register_model(ModelRegistration("ausente", "luna", "modelo"))
    with pytest.raises(ProviderError):
        registry.provider("ausente")
    registry.register_provider("fake", FakeProvider())
    with pytest.raises(ProviderError):
        registry.model("fake", "ausente")
    registry.register_model(
        ModelRegistration("fake", "desligado", None, status=ModelStatus.DISABLED)
    )
    with pytest.raises(ProviderError):
        registry.model("fake", "desligado")


def test_contexto_e_papeis_sao_normalizados_sem_sdk():
    class SystemMessage:
        def __init__(self, content):
            self.content = content

    class AIMessage:
        def __init__(self, content):
            self.content = content

    request = LLMRequest(
        task_type="scientific_reasoning",
        context={"evidencia": "E3"},
        messages=[SystemMessage("regra"), AIMessage("resposta")],
    )
    normalized = normalized_messages(request)
    assert normalized[0]["role"] == "developer"
    assert '"evidencia": "E3"' in normalized[0]["content"]
    assert [item["role"] for item in normalized[1:]] == ["developer", "assistant"]


@pytest.mark.parametrize(
    ("message", "transient", "unavailable"),
    [
        ("429 rate limit", True, False),
        ("503 unavailable", True, False),
        ("404 model not found", False, True),
        ("erro permanente", False, False),
    ],
)
def test_erros_de_provedor_sao_classificados(message, transient, unavailable):
    error = classify_provider_exception(RuntimeError(message), "fake")
    assert error.transient is transient
    assert error.unavailable is unavailable


def test_gateway_padrao_mapeia_aliases_sem_expor_chaves(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "valor-local-nao-versionado")
    monkeypatch.setenv("AL_IADO_OPENAI_MODEL_LUNA", "modelo-luna")
    monkeypatch.setenv("AL_IADO_GEMINI_MODEL_FLASH", "modelo-flash")
    status = build_default_gateway().status()
    aliases = {(item["provider"], item["alias"]): item for item in status["models"]}
    assert aliases[("openai", "luna")]["model_id"] == "modelo-luna"
    assert aliases[("openai", "terra")]["status"] == "disabled"
    assert aliases[("google", "flash")]["model_id"] == "modelo-flash"
    assert "valor-local" not in str(status)
