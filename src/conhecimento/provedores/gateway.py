"""Provider Gateway: uma entrada única para adapters de LLM."""

from __future__ import annotations

import os
from dataclasses import replace
from time import perf_counter
from typing import Iterator

from src.conhecimento.contratos_llm import LLMRequest, LLMResult, LLMStreamChunk
from src.conhecimento.provedores.gemini import (
    MODELO_GEMINI_FUNDO,
    PROVEDORES,
    GeminiProvider,
)
from src.conhecimento.provedores.openai import OpenAIProvider
from src.conhecimento.provedores.registry import (
    ModelRegistration,
    ModelStatus,
    ProviderRegistry,
)


class ProviderGateway:
    def __init__(self, registry: ProviderRegistry | None = None):
        self.registry = registry or ProviderRegistry()

    def register_provider(self, name: str, provider) -> None:
        self.registry.register_provider(name, provider)

    def register_model(self, registration: ModelRegistration) -> None:
        self.registry.register_model(registration)

    def execute(
        self,
        request: LLMRequest,
        *,
        provider: str,
        model_alias: str,
        allow_experimental: bool = False,
    ) -> LLMResult:
        capabilities = {"multimodal"} if request.multimodal else {"text"}
        registration = self.registry.model(
            provider,
            model_alias,
            allow_experimental=allow_experimental,
            capabilities=capabilities,
        )
        adapter = self.registry.provider(provider)
        if not adapter.configured:
            from src.conhecimento.provedores.base import ProviderNotConfiguredError

            raise ProviderNotConfiguredError(provider)
        started = perf_counter()
        result = adapter.generate(request, model_id=str(registration.model_id))
        latency = (perf_counter() - started) * 1000.0
        return replace(result, latency_ms=latency)

    def stream(
        self,
        request: LLMRequest,
        *,
        provider: str,
        model_alias: str,
        allow_experimental: bool = False,
    ) -> Iterator[LLMStreamChunk]:
        capabilities = {"multimodal"} if request.multimodal else {"text"}
        registration = self.registry.model(
            provider,
            model_alias,
            allow_experimental=allow_experimental,
            capabilities=capabilities,
        )
        adapter = self.registry.provider(provider)
        yield from adapter.stream(request, model_id=str(registration.model_id))

    def status(self) -> dict:
        return self.registry.status()


def _model_id(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def build_default_gateway() -> ProviderGateway:
    gateway = ProviderGateway()
    gateway.register_provider("openai", OpenAIProvider())
    gateway.register_provider("google", GeminiProvider())

    openai_models = {
        "luna": _model_id("AL_IADO_OPENAI_MODEL_LUNA"),
        "terra": _model_id("AL_IADO_OPENAI_MODEL_TERRA"),
        "sol": _model_id("AL_IADO_OPENAI_MODEL_SOL"),
    }
    for alias, model_id in openai_models.items():
        gateway.register_model(
            ModelRegistration(
                provider="openai",
                alias=alias,
                model_id=model_id,
                status=ModelStatus.OPERATIONAL if model_id else ModelStatus.DISABLED,
                capabilities=frozenset({"text", "structured_output"}),
            )
        )

    flash_lite = _model_id(
        "AL_IADO_GEMINI_MODEL_FLASH_LITE",
        "AL_IADO_GEMINI_MODEL_FUNDO",
        "AL_IADO_GEMINI_MODEL_AUDITOR",
        default=MODELO_GEMINI_FUNDO,
    )
    flash = _model_id(
        "AL_IADO_GEMINI_MODEL_FLASH",
        "AL_IADO_GEMINI_MODEL",
        default=PROVEDORES["1"]["modelo"],
    )
    gateway.register_model(
        ModelRegistration(
            provider="google",
            alias="flash_lite",
            model_id=flash_lite,
            capabilities=frozenset({"text", "structured_output"}),
        )
    )
    gateway.register_model(
        ModelRegistration(
            provider="google",
            alias="flash",
            model_id=flash,
            capabilities=frozenset({"text", "structured_output", "multimodal"}),
        )
    )
    return gateway


__all__ = ["ProviderGateway", "build_default_gateway"]
