"""Tipos de infraestrutura compartilhados pelos provedores de LLM."""

from __future__ import annotations

import json
from typing import Any, Iterator, Protocol

from src.conhecimento.contratos_llm import LLMRequest, LLMResult, LLMStreamChunk


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        transient: bool = False,
        unavailable: bool = False,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.transient = bool(transient)
        self.unavailable = bool(unavailable)
        self.status_code = status_code


class ProviderNotConfiguredError(ProviderError):
    def __init__(self, provider: str):
        super().__init__(
            f"Provedor {provider} não configurado",
            unavailable=True,
        )


class LLMProvider(Protocol):
    name: str

    @property
    def configured(self) -> bool: ...

    def generate(self, request: LLMRequest, *, model_id: str) -> LLMResult: ...

    def stream(
        self, request: LLMRequest, *, model_id: str
    ) -> Iterator[LLMStreamChunk]: ...


def message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = str(message.get("role") or "user")
    else:
        role = str(getattr(message, "role", "") or "")
        if not role:
            name = type(message).__name__.lower()
            role = "assistant" if "ai" in name else "system" if "system" in name else "user"
    return "developer" if role == "system" else role


def message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", message)


def normalized_messages(request: LLMRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.context is not None:
        context = request.context
        if not isinstance(context, str):
            context = json.dumps(context, ensure_ascii=False, default=str)
        messages.append({"role": "developer", "content": context})
    messages.extend(
        {"role": message_role(item), "content": message_content(item)}
        for item in request.messages
    )
    return messages


def classify_provider_exception(exc: Exception, provider: str) -> ProviderError:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    transient = status in {408, 409, 429} or (status is not None and status >= 500)
    transient = transient or any(
        token in text
        for token in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "temporar",
            "unavailable",
            "high demand",
        )
    )
    unavailable = status in {401, 403, 404} or any(
        token in text for token in ("not configured", "not found", "no longer available")
    )
    return ProviderError(
        f"Falha no provedor {provider}: {exc}",
        transient=transient,
        unavailable=unavailable,
        status_code=status,
    )


__all__ = [
    "LLMProvider",
    "ProviderError",
    "ProviderNotConfiguredError",
    "classify_provider_exception",
    "message_content",
    "message_role",
    "normalized_messages",
]
