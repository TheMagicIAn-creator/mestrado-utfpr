"""Adapter OpenAI para o contrato neutro do ALIAdo."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from src.conhecimento.contratos_llm import (
    LLMRequest,
    LLMResult,
    LLMStreamChunk,
    LLMUsage,
)
from src.conhecimento.provedores.base import (
    ProviderNotConfiguredError,
    classify_provider_exception,
    normalized_messages,
)


def _response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return str(direct)
    parts: list[str] = []
    for output in getattr(response, "output", ()) or ():
        for content in getattr(output, "content", ()) or ():
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "".join(parts)


def _usage(response: Any) -> LLMUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if total is None and input_tokens is not None and output_tokens is not None:
        total = int(input_tokens) + int(output_tokens)
    return LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total)


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, *, client=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self.api_key or self._client is not None)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ProviderNotConfiguredError(self.name)
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ProviderNotConfiguredError(self.name) from exc
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    @staticmethod
    def _kwargs(request: LLMRequest, model_id: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model_id,
            "input": normalized_messages(request),
        }
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(request.max_output_tokens)
        if request.temperature is not None:
            kwargs["temperature"] = float(request.temperature)
        if request.reasoning_level:
            kwargs["reasoning"] = {"effort": request.reasoning_level}
        if request.tools:
            kwargs["tools"] = request.tools
        if request.structured_output:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "aliado_response",
                    "strict": True,
                    "schema": request.structured_output,
                }
            }
        return kwargs

    def generate(self, request: LLMRequest, *, model_id: str) -> LLMResult:
        try:
            response = self._get_client().responses.create(
                **self._kwargs(request, model_id)
            )
            content = _response_text(response)
            structured = json.loads(content) if request.structured_output and content else None
            if structured is not None and not isinstance(structured, dict):
                raise ValueError("A saída estruturada OpenAI deve ser um objeto JSON")
            return LLMResult(
                content=content,
                provider=self.name,
                model=model_id,
                task_type=request.task_type,
                structured_data=structured,
                usage=_usage(response),
            )
        except (ProviderNotConfiguredError, ValueError):
            raise
        except Exception as exc:
            raise classify_provider_exception(exc, self.name) from exc

    def stream(
        self, request: LLMRequest, *, model_id: str
    ) -> Iterator[LLMStreamChunk]:
        try:
            events = self._get_client().responses.create(
                **self._kwargs(request, model_id), stream=True
            )
            for event in events:
                if getattr(event, "type", "") != "response.output_text.delta":
                    continue
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    yield LLMStreamChunk(
                        content=delta,
                        provider=self.name,
                        model=model_id,
                        task_type=request.task_type,
                    )
        except ProviderNotConfiguredError:
            raise
        except Exception as exc:
            raise classify_provider_exception(exc, self.name) from exc


__all__ = ["OpenAIProvider"]
