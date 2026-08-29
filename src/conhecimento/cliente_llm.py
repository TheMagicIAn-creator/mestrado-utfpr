"""Fachada compatível que entrega ao agente apenas o contrato do Router."""

from __future__ import annotations

import json
import threading
from typing import Any

from src.conhecimento.contratos_llm import (
    LLMRequest,
    LLMResult,
    MethodologicalRisk,
    TaskType,
    texto_resultado_llm,
)
from src.conhecimento.roteador_llm import LLMRouter, build_default_router


def _mensagens_normalizadas(mensagens: Any) -> list[Any]:
    if isinstance(mensagens, list):
        if not mensagens:
            raise ValueError("mensagens não pode ser vazia")
        return mensagens
    if isinstance(mensagens, tuple):
        if not mensagens:
            raise ValueError("mensagens não pode ser vazia")
        return list(mensagens)
    if isinstance(mensagens, str):
        if not mensagens.strip():
            raise ValueError("mensagem não pode ser vazia")
        return [{"role": "user", "content": mensagens}]
    return [mensagens]


def _tem_conteudo_multimodal(valor: Any) -> bool:
    if isinstance(valor, dict):
        tipo = str(valor.get("type", "")).lower()
        if tipo in {"image", "image_url", "input_image", "file", "input_file"}:
            return True
        return any(_tem_conteudo_multimodal(item) for item in valor.values())
    if isinstance(valor, (list, tuple)):
        return any(_tem_conteudo_multimodal(item) for item in valor)
    conteudo = getattr(valor, "content", None)
    return conteudo is not None and conteudo is not valor and _tem_conteudo_multimodal(
        conteudo
    )


class RouterLLMFacade:
    """Preserva `invoke`/`stream` enquanto todas as chamadas passam pelo Router."""

    name = "router"
    supports_multimodal = True

    def __init__(
        self,
        router: LLMRouter | None = None,
        *,
        task_type: str = TaskType.SCIENTIFIC_REASONING,
        methodological_risk: str = MethodologicalRisk.MEDIUM,
        reasoning_level: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.router = router or build_default_router()
        self.task_type = str(task_type)
        self.methodological_risk = str(methodological_risk)
        self.reasoning_level = reasoning_level
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._last_result: LLMResult | None = None
        self._last_stream: dict[str, Any] | None = None
        self._status_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        status = self.router.gateway.status()
        providers = status.get("providers")
        if isinstance(providers, dict):
            return any(
                item.get("configured")
                and providers.get(item.get("provider"), {}).get("configured")
                for item in status.get("models", [])
            )
        return any(item.get("configured") for item in status.get("models", []))

    @property
    def provider_label(self) -> str:
        status = self.route_status()
        if status["provider"] and status["model"]:
            return f"{status['provider']}/{status['model']}"
        return "roteamento automático"

    def route_status(self) -> dict[str, Any]:
        with self._status_lock:
            if self._last_result is not None:
                result = self._last_result
                return {
                    "provider": result.provider,
                    "model": result.model,
                    "task_type": result.task_type,
                    "route_reason": result.route_reason,
                    "fallback_used": result.fallback_used,
                    "fallback_kind": result.fallback_kind,
                    "validation_used": result.validation_used,
                    "validation_status": result.validation_status,
                    "request_id": result.request_id,
                }
            if self._last_stream is not None:
                return dict(self._last_stream)
        return {
            "provider": None,
            "model": None,
            "task_type": self.task_type,
            "route_reason": "not_invoked",
            "fallback_used": False,
            "fallback_kind": None,
            "validation_used": False,
            "validation_status": None,
            "request_id": None,
        }

    def _request(
        self,
        mensagens: Any,
        *,
        task_type: str | None = None,
        methodological_risk: str | None = None,
        structured_output: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMRequest:
        normalized = _mensagens_normalizadas(mensagens)
        return LLMRequest(
            task_type=task_type or self.task_type,
            messages=normalized,
            structured_output=structured_output,
            reasoning_level=self.reasoning_level,
            multimodal=_tem_conteudo_multimodal(normalized),
            methodological_risk=methodological_risk or self.methodological_risk,
            max_output_tokens=max_output_tokens or self.max_output_tokens,
            temperature=self.temperature if temperature is None else temperature,
        )

    def invoke(
        self,
        mensagens: Any,
        *,
        task_type: str | None = None,
        methodological_risk: str | None = None,
        structured_output: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        request = self._request(
            mensagens,
            task_type=task_type,
            methodological_risk=methodological_risk,
            structured_output=structured_output,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        result = self.router.execute(request)
        with self._status_lock:
            self._last_result = result
            self._last_stream = None
        return result

    def stream(
        self,
        mensagens: Any,
        *,
        task_type: str | None = None,
        methodological_risk: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ):
        request = self._request(
            mensagens,
            task_type=task_type,
            methodological_risk=methodological_risk,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        for chunk in self.router.stream(request):
            with self._status_lock:
                self._last_result = None
                self._last_stream = {
                    "provider": chunk.provider,
                    "model": chunk.model,
                    "task_type": chunk.task_type,
                    "route_reason": chunk.route_reason,
                    "fallback_used": chunk.fallback_used,
                    "fallback_kind": None,
                    "validation_used": False,
                    "validation_status": None,
                    "request_id": chunk.request_id,
                }
            yield chunk

    def invoke_json(
        self,
        mensagens: Any,
        *,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        task_type: str | None = None,
        methodological_risk: str | None = None,
    ) -> dict[str, Any]:
        result = self.invoke(
            mensagens,
            task_type=task_type,
            methodological_risk=methodological_risk,
            structured_output=schema,
            max_output_tokens=max_tokens,
            temperature=0.0,
        )
        if result.structured_data is not None:
            return result.structured_data
        raw = texto_resultado_llm(result).strip()
        if raw.startswith("```"):
            linhas = raw.splitlines()
            if linhas and linhas[0].strip().lower() in {"```", "```json"}:
                linhas = linhas[1:]
            if linhas and linhas[-1].strip() == "```":
                linhas = linhas[:-1]
            raw = "\n".join(linhas).strip()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("A saída estruturada deve ser um objeto JSON")
        return payload

    def with_task(
        self,
        task_type: str,
        *,
        methodological_risk: str | None = None,
    ) -> "RouterLLMFacade":
        return RouterLLMFacade(
            self.router,
            task_type=task_type,
            methodological_risk=methodological_risk or self.methodological_risk,
            reasoning_level=self.reasoning_level,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )


def build_default_client(
    *,
    task_type: str = TaskType.SCIENTIFIC_REASONING,
    methodological_risk: str = MethodologicalRisk.MEDIUM,
    **kwargs,
) -> RouterLLMFacade:
    return RouterLLMFacade(
        build_default_router(),
        task_type=task_type,
        methodological_risk=methodological_risk,
        **kwargs,
    )


__all__ = ["RouterLLMFacade", "build_default_client"]
