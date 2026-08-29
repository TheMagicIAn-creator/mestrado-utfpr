"""Contratos neutros para inferência por modelos de linguagem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskType(StrEnum):
    SIMPLE_CHAT = "simple_chat"
    FACTUAL_SHORT = "factual_short"
    DOCUMENT_EXTRACTION = "document_extraction"
    MULTIMODAL_ANALYSIS = "multimodal_analysis"
    SCIENTIFIC_REASONING = "scientific_reasoning"
    CRITICAL_REASONING = "critical_reasoning"
    EVIDENCE_AUDIT = "evidence_audit"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    DETERMINISTIC = "deterministic"


class MethodologicalRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class LLMRequest:
    """Pedido independente de SDK encaminhado ao Gateway/Router."""

    task_type: str
    messages: list[Any]
    context: Any | None = None
    tools: list[Any] | None = None
    structured_output: dict[str, Any] | None = None
    reasoning_level: str | None = None
    multimodal: bool = False
    methodological_risk: str = MethodologicalRisk.LOW
    max_cost: float | None = None
    max_latency: float | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.task_type).strip():
            raise ValueError("task_type não pode ser vazio")
        if not isinstance(self.messages, list) or not self.messages:
            raise ValueError("messages deve ser uma lista não vazia")
        if str(self.methodological_risk) not in set(MethodologicalRisk):
            raise ValueError("methodological_risk inválido")
        if self.max_cost is not None and float(self.max_cost) < 0:
            raise ValueError("max_cost não pode ser negativo")
        if self.max_latency is not None and float(self.max_latency) <= 0:
            raise ValueError("max_latency deve ser positivo")
        if self.max_output_tokens is not None and int(self.max_output_tokens) <= 0:
            raise ValueError("max_output_tokens deve ser positivo")


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResult:
    content: str
    provider: str
    model: str
    task_type: str
    latency_ms: float | None = None
    estimated_cost: float | None = None
    fallback_used: bool = False
    validation_used: bool = False
    structured_data: dict[str, Any] | None = None
    usage: LLMUsage | None = None
    attempts: int = 1
    route_reason: str | None = None
    request_id: str | None = None
    fallback_kind: str | None = None
    escalation_used: bool = False
    validation_status: str | None = None
    validation_conflict: bool = False
    reviewer_provider: str | None = None
    reviewer_model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content deve ser texto")
        if not self.provider or not self.model or not self.task_type:
            raise ValueError("provider, model e task_type são obrigatórios")


@dataclass(frozen=True)
class LLMStreamChunk:
    content: str
    provider: str
    model: str
    task_type: str
    request_id: str | None = None
    fallback_used: bool = False
    route_reason: str | None = None


def _partes_de_sequencia(valores):
    for valor in valores:
        yield from _partes_textuais(valor)


def _partes_de_mapeamento(valor: dict):
    tipo = str(valor.get("type", "")).lower()
    if tipo and tipo not in {"text", "output_text"}:
        return
    for chave in ("text", "content", "value"):
        if chave in valor:
            yield from _partes_textuais(valor[chave])
            return


def _partes_de_objeto(valor):
    texto = getattr(valor, "text", None)
    if texto is not None:
        yield from _partes_textuais(texto)
        return
    interno = getattr(valor, "content", None)
    if interno is not None and interno is not valor:
        yield from _partes_textuais(interno)


def _partes_textuais(valor):
    if valor is None:
        return
    if isinstance(valor, str):
        if valor:
            yield valor
        return
    if isinstance(valor, (list, tuple)):
        yield from _partes_de_sequencia(valor)
        return
    if isinstance(valor, dict):
        yield from _partes_de_mapeamento(valor)
        return
    yield from _partes_de_objeto(valor)


def texto_resultado_llm(resposta: Any) -> str:
    """Normaliza texto de resultados neutros, SDKs legados e blocos."""

    conteudo = resposta if isinstance(resposta, (str, list, tuple, dict)) else getattr(
        resposta, "content", resposta
    )
    return "".join(_partes_textuais(conteudo) or ())


__all__ = [
    "LLMRequest",
    "LLMResult",
    "LLMStreamChunk",
    "LLMUsage",
    "MethodologicalRisk",
    "TaskType",
    "texto_resultado_llm",
]
