"""Roteamento explícito, resiliente e auditável para os provedores do ALIAdo."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from time import perf_counter, sleep
from typing import Any
from uuid import uuid4

from src.conhecimento.contratos_llm import (
    LLMRequest,
    LLMResult,
    LLMStreamChunk,
    MethodologicalRisk,
    TaskType,
)
from src.conhecimento.provedores.base import ProviderError
from src.conhecimento.provedores.gateway import ProviderGateway, build_default_gateway

_logger = logging.getLogger(__name__)


class FallbackKind(StrEnum):
    SAME_PROVIDER = "same_provider"
    CROSS_PROVIDER = "cross_provider"


class ValidationStatus(StrEnum):
    AGREED = "agreed"
    REVIEWER_UNAVAILABLE = "reviewer_unavailable"
    CONFLICT_UNRESOLVED = "conflict_unresolved"
    RESOLVED_BY_SOL = "resolved_by_sol"


class RouterError(RuntimeError):
    """Erro de política do Router, independente do SDK de inferência."""


class UnsupportedTaskError(RouterError):
    pass


class DeterministicTaskError(RouterError):
    pass


class RouterStreamInterruptedError(RouterError):
    pass


@dataclass(frozen=True)
class RouteTarget:
    provider: str
    model_alias: str


@dataclass(frozen=True)
class RouteDecision:
    task_type: TaskType
    reason: str
    primary: RouteTarget | None
    same_provider_fallbacks: tuple[RouteTarget, ...] = ()
    cross_provider_fallbacks: tuple[RouteTarget, ...] = ()
    quality_escalations: tuple[RouteTarget, ...] = ()
    validation_reviewer: RouteTarget | None = None


@dataclass(frozen=True)
class RoutingObservation:
    """Evento sem prompts, respostas, anexos ou credenciais."""

    request_id: str
    event: str
    task_type: str
    provider: str | None = None
    model: str | None = None
    route_reason: str | None = None
    fallback: str | None = None
    latency_ms: float | None = None
    estimated_cost: float | None = None
    validation_used: bool = False
    error: str | None = None
    attempts: int = 0


class _TargetFailure(Exception):
    def __init__(self, error: ProviderError, attempts: int):
        super().__init__(type(error).__name__)
        self.error = error
        self.attempts = attempts


@dataclass(frozen=True)
class _Candidate:
    target: RouteTarget
    mode: str


_OPENAI_LUNA = RouteTarget("openai", "luna")
_OPENAI_TERRA = RouteTarget("openai", "terra")
_OPENAI_SOL = RouteTarget("openai", "sol")
_GOOGLE_FLASH_LITE = RouteTarget("google", "flash_lite")
_GOOGLE_FLASH = RouteTarget("google", "flash")

_METHODOLOGICAL_TASKS = {
    TaskType.SCIENTIFIC_REASONING,
    TaskType.CRITICAL_REASONING,
    TaskType.EVIDENCE_AUDIT,
}
_HIGH_RISKS = {MethodologicalRisk.HIGH, MethodologicalRisk.CRITICAL}

_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["agree", "conflict", "uncertain"],
        },
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}


def _standard_decision(task_type: TaskType) -> RouteDecision:
    if task_type in {TaskType.SIMPLE_CHAT, TaskType.FACTUAL_SHORT}:
        return RouteDecision(
            task_type,
            "short_low_latency_task",
            _OPENAI_LUNA,
            (_OPENAI_TERRA, _OPENAI_SOL),
            (_GOOGLE_FLASH_LITE, _GOOGLE_FLASH),
            (_OPENAI_TERRA, _OPENAI_SOL),
        )
    if task_type in {TaskType.DOCUMENT_EXTRACTION, TaskType.MEMORY_CONSOLIDATION}:
        return RouteDecision(
            task_type,
            "high_volume_document_task",
            _GOOGLE_FLASH_LITE,
            (_GOOGLE_FLASH,),
            (_OPENAI_LUNA, _OPENAI_TERRA),
            (_GOOGLE_FLASH,),
        )
    if task_type == TaskType.MULTIMODAL_ANALYSIS:
        return RouteDecision(
            task_type,
            "multimodal_requirement",
            _GOOGLE_FLASH,
            (),
            (_OPENAI_SOL,),
        )
    if task_type in {TaskType.SCIENTIFIC_REASONING, TaskType.EVIDENCE_AUDIT}:
        return RouteDecision(
            task_type,
            "technical_scientific_task",
            _OPENAI_TERRA,
            (_OPENAI_SOL,),
            (_GOOGLE_FLASH,),
            (_OPENAI_SOL,),
        )
    if task_type == TaskType.CRITICAL_REASONING:
        return RouteDecision(
            task_type,
            "exceptional_critical_reasoning",
            _OPENAI_SOL,
            (),
            (_GOOGLE_FLASH,),
        )
    if task_type == TaskType.DETERMINISTIC:
        return RouteDecision(task_type, "deterministic_local_execution", None)
    raise UnsupportedTaskError(f"Tipo de tarefa sem política: {task_type}")


class LLMRouter:
    """Seleciona recursos de inferência sem acoplar regra de negócio ao SDK."""

    def __init__(
        self,
        gateway: ProviderGateway | None = None,
        *,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
        sleeper: Callable[[float], None] = sleep,
        observer: Callable[[RoutingObservation], None] | None = None,
        quality_assessor: Callable[[LLMRequest, LLMResult], bool] | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.gateway = gateway or build_default_gateway()
        self.max_retries = (
            int(os.getenv("AL_IADO_LLM_MAX_RETRIES", "2"))
            if max_retries is None
            else int(max_retries)
        )
        self.backoff_seconds = (
            float(os.getenv("AL_IADO_LLM_BACKOFF_S", "0.5"))
            if backoff_seconds is None
            else float(backoff_seconds)
        )
        if self.max_retries < 0:
            raise ValueError("max_retries não pode ser negativo")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds não pode ser negativo")
        self._sleep = sleeper
        self._observer = observer
        self._quality_assessor = quality_assessor
        self._request_id_factory = request_id_factory or (lambda: uuid4().hex)

    def decide(self, request: LLMRequest) -> RouteDecision:
        try:
            task_type = TaskType(str(request.task_type))
        except ValueError as exc:
            raise UnsupportedTaskError(
                f"Tipo de tarefa sem política: {request.task_type}"
            ) from exc

        if request.multimodal:
            task_type = TaskType.MULTIMODAL_ANALYSIS
        decision = _standard_decision(task_type)
        risk = MethodologicalRisk(str(request.methodological_risk))
        if task_type in _METHODOLOGICAL_TASKS and risk in _HIGH_RISKS:
            return RouteDecision(
                task_type,
                "high_methodological_risk_cross_validation",
                _OPENAI_TERRA,
                (_OPENAI_SOL,),
                (_GOOGLE_FLASH,),
                (_OPENAI_SOL,),
                _GOOGLE_FLASH,
            )
        return decision

    def execute(self, request: LLMRequest) -> LLMResult:
        request_id = self._request_id_factory()
        started = perf_counter()
        decision = self.decide(request)
        self._emit(
            RoutingObservation(
                request_id,
                "route_selected",
                decision.task_type.value,
                provider=decision.primary.provider if decision.primary else None,
                model=decision.primary.model_alias if decision.primary else None,
                route_reason=decision.reason,
            )
        )
        if decision.primary is None:
            self._emit(
                RoutingObservation(
                    request_id,
                    "local_execution_required",
                    decision.task_type.value,
                    route_reason=decision.reason,
                )
            )
            raise DeterministicTaskError(
                "Tarefas determinísticas devem ser executadas localmente, sem LLM"
            )

        result = self._execute_decision(request, decision, request_id)
        if self._quality_assessor is not None and not self._quality_assessor(
            request, result
        ):
            result = self._escalate_quality(request, decision, result, request_id)
        if decision.validation_reviewer is not None:
            result = self._cross_validate(request, decision, result, request_id)

        result = replace(result, latency_ms=(perf_counter() - started) * 1000.0)
        self._emit(
            RoutingObservation(
                request_id,
                "route_completed",
                decision.task_type.value,
                provider=result.provider,
                model=result.model,
                route_reason=result.route_reason,
                fallback=result.fallback_kind,
                latency_ms=result.latency_ms,
                estimated_cost=result.estimated_cost,
                validation_used=result.validation_used,
                attempts=result.attempts,
            )
        )
        return result

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        decision = self.decide(request)
        if decision.primary is None:
            raise DeterministicTaskError(
                "Tarefas determinísticas devem ser executadas localmente, sem LLM"
            )
        if decision.validation_reviewer is not None or self._quality_assessor is not None:
            result = self.execute(request)
            yield LLMStreamChunk(
                result.content,
                result.provider,
                result.model,
                result.task_type,
                request_id=result.request_id,
                fallback_used=result.fallback_used,
                route_reason=result.route_reason,
            )
            return

        request_id = self._request_id_factory()
        started = perf_counter()
        self._emit(
            RoutingObservation(
                request_id,
                "route_selected",
                decision.task_type.value,
                provider=decision.primary.provider,
                model=decision.primary.model_alias,
                route_reason=decision.reason,
            )
        )
        candidates = self._candidates(decision)
        last_error: ProviderError | None = None
        total_attempts = 0
        for candidate in candidates:
            fallback_kind = self._fallback_kind(candidate.mode)
            for retry_index in range(self.max_retries + 1):
                total_attempts += 1
                emitted = False
                try:
                    for chunk in self.gateway.stream(
                        request,
                        provider=candidate.target.provider,
                        model_alias=candidate.target.model_alias,
                    ):
                        emitted = True
                        yield replace(
                            chunk,
                            request_id=request_id,
                            fallback_used=fallback_kind is not None,
                            route_reason=self._result_reason(decision, candidate.mode),
                        )
                    self._emit(
                        RoutingObservation(
                            request_id,
                            "route_completed",
                            decision.task_type.value,
                            provider=candidate.target.provider,
                            model=candidate.target.model_alias,
                            route_reason=self._result_reason(
                                decision, candidate.mode
                            ),
                            fallback=fallback_kind,
                            latency_ms=(perf_counter() - started) * 1000.0,
                            attempts=total_attempts,
                        )
                    )
                    return
                except ProviderError as exc:
                    last_error = exc
                    self._emit_failure(
                        request_id,
                        decision,
                        candidate,
                        exc,
                        retry_index + 1,
                    )
                    if emitted:
                        raise RouterStreamInterruptedError(
                            "O stream foi interrompido após o primeiro fragmento; "
                            "a resposta não será duplicada por retry"
                        ) from exc
                    if exc.transient and retry_index < self.max_retries:
                        self._sleep(self.backoff_seconds * (2**retry_index))
                        continue
                    if exc.transient or exc.unavailable:
                        break
                    raise
        if last_error is not None:
            raise last_error
        raise RouterError("Nenhuma rota de streaming disponível")

    def _execute_decision(
        self,
        request: LLMRequest,
        decision: RouteDecision,
        request_id: str,
    ) -> LLMResult:
        result, _ = self._execute_candidates(
            request,
            decision,
            self._candidates(decision),
            request_id,
        )
        return result

    @staticmethod
    def _candidates(decision: RouteDecision) -> tuple[_Candidate, ...]:
        if decision.primary is None:
            return ()
        return (
            _Candidate(decision.primary, "primary"),
            *(
                _Candidate(target, FallbackKind.SAME_PROVIDER.value)
                for target in decision.same_provider_fallbacks
            ),
            *(
                _Candidate(target, FallbackKind.CROSS_PROVIDER.value)
                for target in decision.cross_provider_fallbacks
            ),
        )

    def _execute_candidates(
        self,
        request: LLMRequest,
        decision: RouteDecision,
        candidates: tuple[_Candidate, ...],
        request_id: str,
        *,
        initial_attempts: int = 0,
    ) -> tuple[LLMResult, int]:
        total_attempts = initial_attempts
        last_error: ProviderError | None = None
        for candidate in candidates:
            if candidate.mode != "primary":
                self._emit(
                    RoutingObservation(
                        request_id,
                        "alternate_route_selected",
                        decision.task_type.value,
                        provider=candidate.target.provider,
                        model=candidate.target.model_alias,
                        route_reason=decision.reason,
                        fallback=self._fallback_kind(candidate.mode),
                        attempts=total_attempts,
                    )
                )
            try:
                raw, used = self._invoke_target(
                    request,
                    decision,
                    candidate,
                    request_id,
                )
            except _TargetFailure as failure:
                total_attempts += failure.attempts
                last_error = failure.error
                if failure.error.transient or failure.error.unavailable:
                    continue
                raise failure.error
            total_attempts += used
            fallback_kind = self._fallback_kind(candidate.mode)
            result = replace(
                raw,
                attempts=total_attempts,
                request_id=request_id,
                fallback_used=fallback_kind is not None,
                fallback_kind=fallback_kind,
                escalation_used=candidate.mode == "quality_escalation",
                route_reason=self._result_reason(decision, candidate.mode),
            )
            return result, total_attempts
        if last_error is not None:
            raise last_error
        raise RouterError("Nenhum modelo elegível para a rota")

    def _invoke_target(
        self,
        request: LLMRequest,
        decision: RouteDecision,
        candidate: _Candidate,
        request_id: str,
    ) -> tuple[LLMResult, int]:
        for retry_index in range(self.max_retries + 1):
            try:
                result = self.gateway.execute(
                    request,
                    provider=candidate.target.provider,
                    model_alias=candidate.target.model_alias,
                )
                return result, retry_index + 1
            except ProviderError as exc:
                self._emit_failure(
                    request_id,
                    decision,
                    candidate,
                    exc,
                    retry_index + 1,
                )
                if exc.transient and retry_index < self.max_retries:
                    self._emit(
                        RoutingObservation(
                            request_id,
                            "retry_scheduled",
                            decision.task_type.value,
                            provider=candidate.target.provider,
                            model=candidate.target.model_alias,
                            route_reason=decision.reason,
                            fallback=self._fallback_kind(candidate.mode),
                            attempts=retry_index + 1,
                        )
                    )
                    self._sleep(self.backoff_seconds * (2**retry_index))
                    continue
                raise _TargetFailure(exc, retry_index + 1) from exc
        raise AssertionError("loop de retry terminou sem resultado")

    def _escalate_quality(
        self,
        request: LLMRequest,
        decision: RouteDecision,
        original: LLMResult,
        request_id: str,
    ) -> LLMResult:
        if not decision.quality_escalations:
            return replace(
                original,
                route_reason=f"{decision.reason}:quality_escalation_unavailable",
            )
        candidates = tuple(
            _Candidate(target, "quality_escalation")
            for target in decision.quality_escalations
        )
        try:
            escalated, _ = self._execute_candidates(
                request,
                decision,
                candidates,
                request_id,
                initial_attempts=original.attempts,
            )
        except ProviderError:
            return replace(
                original,
                route_reason=f"{decision.reason}:quality_escalation_unavailable",
            )
        return replace(escalated, escalation_used=True, fallback_used=False)

    def _cross_validate(
        self,
        request: LLMRequest,
        decision: RouteDecision,
        primary: LLMResult,
        request_id: str,
    ) -> LLMResult:
        reviewer = decision.validation_reviewer
        if reviewer is None:
            return primary
        if primary.provider == reviewer.provider:
            return replace(
                primary,
                validation_status=ValidationStatus.REVIEWER_UNAVAILABLE.value,
                route_reason=(
                    f"{primary.route_reason}:independent_reviewer_unavailable"
                ),
            )
        review_request = self._review_request(request, primary)
        candidate = _Candidate(reviewer, "validation_reviewer")
        try:
            review, used = self._invoke_target(
                review_request,
                decision,
                candidate,
                request_id,
            )
        except _TargetFailure as failure:
            return replace(
                primary,
                attempts=primary.attempts + failure.attempts,
                validation_status=ValidationStatus.REVIEWER_UNAVAILABLE.value,
            )

        attempts = primary.attempts + used
        verdict = self._review_verdict(review)
        self._emit(
            RoutingObservation(
                request_id,
                "cross_validation_completed",
                decision.task_type.value,
                provider=review.provider,
                model=review.model,
                route_reason=decision.reason,
                validation_used=True,
                attempts=attempts,
            )
        )
        if verdict == "agree":
            return replace(
                primary,
                attempts=attempts,
                validation_used=True,
                validation_status=ValidationStatus.AGREED.value,
                reviewer_provider=review.provider,
                reviewer_model=review.model,
            )

        resolution = _Candidate(_OPENAI_SOL, "validation_resolution")
        try:
            resolved, used = self._invoke_target(
                request,
                decision,
                resolution,
                request_id,
            )
        except _TargetFailure as failure:
            return replace(
                primary,
                attempts=attempts + failure.attempts,
                validation_used=True,
                validation_status=ValidationStatus.CONFLICT_UNRESOLVED.value,
                validation_conflict=True,
                reviewer_provider=review.provider,
                reviewer_model=review.model,
            )
        return replace(
            resolved,
            attempts=attempts + used,
            request_id=request_id,
            route_reason="validation_conflict_escalated_to_sol",
            escalation_used=True,
            validation_used=True,
            validation_status=ValidationStatus.RESOLVED_BY_SOL.value,
            validation_conflict=True,
            reviewer_provider=review.provider,
            reviewer_model=review.model,
        )

    @staticmethod
    def _review_request(request: LLMRequest, primary: LLMResult) -> LLMRequest:
        payload = json.dumps(
            {
                "pedido_original": request.messages,
                "conclusao_candidata": primary.content,
            },
            ensure_ascii=False,
            default=str,
        )
        return LLMRequest(
            task_type=TaskType.EVIDENCE_AUDIT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Atue como revisor metodológico independente. Compare o pedido e a "
                        "conclusão abaixo. Classifique como agree, conflict ou uncertain e "
                        f"justifique sem reescrever a resposta.\n\n{payload}"
                    ),
                }
            ],
            structured_output=_REVIEW_SCHEMA,
            methodological_risk=MethodologicalRisk.LOW,
            max_output_tokens=min(request.max_output_tokens or 1200, 2000),
            temperature=0.0,
        )

    @staticmethod
    def _review_verdict(review: LLMResult) -> str:
        data = review.structured_data
        if data is None and review.content:
            try:
                parsed = json.loads(review.content)
                data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                data = None
        verdict = str((data or {}).get("verdict", "uncertain")).strip().lower()
        return verdict if verdict in {"agree", "conflict"} else "uncertain"

    def _emit_failure(
        self,
        request_id: str,
        decision: RouteDecision,
        candidate: _Candidate,
        error: ProviderError,
        attempts: int,
    ) -> None:
        attributes = []
        if error.transient:
            attributes.append("transient")
        if error.unavailable:
            attributes.append("unavailable")
        if error.status_code is not None:
            attributes.append(f"status_{error.status_code}")
        suffix = ":" + ",".join(attributes) if attributes else ""
        self._emit(
            RoutingObservation(
                request_id,
                "route_attempt_failed",
                decision.task_type.value,
                provider=candidate.target.provider,
                model=candidate.target.model_alias,
                route_reason=decision.reason,
                fallback=self._fallback_kind(candidate.mode),
                error=f"{type(error).__name__}{suffix}",
                attempts=attempts,
            )
        )

    def _emit(self, observation: RoutingObservation) -> None:
        _logger.info(
            "llm_route %s",
            json.dumps(asdict(observation), ensure_ascii=True, sort_keys=True),
        )
        if self._observer is None:
            return
        try:
            self._observer(observation)
        except Exception:
            _logger.warning("observador de roteamento falhou; inferência preservada")

    @staticmethod
    def _fallback_kind(mode: str) -> str | None:
        return mode if mode in set(FallbackKind) else None

    @staticmethod
    def _result_reason(decision: RouteDecision, mode: str) -> str:
        return decision.reason if mode == "primary" else f"{decision.reason}:{mode}"


def build_default_router(**kwargs) -> LLMRouter:
    return LLMRouter(build_default_gateway(), **kwargs)


__all__ = [
    "DeterministicTaskError",
    "FallbackKind",
    "LLMRouter",
    "RouteDecision",
    "RouteTarget",
    "RouterError",
    "RouterStreamInterruptedError",
    "RoutingObservation",
    "UnsupportedTaskError",
    "ValidationStatus",
    "build_default_router",
]
