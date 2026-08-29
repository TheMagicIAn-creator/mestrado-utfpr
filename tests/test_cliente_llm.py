from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.conhecimento.cliente_llm import RouterLLMFacade, build_default_client
from src.conhecimento.contratos_llm import (
    LLMResult,
    LLMStreamChunk,
    MethodologicalRisk,
    TaskType,
    texto_resultado_llm,
)


class FakeRouter:
    def __init__(self):
        self.requests = []
        self.gateway = SimpleNamespace(
            status=lambda: {
                "models": [
                    {"provider": "openai", "alias": "terra", "configured": True}
                ]
            }
        )

    def execute(self, request):
        self.requests.append(request)
        structured = {"valor": 7} if request.structured_output else None
        return LLMResult(
            content=json.dumps(structured) if structured else "resposta",
            provider="openai",
            model="modelo-terra",
            task_type=request.task_type,
            structured_data=structured,
            request_id="req-1",
            route_reason="technical_scientific_task",
        )

    def stream(self, request):
        self.requests.append(request)
        yield LLMStreamChunk(
            "res",
            "google",
            "modelo-flash",
            request.task_type,
            request_id="req-stream",
            fallback_used=True,
            route_reason="cross_provider",
        )
        yield LLMStreamChunk(
            "posta",
            "google",
            "modelo-flash",
            request.task_type,
            request_id="req-stream",
            fallback_used=True,
            route_reason="cross_provider",
        )


def test_fachada_normaliza_string_e_expoe_rota_segura():
    router = FakeRouter()
    client = RouterLLMFacade(router)
    result = client.invoke("Explique o resultado")
    assert result.content == "resposta"
    assert router.requests[0].messages == [
        {"role": "user", "content": "Explique o resultado"}
    ]
    assert client.provider_label == "openai/modelo-terra"
    assert client.route_status() == {
        "provider": "openai",
        "model": "modelo-terra",
        "task_type": TaskType.SCIENTIFIC_REASONING,
        "route_reason": "technical_scientific_task",
        "fallback_used": False,
        "fallback_kind": None,
        "validation_used": False,
        "validation_status": None,
        "request_id": "req-1",
    }


def test_fachada_detecta_bloco_multimodal_e_preserva_override_de_tarefa():
    router = FakeRouter()
    client = RouterLLMFacade(router, task_type=TaskType.FACTUAL_SHORT)
    client.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Leia"},
                    {"type": "image_url", "image_url": {"url": "data:image/png"}},
                ],
            }
        ],
        task_type=TaskType.MULTIMODAL_ANALYSIS,
    )
    request = router.requests[0]
    assert request.multimodal is True
    assert request.task_type == TaskType.MULTIMODAL_ANALYSIS


def test_invoke_json_usa_schema_e_retorna_objeto_estruturado():
    router = FakeRouter()
    client = RouterLLMFacade(router)
    schema = {"type": "object", "properties": {"valor": {"type": "integer"}}}
    payload = client.invoke_json(
        [{"content": "retorne JSON"}],
        schema=schema,
        max_tokens=200,
        task_type=TaskType.EVIDENCE_AUDIT,
        methodological_risk=MethodologicalRisk.HIGH,
    )
    assert payload == {"valor": 7}
    request = router.requests[0]
    assert request.structured_output is schema
    assert request.max_output_tokens == 200
    assert request.temperature == 0.0
    assert request.methodological_risk == MethodologicalRisk.HIGH


def test_invoke_json_remove_cerca_markdown_quando_adapter_nao_estrutura():
    class MarkdownRouter(FakeRouter):
        def execute(self, request):
            self.requests.append(request)
            return LLMResult(
                content='```json\n{"ok": true}\n```',
                provider="google",
                model="flash",
                task_type=request.task_type,
            )

    client = RouterLLMFacade(MarkdownRouter())
    assert client.invoke_json("JSON") == {"ok": True}


def test_stream_preserva_fragmentos_e_atualiza_status_dinamico():
    router = FakeRouter()
    client = RouterLLMFacade(router)
    chunks = list(client.stream(({"content": "pergunta"},)))
    assert texto_resultado_llm(chunks) == "resposta"
    assert client.route_status()["provider"] == "google"
    assert client.route_status()["fallback_used"] is True
    assert client.route_status()["request_id"] == "req-stream"


def test_with_task_compartilha_router_sem_compartilhar_estado():
    router = FakeRouter()
    base = RouterLLMFacade(router, task_type=TaskType.SCIENTIFIC_REASONING)
    memory = base.with_task(
        TaskType.MEMORY_CONSOLIDATION,
        methodological_risk=MethodologicalRisk.LOW,
    )
    assert memory.router is router
    assert memory.route_status()["task_type"] == TaskType.MEMORY_CONSOLIDATION
    assert base.route_status()["route_reason"] == "not_invoked"


def test_cliente_declara_configuracao_e_rejeita_mensagens_vazias():
    client = RouterLLMFacade(FakeRouter())
    assert client.configured is True
    assert client.provider_label == "roteamento automático"
    with pytest.raises(ValueError, match="vazia"):
        client.invoke([])
    with pytest.raises(ValueError, match="vazia"):
        client.invoke("  ")


def test_build_default_client_usa_fabrica_do_router(monkeypatch):
    router = FakeRouter()
    monkeypatch.setattr(
        "src.conhecimento.cliente_llm.build_default_router", lambda: router
    )
    client = build_default_client(task_type=TaskType.DOCUMENT_EXTRACTION)
    assert client.router is router
    assert client.task_type == TaskType.DOCUMENT_EXTRACTION


def test_processador_pdf_declara_extracao_documental_e_schema(monkeypatch):
    import src.conhecimento.processador_pdf as processor

    calls = []

    class Client:
        def invoke_json(self, mensagens, **kwargs):
            calls.append((mensagens, kwargs))
            return {"autor": "Torres", "titulo": "Estudo", "ano": "2026"}

    monkeypatch.setattr(processor, "_cliente_metadados", lambda: Client())
    result = processor._extrair_via_llm("texto acadêmico", "estudo.pdf")
    assert result["autor"] == "Torres"
    assert calls[0][1]["task_type"] == TaskType.DOCUMENT_EXTRACTION
    assert calls[0][1]["methodological_risk"] == MethodologicalRisk.LOW
    assert calls[0][1]["schema"]["required"] == ["autor", "titulo", "ano"]
