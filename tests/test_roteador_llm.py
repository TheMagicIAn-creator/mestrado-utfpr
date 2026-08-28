from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from src.conhecimento.contratos_llm import (
    LLMRequest,
    LLMResult,
    LLMStreamChunk,
    MethodologicalRisk,
    TaskType,
)
from src.conhecimento.provedores import (
    ModelRegistration,
    ModelStatus,
    ProviderGateway,
    ProviderRegistry,
)
from src.conhecimento.provedores.base import ProviderError
from src.conhecimento.provedores.gemini import GeminiProvider
from src.conhecimento.roteador_llm import (
    DeterministicTaskError,
    FallbackKind,
    LLMRouter,
    RouterStreamInterruptedError,
    UnsupportedTaskError,
    ValidationStatus,
)


class ScriptedProvider:
    configured = True

    def __init__(self, name, scripts=None, stream_scripts=None):
        self.name = name
        self.scripts = {key: list(value) for key, value in (scripts or {}).items()}
        self.stream_scripts = {
            key: list(value) for key, value in (stream_scripts or {}).items()
        }
        self.calls = []
        self.stream_calls = []

    @staticmethod
    def _take(scripts, model_id, default):
        queue = scripts.get(model_id)
        return queue.pop(0) if queue else default

    def generate(self, request, *, model_id):
        self.calls.append((model_id, request))
        item = self._take(self.scripts, model_id, f"ok:{model_id}")
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, dict):
            return LLMResult(
                content=json.dumps(item),
                provider=self.name,
                model=model_id,
                task_type=request.task_type,
                structured_data=item,
            )
        return LLMResult(
            content=str(item),
            provider=self.name,
            model=model_id,
            task_type=request.task_type,
        )

    def stream(self, request, *, model_id):
        self.stream_calls.append((model_id, request))
        item = self._take(self.stream_scripts, model_id, [f"ok:{model_id}"])
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, tuple) and item[0] == "partial_error":
            yield LLMStreamChunk("parcial", self.name, model_id, request.task_type)
            raise item[1]
        chunks = item if isinstance(item, list) else [str(item)]
        for chunk in chunks:
            yield LLMStreamChunk(str(chunk), self.name, model_id, request.task_type)


def _gateway(
    *,
    openai_scripts=None,
    google_scripts=None,
    openai_stream_scripts=None,
    statuses=None,
):
    registry = ProviderRegistry()
    openai = ScriptedProvider(
        "openai", openai_scripts, stream_scripts=openai_stream_scripts
    )
    google = ScriptedProvider("google", google_scripts)
    registry.register_provider("openai", openai)
    registry.register_provider("google", google)
    statuses = statuses or {}
    for alias in ("luna", "terra", "sol"):
        registry.register_model(
            ModelRegistration(
                "openai",
                alias,
                alias,
                status=statuses.get(("openai", alias), ModelStatus.OPERATIONAL),
                capabilities=frozenset({"text", "structured_output"}),
            )
        )
    registry.register_model(
        ModelRegistration(
            "google",
            "flash_lite",
            "flash_lite",
            status=statuses.get(
                ("google", "flash_lite"), ModelStatus.OPERATIONAL
            ),
            capabilities=frozenset({"text", "structured_output"}),
        )
    )
    registry.register_model(
        ModelRegistration(
            "google",
            "flash",
            "flash",
            status=statuses.get(("google", "flash"), ModelStatus.OPERATIONAL),
            capabilities=frozenset({"text", "structured_output", "multimodal"}),
        )
    )
    return ProviderGateway(registry), openai, google


def _request(task_type=TaskType.SIMPLE_CHAT, **kwargs):
    return LLMRequest(
        task_type=task_type,
        messages=[{"role": "user", "content": "conteúdo de teste"}],
        **kwargs,
    )


@pytest.mark.parametrize(
    ("task_type", "provider", "model"),
    [
        (TaskType.SIMPLE_CHAT, "openai", "luna"),
        (TaskType.FACTUAL_SHORT, "openai", "luna"),
        (TaskType.DOCUMENT_EXTRACTION, "google", "flash_lite"),
        (TaskType.MEMORY_CONSOLIDATION, "google", "flash_lite"),
        (TaskType.MULTIMODAL_ANALYSIS, "google", "flash"),
        (TaskType.SCIENTIFIC_REASONING, "openai", "terra"),
        (TaskType.EVIDENCE_AUDIT, "openai", "terra"),
        (TaskType.CRITICAL_REASONING, "openai", "sol"),
    ],
)
def test_politica_explica_rota_por_tipo_de_tarefa(task_type, provider, model):
    gateway, _, _ = _gateway()
    decision = LLMRouter(gateway, max_retries=0).decide(_request(task_type))
    assert decision.primary is not None
    assert (decision.primary.provider, decision.primary.model_alias) == (
        provider,
        model,
    )
    assert decision.reason


def test_multimodal_tem_precedencia_sobre_rotulo_textual():
    gateway, openai, google = _gateway()
    result = LLMRouter(gateway, max_retries=0).execute(
        _request(TaskType.SCIENTIFIC_REASONING, multimodal=True)
    )
    assert result.provider == "google"
    assert result.model == "flash"
    assert not openai.calls
    assert google.calls[0][0] == "flash"


def test_deterministico_fica_fora_do_llm_e_tarefa_desconhecida_e_rejeitada():
    gateway, openai, google = _gateway()
    router = LLMRouter(gateway, max_retries=0)
    decision = router.decide(_request(TaskType.DETERMINISTIC))
    assert decision.primary is None
    with pytest.raises(DeterministicTaskError):
        router.execute(_request(TaskType.DETERMINISTIC))
    with pytest.raises(UnsupportedTaskError):
        router.decide(_request("sem_politica"))
    assert not openai.calls and not google.calls


def test_retry_ocorre_somente_para_erro_transitorio():
    transient = ProviderError("indisponível", transient=True, status_code=503)
    gateway, openai, _ = _gateway(openai_scripts={"luna": [transient, "ok"]})
    waits = []
    result = LLMRouter(
        gateway,
        max_retries=1,
        backoff_seconds=0.25,
        sleeper=waits.append,
    ).execute(_request())
    assert result.content == "ok"
    assert result.attempts == 2
    assert result.fallback_used is False
    assert waits == [0.25]
    assert [model for model, _ in openai.calls] == ["luna", "luna"]


def test_erro_permanente_nao_e_mascarado_por_fallback():
    permanent = ProviderError("pedido inválido", status_code=400)
    gateway, openai, google = _gateway(openai_scripts={"luna": [permanent]})
    with pytest.raises(ProviderError, match="pedido inválido"):
        LLMRouter(gateway, max_retries=2, backoff_seconds=0).execute(_request())
    assert [model for model, _ in openai.calls] == ["luna"]
    assert not google.calls


def test_fallback_do_mesmo_provedor_e_identificado():
    unavailable = ProviderError("modelo ausente", unavailable=True)
    gateway, openai, _ = _gateway(
        openai_scripts={"luna": [unavailable], "terra": ["resposta terra"]}
    )
    result = LLMRouter(gateway, max_retries=0).execute(_request())
    assert result.model == "terra"
    assert result.fallback_used is True
    assert result.fallback_kind == FallbackKind.SAME_PROVIDER.value
    assert result.attempts == 2
    assert [model for model, _ in openai.calls] == ["luna", "terra"]


def test_fallback_cruzado_so_entra_depois_das_opcoes_do_provedor():
    unavailable = ProviderError("modelo ausente", unavailable=True)
    gateway, openai, google = _gateway(
        openai_scripts={
            "luna": [unavailable],
            "terra": [unavailable],
            "sol": [unavailable],
        },
        google_scripts={"flash_lite": ["resposta google"]},
    )
    result = LLMRouter(gateway, max_retries=0).execute(_request())
    assert result.provider == "google"
    assert result.model == "flash_lite"
    assert result.fallback_kind == FallbackKind.CROSS_PROVIDER.value
    assert result.attempts == 4
    assert [model for model, _ in openai.calls] == ["luna", "terra", "sol"]
    assert [model for model, _ in google.calls] == ["flash_lite"]


def test_modelo_experimental_e_pulado_no_caminho_operacional():
    gateway, openai, _ = _gateway(
        statuses={("openai", "luna"): ModelStatus.EXPERIMENTAL},
        openai_scripts={"terra": ["operacional"]},
    )
    result = LLMRouter(gateway, max_retries=0).execute(_request())
    assert result.model == "terra"
    assert result.fallback_kind == FallbackKind.SAME_PROVIDER.value
    assert [model for model, _ in openai.calls] == ["terra"]


def test_saida_estruturada_exige_capacidade_e_preserva_objeto():
    schema = {"type": "object", "properties": {"valor": {"type": "integer"}}}
    gateway, _, _ = _gateway(openai_scripts={"terra": [{"valor": 7}]})
    result = LLMRouter(gateway, max_retries=0).execute(
        _request(TaskType.SCIENTIFIC_REASONING, structured_output=schema)
    )
    assert result.structured_data == {"valor": 7}


def test_escalonamento_por_qualidade_nao_e_rotulado_como_indisponibilidade():
    gateway, openai, _ = _gateway(
        openai_scripts={"luna": ["rascunho"], "terra": ["resposta revisada"]}
    )
    result = LLMRouter(
        gateway,
        max_retries=0,
        quality_assessor=lambda request, response: False,
    ).execute(_request())
    assert result.content == "resposta revisada"
    assert result.model == "terra"
    assert result.escalation_used is True
    assert result.fallback_used is False
    assert "quality_escalation" in result.route_reason
    assert [model for model, _ in openai.calls] == ["luna", "terra"]


def test_validacao_cruzada_concordante_preserva_conclusao_terra():
    gateway, openai, google = _gateway(
        openai_scripts={"terra": ["conclusão primária"]},
        google_scripts={
            "flash": [{"verdict": "agree", "rationale": "evidência coerente"}]
        },
    )
    result = LLMRouter(gateway, max_retries=0).execute(
        _request(
            TaskType.SCIENTIFIC_REASONING,
            methodological_risk=MethodologicalRisk.HIGH,
        )
    )
    assert result.content == "conclusão primária"
    assert result.model == "terra"
    assert result.validation_used is True
    assert result.validation_status == ValidationStatus.AGREED.value
    assert result.validation_conflict is False
    assert result.reviewer_provider == "google"
    assert [model for model, _ in openai.calls] == ["terra"]
    assert [model for model, _ in google.calls] == ["flash"]


def test_conflito_metodologico_e_escalonado_para_sol():
    gateway, openai, _ = _gateway(
        openai_scripts={
            "terra": ["conclusão primária"],
            "sol": ["síntese crítica"],
        },
        google_scripts={
            "flash": [{"verdict": "conflict", "rationale": "premissa divergente"}]
        },
    )
    result = LLMRouter(gateway, max_retries=0).execute(
        _request(
            TaskType.EVIDENCE_AUDIT,
            methodological_risk=MethodologicalRisk.CRITICAL,
        )
    )
    assert result.content == "síntese crítica"
    assert result.model == "sol"
    assert result.escalation_used is True
    assert result.validation_conflict is True
    assert result.validation_status == ValidationStatus.RESOLVED_BY_SOL.value
    assert result.attempts == 3
    assert [model for model, _ in openai.calls] == ["terra", "sol"]


def test_conflito_nao_resolvido_e_exposto_sem_fabricar_concordancia():
    unavailable = ProviderError("sol indisponível", unavailable=True)
    gateway, _, _ = _gateway(
        openai_scripts={"terra": ["conclusão primária"], "sol": [unavailable]},
        google_scripts={
            "flash": [{"verdict": "uncertain", "rationale": "dados insuficientes"}]
        },
    )
    result = LLMRouter(gateway, max_retries=0).execute(
        _request(
            TaskType.SCIENTIFIC_REASONING,
            methodological_risk=MethodologicalRisk.HIGH,
        )
    )
    assert result.content == "conclusão primária"
    assert result.validation_conflict is True
    assert result.validation_status == ValidationStatus.CONFLICT_UNRESOLVED.value
    assert result.attempts == 3


def test_revisor_indisponivel_fica_visivel_no_resultado():
    unavailable = ProviderError("revisor indisponível", unavailable=True)
    gateway, _, _ = _gateway(
        openai_scripts={"terra": ["conclusão primária"]},
        google_scripts={"flash": [unavailable]},
    )
    result = LLMRouter(gateway, max_retries=0).execute(
        _request(
            TaskType.SCIENTIFIC_REASONING,
            methodological_risk=MethodologicalRisk.HIGH,
        )
    )
    assert result.validation_used is False
    assert result.validation_status == ValidationStatus.REVIEWER_UNAVAILABLE.value
    assert result.attempts == 2


def test_fallback_google_nao_e_validado_pelo_mesmo_provedor():
    unavailable = ProviderError("modelo indisponível", unavailable=True)
    gateway, _, google = _gateway(
        openai_scripts={"terra": [unavailable], "sol": [unavailable]},
        google_scripts={"flash": ["conclusão em fallback"]},
    )
    result = LLMRouter(gateway, max_retries=0).execute(
        _request(
            TaskType.SCIENTIFIC_REASONING,
            methodological_risk=MethodologicalRisk.HIGH,
        )
    )
    assert result.provider == "google"
    assert result.validation_used is False
    assert result.validation_status == ValidationStatus.REVIEWER_UNAVAILABLE.value
    assert "independent_reviewer_unavailable" in result.route_reason
    assert [model for model, _ in google.calls] == ["flash"]


def test_observabilidade_nao_carrega_prompt_resposta_ou_segredo():
    gateway, _, _ = _gateway(openai_scripts={"luna": ["resposta privada"]})
    observations = []
    request = LLMRequest(
        task_type=TaskType.SIMPLE_CHAT,
        messages=[{"role": "user", "content": "OPENAI_API_KEY=segredo-local"}],
    )
    result = LLMRouter(
        gateway,
        max_retries=0,
        observer=observations.append,
        request_id_factory=lambda: "req-auditavel",
    ).execute(request)
    serialized = json.dumps([asdict(item) for item in observations])
    assert result.request_id == "req-auditavel"
    assert "segredo-local" not in serialized
    assert "resposta privada" not in serialized
    assert {item.event for item in observations} >= {
        "route_selected",
        "route_completed",
    }


def test_stream_faz_fallback_somente_antes_do_primeiro_fragmento():
    unavailable = ProviderError("luna indisponível", unavailable=True)
    gateway, openai, _ = _gateway(
        openai_stream_scripts={
            "luna": [unavailable],
            "terra": [["o", "k"]],
        }
    )
    observations = []
    chunks = list(
        LLMRouter(
            gateway, max_retries=0, observer=observations.append
        ).stream(_request())
    )
    assert "".join(item.content for item in chunks) == "ok"
    assert all(item.fallback_used for item in chunks)
    assert [model for model, _ in openai.stream_calls] == ["luna", "terra"]
    completed = [item for item in observations if item.event == "route_completed"]
    assert len(completed) == 1
    assert completed[0].fallback == FallbackKind.SAME_PROVIDER.value
    assert completed[0].attempts == 2


def test_stream_interrompido_nao_duplica_conteudo_por_retry():
    transient = ProviderError("queda tardia", transient=True)
    gateway, openai, _ = _gateway(
        openai_stream_scripts={"luna": [("partial_error", transient)]}
    )
    with pytest.raises(RouterStreamInterruptedError):
        list(
            LLMRouter(gateway, max_retries=2, backoff_seconds=0).stream(_request())
        )
    assert [model for model, _ in openai.stream_calls] == ["luna"]


def test_configuracao_de_retry_rejeita_valores_negativos():
    gateway, _, _ = _gateway()
    with pytest.raises(ValueError, match="max_retries"):
        LLMRouter(gateway, max_retries=-1)
    with pytest.raises(ValueError, match="backoff_seconds"):
        LLMRouter(gateway, backoff_seconds=-0.1)


def test_adapter_gemini_delega_retry_ao_router_e_classifica_sdk():
    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs["model"])
            raise RuntimeError("503 high demand")

    class Client:
        def __init__(self):
            self.models = Models()

    client = Client()
    provider = GeminiProvider(api_key="local", client=client)
    with pytest.raises(ProviderError) as captured:
        provider.generate(_request(TaskType.DOCUMENT_EXTRACTION), model_id="flash")
    assert captured.value.transient is True
    assert client.models.calls == ["flash"]
