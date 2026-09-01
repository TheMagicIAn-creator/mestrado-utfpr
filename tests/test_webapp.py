from __future__ import annotations

import json
from io import BytesIO
from datetime import datetime
from pathlib import Path
from threading import Event, Thread, Timer
from time import monotonic
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter
from starlette.testclient import TestClient

from src.conhecimento.catalogo_bibliografico import salvar_catalogo
from src.webapp.agent_adapter import (
    MAX_ATTACHMENT_BYTES,
    AgentAdapter,
    _Componentes,
    _historico_normalizado,
    _validar_anexos,
)
from src.webapp.app import create_app
from src.webapp.contracts import (
    e3_contract,
    reliability_contract,
    sources_contract,
)
from src.webapp.rendering import render_agent_markdown
from src.webapp.library_service import LibraryService
from src.webapp.scientific_context import scientific_context_for
from src.webapp.session_journal import SessionJournal

ROOT = Path(__file__).resolve().parents[1]


def _sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
        if data:
            events.append((event, json.loads("\n".join(data))))
    return events


@pytest.fixture()
def calls():
    return []


@pytest.fixture()
def client(calls):
    def answer(message, history, attachments):
        calls.append((message, history, attachments))
        return {
            "answer": f"Resposta **verificada**: {message}",
            "images": [],
            "route": "test-double",
        }

    with TestClient(
        create_app(AgentAdapter(answerer=answer), warm_on_startup=False)
    ) as test_client:
        yield test_client


def test_homepage_e_canonicamente_chat_first(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>ALIAdo</title>" in response.text
    assert 'id="view-chat"' in response.text
    assert 'id="chat-form"' in response.text
    assert "/static/app.js" in response.text
    assert "/static/vendor/katex/katex.min.css" in response.text
    assert "/static/vendor/katex/katex.min.js" in response.text
    assert "/static/vendor/katex/auto-render.min.js" in response.text
    assert "plotly" not in response.text.casefold()
    assert "webapp_v2" not in response.text.casefold()
    assert "autoencoder v2" not in response.text.casefold()
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_e3_publica_apenas_denso_e_lstm_no_gpvs(client):
    response = client.get("/api/results/e3")
    assert response.status_code == 200
    data = response.json()
    assert data["contract_version"] == 3
    assert data["dataset"]["name"] == "GPVS-Faults"
    assert set(data["models"]) == {"ae_denso", "ae_lstm"}
    assert data["primary_metrics"] == ["recall", "f1", "precision"]
    assert data["complementary_metrics"] == ["auc_roc", "auc_pr"]
    assert data["metrics"]["ae_denso"]["auc_pr"]["estimate"] == pytest.approx(
        0.8618504150911235
    )
    assert data["metrics"]["ae_lstm"]["auc_pr"]["estimate"] == pytest.approx(
        0.8411642809389649
    )
    assert len(data["trials"]) == 28
    assert len(data["confusion_matrices"]) == 2
    assert set(data["discrimination"]["models"]) == {"ae_denso", "ae_lstm"}
    assert all(
        len(model["roc"]) <= 201
        and len(model["precision_recall"]) <= 201
        for model in data["discrimination"]["models"].values()
    )
    assert len(data["figures"]) == 4
    assert all(item["url"].startswith("/artifacts/comparison/") for item in data["figures"])


def test_endpoint_sintetico_e2_nao_e_mais_publicado(client):
    assert client.get("/api/results/e2").status_code == 404


def test_confiabilidade_publica_quatro_cenarios_fisicos_rastreaveis(client):
    response = client.get("/api/reliability")
    assert response.status_code == 200
    data = response.json()
    assert data["contract_version"] == 2
    assert data["evidence_scope"] == "bibliographic_reliability_only"
    assert "dataset" not in data
    assert data["physical_weibull"]["beta"] is None
    assert data["physical_weibull"]["eta"] is None
    assert data["fmeca"]["status"] == "awaiting_user_fmeca"
    assert {item["component_id"] for item in data["fmeca"]["components"]} == {
        "igbt",
        "sensor_feedback_system",
        "inverter_control_system",
    }
    assert all(item["npr"] is None for item in data["fmeca"]["components"])
    assert data["fmeca"]["boundary"].startswith("A validação E3")
    assert len(data["scenarios"]) == 4
    assert len(data["curve_series"]) == 4
    assert all(len(series["points"]) <= 121 for series in data["curve_series"])
    assert data["failure_rate_distribution"]["status"] == "not_estimable"
    assert data["failure_rate_distribution"]["chart_available"] is False
    assert "histogram" not in data["failure_rate_distribution"]
    assert [figure["title"] for figure in data["figures"][:4]] == [
        "Curva de confiabilidade R(t)",
        "Curva da probabilidade acumulada de falha F(t)",
        "Curva da densidade de probabilidade de falha f(t)",
        "Curva da taxa de falha h(t)",
    ]
    assert data["scenarios"][-1]["lambda_per_hour"] == pytest.approx(2.17e-6)
    assert data["formulas"]["hazard"] == "h(t) = lambda"


def test_contratos_cientificos_grandes_sao_comprimidos(client):
    response = client.get(
        "/api/results/e3",
        headers={"Accept-Encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.json()["contract_version"] == 3


def test_fontes_expoem_dataset_pdf_e_manifestos(client):
    response = client.get("/api/sources")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"]["doi"] == "10.17632/n76t439f65.1"
    assert len(data["dataset"]["raw_files_sha256"]) == 16
    assert data["bibliography"][0]["pdf_page"] == 35
    assert len(data["manifests"]) == 2
    assert len(data["separation_rules"]) == 3


def test_biblioteca_expoe_todos_os_pdfs_e_trechos_sem_contar_manifesto(client):
    response = client.get("/api/library")
    assert response.status_code == 200
    data = response.json()

    assert data["summary"]["documents"] == 45
    assert data["summary"]["indexed_chunks"] == 12556
    assert data["summary"]["portable_index_records"] == 12557
    assert len(data["documents"]) == 45
    assert all(item["url"].startswith("/library-files/") for item in data["documents"])
    assert data["write_policy"]["git_automation"] is False


def _web_library(tmp_path):
    root = tmp_path / "literatura"
    root.mkdir(parents=True)
    catalog = root / "catalogo.json"
    salvar_catalogo(
        catalog,
        {
            "schema_version": 1,
            "catalog_id": "web-test",
            "source_index": {},
            "summary": {
                "documents": 0,
                "indexed_chunks": 0,
                "portable_index_records": 0,
                "categories": {},
                "languages": {},
                "metadata_warnings": 0,
            },
            "documents": [],
        },
    )
    return LibraryService(
        catalog_path=catalog,
        literature_root=root,
        chroma_path=tmp_path / "chroma",
        snapshot_path=tmp_path / "snapshot.jsonl.gz",
        staging_root=tmp_path / "staging",
        start_jobs=False,
        model_factory=lambda: object(),
        indexer=lambda *_args, **_kwargs: {"sucesso": True, "n_chunks": 1},
        snapshot_exporter=lambda: {
            "schema_version": 1,
            "n_chunks": 1,
            "hash_corpus_sha256": "c" * 64,
            "gerado_em_utc": "2026-08-22T00:00:00+00:00",
        },
    )


def _web_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_biblioteca_permite_escrita_apenas_em_loopback(tmp_path):
    library = _web_library(tmp_path)
    app = create_app(
        AgentAdapter(answerer=lambda *_args: {"answer": "ok"}),
        warm_on_startup=False,
        library_service=library,
    )
    with TestClient(app, base_url="http://127.0.0.1") as local:
        assert local.get("/api/library").json()["writable"] is True
        response = local.post(
            "/api/library",
            data={
                "title": "Fonte local",
                "authors": "Rodolfo Torres",
                "year": "2026",
                "category": "confiabilidade",
                "language": "pt",
            },
            files={"file": ("fonte.pdf", _web_pdf(), "application/pdf")},
        )
        assert response.status_code == 202
        job_id = response.json()["job"]["job_id"]
        assert local.get(f"/api/library/jobs/{job_id}").json()["job"]["state"] == "queued"


def test_biblioteca_bloqueia_host_publico_origem_cruzada_e_nuvem(monkeypatch, tmp_path):
    public_library = _web_library(tmp_path / "public")
    public_app = create_app(
        AgentAdapter(answerer=lambda *_args: {"answer": "ok"}),
        warm_on_startup=False,
        library_service=public_library,
    )
    with TestClient(public_app, base_url="https://example.com") as public:
        denied = public.post(
            "/api/library",
            files={"file": ("fonte.pdf", _web_pdf(), "application/pdf")},
        )
        assert denied.status_code == 403

    origin_library = _web_library(tmp_path / "origin")
    origin_app = create_app(
        AgentAdapter(answerer=lambda *_args: {"answer": "ok"}),
        warm_on_startup=False,
        library_service=origin_library,
    )
    with TestClient(origin_app, base_url="http://127.0.0.1") as local:
        denied = local.post(
            "/api/library",
            headers={"Origin": "https://example.com"},
            files={"file": ("fonte.pdf", _web_pdf(), "application/pdf")},
        )
        assert denied.status_code == 403
        malformed = local.post(
            "/api/library",
            headers={"Origin": "http://127.0.0.1:porta-invalida"},
            files={"file": ("fonte.pdf", _web_pdf(), "application/pdf")},
        )
        assert malformed.status_code == 403

    monkeypatch.setenv("AL_IADO_DEPLOYMENT_MODE", "cloud")
    cloud_library = _web_library(tmp_path / "cloud")
    cloud_app = create_app(
        AgentAdapter(answerer=lambda *_args: {"answer": "ok"}),
        warm_on_startup=False,
        library_service=cloud_library,
    )
    with TestClient(cloud_app, base_url="http://127.0.0.1") as cloud:
        assert cloud.get("/api/library").json()["writable"] is False
        assert cloud.post(
            "/api/library",
            files={"file": ("fonte.pdf", _web_pdf(), "application/pdf")},
        ).status_code == 403


def test_status_e_barato_e_nao_carrega_contratos(monkeypatch, client):
    import src.webapp.app as web_app

    monkeypatch.setattr(
        web_app,
        "e3_contract",
        lambda: (_ for _ in ()).throw(AssertionError("não deveria carregar E3")),
    )
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["application"] == "aliado-web"
    assert data["api_version"] == 1
    assert data["agent"]["state"] == "pronto"
    assert data["agent"]["retrieval"] == [
        "semantic",
        "bm25",
        "sessions",
        "obsidian",
    ]
    assert data["identity"]["display_name"] == "Rodolfo"
    assert data["identity"]["timezone"] == "America/Sao_Paulo"
    assert data["identity"]["greeting"] in {"Bom dia", "Boa tarde", "Boa noite"}


def test_versao_e_entrypoint_sao_canonicos(client):
    response = client.get("/api/version")
    assert response.json() == {
        "application": "aliado-web",
        "name": "ALIAdo",
        "version": "4.0.1",
        "api_version": 1,
        "interface": "asgi",
    }
    launcher = (ROOT / "src/webapp/__main__.py").read_text(encoding="utf-8")
    root_app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "src.webapp.launcher" in launcher
    assert "from src.webapp.app import app" in root_app


def test_chat_stream_emite_status_delta_e_done(client, calls):
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(20)
    ]
    response = client.post(
        "/api/chat/stream",
        json={"message": "Interprete o AUC", "history": history},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    names = [name for name, _payload in events]
    assert names[:2] == ["status", "status"]
    assert "delta" in names
    assert names[-1] == "done"
    done = events[-1][1]
    assert done["route"] == "test-double"
    assert "<strong>verificada</strong>" in done["answer_html"]
    message, received_history, attachments = calls[-1]
    assert message == "Interprete o AUC"
    assert len(received_history) == 16
    assert received_history[0]["content"] == "4"
    assert attachments == []


def test_saudacao_local_responde_antes_do_aquecimento(monkeypatch, tmp_path):
    import src.conhecimento.base_runtime as base_runtime
    import src.conhecimento.multiagente as multiagent

    started = Event()
    release = Event()

    def slow_base(*, sincronizar_obsidian_local=True, embeddings_baixo_consumo=False):
        assert sincronizar_obsidian_local is False
        assert embeddings_baixo_consumo is True
        started.set()
        release.wait(timeout=3)
        return SimpleNamespace(
            perfil="perfil",
            modelo_embeddings=object(),
            literatura=object(),
            sessoes=object(),
            obsidian=object(),
            indice_lexical=object(),
            modo_consulta=True,
        )

    monkeypatch.setattr(base_runtime, "carregar_base_conhecimento", slow_base)
    monkeypatch.setattr(
        multiagent,
        "criar_equipe_agentes",
        lambda: SimpleNamespace(conversa=object(), auditor=object()),
    )
    adapter = AgentAdapter(session_journal=SessionJournal(pasta=tmp_path))
    warm = adapter.warm_background()
    assert warm is not None
    assert started.wait(timeout=1)
    safety_release = Timer(1.5, release.set)
    safety_release.daemon = True
    safety_release.start()

    with TestClient(create_app(adapter, warm_on_startup=False)) as test_client:
        start = monotonic()
        response = test_client.post(
            "/api/chat/stream",
            json={"message": "oi", "history": [], "session_id": "sessao_rapida"},
        )
        duration = monotonic() - start

    release.set()
    warm.join(timeout=2)
    done = _sse_events(response.text)[-1][1]
    assert done["route"] == "local"
    assert done["response_ms"] < 500
    assert duration < 0.5


def test_pergunta_social_usa_horario_nome_e_historico_sem_rag(monkeypatch):
    import src.conhecimento.agente_interacao as interaction

    monkeypatch.setenv("AL_IADO_USER_NAME", "Rodolfo Torres")
    monkeypatch.setattr(
        interaction,
        "agora_local",
        lambda: datetime.fromisoformat("2026-08-22T08:30:00-03:00"),
    )

    first = interaction.resposta_interacao_simples("Tudo bem?")
    continued = interaction.resposta_interacao_simples(
        "tudo bem?",
        [{"role": "assistant", "content": "Resposta anterior"}],
    )
    technical = interaction.resposta_interacao_simples(
        "Tudo bem com o autoencoder?"
    )

    assert first == (
        "Bom dia, Rodolfo Torres! Tudo bem por aqui. "
        "Em que parte da pesquisa trabalhamos hoje?"
    )
    assert continued == "Tudo bem por aqui, Rodolfo Torres. E com você?"
    assert technical is None


def test_chat_multipart_sanitiza_nome_e_limita_anexos(client, calls):
    response = client.post(
        "/api/chat/stream",
        data={"message": "Leia o anexo", "history": "[]"},
        files=[("files", ("../evidencia.txt", b"conteudo", "text/plain"))],
    )
    assert response.status_code == 200
    assert _sse_events(response.text)[-1][0] == "done"
    assert calls[-1][2] == [("evidencia.txt", b"conteudo")]

    excessive = client.post(
        "/api/chat/stream",
        data={"message": "Leia", "history": "[]"},
        files=[
            ("files", (f"{index}.txt", b"x", "text/plain"))
            for index in range(5)
        ],
    )
    assert excessive.status_code == 400
    assert "No máximo 4" in excessive.json()["detail"]


def test_limites_do_adaptador_sao_deterministicos():
    history = [{"role": "user", "content": str(index)} for index in range(30)]
    assert len(_historico_normalizado(history)) == 16
    with pytest.raises(ValueError, match="15 MB"):
        _validar_anexos([("grande.bin", b"x" * (MAX_ATTACHMENT_BYTES + 1))])
    with pytest.raises(ValueError, match="No máximo 4"):
        _validar_anexos([(f"{index}.txt", b"x") for index in range(5)])


def test_agente_reutiliza_pipeline_rag_completo(monkeypatch):
    import src.conhecimento.agente as agent
    import src.conhecimento.ferramentas as tools

    captured = {}
    monkeypatch.setattr(tools, "decidir_acao", lambda _message, _llm: {"usar_ferramenta": False})

    def fake_question(**kwargs):
        captured.update(kwargs)
        return "Resposta do RAG compartilhado"

    monkeypatch.setattr(agent, "perguntar", fake_question)
    components = _Componentes(
        perfil="perfil",
        modelo_embeddings=object(),
        literatura=object(),
        sessoes=object(),
        obsidian=object(),
        indice_lexical=object(),
        llm=object(),
        auditor=SimpleNamespace(),
        modo_consulta=True,
    )
    adapter = AgentAdapter()
    adapter._components = components
    adapter._state = "pronto"

    response = adapter.answer("Compare o Autoencoder Denso com o AE-LSTM.")

    assert response["answer"] == "Resposta do RAG compartilhado"
    assert response["scientific_contract"] == "canonical"
    assert captured["modelo_embeddings"] is components.modelo_embeddings
    assert captured["colecao"] is components.literatura
    assert captured["colecao_sessoes"] is components.sessoes
    assert captured["colecao_obsidian"] is components.obsidian
    assert captured["indice_lexical"] is components.indice_lexical
    assert captured["auditor"] is components.auditor
    assert captured["streaming"] is False
    assert "Dataset experimental único" in captured["contexto_autoritativo"]
    assert "PCA" not in captured["contexto_autoritativo"]


def test_agente_encaminha_chunks_reais_do_llm(monkeypatch):
    import src.conhecimento.agente as agent
    import src.conhecimento.ferramentas as tools

    captured = {}
    chunks = []
    monkeypatch.setattr(tools, "decidir_acao", lambda _message, _llm: {"usar_ferramenta": False})

    def fake_question(**kwargs):
        captured.update(kwargs)
        kwargs["on_chunk"]("Primeiro ")
        kwargs["on_chunk"]("fragmento.")
        return "Primeiro fragmento."

    monkeypatch.setattr(agent, "perguntar", fake_question)
    adapter = AgentAdapter()
    adapter._components = _Componentes(
        perfil="perfil",
        modelo_embeddings=object(),
        literatura=object(),
        sessoes=object(),
        obsidian=object(),
        indice_lexical=object(),
        llm=object(),
        auditor=SimpleNamespace(),
        modo_consulta=True,
    )
    adapter._state = "pronto"

    response = adapter.answer(
        "Explique a confiabilidade.",
        on_chunk=chunks.append,
    )

    assert response["answer"] == "Primeiro fragmento."
    assert chunks == ["Primeiro ", "fragmento."]
    assert captured["streaming"] is True
    assert captured["on_chunk"] is not None


def test_status_publica_rota_real_sem_rotulo_fixo_de_provedor():
    adapter = AgentAdapter(answerer=lambda *_args: "ok")
    adapter._components = _Componentes(
        perfil="perfil",
        modelo_embeddings=object(),
        literatura=object(),
        sessoes=object(),
        obsidian=object(),
        indice_lexical=object(),
        llm=SimpleNamespace(
            route_status=lambda: {
                "provider": "openai",
                "model": "modelo-terra",
                "task_type": "scientific_reasoning",
                "route_reason": "technical_scientific_task",
                "fallback_used": False,
                "validation_used": False,
            }
        ),
        auditor=SimpleNamespace(),
        modo_consulta=True,
    )
    status = adapter.status()
    assert status["provider"] == "openai"
    assert status["model"] == "modelo-terra"
    assert status["routing"]["route_reason"] == "technical_scientific_task"


def test_manutencao_da_sessao_nao_bloqueia_resposta(monkeypatch, tmp_path):
    import src.conhecimento.agente as agent

    indexing_started = Event()
    release_indexing = Event()
    indexing_finished = Event()

    def slow_index(_path, _model):
        indexing_started.set()
        release_indexing.wait(timeout=3)
        indexing_finished.set()

    monkeypatch.setattr(agent, "perguntar", lambda **_kwargs: "Resposta científica pronta")
    adapter = AgentAdapter(session_journal=SessionJournal(pasta=tmp_path, indexer=slow_index))
    adapter._components = _Componentes(
        perfil="perfil",
        modelo_embeddings=object(),
        literatura=object(),
        sessoes=object(),
        obsidian=object(),
        indice_lexical=object(),
        llm=object(),
        auditor=SimpleNamespace(),
        modo_consulta=True,
    )
    adapter._state = "pronto"

    start = monotonic()
    response = adapter.answer(
        "Compare o Autoencoder Denso com o AE-LSTM.",
        session_id="sessao_manutencao",
    )
    duration = monotonic() - start

    assert response["maintenance_scheduled"] is True
    assert duration < 1.0
    assert indexing_started.wait(timeout=1)
    release_indexing.set()
    assert indexing_finished.wait(timeout=1)


def test_diario_canonico_persiste_e_reindexa_turnos(tmp_path):
    indexed = []
    journal = SessionJournal(
        pasta=tmp_path,
        indexer=lambda path, model: indexed.append((path, model)),
    )
    model = object()
    first = journal.record("sessao_12345678", "Qual foi o AUC?", "AUC 0,861.", [], model)
    second = journal.record("sessao_12345678", "E o LSTM?", "AUC 0,841.", [], model)

    assert first is not None and second is not None
    assert first["path"] == second["path"]
    assert second["interaction"] == 2
    path = Path(second["path"])
    if not path.is_absolute():
        path = ROOT / path
    text = path.read_text(encoding="utf-8")
    assert "# Sessão Web" in text
    assert "tipo: sessao-web" in text
    assert "V2" not in text
    assert len(indexed) == 2


def test_catalogo_de_conversas_preserva_historico_e_imagens(tmp_path):
    journal = SessionJournal(pasta=tmp_path, indexer=lambda *_args: None)
    session_id = "sessao_catalogo_123"
    recorded = journal.record(
        session_id,
        "Mostre a curva de confiabilidade.",
        "Segue a figura solicitada.",
        [
            {
                "caption": "Curva de confiabilidade R(t)",
                "url": "/artifacts/reliability/confiabilidade.png",
            }
        ],
        object(),
    )

    detail = journal.get_conversation(session_id)
    assert recorded is not None
    assert detail["status"] == "active"
    assert detail["messages"][-1]["images"] == [
        {
            "caption": "Curva de confiabilidade R(t)",
            "url": "/artifacts/reliability/confiabilidade.png",
        }
    ]

    assert journal.rename(session_id, "Confiabilidade física")["title"] == (
        "Confiabilidade física"
    )
    assert journal.archive(session_id)["status"] == "archived"
    assert journal.list_conversations("active") == []
    assert journal.list_conversations("archived")[0]["id"] == session_id
    assert journal.restore(session_id)["status"] == "active"

    transcript = Path(recorded["path"])
    if not transcript.is_absolute():
        transcript = ROOT / transcript
    journal.delete(session_id)
    assert transcript.is_file()
    assert journal.list_conversations("active") == []
    assert journal.list_conversations("archived") == []
    with pytest.raises(KeyError):
        journal.get_conversation(session_id)


def test_api_de_conversas_cobre_ciclo_completo_e_origem(tmp_path):
    journal = SessionJournal(pasta=tmp_path / "sessions", indexer=lambda *_args: None)
    session_id = "sessao_api_12345"
    recorded = journal.record(
        session_id,
        "Analise a referência específica.",
        "Análise concluída.",
        [],
        object(),
    )
    adapter = AgentAdapter(
        answerer=lambda *_args: {"answer": "ok", "images": [], "route": "test"},
        session_journal=journal,
    )
    app = create_app(
        adapter,
        warm_on_startup=False,
        library_service=_web_library(tmp_path / "library"),
    )

    with TestClient(app, base_url="http://127.0.0.1") as local:
        active = local.get("/api/conversations?status=active")
        assert active.status_code == 200
        assert active.json()["conversations"][0]["id"] == session_id
        assert local.get(f"/api/conversations/{session_id}").status_code == 200

        renamed = local.patch(
            f"/api/conversations/{session_id}",
            json={"title": "Referência selecionada"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["conversation"]["title"] == "Referência selecionada"

        archived = local.post(f"/api/conversations/{session_id}/archive")
        assert archived.json()["conversation"]["status"] == "archived"
        assert local.get("/api/conversations?status=active").json()[
            "conversations"
        ] == []
        restored = local.post(f"/api/conversations/{session_id}/restore")
        assert restored.json()["conversation"]["status"] == "active"

        denied = local.patch(
            f"/api/conversations/{session_id}",
            headers={"Origin": "https://example.com"},
            json={"title": "Origem indevida"},
        )
        assert denied.status_code == 403
        deleted = local.delete(f"/api/conversations/{session_id}")
        assert deleted.json()["conversation"]["memory_retained"] is True
        assert local.get(f"/api/conversations/{session_id}").status_code == 404

    transcript = Path(recorded["path"])
    if not transcript.is_absolute():
        transcript = ROOT / transcript
    assert transcript.is_file()


def test_renderizacao_markdown_bloqueia_html_bruto():
    rendered = render_agent_markdown("**seguro** <script>alert(1)</script>")
    assert "<strong>seguro</strong>" in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_renderizacao_em_lote_preserva_latex_e_nao_confia_em_html(client):
    response = client.post(
        "/api/render",
        json={
            "messages": [
                {
                    "id": "equacao",
                    "content": (
                        r"**Confiabilidade:** \(R(t)=e^{-\lambda t}\) "
                        r"<img src=x onerror=alert(1)>"
                    ),
                },
                {"id": "codigo", "content": r"`\(nao renderizar\)`"},
            ]
        },
    )

    assert response.status_code == 200
    rendered = {item["id"]: item["html"] for item in response.json()["messages"]}
    assert "<strong>Confiabilidade:</strong>" in rendered["equacao"]
    assert r"\(R(t)=e^{-\lambda t}\)" in rendered["equacao"]
    assert "<img" not in rendered["equacao"]
    assert "&lt;img" in rendered["equacao"]
    assert r"<code>\(nao renderizar\)</code>" in rendered["codigo"]


def test_renderizacao_em_lote_rejeita_formato_e_volume_invalidos(client):
    invalid = client.post("/api/render", json={"messages": "texto"})
    excessive = client.post(
        "/api/render",
        json={
            "messages": [
                {"id": str(index), "content": "x"} for index in range(101)
            ]
        },
    )

    assert invalid.status_code == 400
    assert excessive.status_code == 400
    assert "No máximo 100" in excessive.json()["detail"]


def test_contexto_cientifico_reconcilia_comparacao_e_confiabilidade():
    context = scientific_context_for(
        "Compare o autoencoder denso com o AE-LSTM e explique a taxa de falha"
    )
    assert context is not None
    assert "0.384146" in context
    assert "0.386702" in context
    assert "0.861850" in context
    assert "0.841164" in context
    assert "2.170e-06 h^-1" in context
    assert "SMD95" not in context
    assert "a_det" not in context
    assert "PCA" not in context


def test_contratos_diretos_sao_coerentes_e_cacheaveis():
    assert e3_contract()["dataset"]["name"] == "GPVS-Faults"
    assert reliability_contract()["hours_per_year"] == 8760.0
    assert sources_contract()["dataset"]["experiments"] == 16


def test_pacote_web_nao_conserva_identificadores_legados():
    package = ROOT / "src" / "webapp"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in package.rglob("*")
        if path.suffix in {".py", ".html", ".css", ".js"}
    ).casefold()
    assert "webapp_v2" not in source
    assert "autoencoder_v2" not in source
    assert "resultados/v2" not in source
    assert "resultados\\v2" not in source
    assert "plotly" not in source
