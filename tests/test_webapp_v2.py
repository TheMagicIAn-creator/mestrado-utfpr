from __future__ import annotations

from collections import Counter
from pathlib import Path
from threading import Event, Thread, Timer
from time import monotonic
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from src.webapp_v2.agent_adapter import (
    MAX_ATTACHMENT_BYTES,
    AgentAdapter,
    _Componentes,
    _historico_normalizado,
    _validar_anexos,
)
from src.webapp_v2.app import create_app
from src.webapp_v2.contracts import dashboard_contract, reliability_curves_contract
from src.webapp_v2.launcher import bloquear_execucao_streamlit
from src.webapp_v2.rendering import render_agent_markdown
from src.webapp_v2.scientific_context import scientific_context_for
from src.webapp_v2.session_journal import SessionJournal

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture()
def chamadas():
    return []


@pytest.fixture()
def client(chamadas):
    def responder(mensagem, historico, anexos):
        chamadas.append((mensagem, historico, anexos))
        return {
            "answer": f"Resposta verificada: {mensagem}",
            "images": [],
            "route": "test-double",
        }

    with TestClient(create_app(AgentAdapter(answerer=responder))) as test_client:
        yield test_client


def test_homepage_e_asgi_sem_streamlit(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ALIAdo PV | Web V2" in response.text
    assert "/static/app.js" in response.text
    assert "streamlit" not in response.text.lower()
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_dashboard_publica_apenas_contratos_cientificos_v2(client):
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    dados = response.json()
    assert dados["schema_version"] == 2
    assert dados["project"]["dataset"] == "GPVS-Faults"
    assert dados["overview"]["metrics"]["auc_roc"]["mean"] == pytest.approx(
        0.7775582664897344
    )
    assert dados["autoencoder"]["threshold"]["value"] == pytest.approx(
        0.4404987422783545
    )
    assert len(dados["autoencoder"]["trials"]) == 28
    assert {item["method"] for item in dados["autoencoder"]["trials"]} == {
        "autoencoder_v2",
        "pca",
    }
    assert dados["reliability"]["physical_weibull"]["beta"] is None
    assert dados["reliability"]["physical_weibull"]["eta"] is None
    assert [item["npr"] for item in dados["fmeca"]["components"]] == [315, 90, 30]


def test_curvas_reconciliam_probabilidades_e_grade(client):
    response = client.get("/api/reliability/curves")
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert len(rows) == 5 * 401
    assert set(Counter(item["scenario_id"] for item in rows).values()) == {401}
    for item in rows:
        assert item["reliability"] + item["cumulative_failure_probability"] == pytest.approx(
            1.0
        )
        assert item["failure_density_per_year"] >= 0
        assert item["hazard_per_year"] > 0


def test_downloads_e_bundle_plotly_sao_servidos_localmente(client):
    image = client.get("/artifacts/autoencoder/calibracao_limiar.png")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert len(image.content) > 50_000

    bundle = client.get("/vendor/plotly.min.js")
    assert bundle.status_code == 200
    assert "Plotly" in bundle.text
    assert len(bundle.content) > 1_000_000


def test_health_reflete_contratos_e_inicializacao_sob_demanda(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    dados = response.json()
    assert dados["application"] == "aliado-pv-web"
    assert dados["version"] == "2.1.0"
    assert dados["schema_version"] == 2
    assert dados["status"] == "ok"
    assert dados["contracts"] == {
        "autoencoder_v2": "ready",
        "reliability_v2": "ready",
    }
    assert dados["agent"]["state"] == "ready"
    assert dados["agent"]["lazy_initialization"] is True
    assert dados["agent"]["engine"] == "src.conhecimento.agente.perguntar"
    assert dados["agent"]["retrieval"] == [
        "semantic",
        "bm25",
        "sessions",
        "obsidian",
    ]


def test_versao_e_aquecimento_identificam_a_v2(client):
    version = client.get("/api/version")
    assert version.status_code == 200
    assert version.json() == {
        "application": "aliado-pv-web",
        "name": "ALIAdo PV",
        "version": "2.1.0",
        "schema_version": 2,
        "interface": "asgi",
    }

    warmup = client.post("/api/agent/initialize")
    assert warmup.status_code == 200
    assert warmup.json()["agent"]["state"] == "ready"


def test_chat_json_limita_historico_sem_recalcular_resultados(client, chamadas):
    historico = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(20)
    ]
    response = client.post(
        "/api/chat",
        json={"message": "Interprete o AUC", "history": historico},
    )
    assert response.status_code == 200
    assert response.json()["route"] == "test-double"
    assert "Resposta verificada" in response.json()["answer_html"]
    mensagem, recebido, anexos = chamadas[-1]
    assert mensagem == "Interprete o AUC"
    assert len(recebido) == 16
    assert recebido[0]["content"] == "4"
    assert anexos == []


def test_saudacao_local_nao_aguarda_aquecimento_da_base(monkeypatch, tmp_path):
    import src.conhecimento.base_runtime as base_runtime
    import src.conhecimento.multiagente as multiagente

    iniciou = Event()
    liberar = Event()

    def carregar_base_lenta(
        *,
        sincronizar_obsidian_local=True,
        embeddings_baixo_consumo=False,
    ):
        assert sincronizar_obsidian_local is False
        assert embeddings_baixo_consumo is True
        iniciou.set()
        liberar.wait(timeout=3)
        return SimpleNamespace(
            perfil="perfil",
            modelo_embeddings=object(),
            literatura=object(),
            sessoes=object(),
            obsidian=object(),
            indice_lexical=object(),
            modo_consulta=True,
        )

    monkeypatch.setattr(base_runtime, "carregar_base_conhecimento", carregar_base_lenta)
    monkeypatch.setattr(
        multiagente,
        "criar_equipe_agentes",
        lambda: SimpleNamespace(conversa=object(), auditor=object()),
    )
    adapter = AgentAdapter(session_journal=SessionJournal(pasta=tmp_path))
    aquecimento = Thread(target=adapter.initialize, daemon=True)
    aquecimento.start()
    assert iniciou.wait(timeout=1)

    # Evita deadlock em uma regressao: a implementacao antiga so respondia
    # depois que este timer liberava o carregamento pesado.
    liberacao_de_seguranca = Timer(1.5, liberar.set)
    liberacao_de_seguranca.daemon = True
    liberacao_de_seguranca.start()
    with TestClient(create_app(adapter)) as test_client:
        inicio = monotonic()
        response = test_client.post(
            "/api/chat",
            json={"message": "oi", "history": [], "session_id": "sessao_rapida"},
        )
        duracao = monotonic() - inicio

    liberar.set()
    aquecimento.join(timeout=2)
    assert response.status_code == 200
    assert response.json()["route"] == "local"
    assert response.json()["response_ms"] < 800
    assert duracao < 0.8


def test_chat_multipart_sanitiza_nome_do_anexo(client, chamadas):
    response = client.post(
        "/api/chat",
        data={"message": "Leia o anexo", "history": "[]"},
        files=[("files", ("../evidencia.txt", b"conteudo", "text/plain"))],
    )
    assert response.status_code == 200
    assert chamadas[-1][2] == [("evidencia.txt", b"conteudo")]


def test_chat_multipart_rejeita_excesso_antes_do_adaptador(client):
    response = client.post(
        "/api/chat",
        data={"message": "Leia", "history": "[]"},
        files=[
            ("files", (f"{index}.txt", b"x", "text/plain"))
            for index in range(5)
        ],
    )
    assert response.status_code == 400
    assert "No máximo 4" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "", "history": []},
        {"message": "ok", "history": "nao-e-lista"},
        {"message": "ok", "history": [{"role": "system", "content": "x"}]},
    ],
)
def test_chat_rejeita_entradas_invalidas(client, payload):
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_chat_rejeita_identificador_de_sessao_inseguro(client):
    response = client.post(
        "/api/chat",
        json={"message": "Interprete", "history": [], "session_id": "../fora"},
    )
    assert response.status_code == 400
    assert "session_id invalido" in response.json()["detail"]


def test_limites_do_adaptador_sao_deterministicos():
    historico = [{"role": "user", "content": str(index)} for index in range(30)]
    assert len(_historico_normalizado(historico)) == 16
    with pytest.raises(ValueError, match="15 MB"):
        _validar_anexos([("grande.bin", b"x" * (MAX_ATTACHMENT_BYTES + 1))])
    with pytest.raises(ValueError, match="No máximo 4"):
        _validar_anexos([(f"{index}.txt", b"x") for index in range(5)])


def test_v2_reutiliza_pipeline_rag_completo_da_v1(monkeypatch):
    import src.conhecimento.agente as agente
    import src.conhecimento.ferramentas as ferramentas

    capturado = {}
    monkeypatch.setattr(
        ferramentas,
        "decidir_acao",
        lambda _mensagem, _llm: {"usar_ferramenta": False},
    )

    def perguntar_falso(**kwargs):
        capturado.update(kwargs)
        return "Resposta do RAG compartilhado"

    monkeypatch.setattr(agente, "perguntar", perguntar_falso)
    componentes = _Componentes(
        perfil="perfil-v2",
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
    adapter._components = componentes
    adapter._state = "ready"

    resposta = adapter.answer(
        "Interprete o resultado experimental",
        [{"role": "user", "content": "Contexto anterior"}],
    )

    assert resposta["answer"] == "Resposta do RAG compartilhado"
    assert capturado["modelo_embeddings"] is componentes.modelo_embeddings
    assert capturado["colecao"] is componentes.literatura
    assert capturado["colecao_sessoes"] is componentes.sessoes
    assert capturado["colecao_obsidian"] is componentes.obsidian
    assert capturado["indice_lexical"] is componentes.indice_lexical
    assert capturado["llm"] is componentes.llm
    assert capturado["auditor"] is componentes.auditor
    assert capturado["streaming"] is False
    assert "Nao invente valores" in capturado["contexto_autoritativo"]
    assert "nao ha superioridade global" in capturado["contexto_autoritativo"].lower()


def test_v2_evitar_roteador_llm_em_pergunta_discursiva_canonica(monkeypatch):
    import src.conhecimento.agente as agente
    import src.conhecimento.ferramentas as ferramentas

    def roteador_nao_esperado(*_args):
        raise AssertionError("roteador semantico nao deveria ser chamado")

    monkeypatch.setattr(ferramentas, "decidir_acao", roteador_nao_esperado)
    monkeypatch.setattr(
        agente,
        "perguntar",
        lambda **_kwargs: "Comparacao respondida pelo RAG compartilhado",
    )
    adapter = AgentAdapter()
    adapter._components = _Componentes(
        perfil="perfil-v2",
        modelo_embeddings=object(),
        literatura=object(),
        sessoes=object(),
        obsidian=object(),
        indice_lexical=object(),
        llm=object(),
        auditor=SimpleNamespace(),
        modo_consulta=True,
    )
    adapter._state = "ready"

    resposta = adapter.answer("Compare o autoencoder V2 com o PCA.")

    assert resposta["route"] == "rag"
    assert resposta["scientific_contract"] == "v2"


def test_manutencao_da_sessao_nao_bloqueia_resposta(monkeypatch, tmp_path):
    import src.conhecimento.agente as agente

    indexacao_iniciou = Event()
    liberar_indexacao = Event()
    indexacao_terminou = Event()

    def indexar_lento(_caminho, _modelo):
        indexacao_iniciou.set()
        liberar_indexacao.wait(timeout=3)
        indexacao_terminou.set()

    monkeypatch.setattr(
        agente,
        "perguntar",
        lambda **_kwargs: "Resposta cientifica pronta",
    )
    adapter = AgentAdapter(
        session_journal=SessionJournal(pasta=tmp_path, indexer=indexar_lento)
    )
    adapter._components = _Componentes(
        perfil="perfil-v2",
        modelo_embeddings=object(),
        literatura=object(),
        sessoes=object(),
        obsidian=object(),
        indice_lexical=object(),
        llm=object(),
        auditor=SimpleNamespace(),
        modo_consulta=True,
    )
    adapter._state = "ready"

    inicio = monotonic()
    resposta = adapter.answer(
        "Compare o autoencoder V2 com o PCA.",
        session_id="sessao_manutencao",
    )
    duracao = monotonic() - inicio

    assert resposta["maintenance_scheduled"] is True
    assert duracao < 1.0
    assert indexacao_iniciou.wait(timeout=1)
    liberar_indexacao.set()
    assert indexacao_terminou.wait(timeout=1)


def test_obsidian_e_sincronizado_em_background_com_intervalo(monkeypatch, tmp_path):
    import src.conhecimento.agente as agente
    import src.conhecimento.obsidian as obsidian

    sincronizacoes = []
    concluiu = Event()
    monkeypatch.setenv("AL_IADO_OBSIDIAN_SYNC_INTERVAL_S", "300")
    monkeypatch.setattr(
        agente,
        "perguntar",
        lambda **_kwargs: "Resposta cientifica pronta",
    )

    def sincronizar(colecao, modelo):
        sincronizacoes.append((colecao, modelo))
        concluiu.set()

    monkeypatch.setattr(obsidian, "sincronizar_obsidian", sincronizar)
    componentes = _Componentes(
        perfil="perfil-v2",
        modelo_embeddings=object(),
        literatura=object(),
        sessoes=object(),
        obsidian=object(),
        indice_lexical=object(),
        llm=object(),
        auditor=SimpleNamespace(),
        modo_consulta=False,
    )
    adapter = AgentAdapter(
        session_journal=SessionJournal(
            pasta=tmp_path,
            indexer=lambda *_args: None,
        )
    )
    adapter._components = componentes
    adapter._state = "ready"

    primeira = adapter.answer("Compare o autoencoder V2 com o PCA.")
    assert primeira["maintenance_scheduled"] is True
    assert concluiu.wait(timeout=1)
    adapter.answer("Explique a calibracao do limiar.")

    assert sincronizacoes == [
        (componentes.obsidian, componentes.modelo_embeddings),
    ]


def test_runtime_preserva_sincronizacao_local_por_padrao():
    import inspect

    from src.conhecimento.base_runtime import carregar_base_conhecimento

    parametro = inspect.signature(carregar_base_conhecimento).parameters[
        "sincronizar_obsidian_local"
    ]
    assert parametro.default is True
    assert (
        inspect.signature(carregar_base_conhecimento)
        .parameters["embeddings_baixo_consumo"]
        .default
        is False
    )


def test_diario_v2_persiste_e_reindexa_turnos_sem_streamlit(tmp_path):
    indexados = []
    journal = SessionJournal(
        pasta=tmp_path,
        indexer=lambda caminho, modelo: indexados.append((caminho, modelo)),
    )
    modelo = object()

    primeiro = journal.record(
        "sessao_12345678",
        "Qual foi o AUC?",
        "O AUC foi 0,778.",
        [],
        modelo,
    )
    segundo = journal.record(
        "sessao_12345678",
        "E o PCA?",
        "O PCA obteve AUC superior neste protocolo.",
        [],
        modelo,
    )

    assert primeiro is not None
    assert segundo is not None
    assert primeiro["path"] == segundo["path"]
    assert segundo["interaction"] == 2
    texto = Path(segundo["path"]).read_text(encoding="utf-8")
    assert "tipo: sessao-web-v2" in texto
    assert "## Interacao 1" in texto
    assert "## Interacao 2" in texto
    assert len(indexados) == 2
    assert all(item[1] is modelo for item in indexados)


def test_contexto_cientifico_reconcilia_agente_com_graficos_v2():
    contexto = scientific_context_for("Compare o autoencoder com o PCA")
    assert contexto is not None
    assert "0.777558" in contexto
    assert "0.788" in contexto
    assert "nao ha superioridade global" in contexto.lower()
    assert "nao use sobreposicao de ic95% como prova de equivalencia" in contexto.lower()
    assert "beta=None, eta=None" in contexto
    assert "Contator AC NPR=315" in contexto
    assert scientific_context_for("Ola, tudo bem?") is None


def test_frontend_nao_recalcula_metricas_nem_fixa_faixas_logaritmicas():
    javascript = (RAIZ / "src/webapp_v2/static/app.js").read_text(encoding="utf-8")
    html = (RAIZ / "src/webapp_v2/templates/index.html").read_text(encoding="utf-8")
    assert 'mode: "lines+markers"' not in javascript
    assert 'yaxis.range = [-7' not in javascript
    assert 'yaxis.range = [-1.3' not in javascript
    assert "renderAllCharts" not in javascript
    assert "renderChartsForView(activeView)" in javascript
    assert "streamlit" not in html.lower()
    assert "fetchJSON(\"/api/dashboard\")" in javascript
    assert 'fetchJSON("/api/agent/initialize"' in javascript
    assert 'view.dataset.view === "agent"' in javascript
    assert "await warmupAgent();" not in javascript
    assert "typing-indicator" in javascript
    assert 'event.key === "Enter" && !event.shiftKey' in javascript
    assert "if (!state.chatRequestActive) updateAgentStatus(result.agent);" in javascript
    assert "message-avatar" in html


def test_resposta_markdown_preserva_formato_sem_executar_html_bruto():
    html = render_agent_markdown(
        "## Resultado\n\n| Métrica | Valor |\n|---|---:|\n| AUC | 0,778 |\n\n"
        "linha 1<br>linha 2\n\n<script>alert('x')</script>"
    )
    assert "<h2>Resultado</h2>" in html
    assert "<table>" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;br&gt;" not in html


def test_streamlit_e_bloqueado_no_entrypoint_v2(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "streamlit", object())
    with pytest.raises(RuntimeError, match="python -m src.webapp_v2"):
        bloquear_execucao_streamlit()


def test_webapp_v2_nao_importa_interface_legada_nem_resultados_sem_versao():
    pasta = RAIZ / "src" / "webapp_v2"
    fontes = "\n".join(
        caminho.read_text(encoding="utf-8")
        for caminho in pasta.rglob("*.py")
    ).lower()
    assert "src.interface" not in fontes
    assert "import streamlit" not in fontes

    contratos = (pasta / "contracts.py").read_text(encoding="utf-8")
    assert 'RAIZ / "resultados" / "v2" / "autoencoder"' in contratos
    assert 'RAIZ / "resultados" / "v2" / "confiabilidade"' in contratos


def test_contratos_em_cache_mantem_identidade_e_contagens():
    assert dashboard_contract() is dashboard_contract()
    assert reliability_curves_contract() is reliability_curves_contract()
    assert len(dashboard_contract()["reliability"]["scenarios"]) == 5


def test_modulo_de_base_pode_ser_importado_sem_carregar_chromadb():
    import sys

    sys.modules.pop("src.conhecimento.base_runtime", None)
    antes = "chromadb" in sys.modules
    __import__("src.conhecimento.base_runtime")
    assert ("chromadb" in sys.modules) is antes
