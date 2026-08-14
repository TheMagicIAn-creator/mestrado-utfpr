from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from src.webapp.agent_adapter import (
    MAX_ATTACHMENT_BYTES,
    AgentAdapter,
    _historico_normalizado,
    _validar_anexos,
)
from src.webapp.app import create_app
from src.webapp.contracts import dashboard_contract, reliability_curves_contract

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
    assert "ALIAdo PV | Resultados V2" in response.text
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
    assert dados["status"] == "ok"
    assert dados["contracts"] == {
        "autoencoder_v2": "ready",
        "reliability_v2": "ready",
    }
    assert dados["agent"]["state"] == "ready"
    assert dados["agent"]["lazy_initialization"] is True


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
    mensagem, recebido, anexos = chamadas[-1]
    assert mensagem == "Interprete o AUC"
    assert len(recebido) == 16
    assert recebido[0]["content"] == "4"
    assert anexos == []


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


def test_limites_do_adaptador_sao_deterministicos():
    historico = [{"role": "user", "content": str(index)} for index in range(30)]
    assert len(_historico_normalizado(historico)) == 16
    with pytest.raises(ValueError, match="15 MB"):
        _validar_anexos([("grande.bin", b"x" * (MAX_ATTACHMENT_BYTES + 1))])
    with pytest.raises(ValueError, match="No máximo 4"):
        _validar_anexos([(f"{index}.txt", b"x") for index in range(5)])


def test_frontend_nao_recalcula_metricas_nem_fixa_faixas_logaritmicas():
    javascript = (RAIZ / "src/webapp/static/app.js").read_text(encoding="utf-8")
    html = (RAIZ / "src/webapp/templates/index.html").read_text(encoding="utf-8")
    assert 'mode: "lines+markers"' not in javascript
    assert 'yaxis.range = [-7' not in javascript
    assert 'yaxis.range = [-1.3' not in javascript
    assert "renderAllCharts" not in javascript
    assert "renderChartsForView(activeView)" in javascript
    assert "streamlit" not in html.lower()
    assert "fetchJSON(\"/api/dashboard\")" in javascript


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
