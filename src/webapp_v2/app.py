"""Aplicação ASGI do ALIAdo PV."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.core.config import RAIZ_PROJETO
from src.webapp_v2 import APP_ID, APP_NAME, APP_VERSION, SCHEMA_VERSION
from src.webapp_v2.agent_adapter import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    AgentAdapter,
    AgenteIndisponivel,
)
from src.webapp_v2.contracts import (
    AUTOENCODER,
    CONFIABILIDADE,
    DETECTABILIDADE,
    ContratoWebInvalido,
    dashboard_contract,
    reliability_curves_contract,
)
from src.webapp_v2.rendering import render_agent_markdown

RAIZ = Path(RAIZ_PROJETO)
WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
INDEX_HTML = WEB_ROOT / "templates" / "index.html"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "font-src 'self'; connect-src 'self'; frame-ancestors 'none';",
        )
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response


def _contract_error(exc: Exception) -> JSONResponse:
    return JSONResponse(
        {"error": "scientific_contract_unavailable", "detail": str(exc)},
        status_code=503,
    )


async def homepage(_request: Request) -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


async def dashboard_api(_request: Request) -> JSONResponse:
    try:
        contract = await run_in_threadpool(dashboard_contract)
    except ContratoWebInvalido as exc:
        return _contract_error(exc)
    return JSONResponse(contract)


async def reliability_curves_api(_request: Request) -> JSONResponse:
    try:
        contract = await run_in_threadpool(reliability_curves_contract)
    except ContratoWebInvalido as exc:
        return _contract_error(exc)
    return JSONResponse(contract)


async def version_api(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "application": APP_ID,
            "name": APP_NAME,
            "version": APP_VERSION,
            "schema_version": SCHEMA_VERSION,
            "interface": "asgi",
        }
    )


async def plotly_bundle(_request: Request) -> Response:
    from plotly.offline import get_plotlyjs

    return Response(
        get_plotlyjs(),
        media_type="text/javascript",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


async def _chat_payload(
    request: Request,
) -> tuple[str, list, list[tuple[str, bytes]], str | None]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("application/json"):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("O corpo JSON deve ser um objeto")
        return (
            payload.get("message", ""),
            payload.get("history", []),
            [],
            payload.get("session_id"),
        )

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        historico_bruto = str(form.get("history") or "[]")
        try:
            historico = json.loads(historico_bruto)
        except json.JSONDecodeError as exc:
            raise ValueError("Histórico JSON inválido") from exc
        uploads = [
            upload for upload in form.getlist("files") if isinstance(upload, UploadFile)
        ]
        if len(uploads) > MAX_ATTACHMENTS:
            raise ValueError(f"No máximo {MAX_ATTACHMENTS} anexos por mensagem")
        anexos = []
        for upload in uploads:
            dados = await upload.read(MAX_ATTACHMENT_BYTES + 1)
            anexos.append((upload.filename or "anexo", dados))
            await upload.close()
        return (
            str(form.get("message") or ""),
            historico,
            anexos,
            str(form.get("session_id") or "") or None,
        )

    raise ValueError("Use application/json ou multipart/form-data")


def create_app(agent_adapter: AgentAdapter | None = None) -> Starlette:
    adapter = agent_adapter or AgentAdapter()

    async def health_api(_request: Request) -> JSONResponse:
        try:
            await run_in_threadpool(dashboard_contract)
            contracts = {"autoencoder_v2": "ready", "reliability_v2": "ready"}
            status = "ok"
        except ContratoWebInvalido as exc:
            contracts = {"autoencoder_v2": "error", "reliability_v2": "error"}
            status = "degraded"
            contracts["detail"] = str(exc)
        return JSONResponse(
            {
                "application": APP_ID,
                "version": APP_VERSION,
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "contracts": contracts,
                "agent": adapter.status(),
            }
        )

    async def chat_api(request: Request) -> JSONResponse:
        try:
            inicio = perf_counter()
            mensagem, historico, anexos, session_id = await _chat_payload(request)
            resposta = await run_in_threadpool(
                adapter.answer,
                mensagem,
                historico,
                anexos,
                session_id,
            )
            return JSONResponse(
                {
                    "status": "ok",
                    **resposta,
                    "answer_html": render_agent_markdown(resposta["answer"]),
                    "agent": adapter.status(),
                    "response_ms": round((perf_counter() - inicio) * 1000, 1),
                }
            )
        except (ValueError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": "invalid_request", "detail": str(exc)}, status_code=400)
        except AgenteIndisponivel as exc:
            return JSONResponse(
                {"error": "agent_unavailable", "detail": str(exc)}, status_code=503
            )

    async def initialize_agent_api(_request: Request) -> JSONResponse:
        try:
            status = await run_in_threadpool(adapter.initialize)
            return JSONResponse({"status": "ok", "agent": status})
        except AgenteIndisponivel as exc:
            return JSONResponse(
                {"error": "agent_unavailable", "detail": str(exc)}, status_code=503
            )

    async def reset_agent_api(_request: Request) -> JSONResponse:
        adapter.reset()
        return JSONResponse({"status": "ok", "agent": adapter.status()})

    routes = [
        Route("/", homepage, methods=["GET"]),
        Route("/api/dashboard", dashboard_api, methods=["GET"]),
        Route("/api/reliability/curves", reliability_curves_api, methods=["GET"]),
        Route("/api/version", version_api, methods=["GET"]),
        Route("/api/health", health_api, methods=["GET"]),
        Route("/api/chat", chat_api, methods=["POST"]),
        Route("/api/agent/initialize", initialize_agent_api, methods=["POST"]),
        Route("/api/agent/reset", reset_agent_api, methods=["POST"]),
        Route("/vendor/plotly.min.js", plotly_bundle, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=STATIC_ROOT), name="static"),
        Mount(
            "/artifacts/autoencoder",
            app=StaticFiles(directory=AUTOENCODER),
            name="artifacts-autoencoder",
        ),
        Mount(
            "/artifacts/reliability",
            app=StaticFiles(directory=CONFIABILIDADE),
            name="artifacts-reliability",
        ),
        *(
            [
                Mount(
                    "/artifacts/detectability",
                    app=StaticFiles(directory=DETECTABILIDADE),
                    name="artifacts-detectability",
                )
            ]
            # A pasta só existe depois de `python -m src.ml.macro_weibull`.
            # StaticFiles com diretório inexistente estoura na criação do app,
            # e isso derrubaria a aplicação inteira por causa de um artefato
            # opcional.
            if DETECTABILIDADE.is_dir() else []
        ),
    ]
    app = Starlette(
        debug=False,
        routes=routes,
        middleware=[Middleware(SecurityHeadersMiddleware)],
    )
    app.state.agent_adapter = adapter
    return app


app = create_app()
