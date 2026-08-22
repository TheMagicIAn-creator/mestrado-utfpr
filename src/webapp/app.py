"""Aplicação ASGI canônica do ALIAdo."""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.core.logs import get_logger
from src.core.identidade import nome_pesquisador
from src.core.tempo import fuso_projeto, saudacao_periodo
from src.webapp import API_VERSION, APP_ID, APP_NAME, APP_VERSION
from src.webapp.agent_adapter import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    AgentAdapter,
    AgenteIndisponivel,
)
from src.webapp.contracts import (
    COMPARISON,
    LITERATURE,
    MANIFESTS,
    RELIABILITY,
    ContratoWebInvalido,
    contracts_status,
    e2_contract,
    e3_contract,
    reliability_contract,
    sources_contract,
    warm_contracts_background,
)
from src.webapp.rendering import render_agent_markdown, render_agent_messages

WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
INDEX_HTML = WEB_ROOT / "templates" / "index.html"
_LOGGER = get_logger("webapp.http")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


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
        elif request.url.path.startswith(("/static/", "/artifacts/")):
            response.headers.setdefault("Cache-Control", "public, max-age=3600")
        return response


def _contract_error(exc: Exception) -> JSONResponse:
    return JSONResponse(
        {"error": "scientific_contract_unavailable", "detail": str(exc)},
        status_code=503,
    )


async def homepage(_request: Request) -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


async def _contract_response(loader) -> JSONResponse:
    try:
        return JSONResponse(await run_in_threadpool(loader))
    except ContratoWebInvalido as exc:
        return _contract_error(exc)


async def e3_api(_request: Request) -> JSONResponse:
    return await _contract_response(e3_contract)


async def e2_api(_request: Request) -> JSONResponse:
    return await _contract_response(e2_contract)


async def reliability_api(_request: Request) -> JSONResponse:
    return await _contract_response(reliability_contract)


async def sources_api(_request: Request) -> JSONResponse:
    return await _contract_response(sources_contract)


async def version_api(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "application": APP_ID,
            "name": APP_NAME,
            "version": APP_VERSION,
            "api_version": API_VERSION,
            "interface": "asgi",
        }
    )


async def render_api(request: Request) -> JSONResponse:
    """Renderiza historico Markdown sem confiar em HTML persistido no cliente."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("O corpo JSON deve ser um objeto")
        rendered = render_agent_messages(payload.get("messages", []))
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"error": "invalid_request", "detail": str(exc)},
            status_code=400,
        )
    return JSONResponse({"messages": rendered})


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
        history_raw = str(form.get("history") or "[]")
        try:
            history = json.loads(history_raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Histórico JSON inválido") from exc
        uploads = [
            upload for upload in form.getlist("files") if isinstance(upload, UploadFile)
        ]
        if len(uploads) > MAX_ATTACHMENTS:
            raise ValueError(f"No máximo {MAX_ATTACHMENTS} anexos por mensagem")
        attachments = []
        for upload in uploads:
            data = await upload.read(MAX_ATTACHMENT_BYTES + 1)
            attachments.append((upload.filename or "anexo", data))
            await upload.close()
        return (
            str(form.get("message") or ""),
            history,
            attachments,
            str(form.get("session_id") or "") or None,
        )

    raise ValueError("Use application/json ou multipart/form-data")


def _sse(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _text_chunks(text: str, target_size: int = 96):
    buffer = ""
    for token in re.findall(r"\S+\s*", text):
        buffer += token
        if len(buffer) >= target_size or "\n\n" in buffer:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def _citations(text: str) -> list[dict[str, str]]:
    seen = set()
    citations = []
    for label, url in _MARKDOWN_LINK.findall(text):
        if url in seen:
            continue
        seen.add(url)
        citations.append({"label": label.strip(), "url": url})
    return citations


def create_app(
    agent_adapter: AgentAdapter | None = None,
    *,
    warm_on_startup: bool = True,
) -> Starlette:
    adapter = agent_adapter or AgentAdapter()
    started_at = perf_counter()

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        if warm_on_startup:
            adapter.warm_background()
            warm_contracts_background()
        yield

    async def status_api(_request: Request) -> JSONResponse:
        agent = adapter.status()
        contracts = contracts_status()
        states = {agent["state"], contracts["state"]}
        if "degradado" in states:
            state = "degradado"
        elif states == {"pronto"}:
            state = "pronto"
        else:
            state = "iniciando"
        return JSONResponse(
            {
                "application": APP_ID,
                "version": APP_VERSION,
                "api_version": API_VERSION,
                "state": state,
                "uptime_ms": round((perf_counter() - started_at) * 1000, 1),
                "agent": agent,
                "contracts": contracts,
                "identity": {
                    "display_name": nome_pesquisador(),
                    "greeting": saudacao_periodo(),
                    "timezone": getattr(fuso_projeto(), "key", str(fuso_projeto())),
                },
            }
        )

    async def chat_stream_api(request: Request):
        try:
            message, history, attachments, session_id = await _chat_payload(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return JSONResponse(
                {"error": "invalid_request", "detail": str(exc)},
                status_code=400,
            )

        request_id = uuid4().hex[:12]

        async def event_stream():
            started = perf_counter()
            yield _sse(
                "status",
                {
                    "request_id": request_id,
                    "phase": "recebido",
                    "message": "Preparando contexto",
                },
            )
            try:
                yield _sse(
                    "status",
                    {
                        "request_id": request_id,
                        "phase": "consultando",
                        "message": (
                            "Respondendo localmente"
                            if not attachments and len(str(message)) < 80
                            else "Consultando base acadêmica"
                        ),
                    },
                )
                response = await run_in_threadpool(
                    adapter.answer,
                    message,
                    history,
                    attachments,
                    session_id,
                )
                answer = str(response["answer"])
                for chunk in _text_chunks(answer):
                    if await request.is_disconnected():
                        _LOGGER.info("chat cancelado request_id=%s", request_id)
                        return
                    yield _sse("delta", {"request_id": request_id, "text": chunk})
                    await asyncio.sleep(0.008)
                elapsed_ms = round((perf_counter() - started) * 1000, 1)
                done = {
                    **response,
                    "answer_html": render_agent_markdown(answer),
                    "citations": _citations(answer),
                    "request_id": request_id,
                    "response_ms": elapsed_ms,
                    "agent": adapter.status(),
                }
                yield _sse("done", done)
                _LOGGER.info(
                    "chat concluido request_id=%s duration_ms=%.1f route=%s",
                    request_id,
                    elapsed_ms,
                    response.get("route", "unknown"),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                yield _sse(
                    "error",
                    {
                        "request_id": request_id,
                        "code": "invalid_request",
                        "detail": str(exc),
                    },
                )
            except AgenteIndisponivel as exc:
                yield _sse(
                    "error",
                    {
                        "request_id": request_id,
                        "code": "agent_unavailable",
                        "detail": str(exc),
                    },
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    routes = [
        Route("/", homepage, methods=["GET"]),
        Route("/api/chat/stream", chat_stream_api, methods=["POST"]),
        Route("/api/render", render_api, methods=["POST"]),
        Route("/api/status", status_api, methods=["GET"]),
        Route("/api/health", status_api, methods=["GET"]),
        Route("/api/results/e2", e2_api, methods=["GET"]),
        Route("/api/results/e3", e3_api, methods=["GET"]),
        Route("/api/reliability", reliability_api, methods=["GET"]),
        Route("/api/sources", sources_api, methods=["GET"]),
        Route("/api/version", version_api, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=STATIC_ROOT), name="static"),
        Mount(
            "/artifacts/comparison",
            app=StaticFiles(directory=COMPARISON),
            name="artifacts-comparison",
        ),
        Mount(
            "/artifacts/reliability",
            app=StaticFiles(directory=RELIABILITY),
            name="artifacts-reliability",
        ),
        Mount(
            "/artifacts/manifests",
            app=StaticFiles(directory=MANIFESTS),
            name="artifacts-manifests",
        ),
        Mount(
            "/sources/inversores-pv",
            app=StaticFiles(directory=LITERATURE),
            name="sources-literature",
        ),
    ]
    app = Starlette(
        debug=False,
        routes=routes,
        middleware=[Middleware(SecurityHeadersMiddleware)],
        lifespan=lifespan,
    )
    app.state.agent_adapter = adapter
    return app


app = create_app()


__all__ = ["app", "create_app"]
