"""Aplicação ASGI canônica do ALIAdo."""

from __future__ import annotations

import asyncio
import json
import os
import re
from ipaddress import ip_address
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
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
    e3_contract,
    reliability_contract,
    sources_contract,
    warm_contracts_background,
)
from src.webapp.rendering import render_agent_markdown, render_agent_messages
from src.webapp.library_service import (
    LibraryDuplicateError,
    LibraryError,
    LibraryNotFoundError,
    LibraryService,
    MAX_LIBRARY_PDF_BYTES,
    library_is_read_only,
)
from src.conhecimento.catalogo_bibliografico import (
    CATEGORIAS,
    IDIOMAS,
    CatalogoBibliograficoInvalido,
)

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


def homepage(_request: Request) -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


async def _contract_response(loader) -> JSONResponse:
    try:
        return JSONResponse(await run_in_threadpool(loader))
    except ContratoWebInvalido as exc:
        return _contract_error(exc)


async def e3_api(_request: Request) -> JSONResponse:
    return await _contract_response(e3_contract)


async def reliability_api(_request: Request) -> JSONResponse:
    return await _contract_response(reliability_contract)


async def sources_api(_request: Request) -> JSONResponse:
    return await _contract_response(sources_contract)


def version_api(_request: Request) -> JSONResponse:
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
    except ValueError as exc:
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


def _loopback_host(value: str | None) -> bool:
    host = str(value or "").strip().casefold().strip("[]")
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _same_origin(request: Request) -> tuple[bool, str | None]:
    origin = request.headers.get("origin")
    if not origin:
        return True, None
    try:
        parsed = urlsplit(origin)
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False, "A origem da requisicao e invalida."
    request_port = request.url.port or (
        443 if request.url.scheme == "https" else 80
    )
    matches = (
        parsed.scheme == request.url.scheme
        and parsed.hostname == request.url.hostname
        and origin_port == request_port
    )
    if not matches:
        return False, "A origem da requisicao nao corresponde ao ALIAdo local."
    return True, None


def _library_write_access(request: Request) -> tuple[bool, str | None]:
    read_only, reason = library_is_read_only()
    if read_only:
        return False, reason

    url_host = request.url.hostname
    if not _loopback_host(url_host):
        return False, "A biblioteca so pode ser alterada em localhost."

    client_host = request.client.host if request.client else ""
    if not _loopback_host(client_host) and client_host != "testclient":
        return False, "A conexao de escrita nao se originou do computador local."

    return _same_origin(request)


def create_app(
    agent_adapter: AgentAdapter | None = None,
    *,
    warm_on_startup: bool = True,
    library_service: LibraryService | None = None,
) -> Starlette:
    library = library_service or LibraryService()
    adapter = agent_adapter or AgentAdapter(library_service=library)
    adapter.configure_library_service(library)
    conversations = adapter.session_journal
    started_at = perf_counter()

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        if warm_on_startup:
            adapter.warm_background()
            warm_contracts_background()
        yield
        library.close()

    def status_api(_request: Request) -> JSONResponse:
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
        except ValueError as exc:
            return JSONResponse(
                {"error": "invalid_request", "detail": str(exc)},
                status_code=400,
            )

        request_id = uuid4().hex[:12]
        library_write_allowed, library_write_reason = _library_write_access(request)

        async def event_stream():
            started = perf_counter()
            yield _sse(
                "status",
                {
                    "request_id": request_id,
                    "phase": "recebido",
                    "message": "Preparando resposta",
                },
            )
            try:
                yield _sse(
                    "status",
                    {
                        "request_id": request_id,
                        "phase": "consultando",
                        "message": "Consultando contexto acadêmico",
                    },
                )
                chunks: asyncio.Queue[str] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def publish_chunk(text: str) -> None:
                    if text:
                        loop.call_soon_threadsafe(chunks.put_nowait, text)

                answer_task = asyncio.create_task(
                    run_in_threadpool(
                        adapter.answer,
                        message,
                        history,
                        attachments,
                        session_id,
                        on_chunk=publish_chunk,
                        library_write_allowed=library_write_allowed,
                        library_write_reason=library_write_reason,
                    )
                )
                streamed = False
                while not answer_task.done() or not chunks.empty():
                    if await request.is_disconnected():
                        _LOGGER.info("chat cancelado request_id=%s", request_id)
                        answer_task.cancel()
                        return
                    try:
                        chunk = await asyncio.wait_for(chunks.get(), timeout=0.1)
                    except TimeoutError:
                        continue
                    streamed = True
                    yield _sse("delta", {"request_id": request_id, "text": chunk})

                response = await answer_task
                answer = str(response["answer"])
                if not streamed:
                    for chunk in _text_chunks(answer):
                        if await request.is_disconnected():
                            _LOGGER.info("chat cancelado request_id=%s", request_id)
                            return
                        yield _sse(
                            "delta", {"request_id": request_id, "text": chunk}
                        )
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
            except ValueError as exc:
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

    async def conversations_api(request: Request) -> JSONResponse:
        status = str(request.query_params.get("status") or "active")
        try:
            items = await run_in_threadpool(conversations.list_conversations, status)
        except ValueError as exc:
            return JSONResponse(
                {"error": "invalid_conversation_status", "detail": str(exc)},
                status_code=400,
            )
        return JSONResponse(
            {
                "conversations": items,
                "status": status,
                "memory_policy": "non_destructive_audit_log",
            }
        )

    async def conversation_detail_api(request: Request) -> JSONResponse:
        try:
            conversation = await run_in_threadpool(
                conversations.get_conversation,
                request.path_params["session_id"],
            )
        except (KeyError, ValueError):
            return JSONResponse(
                {
                    "error": "conversation_not_found",
                    "detail": "Conversa nao encontrada.",
                },
                status_code=404,
            )
        return JSONResponse({"conversation": conversation})

    def conversation_write_denied(request: Request) -> JSONResponse | None:
        allowed, reason = _same_origin(request)
        if allowed:
            return None
        return JSONResponse(
            {"error": "cross_origin_write", "detail": reason},
            status_code=403,
        )

    async def conversation_patch_api(request: Request) -> JSONResponse:
        denied = conversation_write_denied(request)
        if denied:
            return denied
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("O corpo JSON deve ser um objeto.")
            conversation = await run_in_threadpool(
                conversations.rename,
                request.path_params["session_id"],
                str(payload.get("title") or ""),
            )
        except KeyError:
            return JSONResponse(
                {
                    "error": "conversation_not_found",
                    "detail": "Conversa nao encontrada.",
                },
                status_code=404,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return JSONResponse(
                {"error": "invalid_conversation", "detail": str(exc)},
                status_code=400,
            )
        return JSONResponse({"conversation": conversation})

    async def conversation_status_api(request: Request, status: str) -> JSONResponse:
        denied = conversation_write_denied(request)
        if denied:
            return denied
        action = {
            "archived": conversations.archive,
            "active": conversations.restore,
            "deleted": conversations.delete,
        }[status]
        try:
            conversation = await run_in_threadpool(
                action,
                request.path_params["session_id"],
            )
        except (KeyError, ValueError):
            return JSONResponse(
                {
                    "error": "conversation_not_found",
                    "detail": "Conversa nao encontrada.",
                },
                status_code=404,
            )
        return JSONResponse({"conversation": conversation})

    async def conversation_archive_api(request: Request) -> JSONResponse:
        return await conversation_status_api(request, "archived")

    async def conversation_restore_api(request: Request) -> JSONResponse:
        return await conversation_status_api(request, "active")

    async def conversation_delete_api(request: Request) -> JSONResponse:
        return await conversation_status_api(request, "deleted")

    async def library_api(request: Request) -> JSONResponse:
        try:
            catalog, provenance = await asyncio.gather(
                run_in_threadpool(library.catalog),
                run_in_threadpool(sources_contract),
            )
        except (CatalogoBibliograficoInvalido, ContratoWebInvalido) as exc:
            return JSONResponse(
                {"error": "library_unavailable", "detail": str(exc)},
                status_code=503,
            )
        writable, reason = _library_write_access(request)
        return JSONResponse(
            {
                **catalog,
                "writable": writable,
                "write_policy": {
                    "scope": "local_loopback_only",
                    "git_automation": False,
                    "reason": reason,
                },
                "categories": list(CATEGORIAS),
                "languages": list(IDIOMAS),
                "provenance": provenance,
            }
        )

    def write_denied(request: Request) -> JSONResponse | None:
        allowed, reason = _library_write_access(request)
        if allowed:
            return None
        return JSONResponse(
            {"error": "library_read_only", "detail": reason}, status_code=403
        )

    async def library_add_api(request: Request) -> JSONResponse:
        denied = write_denied(request)
        if denied:
            return denied
        try:
            form = await request.form()
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise LibraryError("Selecione um arquivo PDF.")
            try:
                data = await upload.read(MAX_LIBRARY_PDF_BYTES + 1)
                metadata = {
                    key: form.get(key)
                    for key in (
                        "title",
                        "authors",
                        "year",
                        "category",
                        "language",
                    )
                    if form.get(key) not in {None, ""}
                }
                job = await run_in_threadpool(
                    library.queue_pdf,
                    upload.filename or "fonte.pdf",
                    data,
                    metadata,
                )
            finally:
                await upload.close()
            return JSONResponse({"job": job}, status_code=202)
        except LibraryDuplicateError as exc:
            return JSONResponse(
                {
                    "error": "duplicate_pdf",
                    "detail": str(exc),
                    "document": exc.document,
                },
                status_code=409,
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": "invalid_library_pdf", "detail": str(exc)},
                status_code=400,
            )

    async def library_patch_api(request: Request) -> JSONResponse:
        denied = write_denied(request)
        if denied:
            return denied
        try:
            patch = await request.json()
            if not isinstance(patch, dict):
                raise LibraryError("O corpo JSON deve ser um objeto.")
            document = await run_in_threadpool(
                library.update_document,
                request.path_params["source_id"],
                patch,
            )
            return JSONResponse({"document": document})
        except LibraryNotFoundError as exc:
            return JSONResponse(
                {"error": "source_not_found", "detail": str(exc)},
                status_code=404,
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": "invalid_metadata", "detail": str(exc)},
                status_code=400,
            )

    async def library_reindex_api(request: Request) -> JSONResponse:
        denied = write_denied(request)
        if denied:
            return denied
        try:
            job = await run_in_threadpool(
                library.queue_reindex, request.path_params["source_id"]
            )
            return JSONResponse({"job": job}, status_code=202)
        except LibraryNotFoundError as exc:
            return JSONResponse(
                {"error": "source_not_found", "detail": str(exc)},
                status_code=404,
            )

    def library_job_api(request: Request) -> JSONResponse:
        try:
            job = library.get_job(request.path_params["job_id"])
            return JSONResponse({"job": job})
        except LibraryNotFoundError as exc:
            return JSONResponse(
                {"error": "job_not_found", "detail": str(exc)},
                status_code=404,
            )

    routes = [
        Route("/", homepage, methods=["GET"]),
        Route("/api/chat/stream", chat_stream_api, methods=["POST"]),
        Route("/api/render", render_api, methods=["POST"]),
        Route("/api/conversations", conversations_api, methods=["GET"]),
        Route(
            "/api/conversations/{session_id}",
            conversation_detail_api,
            methods=["GET"],
        ),
        Route(
            "/api/conversations/{session_id}",
            conversation_patch_api,
            methods=["PATCH"],
        ),
        Route(
            "/api/conversations/{session_id}",
            conversation_delete_api,
            methods=["DELETE"],
        ),
        Route(
            "/api/conversations/{session_id}/archive",
            conversation_archive_api,
            methods=["POST"],
        ),
        Route(
            "/api/conversations/{session_id}/restore",
            conversation_restore_api,
            methods=["POST"],
        ),
        Route("/api/status", status_api, methods=["GET"]),
        Route("/api/health", status_api, methods=["GET"]),
        Route("/api/results/e3", e3_api, methods=["GET"]),
        Route("/api/reliability", reliability_api, methods=["GET"]),
        Route("/api/library", library_api, methods=["GET"]),
        Route("/api/library", library_add_api, methods=["POST"]),
        Route(
            "/api/library/{source_id}", library_patch_api, methods=["PATCH"]
        ),
        Route(
            "/api/library/{source_id}/reindex",
            library_reindex_api,
            methods=["POST"],
        ),
        Route(
            "/api/library/jobs/{job_id}", library_job_api, methods=["GET"]
        ),
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
        Mount(
            "/library-files",
            app=StaticFiles(directory=library.literature_root),
            name="library-files",
        ),
    ]
    app = Starlette(
        debug=False,
        routes=routes,
        middleware=[
            Middleware(SecurityHeadersMiddleware),
            Middleware(GZipMiddleware, minimum_size=1000),
        ],
        lifespan=lifespan,
    )
    app.state.agent_adapter = adapter
    app.state.library_service = library
    return app


app = create_app()


__all__ = ["app", "create_app"]
