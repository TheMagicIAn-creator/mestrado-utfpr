"""Inicialização explícita da aplicação web canônica."""

from __future__ import annotations

import os


def main(asgi_app=None) -> None:
    """Inicia o servidor local com configuração previsível."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from src.core.utils import configurar_saida_utf8

    configurar_saida_utf8()

    if asgi_app is None:
        from src.webapp.app import app as asgi_app

    import uvicorn

    uvicorn.run(
        asgi_app,
        host=os.getenv("AL_IADO_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("AL_IADO_LOG_LEVEL", "info"),
    )
