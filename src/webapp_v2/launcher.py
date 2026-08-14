"""Inicializacao explicita da aplicacao web V2."""

from __future__ import annotations

import os
import sys


def executando_no_streamlit() -> bool:
    """Detecta o executor legado sem importar Streamlit como dependencia."""
    return any(
        nome == "streamlit" or nome.startswith("streamlit.")
        for nome in sys.modules
    )


def bloquear_execucao_streamlit() -> None:
    if executando_no_streamlit():
        raise RuntimeError(
            "app.py pertence ao ALIAdo PV Web V2 e nao pode ser executado pelo "
            "Streamlit. Use: python -m src.webapp_v2"
        )


def main(asgi_app=None) -> None:
    """Inicia o servidor local V2 com configuracao previsivel."""
    bloquear_execucao_streamlit()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from src.core.utils import configurar_saida_utf8

    configurar_saida_utf8()

    if asgi_app is None:
        from src.webapp_v2.app import app as asgi_app

    import uvicorn

    uvicorn.run(
        asgi_app,
        host=os.getenv("AL_IADO_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("AL_IADO_LOG_LEVEL", "info"),
    )
