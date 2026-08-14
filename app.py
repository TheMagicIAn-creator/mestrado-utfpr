"""Entrada ASGI da aplicacao web ALIAdo PV."""

# ANTES de qualquer import pesado: permitir runtimes OpenMP duplicados
# (torch/numpy/onnxruntime/Orange) para não crashar com access violation no
# Windows durante o carregamento dos modelos.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Blindar stdout/stderr contra caracteres fora do cp1252 no Windows.
from src.core.utils import configurar_saida_utf8

configurar_saida_utf8()

from src.webapp.app import app  # noqa: E402,I001 - ambiente precede imports pesados


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("AL_IADO_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("AL_IADO_LOG_LEVEL", "info"),
    )
