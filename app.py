"""Entrada Streamlit do Al IAdo PV."""

# ANTES de qualquer import pesado: permitir runtimes OpenMP duplicados
# (torch/numpy/onnxruntime/Orange) para não crashar com access violation no
# Windows durante o carregamento dos modelos.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Blindar stdout/stderr contra emoji no Windows (cp1252).
from src.core.utils import configurar_saida_utf8

configurar_saida_utf8()

from src.interface.streamlit_app import main


if __name__ == "__main__":
    main()
