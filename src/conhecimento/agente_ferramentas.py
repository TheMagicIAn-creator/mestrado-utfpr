"""
agente_ferramentas.py — compatibilidade retroativa.
Todo o conteúdo foi consolidado em ferramentas.py.
"""
from src.conhecimento.ferramentas import (  # noqa: F401
    ESPEC_FERRAMENTAS,
    executar_ferramenta,
    decidir_acao,
    comentar_resultado,
    processar_com_ferramentas,
)
