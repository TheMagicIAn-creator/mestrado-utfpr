"""
Sprint 4 — logging estruturado (item 10.1).

get_logger escreve em logs/al_iado_pv.log (arquivo rotativo, não versionado).
"""

import logging

from src.core.logs import ARQUIVO_LOG, get_logger


def test_get_logger_escreve_no_arquivo():
    log = get_logger("teste")
    assert log.name == "al_iado_pv.teste"

    log.error("mensagem estruturada de teste 4711")
    for h in logging.getLogger("al_iado_pv").handlers:
        h.flush()

    assert ARQUIVO_LOG.exists()
    conteudo = ARQUIVO_LOG.read_text(encoding="utf-8")
    assert "4711" in conteudo
    assert "ERROR" in conteudo
    assert "al_iado_pv.teste" in conteudo
