"""
logs.py — Al IAdo PV / Sprint 4 (robustez)

Logging estruturado com arquivo rotativo em logs/al_iado_pv.log. Substitui
exceções silenciosas por registros rastreáveis (módulo, operação, exceção).

Uso:
    from src.core.logs import get_logger
    log = get_logger("experimentos")
    log.info("rodando %s", chave)
    log.exception("falhou ao gerar gráfico")   # registra o traceback

O arquivo de log NÃO é versionado (ver .gitignore).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.core.config import RAIZ_PROJETO

PASTA_LOGS = Path(RAIZ_PROJETO) / "logs"
ARQUIVO_LOG = PASTA_LOGS / "al_iado_pv.log"
_RAIZ_LOGGER = "al_iado_pv"
_configurado = False


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Configura o handler rotativo uma única vez (idempotente)."""
    global _configurado
    if _configurado:
        return
    try:
        PASTA_LOGS.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            ARQUIVO_LOG, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        root = logging.getLogger(_RAIZ_LOGGER)
        root.setLevel(nivel)
        if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
            root.addHandler(handler)
        root.propagate = False
    except Exception:
        # Logging nunca deve derrubar a aplicação.
        pass
    _configurado = True


def get_logger(nome: str) -> logging.Logger:
    """Retorna um logger filho (al_iado_pv.<nome>) já configurado."""
    configurar_logging()
    return logging.getLogger(f"{_RAIZ_LOGGER}.{nome}")
