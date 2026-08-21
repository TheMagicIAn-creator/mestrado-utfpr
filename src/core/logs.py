"""
logs.py — Al IAdo PV / Sprint 4 (robustez)

Logging estruturado no console. Um arquivo rotativo pode ser habilitado por
``AL_IADO_LOG_FILE=1`` quando uma execução precisar de persistência local.

Uso:
    from src.core.logs import get_logger
    log = get_logger("experimentos")
    log.info("rodando %s", chave)
    log.exception("falhou ao gerar gráfico")   # registra o traceback

O arquivo de log NÃO é versionado (ver .gitignore).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.core.config import RAIZ_PROJETO

PASTA_LOGS = Path(RAIZ_PROJETO) / "logs"
ARQUIVO_LOG = PASTA_LOGS / "al_iado_pv.log"
_RAIZ_LOGGER = "al_iado_pv"
_configurado = False
_erro_configuracao: str | None = None

# Emojis/pictogramas herdados dos prints do ML poluem o LOG (e atrapalham
# grep/Get-Content). Removemos do registro — a interface do chat NÃO passa por
# aqui, então os emojis seguem na conversa. Acentos (á, ç, ã) e box-drawing
# (═ ─) são PRESERVADOS: são texto legível, não símbolo.
_RX_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictogramas, emoticons, transporte, suplementares
    "☀-➿"          # símbolos diversos + dingbats (⚡ ✅ ✂ ☑)
    "⌀-⏿"          # técnicos (⏹ ⏳ ⌛)
    "⬀-⯿"          # setas/estrelas decorativas
    "︀-️"          # seletores de variação (emoji presentation)
    "™ℹ"           # ™ ℹ
    "]+"
)  # NÃO inclui box-drawing (─ ═), setas de texto (→) nem acentos — preservados


def limpar_simbolos(texto: str) -> str:
    """Remove emojis/pictogramas, preservando acentos e box-drawing."""
    return re.sub(r"[ \t]{2,}", " ", _RX_EMOJI.sub("", texto))


class _FormatadorSemEmoji(logging.Formatter):
    """Formatter que mantém o log em texto limpo (sem emojis)."""

    def format(self, record: logging.LogRecord) -> str:
        return limpar_simbolos(super().format(record))


def _arquivo_solicitado() -> bool:
    return os.getenv("AL_IADO_LOG_FILE", "").strip().lower() in {
        "1",
        "true",
        "sim",
        "yes",
    }


def configurar_logging(
    nivel: int = logging.INFO,
    *,
    arquivo: bool | None = None,
) -> None:
    """Configura console e, somente quando solicitado, arquivo rotativo."""
    global _configurado, _erro_configuracao
    try:
        root = logging.getLogger(_RAIZ_LOGGER)
        root.setLevel(nivel)
        if not any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, RotatingFileHandler)
            for handler in root.handlers
        ):
            console = logging.StreamHandler()
            console.setLevel(nivel)
            console.setFormatter(_FormatadorSemEmoji("%(levelname)s | %(name)s | %(message)s"))
            root.addHandler(console)
        usar_arquivo = _arquivo_solicitado() if arquivo is None else arquivo
        if usar_arquivo and not any(
            isinstance(handler, RotatingFileHandler) for handler in root.handlers
        ):
            PASTA_LOGS.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                ARQUIVO_LOG, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            handler.setFormatter(
                _FormatadorSemEmoji(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                )
            )
            root.addHandler(handler)
        root.propagate = False
    except Exception as exc:
        # Logging nunca deve derrubar a aplicação, mas a falha precisa aparecer.
        _erro_configuracao = str(exc)
        if sys.__stderr__ is not None:
            sys.__stderr__.write(f"[logging indisponível] {_erro_configuracao}\n")
    _configurado = True


def get_logger(nome: str) -> logging.Logger:
    """Retorna um logger filho (al_iado_pv.<nome>) já configurado."""
    configurar_logging()
    return logging.getLogger(f"{_RAIZ_LOGGER}.{nome}")


def adaptar_logger_como_print(logger: logging.Logger):
    """Cria um adaptador compativel com ``print`` para scripts de ML."""

    def registrar(*args, sep=" ", end="\n", flush=None):
        del end, flush  # aceitos por compatibilidade com chamadas existentes
        texto = sep.join(str(arg) for arg in args)
        if not texto.strip():
            return
        if texto.startswith("\r"):
            logger.debug(texto.strip())
            return
        logger.info(texto.rstrip("\n"))

    return registrar


def habilitar_console(nivel: int = logging.INFO) -> None:
    """
    Espelha os logs também no TERMINAL (formato curto, sem timestamp).

    Garante stdout/stderr em UTF-8 e eleva o nível do console. Idempotente.
    """
    try:
        from src.core.utils import configurar_saida_utf8

        configurar_saida_utf8()
    except Exception as exc:  # noqa: BLE001 — nunca bloquear a execução manual
        logging.getLogger(_RAIZ_LOGGER).debug(
            "não foi possível reconfigurar stdout/stderr: %s", exc
        )
    configurar_logging(nivel)
    root = logging.getLogger(_RAIZ_LOGGER)
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, RotatingFileHandler
        ):
            handler.setLevel(nivel)
