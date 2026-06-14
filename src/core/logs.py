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
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.core.config import RAIZ_PROJETO

PASTA_LOGS = Path(RAIZ_PROJETO) / "logs"
ARQUIVO_LOG = PASTA_LOGS / "al_iado_pv.log"
_RAIZ_LOGGER = "al_iado_pv"
_configurado = False

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
        handler.setFormatter(_FormatadorSemEmoji(
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


def habilitar_console(nivel: int = logging.INFO) -> None:
    """
    Espelha os logs também no TERMINAL (formato curto, sem timestamp).

    Os módulos de ML logam em arquivo por padrão (terminal silencioso no app).
    Scripts executados manualmente (``python src/ml/autoencoder.py``) chamam
    isto no bloco ``__main__`` para o pesquisador continuar vendo o progresso.
    Também garante stdout/stderr em UTF-8 (Windows cp1252 quebra com acentos
    e emojis em prints/help). Idempotente.
    """
    try:
        from src.core.utils import configurar_saida_utf8

        configurar_saida_utf8()
    except Exception:  # noqa: BLE001 — nunca bloquear a execução manual
        pass
    configurar_logging()
    root = logging.getLogger(_RAIZ_LOGGER)
    ja_tem = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    )
    if not ja_tem:
        console = logging.StreamHandler()
        console.setLevel(nivel)
        console.setFormatter(_FormatadorSemEmoji("%(message)s"))
        root.addHandler(console)
