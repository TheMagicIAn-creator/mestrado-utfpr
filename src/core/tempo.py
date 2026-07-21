"""Relogio central do projeto, independente do fuso do servidor."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


FUSO_PADRAO = "America/Sao_Paulo"


def fuso_projeto() -> ZoneInfo:
    nome = os.getenv("AL_IADO_TIMEZONE", FUSO_PADRAO).strip() or FUSO_PADRAO
    try:
        return ZoneInfo(nome)
    except ZoneInfoNotFoundError:
        return ZoneInfo(FUSO_PADRAO)


def agora_local() -> datetime:
    """Data/hora consciente do fuso configurado para o pesquisador."""
    return datetime.now(fuso_projeto())


def agora_utc() -> datetime:
    """Data/hora UTC para trilhas de auditoria explicitamente universais."""
    return datetime.now(timezone.utc)
