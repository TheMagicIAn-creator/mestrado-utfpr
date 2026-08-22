"""Identidade configuravel do pesquisador na interface do ALIAdo."""

from __future__ import annotations

import os


NOME_PESQUISADOR_PADRAO = "Rodolfo"


def nome_pesquisador() -> str:
    """Retorna um nome curto e seguro para saudacoes e rotulos locais."""
    nome = " ".join(
        os.getenv("AL_IADO_USER_NAME", NOME_PESQUISADOR_PADRAO).split()
    ).strip()
    return (nome or NOME_PESQUISADOR_PADRAO)[:80]


__all__ = ["NOME_PESQUISADOR_PADRAO", "nome_pesquisador"]
