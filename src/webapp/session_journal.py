"""Registro e indexação das conversas da aplicação canônica."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Callable

from src.core.config import PASTA_CHROMADB, RAIZ_PROJETO
from src.core.logs import get_logger
from src.core.tempo import agora_local

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
_logger = get_logger("webapp.session_journal")


def _indexar(caminho: Path, modelo_embeddings) -> None:
    from src.conhecimento.indexador import indexar_sessao

    indexar_sessao(caminho, modelo_embeddings, PASTA_CHROMADB)


class SessionJournal:
    """Mantem um arquivo por sessao do navegador e o atualiza no RAG."""

    def __init__(
        self,
        pasta: Path | None = None,
        indexer: Callable[[Path, object], None] = _indexar,
    ):
        self._pasta = pasta or Path(RAIZ_PROJETO) / "notas" / "sessoes"
        self._indexer = indexer
        self._lock = threading.RLock()
        self._paths: dict[str, Path] = {}
        self._counts: dict[str, int] = {}

    @staticmethod
    def validar_session_id(session_id: str | None) -> str | None:
        valor = str(session_id or "").strip()
        if not valor:
            return None
        if not _SESSION_ID.fullmatch(valor):
            raise ValueError("session_id invalido")
        return valor

    def _path(self, session_id: str) -> Path:
        caminho = self._paths.get(session_id)
        if caminho is not None:
            return caminho
        agora = agora_local()
        self._pasta.mkdir(parents=True, exist_ok=True)
        caminho = self._pasta / (
            f"{agora:%Y-%m-%d_%H-%M-%S}_{session_id[:8]}_sessao_web.md"
        )
        caminho.write_text(
            (
                "---\n"
                f"data: {agora:%Y-%m-%d}\n"
                f"hora: {agora:%H:%M}\n"
                "tipo: sessao-web\n"
                "tags: [aliado, sessao, web, mestrado]\n"
                "---\n\n"
                f"# Sessão Web - {agora:%d/%m/%Y %H:%M}\n\n"
            ),
            encoding="utf-8",
        )
        self._paths[session_id] = caminho
        self._counts[session_id] = 0
        return caminho

    def record(
        self,
        session_id: str | None,
        question: str,
        answer: str,
        images: list[dict],
        modelo_embeddings,
    ) -> dict | None:
        session_id = self.validar_session_id(session_id)
        if session_id is None:
            return None

        with self._lock:
            caminho = self._path(session_id)
            numero = self._counts[session_id] + 1
            self._counts[session_id] = numero
            linhas_imagens = "\n".join(
                f"- {item.get('caption', 'Figura')}: "
                f"{item.get('url') or item.get('path') or ''}"
                for item in images
            )
            bloco = (
                f"---\n\n## Interacao {numero}\n\n"
                f"**Rodolfo:** {question}\n\n"
                f"**ALIAdo PV:**\n\n{answer}\n\n"
            )
            if linhas_imagens:
                bloco += f"**Imagens exibidas:**\n{linhas_imagens}\n\n"
            with caminho.open("a", encoding="utf-8") as arquivo:
                arquivo.write(bloco)

        try:
            self._indexer(caminho, modelo_embeddings)
        except Exception as exc:
            _logger.warning("sessão salva, mas não indexada: %s", exc)

        self._persistir_na_nuvem(caminho, numero)
        try:
            caminho_publico = caminho.relative_to(RAIZ_PROJETO)
        except ValueError:
            caminho_publico = caminho
        return {
            "id": session_id,
            "interaction": numero,
            "path": str(caminho_publico).replace("\\", "/"),
        }

    @staticmethod
    def _persistir_na_nuvem(caminho: Path, numero: int) -> None:
        try:
            passo = max(1, int(os.getenv("AL_IADO_CONSOLIDAR_A_CADA", "6")))
        except ValueError:
            passo = 6
        if numero % passo:
            return
        try:
            from src.conhecimento.persistencia_nuvem import (
                persistencia_ativa,
                persistir_arquivo,
            )

            if persistencia_ativa():
                persistir_arquivo(
                    caminho,
                    mensagem=f"chore(sessao): atualiza sessão web ({numero} interações)",
                    alvo="sessao",
                )
        except Exception as exc:
            _logger.warning("sessão não persistida na nuvem: %s", exc)
