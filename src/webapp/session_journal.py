"""Registro e indexação das conversas da aplicação canônica."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Callable

from src.core.config import PASTA_CHROMADB, RAIZ_PROJETO
from src.core.logs import get_logger
from src.core.tempo import agora_local

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
_CONVERSATION_STATES = {"active", "archived", "deleted"}
_logger = get_logger("webapp.session_journal")


def _indexar(caminho: Path, modelo_embeddings) -> None:
    from src.conhecimento.indexador import indexar_sessao

    indexar_sessao(caminho, modelo_embeddings, PASTA_CHROMADB)


class SessionJournal:
    """Mantem um arquivo por conversa e um catalogo de estado nao destrutivo."""

    def __init__(
        self,
        pasta: Path | None = None,
        indexer: Callable[[Path, object], None] = _indexar,
    ):
        self._pasta = pasta or Path(RAIZ_PROJETO) / "notas" / "sessoes"
        self._catalog_path = self._pasta / "catalogo_conversas.json"
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

    @staticmethod
    def _title(question: str) -> str:
        compact = re.sub(r"\s+", " ", str(question or "")).strip()
        return compact[:80] or "Nova conversa"

    def _empty_catalog(self) -> dict:
        return {"version": 1, "conversations": []}

    def _load_catalog(self) -> dict:
        if not self._catalog_path.is_file():
            return self._empty_catalog()
        try:
            payload = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("catalogo de conversas indisponivel: %s", exc)
            return self._empty_catalog()
        conversations = payload.get("conversations")
        if payload.get("version") != 1 or not isinstance(conversations, list):
            _logger.warning("catalogo de conversas com schema invalido")
            return self._empty_catalog()
        return payload

    def _save_catalog(self, payload: dict) -> None:
        self._pasta.mkdir(parents=True, exist_ok=True)
        temporary = self._catalog_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self._catalog_path)

    @staticmethod
    def _entry(catalog: dict, session_id: str) -> dict | None:
        return next(
            (
                item
                for item in catalog.get("conversations", [])
                if item.get("id") == session_id
            ),
            None,
        )

    def _catalog_path_value(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self._pasta.parent.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    def _path_from_entry(self, entry: dict) -> Path | None:
        raw = str(entry.get("path") or "").strip()
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._pasta.parent / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self._pasta.parent.resolve())
        except (OSError, ValueError):
            return None
        return resolved

    def _path(self, session_id: str, question: str = "") -> Path:
        caminho = self._paths.get(session_id)
        if caminho is not None:
            return caminho

        catalog = self._load_catalog()
        entry = self._entry(catalog, session_id)
        if entry is not None:
            existing = self._path_from_entry(entry)
            if existing is not None and existing.is_file():
                self._paths[session_id] = existing
                self._counts[session_id] = int(entry.get("interaction_count") or 0)
                return existing

        agora = agora_local()
        self._pasta.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
        caminho = self._pasta / (
            f"{agora:%Y-%m-%d_%H-%M-%S}_{digest}_sessao_web.md"
        )
        caminho.write_text(
            (
                "---\n"
                f"data: {agora:%Y-%m-%d}\n"
                f"hora: {agora:%H:%M}\n"
                "tipo: sessao-web\n"
                f"session_id: {session_id}\n"
                "estado: ativa\n"
                "tags: [aliado, sessao, web, mestrado]\n"
                "---\n\n"
                f"# {self._title(question)}\n\n"
            ),
            encoding="utf-8",
        )
        self._paths[session_id] = caminho
        self._counts[session_id] = 0
        return caminho

    def _update_entry(
        self,
        session_id: str,
        *,
        path: Path | None = None,
        title: str | None = None,
        status: str | None = None,
        interaction_count: int | None = None,
    ) -> dict:
        catalog = self._load_catalog()
        entry = self._entry(catalog, session_id)
        now = agora_local().isoformat()
        if entry is None:
            entry = {
                "id": session_id,
                "title": title or "Nova conversa",
                "status": status or "active",
                "created_at": now,
                "updated_at": now,
                "interaction_count": interaction_count or 0,
                "path": self._catalog_path_value(path) if path else "",
            }
            catalog["conversations"].append(entry)
        else:
            if title and entry.get("title") in {None, "", "Nova conversa"}:
                entry["title"] = title
            if path is not None:
                entry["path"] = self._catalog_path_value(path)
            if status is not None:
                entry["status"] = status
            if interaction_count is not None:
                entry["interaction_count"] = interaction_count
            entry["updated_at"] = now
        self._save_catalog(catalog)
        return dict(entry)

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
            caminho = self._path(session_id, question)
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
            entry = self._update_entry(
                session_id,
                path=caminho,
                title=self._title(question),
                interaction_count=numero,
            )

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
            "title": entry["title"],
            "status": entry["status"],
        }

    def list_conversations(self, status: str = "active") -> list[dict]:
        if status not in {"active", "archived"}:
            raise ValueError("status deve ser active ou archived")
        with self._lock:
            catalog = self._load_catalog()
            items = [
                {
                    "id": item.get("id"),
                    "title": item.get("title") or "Nova conversa",
                    "status": item.get("status") or "active",
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "interaction_count": int(item.get("interaction_count") or 0),
                }
                for item in catalog.get("conversations", [])
                if item.get("status", "active") == status
            ]
        return sorted(
            items,
            key=lambda item: str(item.get("updated_at") or ""),
            reverse=True,
        )

    @staticmethod
    def _messages_from_markdown(text: str) -> list[dict[str, str]]:
        blocks = re.split(
            r"\n---\s*\n+## Intera(?:cao|ção)\s+\d+\s*\n+",
            text,
            flags=re.IGNORECASE,
        )[1:]
        messages: list[dict[str, str]] = []
        for block in blocks:
            match = re.match(
                r"\*\*Rodolfo:\*\*\s*(.*?)\n\n"
                r"\*\*ALIAdo(?: PV)?:\*\*\s*\n\n(.*)",
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if not match:
                continue
            question = match.group(1).strip()
            answer = re.split(
                r"\n\n\*\*Imagens exibidas:\*\*",
                match.group(2),
                maxsplit=1,
            )[0].strip()
            if question:
                messages.append({"role": "user", "content": question})
            if answer:
                messages.append({"role": "assistant", "content": answer})
        return messages

    def get_conversation(self, session_id: str) -> dict:
        session_id = self.validar_session_id(session_id) or ""
        with self._lock:
            catalog = self._load_catalog()
            entry = self._entry(catalog, session_id)
            if entry is None or entry.get("status") == "deleted":
                raise KeyError(session_id)
            path = self._path_from_entry(entry)
            text = path.read_text(encoding="utf-8") if path and path.is_file() else ""
            return {
                "id": session_id,
                "title": entry.get("title") or "Nova conversa",
                "status": entry.get("status") or "active",
                "created_at": entry.get("created_at"),
                "updated_at": entry.get("updated_at"),
                "interaction_count": int(entry.get("interaction_count") or 0),
                "messages": self._messages_from_markdown(text),
            }

    def set_status(self, session_id: str, status: str) -> dict:
        session_id = self.validar_session_id(session_id) or ""
        if status not in _CONVERSATION_STATES:
            raise ValueError("estado de conversa invalido")
        with self._lock:
            catalog = self._load_catalog()
            entry = self._entry(catalog, session_id)
            if entry is None or entry.get("status") == "deleted":
                raise KeyError(session_id)
            updated = self._update_entry(session_id, status=status)
        self._persistir_estado_na_nuvem()
        return {
            "id": updated["id"],
            "title": updated.get("title") or "Nova conversa",
            "status": updated["status"],
            "memory_retained": True,
        }

    def archive(self, session_id: str) -> dict:
        return self.set_status(session_id, "archived")

    def restore(self, session_id: str) -> dict:
        return self.set_status(session_id, "active")

    def delete(self, session_id: str) -> dict:
        """Oculta a conversa sem remover o transcrito da memoria auditavel."""
        return self.set_status(session_id, "deleted")

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

    def _persistir_estado_na_nuvem(self) -> None:
        try:
            from src.conhecimento.persistencia_nuvem import (
                persistencia_ativa,
                persistir_arquivo,
            )

            if persistencia_ativa() and self._catalog_path.is_file():
                persistir_arquivo(
                    self._catalog_path,
                    mensagem="chore(sessao): atualiza catalogo de conversas",
                    alvo="sessao",
                )
        except Exception as exc:
            _logger.warning("catalogo de conversas nao persistido na nuvem: %s", exc)
