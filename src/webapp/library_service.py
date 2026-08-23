"""Biblioteca bibliografica local, editavel e reconstruivel do ALIAdo."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from pypdf import PdfReader

from src.conhecimento.catalogo_bibliografico import (
    CATEGORIAS,
    IDIOMAS,
    CatalogoBibliograficoInvalido,
    CatalogoStore,
    limpar_metadado,
    sha256_arquivo,
)
from src.core.utils import parsear_nome_arquivo


MAX_LIBRARY_PDF_BYTES = 15 * 1024 * 1024
_ACTIVE_STATES = {"queued", "running"}
_TRUE_VALUES = {"1", "true", "yes", "sim", "on"}


class LibraryError(ValueError):
    """Erro de entrada ou operacao da biblioteca."""


class LibraryDuplicateError(LibraryError):
    def __init__(self, document: dict):
        super().__init__("Este PDF ja esta catalogado pelo mesmo SHA-256.")
        self.document = document


class LibraryNotFoundError(LibraryError):
    """Fonte ou trabalho de indexacao nao encontrado."""


def library_is_read_only() -> tuple[bool, str | None]:
    explicit = os.getenv("AL_IADO_LIBRARY_READ_ONLY", "").strip().casefold()
    deployment = os.getenv("AL_IADO_DEPLOYMENT_MODE", "local").strip().casefold()
    if explicit in _TRUE_VALUES:
        return True, "A escrita foi desativada por AL_IADO_LIBRARY_READ_ONLY."
    if deployment in {"cloud", "nuvem", "public", "readonly", "read-only"}:
        return True, "Implantacoes em nuvem mantem a biblioteca somente para leitura."
    return False, None


def _agora_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _authors(value) -> list[str]:
    if isinstance(value, list):
        values = value
    else:
        values = re.split(r"\s*;\s*", limpar_metadado(value, limite=1200))
    result = []
    for value in values:
        author = limpar_metadado(value, limite=220)
        if author and author not in result:
            result.append(author)
    return result or ["Autor desconhecido"]


def _year(value) -> tuple[int | None, str]:
    text = limpar_metadado(value, limite=20)
    if re.fullmatch(r"(?:18|19|20)\d{2}", text):
        return int(text), "informado"
    return None, "desconhecido"


def _citation(authors: list[str], year: int | None, title: str) -> str:
    author = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
    return f"{author} ({year if year is not None else 's.d.'}) - {title}"


def _safe_title(value: str) -> bool:
    text = value.casefold()
    rejected = ("template for", "microsoft word", "untitled", "acrobat")
    return len(value) >= 12 and not any(marker in text for marker in rejected)


def _safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise LibraryError("Caminho da fonte fora da biblioteca.")
    return candidate


class LibraryService:
    """Coordena catalogo, PDFs, ChromaDB e snapshot em uma fila serial."""

    def __init__(
        self,
        *,
        catalog_path: Path | None = None,
        literature_root: Path | None = None,
        chroma_path: Path | None = None,
        snapshot_path: Path | None = None,
        staging_root: Path | None = None,
        start_jobs: bool = True,
        model_factory=None,
        indexer=None,
        snapshot_exporter=None,
    ) -> None:
        from src.core.config import (
            ARQUIVO_INDICE_LITERATURA,
            PASTA_CHROMADB,
            PASTA_LITERATURA,
        )

        self.literature_root = Path(literature_root or PASTA_LITERATURA).resolve()
        self.catalog_path = Path(
            catalog_path or self.literature_root / "catalogo.json"
        ).resolve()
        self.chroma_path = Path(chroma_path or PASTA_CHROMADB).resolve()
        self.snapshot_path = Path(
            snapshot_path or ARQUIVO_INDICE_LITERATURA
        ).resolve()
        self.staging_root = Path(
            staging_root
            or Path(tempfile.gettempdir()) / "aliado-library-uploads"
        ).resolve()
        self.store = CatalogoStore(self.catalog_path)
        self.start_jobs = bool(start_jobs)
        self.model_factory = model_factory or self._default_model_factory
        self.indexer = indexer or self._default_indexer
        self.snapshot_exporter = snapshot_exporter or self._export_snapshot
        self._jobs: dict[str, dict] = {}
        self._active_by_source: dict[str, str] = {}
        self._lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._model = None
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="aliado-library")
            if self.start_jobs
            else None
        )

    @staticmethod
    def _default_model_factory():
        from src.conhecimento.embeddings import criar_modelo_embeddings

        return criar_modelo_embeddings(modo_consulta=True)

    @staticmethod
    def _default_indexer(*args, **kwargs):
        from src.conhecimento.indexador import indexar_pdf_unico

        return indexar_pdf_unico(*args, **kwargs)

    def _get_model(self):
        with self._model_lock:
            if self._model is None:
                self._model = self.model_factory()
            return self._model

    @staticmethod
    def _public_job(job: dict) -> dict:
        return {key: value for key, value in job.items() if not key.startswith("_")}

    def _set_job(self, job_id: str, **changes) -> dict:
        with self._lock:
            job = self._jobs[job_id]
            job.update(changes)
            job["updated_at"] = _agora_utc()
            return self._public_job(job)

    def _new_job(self, kind: str, source_id: str, **private) -> dict:
        job_id = uuid4().hex
        now = _agora_utc()
        job = {
            "job_id": job_id,
            "kind": kind,
            "source_id": source_id,
            "state": "queued",
            "phase": "queued",
            "progress": 0,
            "message": "Aguardando processamento local.",
            "warnings": [],
            "created_at": now,
            "updated_at": now,
            **{f"_{key}": value for key, value in private.items()},
        }
        self._jobs[job_id] = job
        self._active_by_source[source_id] = job_id
        self._trim_jobs()
        if self._executor is not None:
            self._executor.submit(self.run_job, job_id)
        return self._public_job(job)

    def _trim_jobs(self) -> None:
        completed = [
            job
            for job in self._jobs.values()
            if job["state"] not in _ACTIVE_STATES
        ]
        for job in sorted(completed, key=lambda item: item["updated_at"])[:-96]:
            self._jobs.pop(job["job_id"], None)

    def catalog(self) -> dict:
        payload = self.store.load()
        documents = []
        for item in payload["documents"]:
            document = dict(item)
            document["url"] = "/library-files/" + quote(
                str(item["relative_path"]), safe="/"
            )
            documents.append(document)
        return {**payload, "documents": documents}

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            try:
                return self._public_job(self._jobs[job_id])
            except KeyError as exc:
                raise LibraryNotFoundError("Trabalho nao encontrado.") from exc

    def queue_pdf(self, filename: str, data: bytes, metadata: dict | None = None) -> dict:
        clean_name = Path(str(filename or "fonte.pdf").replace("\\", "/")).name
        if Path(clean_name).suffix.casefold() != ".pdf":
            raise LibraryError("A biblioteca aceita somente arquivos PDF.")
        if not data or len(data) > MAX_LIBRARY_PDF_BYTES:
            raise LibraryError("O PDF deve ter no maximo 15 MB.")
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise LibraryError("O arquivo enviado nao possui assinatura PDF valida.")

        source_id = _hash_bytes(data)
        try:
            raise LibraryDuplicateError(self.store.get(source_id))
        except KeyError:
            pass

        with self._lock:
            active = self._active_by_source.get(source_id)
            if active and self._jobs[active]["state"] in _ACTIVE_STATES:
                return self._public_job(self._jobs[active])

            self.staging_root.mkdir(parents=True, exist_ok=True)
            staging = self.staging_root / f"{source_id}-{uuid4().hex[:8]}.pdf"
            staging.write_bytes(data)
            return self._new_job(
                "add",
                source_id,
                staging_path=str(staging),
                original_name=clean_name,
                metadata=dict(metadata or {}),
            )

    def queue_reindex(self, source_id: str) -> dict:
        try:
            self.store.get(source_id)
        except KeyError as exc:
            raise LibraryNotFoundError("Fonte nao encontrada.") from exc
        with self._lock:
            active = self._active_by_source.get(source_id)
            if active and self._jobs[active]["state"] in _ACTIVE_STATES:
                return self._public_job(self._jobs[active])
            return self._new_job("reindex", source_id)

    def run_job(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise LibraryNotFoundError("Trabalho nao encontrado.")
            job = self._jobs[job_id]
            if job["state"] != "queued":
                return self._public_job(job)
            job["state"] = "running"
            job["phase"] = "preparing"
            job["progress"] = 5
            job["message"] = "Validando metadados e arquivo."
            job["updated_at"] = _agora_utc()

        try:
            warnings = (
                self._run_add(job_id)
                if job["kind"] == "add"
                else self._run_reindex(job_id)
            )
            return self._set_job(
                job_id,
                state="completed",
                phase="completed",
                progress=100,
                message=(
                    "Fonte pronta; o snapshot requer revisao."
                    if warnings
                    else "Fonte e indices atualizados."
                ),
                warnings=warnings,
            )
        except Exception as exc:
            return self._set_job(
                job_id,
                state="failed",
                phase="failed",
                progress=100,
                message=limpar_metadado(exc, limite=500) or "Falha no processamento.",
            )
        finally:
            with self._lock:
                if self._active_by_source.get(job["source_id"]) == job_id:
                    self._active_by_source.pop(job["source_id"], None)
                staging = job.get("_staging_path")
            if staging:
                Path(staging).unlink(missing_ok=True)

    def _metadata_for_upload(self, job: dict, staging: Path) -> dict:
        requested = dict(job.get("_metadata") or {})
        fallback = parsear_nome_arquivo(job["_original_name"])
        internal_title = ""
        internal_author = ""
        text = ""
        warnings: list[str] = []
        try:
            reader = PdfReader(str(staging))
            internal = reader.metadata or {}
            internal_title = limpar_metadado(internal.get("/Title"), limite=800)
            internal_author = limpar_metadado(internal.get("/Author"), limite=800)
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages[:3]
            ).strip()
        except Exception:
            warnings.append("metadado_pdf_interno_ilegivel")

        title = limpar_metadado(requested.get("title"), limite=800)
        if not title and _safe_title(internal_title):
            title = internal_title
        if not title:
            title = limpar_metadado(fallback.get("titulo") or Path(job["_original_name"]).stem)

        requested_authors = requested.get("authors")
        if requested_authors:
            authors = _authors(requested_authors)
        elif internal_author:
            authors = _authors(re.sub(r"\s*(?:,|\band\b|&)\s*", ";", internal_author))
        else:
            authors = _authors(fallback.get("autor"))

        year, year_status = _year(requested.get("year") or fallback.get("ano"))
        category = limpar_metadado(requested.get("category"), limite=80)
        if category and category not in CATEGORIAS:
            raise LibraryError("Categoria bibliografica invalida.")
        if not category:
            from src.conhecimento.processador_pdf import classificar_tema

            category = classificar_tema(job["_original_name"], text)

        language = limpar_metadado(requested.get("language"), limite=20).casefold()
        if language and language not in IDIOMAS:
            raise LibraryError("Idioma bibliografico invalido.")
        if not language:
            from src.conhecimento.indexador import detectar_idioma_texto

            language = detectar_idioma_texto(text) if text else "desconhecido"
            if language not in IDIOMAS:
                language = "desconhecido"

        return {
            "title": title,
            "authors": authors,
            "year": year,
            "year_status": year_status,
            "category": category,
            "language": language,
            "warnings": warnings,
            "origin": "upload_user" if any(requested.values()) else "upload_pdf",
        }

    def _destination_for(self, metadata: dict, source_id: str) -> Path:
        from src.conhecimento.processador_pdf import gerar_nome_padronizado

        year = str(metadata["year"] or "0000")
        filename = gerar_nome_padronizado(
            metadata["authors"][0], metadata["title"], year
        )
        relative = f"{metadata['category']}/{filename}"
        destination = _safe_path(self.literature_root, relative)
        if destination.exists() and sha256_arquivo(destination) != source_id:
            destination = destination.with_name(
                f"{destination.stem}-{source_id[:8]}{destination.suffix}"
            )
        return destination

    def _run_add(self, job_id: str) -> list[str]:
        with self._lock:
            job = self._jobs[job_id]
        staging = Path(job["_staging_path"])
        metadata = self._metadata_for_upload(job, staging)
        destination = self._destination_for(metadata, job["source_id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".{uuid4().hex}.tmp")

        self._set_job(
            job_id,
            phase="copying",
            progress=20,
            message="Copiando PDF para a categoria selecionada.",
        )
        shutil.copyfile(staging, temporary)
        os.replace(temporary, destination)

        relative = destination.relative_to(self.literature_root).as_posix()
        item = {
            "source_id": job["source_id"],
            "sha256": job["source_id"],
            "file_name": destination.name,
            "relative_path": relative,
            "title": metadata["title"],
            "authors": metadata["authors"],
            "year": metadata["year"],
            "year_status": metadata["year_status"],
            "category": metadata["category"],
            "language": metadata["language"],
            "chunk_count": 0,
            "size_bytes": destination.stat().st_size,
            "citation": _citation(
                metadata["authors"], metadata["year"], metadata["title"]
            ),
            "metadata_origin": metadata["origin"],
            "metadata_edited": bool(job.get("_metadata")),
            "index_status": "indexing",
            "extraction_warnings": metadata["warnings"],
            "created_at": _agora_utc(),
        }
        try:
            self.store.add(item)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        return self._index_document(job_id, item, destination, force=False)

    def _run_reindex(self, job_id: str) -> list[str]:
        with self._lock:
            source_id = self._jobs[job_id]["source_id"]
        try:
            item = self.store.get(source_id)
        except KeyError as exc:
            raise LibraryNotFoundError("Fonte nao encontrada.") from exc
        path = _safe_path(self.literature_root, item["relative_path"])
        if not path.is_file() or sha256_arquivo(path) != source_id:
            raise LibraryError("O PDF local nao corresponde ao SHA-256 catalogado.")
        return self._index_document(job_id, item, path, force=True)

    def _index_document(
        self, job_id: str, item: dict, path: Path, *, force: bool
    ) -> list[str]:
        self._set_job(
            job_id,
            phase="indexing",
            progress=38,
            message="Gerando trechos e embeddings da fonte.",
        )
        override = {
            "autor": "; ".join(item["authors"]),
            "titulo": item["title"],
            "ano": str(item.get("year") or "0000"),
            "citacao": item["citation"],
            "idioma": item["language"],
            "pasta": item["category"],
        }
        result = self.indexer(
            path,
            self._get_model(),
            self.chroma_path,
            forcar=force,
            metadados_override=override,
        )
        if not result.get("sucesso"):
            detail = result.get("erro") or "O indexador nao concluiu a fonte."
            self.store.update_index(
                item["source_id"],
                chunk_count=int(result.get("n_chunks", 0)),
                status="index_failed",
                warning="falha_na_indexacao",
            )
            raise LibraryError(str(detail))

        chunk_count = int(result.get("n_chunks", 0))
        self.store.update_index(
            item["source_id"],
            chunk_count=chunk_count,
            status="indexed_snapshot_stale",
            warning="snapshot_portatil_desatualizado",
        )
        self._set_job(
            job_id,
            phase="snapshot",
            progress=82,
            message="Atualizando snapshot portatil e busca lexical.",
        )
        try:
            snapshot = self.snapshot_exporter()
        except Exception as exc:
            self.store.update_index(
                item["source_id"],
                chunk_count=chunk_count,
                status="indexed_snapshot_stale",
                warning="snapshot_portatil_desatualizado",
            )
            return [f"Snapshot portatil desatualizado: {limpar_metadado(exc)}"]

        self.store.update_index(
            item["source_id"],
            chunk_count=chunk_count,
            status="indexed",
            snapshot=snapshot,
        )
        return []

    def _export_snapshot(self) -> dict:
        import chromadb

        from src.conhecimento.indice_lexical import IndiceLexicalSQLite
        from src.conhecimento.indice_portatil import exportar_colecao, hash_corpus_pdfs
        from src.conhecimento.index_lock import lock_indexacao
        from src.core.config import MODELO_EMBEDDINGS, NOME_COLECAO

        with lock_indexacao():
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            collection = client.get_or_create_collection(
                name=NOME_COLECAO,
                metadata={"hnsw:space": "cosine"},
            )
            corpus_hash, documents = hash_corpus_pdfs(self.literature_root)
            snapshot = exportar_colecao(
                collection,
                self.snapshot_path,
                modelo_embeddings=MODELO_EMBEDDINGS,
                hash_corpus=corpus_hash,
                n_documentos=documents,
            )
            IndiceLexicalSQLite().sincronizar(
                collection,
                versao=corpus_hash,
            )
            return snapshot

    def update_document(self, source_id: str, patch: dict) -> dict:
        catalog_before = self.store.load()
        current = next(
            (
                dict(item)
                for item in catalog_before["documents"]
                if item["source_id"] == source_id
            ),
            None,
        )
        if current is None:
            raise LibraryNotFoundError("Fonte nao encontrada.")

        allowed = {"title", "authors", "year", "category", "language"}
        unknown = set(patch) - allowed
        if unknown:
            raise LibraryError("Campos nao editaveis: " + ", ".join(sorted(unknown)))
        if not patch:
            raise LibraryError("Nenhum metadado foi informado.")

        old_path = _safe_path(self.literature_root, current["relative_path"])
        new_path = old_path
        category = patch.get("category")
        if category and category != current["category"]:
            if category not in CATEGORIAS:
                raise LibraryError("Categoria bibliografica invalida.")
            new_path = _safe_path(
                self.literature_root, f"{category}/{old_path.name}"
            )
            if new_path.exists() and new_path != old_path:
                raise LibraryError("Ja existe um PDF com esse nome na categoria.")
            new_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_path, new_path)

        try:
            updated = self.store.update(source_id, patch)
            if new_path != old_path:
                updated = self.store.update(
                    source_id,
                    {
                        "relative_path": new_path.relative_to(
                            self.literature_root
                        ).as_posix(),
                        "file_name": new_path.name,
                    },
                    internal=True,
                )
            return updated
        except Exception:
            self.store.save(catalog_before)
            if new_path != old_path and new_path.exists() and not old_path.exists():
                old_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(new_path, old_path)
            raise

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=False)


__all__ = [
    "LibraryDuplicateError",
    "LibraryError",
    "LibraryNotFoundError",
    "LibraryService",
    "MAX_LIBRARY_PDF_BYTES",
    "library_is_read_only",
]
