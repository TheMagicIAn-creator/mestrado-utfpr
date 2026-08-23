from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.conhecimento.catalogo_bibliografico import salvar_catalogo
from src.webapp.library_service import LibraryDuplicateError, LibraryService


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_metadata({"/Title": "Fonte Academica de Teste", "/Author": "Ana Silva"})
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _empty_catalog(path: Path) -> None:
    salvar_catalogo(
        path,
        {
            "schema_version": 1,
            "catalog_id": "test-library",
            "source_index": {},
            "summary": {
                "documents": 0,
                "indexed_chunks": 0,
                "portable_index_records": 0,
                "categories": {},
                "languages": {},
                "metadata_warnings": 0,
            },
            "documents": [],
        },
    )


@pytest.fixture()
def service(tmp_path):
    root = tmp_path / "literatura"
    root.mkdir()
    catalog = root / "catalogo.json"
    _empty_catalog(catalog)
    calls = []

    def indexer(path, _model, _chroma, **kwargs):
        calls.append((path, kwargs))
        return {"sucesso": True, "n_chunks": 3, "pulou": False}

    def snapshot():
        return {
            "schema_version": 1,
            "n_chunks": 3,
            "hash_corpus_sha256": "b" * 64,
            "gerado_em_utc": "2026-08-22T12:00:00+00:00",
        }

    instance = LibraryService(
        catalog_path=catalog,
        literature_root=root,
        chroma_path=tmp_path / "chroma",
        snapshot_path=tmp_path / "snapshot.jsonl.gz",
        staging_root=tmp_path / "staging",
        start_jobs=False,
        model_factory=lambda: object(),
        indexer=indexer,
        snapshot_exporter=snapshot,
    )
    instance.test_calls = calls
    yield instance
    instance.close()


def _queue(service: LibraryService):
    return service.queue_pdf(
        "../fonte.pdf",
        _pdf_bytes(),
        {
            "title": "Fonte Academica de Teste",
            "authors": "Ana Silva; Bruno Souza",
            "year": "2025",
            "category": "confiabilidade",
            "language": "pt",
        },
    )


def test_upload_copia_catalogo_indexa_e_exporta_snapshot(service):
    queued = _queue(service)
    completed = service.run_job(queued["job_id"])
    catalog = service.catalog()
    document = catalog["documents"][0]

    assert completed["state"] == "completed"
    assert completed["progress"] == 100
    assert catalog["summary"]["documents"] == 1
    assert catalog["summary"]["indexed_chunks"] == 3
    assert document["authors"] == ["Ana Silva", "Bruno Souza"]
    assert document["index_status"] == "indexed"
    assert document["url"].startswith("/library-files/confiabilidade/")
    assert (service.literature_root / document["relative_path"]).is_file()
    assert service.test_calls[0][1]["forcar"] is False


def test_upload_rejeita_duplicidade_por_hash(service):
    first = _queue(service)
    service.run_job(first["job_id"])

    with pytest.raises(LibraryDuplicateError):
        _queue(service)


def test_reindexacao_e_idempotente_enquanto_esta_na_fila(service):
    added = _queue(service)
    service.run_job(added["job_id"])
    source_id = service.catalog()["documents"][0]["source_id"]

    first = service.queue_reindex(source_id)
    second = service.queue_reindex(source_id)
    completed = service.run_job(first["job_id"])

    assert first["job_id"] == second["job_id"]
    assert completed["state"] == "completed"
    assert service.test_calls[-1][1]["forcar"] is True


def test_edicao_de_categoria_move_pdf_e_marca_metadados_stale(service):
    added = _queue(service)
    service.run_job(added["job_id"])
    before = service.catalog()["documents"][0]
    old_path = service.literature_root / before["relative_path"]

    updated = service.update_document(
        before["source_id"],
        {"category": "manutencao", "title": "Fonte revisada"},
    )
    new_path = service.literature_root / updated["relative_path"]

    assert not old_path.exists()
    assert new_path.is_file()
    assert updated["category"] == "manutencao"
    assert updated["index_status"] == "metadata_stale"


def test_falha_do_snapshot_preserva_indice_e_publica_ressalva(tmp_path):
    root = tmp_path / "literatura"
    root.mkdir()
    catalog = root / "catalogo.json"
    _empty_catalog(catalog)
    service = LibraryService(
        catalog_path=catalog,
        literature_root=root,
        chroma_path=tmp_path / "chroma",
        snapshot_path=tmp_path / "snapshot.jsonl.gz",
        staging_root=tmp_path / "staging",
        start_jobs=False,
        model_factory=lambda: object(),
        indexer=lambda *_args, **_kwargs: {"sucesso": True, "n_chunks": 2},
        snapshot_exporter=lambda: (_ for _ in ()).throw(RuntimeError("disco cheio")),
    )
    try:
        queued = _queue(service)
        completed = service.run_job(queued["job_id"])
        document = service.catalog()["documents"][0]
    finally:
        service.close()

    assert completed["state"] == "completed"
    assert completed["warnings"]
    assert document["index_status"] == "indexed_snapshot_stale"
