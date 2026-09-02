from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.conhecimento.catalogo_bibliografico import (
    CatalogoBibliograficoInvalido,
    CatalogoStore,
    carregar_catalogo,
    construir_catalogo,
    salvar_catalogo,
    sha256_arquivo,
)


ROOT = Path(__file__).resolve().parents[1]


def _pdf(path: Path, *, title: str, author: str) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_metadata({"/Title": title, "/Author": author})
    with path.open("wb") as stream:
        writer.write(stream)


def _snapshot(path: Path, pdf: Path, chunks: int = 2) -> None:
    source_id = sha256_arquivo(pdf)
    header = {
        "tipo": "manifesto_indice_portatil",
        "schema_version": 1,
        "n_chunks": chunks,
        "hash_corpus_sha256": "f" * 64,
        "gerado_em_utc": "2026-08-22T00:00:00+00:00",
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(header) + "\n")
        for index in range(chunks):
            stream.write(
                json.dumps(
                    {
                        "tipo": "chunk_indice_portatil",
                        "metadata": {
                            "arquivo_hash": source_id,
                            "arquivo": pdf.name,
                            "titulo": "Title Truncated By Filename",
                            "autor": "Silva",
                            "ano": "2025",
                            "pasta": "inversores-pv",
                            "idioma": "en",
                            "chunk_index": index,
                        },
                    }
                )
                + "\n"
            )


def test_catalogo_prefere_metadado_interno_plausivel_sem_alterar_pdf(tmp_path):
    literature = tmp_path / "literatura" / "inversores-pv"
    literature.mkdir(parents=True)
    pdf = literature / "silva_title-truncated-by-filename_2025.pdf"
    _pdf(
        pdf,
        title=(
            "Title Truncated By Filename: Complete Study of Photovoltaic "
            "Inverter Reliability"
        ),
        author="Ana Silva and Bruno Souza",
    )
    before = sha256_arquivo(pdf)
    snapshot = tmp_path / "index.jsonl.gz"
    _snapshot(snapshot, pdf)

    catalog = construir_catalogo(tmp_path / "literatura", snapshot)
    document = catalog["documents"][0]

    assert document["title"] == (
        "Title Truncated By Filename: Complete Study of Photovoltaic "
        "Inverter Reliability"
    )
    assert document["authors"] == ["Ana Silva", "Bruno Souza"]
    assert document["metadata_origin"] == "pdf_interno"
    assert catalog["summary"]["indexed_chunks"] == 2
    assert catalog["summary"]["portable_index_records"] == 3
    assert sha256_arquivo(pdf) == before


def test_catalogo_rejeita_titulo_interno_de_template(tmp_path):
    literature = tmp_path / "literatura" / "confiabilidade"
    literature.mkdir(parents=True)
    pdf = literature / "cristaldi_root-cause-analysis_2017.pdf"
    _pdf(pdf, title="Template for an Acta IMEKO event paper", author="Template Team")

    catalog = construir_catalogo(tmp_path / "literatura")
    document = catalog["documents"][0]

    assert document["title"] == "Root Cause Analysis"
    assert document["authors"] == ["Cristaldi"]
    assert document["metadata_origin"] == "nome_arquivo"


def test_catalogo_versionado_cobre_corpus_real_sem_confundir_manifesto_com_chunk():
    catalog = carregar_catalogo(ROOT / "literatura" / "catalogo.json")

    assert catalog["summary"]["documents"] == len(catalog["documents"])
    assert catalog["summary"]["documents"] >= 45
    assert catalog["summary"]["indexed_chunks"] >= 12556
    assert (
        catalog["summary"]["portable_index_records"]
        >= catalog["summary"]["indexed_chunks"]
    )
    assert catalog["summary"]["metadata_warnings"] == 4
    assert all(len(item["source_id"]) == 64 for item in catalog["documents"])
    assert all("\ufffd" not in item["title"] for item in catalog["documents"])
    assert all(
        "\ufffd" not in author
        for item in catalog["documents"]
        for author in item["authors"]
    )
    kull = next(
        item
        for item in catalog["documents"]
        if item["source_id"]
        == "6a3d079ab53021fc6920608048113153605e90581548bc5e5bff0914b46d102c"
    )
    assert kull["year"] == 2025
    assert kull["chunk_count"] == 0
    assert kull["index_status"] == "pending_benchmark_regression"
    assert "benchmark_r6_regression_not_promoted" in kull["extraction_warnings"]
    assert kull["file_name"].endswith("_2025.pdf")


def test_store_edita_metadados_e_marca_indice_como_stale(tmp_path):
    source_id = "a" * 64
    catalog_path = tmp_path / "catalogo.json"
    payload = {
        "schema_version": 1,
        "catalog_id": "test",
        "source_index": {},
        "summary": {
            "documents": 1,
            "indexed_chunks": 3,
            "portable_index_records": 4,
        },
        "documents": [
            {
                "source_id": source_id,
                "sha256": source_id,
                "title": "Titulo inicial",
                "authors": ["Autora"],
                "year": 2025,
                "year_status": "informado",
                "category": "confiabilidade",
                "language": "pt",
                "citation": "Autora (2025) - Titulo inicial",
                "chunk_count": 3,
                "index_status": "indexed",
                "extraction_warnings": [],
                "relative_path": "confiabilidade/fonte.pdf",
                "file_name": "fonte.pdf",
            }
        ],
    }
    salvar_catalogo(catalog_path, payload)

    updated = CatalogoStore(catalog_path).update(
        source_id, {"title": "Titulo revisado"}
    )

    assert updated["title"] == "Titulo revisado"
    assert updated["metadata_edited"] is True
    assert updated["index_status"] == "metadata_stale"


def test_catalogo_recusa_destino_com_nome_controlado(tmp_path):
    payload = {
        "schema_version": 1,
        "catalog_id": "test",
        "source_index": {},
        "summary": {"documents": 0, "indexed_chunks": 0},
        "documents": [],
    }

    with pytest.raises(CatalogoBibliograficoInvalido, match="nome fixo"):
        salvar_catalogo(tmp_path / "../../catalogo-injetado.json", payload)
