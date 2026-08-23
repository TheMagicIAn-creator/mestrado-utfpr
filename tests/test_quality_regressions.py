from __future__ import annotations

import hashlib

import pytest

from src.conhecimento.agente_interacao import _chave_citacao
from src.conhecimento.catalogo_bibliografico import CATEGORIAS, IDIOMAS
from src.conhecimento.indexador import (
    normalizar_texto_pdf,
    remover_chunks_duplicados,
    remover_itens_duplicados,
)
from src.webapp.library_service import (
    LibraryError,
    _pdf_metadata,
    _upload_authors,
    _upload_category,
    _upload_language,
)


def test_chave_citacao_prioriza_sha256_e_mantem_legado():
    sha256 = "b" * 64
    sha1 = "a" * 40

    atual = _chave_citacao(
        {"arquivo": "fonte.pdf", "chunk_sha256": sha256},
        "conteudo atual",
    )
    legado = _chave_citacao(
        {"arquivo": "fonte.pdf", "chunk_sha1": sha1},
        "conteudo legado",
    )
    fallback = _chave_citacao({"arquivo": "fonte.pdf"}, "conteudo sem hash")

    esperado = hashlib.sha256("conteudo sem hash".encode("utf-8")).hexdigest()[:16]
    assert atual.endswith(sha256[:16])
    assert legado.endswith(sha1[:16])
    assert fallback.endswith(esperado)


def test_normalizacao_e_deduplicacao_textual_sao_deterministicas():
    texto = "confiabili-\n dade  , taxa de falha"
    assert normalizar_texto_pdf(texto) == "confiabilidade, taxa de falha"

    chunks = remover_chunks_duplicados(
        [" Falha no lado CA ", "falha   no lado ca", "Outro evento"]
    )
    itens = remover_itens_duplicados(
        [("Falha no lado CA", 1, 1), ("falha   no lado ca", 2, 2)]
    )

    assert chunks == ["Falha no lado CA", "Outro evento"]
    assert itens == [("Falha no lado CA", 1, 1)]


def test_metadados_de_upload_usam_fallbacks_controlados(tmp_path):
    arquivo_invalido = tmp_path / "fonte-invalida.pdf"
    arquivo_invalido.write_bytes(b"nao e um PDF")

    titulo, autor, texto, alertas = _pdf_metadata(arquivo_invalido)
    autores_internos = _upload_authors(
        None,
        "Ana Silva and Bruno Souza & Carla Lima",
        None,
    )
    autores_fallback = _upload_authors(None, "", "Autor Base")
    categoria = _upload_category(
        None,
        "analise-fmeca.pdf",
        "analise de confiabilidade e manutencao por FMECA",
    )
    idioma = _upload_language(
        None,
        "Este artigo apresenta uma analise de confiabilidade do inversor.",
    )

    assert (titulo, autor, texto) == ("", "", "")
    assert alertas == ["metadado_pdf_interno_ilegivel"]
    assert autores_internos == ["Ana Silva", "Bruno Souza", "Carla Lima"]
    assert autores_fallback == ["Autor Base"]
    assert categoria in CATEGORIAS
    assert idioma in IDIOMAS


def test_metadados_de_upload_rejeitam_vocabulario_desconhecido():
    with pytest.raises(LibraryError, match="Categoria"):
        _upload_category("categoria-invalida", "fonte.pdf", "")
    with pytest.raises(LibraryError, match="Idioma"):
        _upload_language("idioma-invalido", "")
