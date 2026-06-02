"""
Sprint 4 — 7.1 RAG por página.

Garante que a indexação é PAGE-AWARE: cada chunk carrega a página de origem
(``pagina_inicio``/``pagina_fim``) no metadado, habilitando a citação com
página — "Autor (ano, p. X)".

Torch-free: usa um modelo de embeddings FAKE (determinístico) e um ChromaDB
temporário, de modo que o caminho completo de ``indexar_pdf_unico`` rode no CI
sem carregar ``sentence_transformers``/torch. O PDF de fixture é gerado com
matplotlib (texto real, extraível pelo pypdf).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.conhecimento.indexador import (
    ler_pdf_paginas,
    remover_itens_duplicados,
    indexar_pdf_unico,
)

# Marcadores únicos por página — curtos o bastante para um chunk por página.
MARCADORES = ["ALFAPRIMEIRA", "BETASEGUNDA", "GAMATERCEIRA"]


@pytest.fixture(scope="module")
def pdf_tres_paginas(tmp_path_factory) -> Path:
    """Gera um PDF de 3 páginas com um marcador distinto por página."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    destino = tmp_path_factory.mktemp("fixture_pdf") / "Doc-Teste (2024) Pagina.pdf"
    with PdfPages(str(destino)) as pdf:
        for i, marca in enumerate(MARCADORES, 1):
            # Texto longo (> 80 chars, o filtro mínimo de chunk) e multilinha
            # para não estourar a largura da figura.
            linhas = [f"Pagina {i} marcador {marca}"]
            for k in range(8):
                linhas.append(
                    f"linha {k} conteudo de teste de indexacao por pagina do Al IAdo PV"
                )
            fig = plt.figure(figsize=(8, 11))
            fig.text(0.1, 0.92, "\n".join(linhas), fontsize=12,
                     va="top", family="monospace")
            pdf.savefig(fig)
            plt.close(fig)
    return destino


class _FakeEmbeddings:
    """Embeddings determinísticos, não-degenerados, sem torch."""

    def encode(self, textos):
        linhas = []
        for t in textos:
            base = (sum(map(ord, t)) % 97 + 1) / 100.0
            linhas.append([base + j * 0.001 for j in range(8)])
        return np.array(linhas, dtype=float)


def test_ler_pdf_paginas_preserva_pagina(pdf_tres_paginas):
    paginas = ler_pdf_paginas(pdf_tres_paginas)
    assert len(paginas) == 3
    # numeração começa em 1 e é crescente
    assert [n for _, n in paginas] == [1, 2, 3]
    # cada marcador cai na sua página
    for (texto, n), marca in zip(paginas, MARCADORES):
        assert marca in texto, f"esperava {marca} na página {n}: {texto!r}"


def test_remover_itens_duplicados_preserva_primeira_pagina():
    itens = [
        ("texto repetido", 2, 2),
        ("texto repetido", 5, 5),   # duplicado → descartado
        ("outro texto", 7, 7),
        ("   ", 9, 9),              # vazio → descartado
    ]
    out = remover_itens_duplicados(itens)
    assert out == [("texto repetido", 2, 2), ("outro texto", 7, 7)]


def test_indexar_pdf_unico_grava_pagina(pdf_tres_paginas, tmp_path):
    import chromadb
    from src.core.config import NOME_COLECAO

    pasta_chroma = tmp_path / "chroma"
    resultado = indexar_pdf_unico(pdf_tres_paginas, _FakeEmbeddings(), pasta_chroma)

    assert resultado["sucesso"] is True, resultado
    assert resultado["n_chunks"] >= 3

    cli = chromadb.PersistentClient(path=str(pasta_chroma))
    col = cli.get_or_create_collection(NOME_COLECAO)
    dados = col.get(include=["documents", "metadatas"])
    metadados = dados["metadatas"]
    documentos = dados["documents"]
    assert metadados, "coleção vazia após indexação"

    # toda metadado tem página válida (>= 1) e fim >= início
    for meta in metadados:
        assert "pagina_inicio" in meta and "pagina_fim" in meta
        assert int(meta["pagina_inicio"]) >= 1
        assert int(meta["pagina_fim"]) >= int(meta["pagina_inicio"])

    # atribuição correta: o chunk com cada marcador aponta para a sua página
    pagina_por_marca = {}
    for doc, meta in zip(documentos, metadados):
        for i, marca in enumerate(MARCADORES, 1):
            if marca in doc:
                pagina_por_marca[i] = int(meta["pagina_inicio"])
    assert pagina_por_marca.get(1) == 1
    assert pagina_por_marca.get(2) == 2
    assert pagina_por_marca.get(3) == 3


def test_formatar_intervalo_paginas_se_disponivel():
    """O formatador de páginas vive em agente.py (que puxa torch); roda local,
    pula no CI torch-free."""
    pytest.importorskip("sentence_transformers")
    from src.conhecimento.agente import (
        _formatar_intervalo_paginas as f,
        _paginas_do_intervalo,
    )

    assert f({3}) == "3"
    assert f({3, 4, 5, 8}) == "3–5, 8"
    assert f({12, 13, 20, 21, 22}) == "12–13, 20–22"
    assert f(set()) == ""
    assert f({0, None, "", -1, 7}) == "7"
    assert _paginas_do_intervalo(3, 5) == [3, 4, 5]
    assert f(_paginas_do_intervalo("10", "12")) == "10–12"
    assert _paginas_do_intervalo("", 12) == []
