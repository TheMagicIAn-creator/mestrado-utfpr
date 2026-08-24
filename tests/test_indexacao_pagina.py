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
    trecho_auditavel,
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


def test_trecho_auditavel_limpa_e_limita_texto():
    texto = "  Falha no sensor CA.  " + ("conteudo tecnico " * 40)
    trecho = trecho_auditavel(texto, limite=150)

    assert trecho.startswith("Falha no sensor CA")
    assert "\n" not in trecho
    assert len(trecho) <= 153


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
        assert "pagina_rotulo" in meta
        assert meta.get("trecho")
        assert len(str(meta.get("chunk_sha256", ""))) == 64
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

    reindexado = indexar_pdf_unico(
        pdf_tres_paginas,
        _FakeEmbeddings(),
        pasta_chroma,
        forcar=True,
        metadados_override={
            "autor": "Autora Curada",
            "titulo": "Titulo Curado",
            "ano": "2026",
            "citacao": "Autora Curada (2026) - Titulo Curado",
            "idioma": "pt",
            "pasta": "confiabilidade",
        },
    )
    assert reindexado["sucesso"] is True
    atualizado = col.get(include=["metadatas"])["metadatas"]
    assert {meta["titulo"] for meta in atualizado} == {"Titulo Curado"}
    assert {meta["autor"] for meta in atualizado} == {"Autora Curada"}


def test_formatar_intervalo_paginas_se_disponivel():
    """O formatador de páginas vive em agente.py (que puxa torch); roda local,
    pula no CI torch-free."""
    pytest.importorskip("sentence_transformers")
    from src.conhecimento.agente import (
        _formatar_intervalo_paginas as f,
        _paginas_do_intervalo,
        _entrada_citacao,
        _rotulo_paginas_meta,
        _rerankar,
        _trecho_relevante,
        eh_query_de_revisao,
    )

    assert f({3}) == "3"
    assert f({3, 4, 5, 8}) == "3–5, 8"
    assert f({12, 13, 20, 21, 22}) == "12–13, 20–22"
    assert f(set()) == ""
    assert f({0, None, "", -1, 7}) == "7"
    assert _paginas_do_intervalo(3, 5) == [3, 4, 5]
    assert f(_paginas_do_intervalo("10", "12")) == "10–12"
    assert _paginas_do_intervalo("", 12) == []
    assert eh_query_de_revisao(
        "Cite artigos sobre deteccao de anomalias em inversores fotovoltaicos."
    )

    meta = {
        "citacao": "Autor (2026)",
        "pagina_inicio": 12,
        "pagina_fim": 12,
        "pagina_rotulo": "A-3",
        "chunk_sha1": "a" * 40,
        "trecho": "O dataset de Paderborn descreve um inversor saudavel.",
    }
    doc = (
        "O dataset de Paderborn descreve um inversor saudavel usado como "
        "referencia de normalidade. Outro paragrafo fala de detalhes laterais."
    )
    assert _rotulo_paginas_meta(meta) == "p. 12 (rotulo PDF: A-3)"
    assert "Paderborn" in _trecho_relevante(doc, "O que diz sobre Paderborn?", meta)
    fonte = _entrada_citacao(meta, doc, "O que diz sobre Paderborn?")
    assert "Autor (2026)" in fonte
    assert "p. 12" in fonte
    assert "trecho:" in fonte

    candidatos = [
        (
            "Tabela FMEA e RCM para sistema fotovoltaico com indice de deteccao.",
            {
                "citacao": "Torres (2024)",
                "titulo": "RCM em sistema fotovoltaico",
                "arquivo": "torres_rcm_2024.pdf",
                "autor": "Torres",
                "pasta": "confiabilidade",
            },
        ),
        (
            "Anomaly detection in solar PV inverter using isolation forest and machine learning.",
            {
                "citacao": "Sharma (2026)",
                "titulo": "Self tuning isolation forest for PV inverter anomaly detection",
                "arquivo": "sharma_a-self-tuning-reinforcement-learning-driven-isolation-forest_2026.pdf",
                "autor": "Sharma",
                "pasta": "ml-preditivo",
            },
        ),
    ]
    melhores = _rerankar(
        candidatos,
        "Cite artigos sobre deteccao de anomalias em inversores fotovoltaicos.",
        n_final=2,
        max_por_fonte=1,
    )
    assert melhores[0][1]["autor"] == "Sharma"
