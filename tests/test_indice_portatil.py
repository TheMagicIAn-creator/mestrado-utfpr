import gzip
import json

import pytest

from src.conhecimento.indice_portatil import (
    IndicePortatilInvalido,
    exportar_colecao,
    importar_colecao,
    ler_manifesto,
)


class ColecaoFalsa:
    name = "literatura_pv"

    def __init__(self, itens=None):
        self.itens = {item["id"]: item for item in (itens or [])}

    def count(self):
        return len(self.itens)

    def get(self, limit=100, offset=0, include=None):
        lote = list(self.itens.values())[offset:offset + limit]
        return {
            "ids": [item["id"] for item in lote],
            "documents": [item["documento"] for item in lote],
            "metadatas": [item["metadata"] for item in lote],
            "embeddings": [item["embedding"] for item in lote],
        }

    def upsert(self, ids, documents, metadatas, embeddings):
        for chunk_id, documento, metadata, embedding in zip(
            ids, documents, metadatas, embeddings
        ):
            self.itens[chunk_id] = {
                "id": chunk_id,
                "documento": documento,
                "metadata": metadata,
                "embedding": embedding,
            }


@pytest.fixture
def itens():
    return [
        {
            "id": "paper-a-1",
            "documento": "Trecho auditável da página 1.",
            "metadata": {"arquivo": "paper-a.pdf", "pagina": 1},
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "id": "paper-b-4",
            "documento": "Anomaly detection in PV inverters.",
            "metadata": {"arquivo": "paper-b.pdf", "pagina": 4},
            "embedding": [0.4, 0.5, 0.6],
        },
    ]


def test_roundtrip_indice_portatil(tmp_path, itens):
    origem = ColecaoFalsa(itens)
    destino = tmp_path / "indice.jsonl.gz"
    exportado = exportar_colecao(
        origem,
        destino,
        modelo_embeddings="modelo-multilingue",
        hash_corpus="abc123",
        n_documentos=2,
        tamanho_lote=1,
    )

    assert exportado["n_chunks"] == 2
    assert destino.stat().st_size > 0
    assert ler_manifesto(destino)["hash_corpus_sha256"] == "abc123"

    restaurada = ColecaoFalsa()
    importado = importar_colecao(restaurada, destino, tamanho_lote=1)
    assert importado["importados"] == 2
    assert restaurada.itens == origem.itens


def test_importacao_recusa_colecao_parcial(tmp_path, itens):
    destino = tmp_path / "indice.jsonl.gz"
    exportar_colecao(
        ColecaoFalsa(itens),
        destino,
        modelo_embeddings="modelo",
        hash_corpus="hash",
        n_documentos=2,
    )

    with pytest.raises(IndicePortatilInvalido, match="parcialmente preenchida"):
        importar_colecao(ColecaoFalsa(itens[:1]), destino)


def test_mesclagem_completa_snapshot_e_preserva_registro_novo(tmp_path, itens):
    destino = tmp_path / "indice.jsonl.gz"
    exportar_colecao(
        ColecaoFalsa(itens),
        destino,
        modelo_embeddings="modelo",
        hash_corpus="hash",
        n_documentos=2,
    )
    registro_novo = {
        "id": "sessao-runtime",
        "documento": "Sessão criada depois do deploy.",
        "metadata": {"caminho_obsidian": "sessoes/2026-07-21.md"},
        "embedding": [0.7, 0.8, 0.9],
    }
    colecao = ColecaoFalsa([itens[0], registro_novo])

    resultado = importar_colecao(colecao, destino, mesclar=True, tamanho_lote=1)

    assert resultado["importados"] == 1
    assert resultado["preservados"] == 1
    assert colecao.count() == 3
    assert set(colecao.itens) == {"paper-a-1", "paper-b-4", "sessao-runtime"}


def test_manifesto_rejeita_arquivo_que_nao_e_gzip(tmp_path):
    caminho = tmp_path / "quebrado.jsonl.gz"
    caminho.write_text(json.dumps({"tipo": "qualquer"}), encoding="utf-8")
    with pytest.raises(IndicePortatilInvalido, match="ilegível"):
        ler_manifesto(caminho)


def test_importador_preserva_compatibilidade_com_snapshot_legado(tmp_path):
    caminho = tmp_path / "legado.jsonl.gz"
    manifesto = {
        "tipo": "manifesto_indice_literatura",
        "schema_version": 1,
        "colecao": "literatura_pv",
        "n_chunks": 1,
        "n_documentos": 1,
    }
    chunk = {
        "tipo": "chunk_literatura",
        "id": "legado-1",
        "documento": "Conteúdo legado.",
        "metadata": {"arquivo": "legado.pdf"},
        "embedding": [0.1, 0.2],
    }
    with gzip.open(caminho, "wt", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(manifesto) + "\n")
        arquivo.write(json.dumps(chunk) + "\n")

    colecao = ColecaoFalsa()
    resultado = importar_colecao(colecao, caminho)

    assert resultado["importados"] == 1
    assert colecao.count() == 1
