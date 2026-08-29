import gzip
import json

import pytest

from src.conhecimento.indice_portatil import (
    IndicePortatilInvalido,
    atualizar_metadados_snapshot,
    exportar_colecao,
    importar_colecao,
    ler_manifesto,
    migrar_snapshot_v2,
    validar_snapshot,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


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
            "id": f"{HASH_A}__chunk_00000",
            "documento": "Trecho auditável da página 1.",
            "metadata": {
                "arquivo": "paper-a.pdf",
                "arquivo_hash": HASH_A,
                "pagina_inicio": 1,
                "pagina_fim": 1,
                "chunk_index": 0,
                "total_chunks": 1,
                "idioma": "pt",
                "autor": "Autor A",
            },
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "id": f"{HASH_B}__chunk_00000",
            "documento": "Anomaly detection in PV inverters.",
            "metadata": {
                "arquivo": "paper-b.pdf",
                "arquivo_hash": HASH_B,
                "pagina_inicio": 4,
                "pagina_fim": 4,
                "chunk_index": 0,
                "total_chunks": 1,
                "idioma": "en",
                "autor": "Author B",
            },
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
    assert exportado["schema_version"] == 2
    assert destino.stat().st_size > 0
    assert ler_manifesto(destino)["hash_corpus_sha256"] == "abc123"
    assert validar_snapshot(destino)["chunks_validados"] == 2

    restaurada = ColecaoFalsa()
    importado = importar_colecao(restaurada, destino, tamanho_lote=1)
    assert importado["importados"] == 2
    assert set(restaurada.itens) == set(origem.itens)
    for chunk_id, original in origem.itens.items():
        restaurado = restaurada.itens[chunk_id]
        assert restaurado["documento"] == original["documento"]
        assert restaurado["embedding"] == original["embedding"]
        assert restaurado["metadata"]["arquivo_hash"] == original["metadata"]["arquivo_hash"]


def test_atualiza_metadados_sem_alterar_texto_ou_embedding(tmp_path, itens):
    destino = tmp_path / "indice.jsonl.gz"
    exportar_colecao(
        ColecaoFalsa(itens),
        destino,
        modelo_embeddings="modelo",
        hash_corpus="hash-antigo",
        n_documentos=2,
    )

    resultado = atualizar_metadados_snapshot(
        destino,
        {
            "paper-a.pdf": {
                "arquivo": "autor_paper-a_2026.pdf",
                "autor": "Autor",
                "ano": "2026",
            }
        },
        hash_corpus="hash-novo",
    )
    restaurada = ColecaoFalsa()
    importar_colecao(restaurada, destino)

    assert resultado["chunks_atualizados"] == 1
    assert ler_manifesto(destino)["hash_corpus_sha256"] == "hash-novo"
    chunk_id = itens[0]["id"]
    assert restaurada.itens[chunk_id]["documento"] == itens[0]["documento"]
    assert restaurada.itens[chunk_id]["embedding"] == itens[0]["embedding"]
    assert restaurada.itens[chunk_id]["metadata"]["ano"] == "2026"


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
    assert set(colecao.itens) == {itens[0]["id"], itens[1]["id"], "sessao-runtime"}


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


def test_migracao_v1_para_v2_preserva_texto_embedding_e_ids(tmp_path):
    caminho = tmp_path / "indice.jsonl.gz"
    manifesto = {
        "tipo": "manifesto_indice_portatil",
        "schema_version": 1,
        "colecao": "literatura_pv",
        "modelo_embeddings": "modelo",
        "n_chunks": 2,
        "n_documentos": 1,
        "hash_corpus_sha256": "corpus",
    }
    registros = [
        {
            "tipo": "chunk_indice_portatil",
            "id": f"{HASH_A}__chunk_{indice:05d}",
            "documento": f"Trecho {indice}",
            "metadata": {
                "arquivo": "paper-a.pdf",
                "arquivo_hash": HASH_A,
                "pagina_inicio": indice + 1,
                "pagina_fim": indice + 1,
                "chunk_index": indice,
                "total_chunks": 2,
                "idioma": "pt",
                "autor": "Autor A",
            },
            "embedding": [0.1 + indice, 0.2 + indice],
        }
        for indice in range(2)
    ]
    with gzip.open(caminho, "wt", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(manifesto) + "\n")
        for registro in registros:
            arquivo.write(json.dumps(registro) + "\n")

    resultado = migrar_snapshot_v2(caminho)

    assert resultado["schema_version"] == 2
    assert resultado["migrado_de_schema_version"] == 1
    assert resultado["chunks_validados"] == 2
    assert resultado["conteudo_preservado"] is True
    assert (
        resultado["hash_conteudo_origem_sha256"]
        == resultado["hash_conteudo_retrieval_sha256"]
    )
    with gzip.open(caminho, "rt", encoding="utf-8") as arquivo:
        manifesto_v2 = json.loads(next(arquivo))
        chunks_v2 = [json.loads(linha) for linha in arquivo]
    assert manifesto_v2["retrieval_text_strategy"] == "identity_raw_text"
    assert [item["chunk_id"] for item in chunks_v2] == [item["id"] for item in registros]
    assert [item["raw_text"] for item in chunks_v2] == [item["documento"] for item in registros]
    assert [item["retrieval_text"] for item in chunks_v2] == [item["documento"] for item in registros]
    assert [item["embedding"] for item in chunks_v2] == [item["embedding"] for item in registros]
    assert chunks_v2[0]["metadata"]["next_chunk_id"] == registros[1]["id"]
    assert chunks_v2[1]["metadata"]["prev_chunk_id"] == registros[0]["id"]
    assert migrar_snapshot_v2(caminho)["ja_estava_pronto"] is True


def test_validacao_v2_rejeita_retrieval_text_alterado(tmp_path, itens):
    caminho = tmp_path / "indice.jsonl.gz"
    exportar_colecao(
        ColecaoFalsa(itens),
        caminho,
        modelo_embeddings="modelo",
        hash_corpus="hash",
        n_documentos=2,
    )
    adulterado = tmp_path / "adulterado.jsonl.gz"
    with gzip.open(caminho, "rt", encoding="utf-8") as origem, gzip.open(
        adulterado, "wt", encoding="utf-8"
    ) as destino:
        destino.write(next(origem))
        primeiro = json.loads(next(origem))
        primeiro["retrieval_text"] += " contexto não autorizado"
        destino.write(json.dumps(primeiro) + "\n")
        for linha in origem:
            destino.write(linha)

    with pytest.raises(IndicePortatilInvalido, match="idêntico"):
        validar_snapshot(adulterado)
