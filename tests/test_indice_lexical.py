from __future__ import annotations

from src.conhecimento.indice_lexical import IndiceLexicalSQLite


class _Colecao:
    def __init__(self):
        self.itens = [
            (
                "stender-1",
                "Stender descreve o dataset Paderborn do inversor IGBT saudavel.",
                {"autor": "Stender", "arquivo": "stender.pdf"},
            ),
            (
                "fmeca-1",
                "A FMECA prioriza o contator AC pelo NPR e pelo efeito da falha.",
                {"autor": "Torres", "arquivo": "torres.pdf"},
            ),
            (
                "ml-1",
                "Deteccao de anomalias com autoencoder e Isolation Forest.",
                {"autor": "Ibrahim", "arquivo": "ibrahim.pdf"},
            ),
        ]

    def count(self):
        return len(self.itens)

    def get(self, limit, offset, include):
        lote = self.itens[offset:offset + limit]
        return {
            "ids": [i[0] for i in lote],
            "documents": [i[1] for i in lote],
            "metadatas": [i[2] for i in lote],
        }


def test_indice_bm25_sincroniza_e_recupera_termo_exato(tmp_path):
    indice = IndiceLexicalSQLite(tmp_path / "fts.sqlite3")
    colecao = _Colecao()

    primeiro = indice.sincronizar(colecao, versao="v1", tamanho_lote=2)
    segundo = indice.sincronizar(colecao, versao="v1", tamanho_lote=2)
    resultados = indice.buscar("O que Stender diz sobre Paderborn?", limite=5)

    assert indice.disponivel is True
    assert primeiro["reconstruido"] is True
    assert segundo["reconstruido"] is False
    assert resultados[0].chunk_id == "stender-1"
    assert resultados[0].metadata["autor"] == "Stender"


def test_indice_bm25_remove_acentos_da_consulta(tmp_path):
    indice = IndiceLexicalSQLite(tmp_path / "fts.sqlite3")
    indice.sincronizar(_Colecao(), versao="v1")

    resultados = indice.buscar("detecção de anomalias", limite=5)

    assert resultados
    assert resultados[0].chunk_id == "ml-1"
