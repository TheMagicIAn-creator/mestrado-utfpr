"""
As Camadas 2 e 3 do RAG — fusão RRF, diversificação e reranking — sem ChromaDB.

POR QUE ESTE TESTE EXISTE
=========================
`_busca_hibrida`, `_diversificar_por_fonte` e `_rerankar` são o miolo da
recuperação: decidem QUAIS trechos chegam ao prompt do Gemini e em que ordem.
Até aqui não havia um único teste sobre eles, porque exercitá-los parecia exigir
ChromaDB real — e o CI é leve.

Não exige. A coleção é usada por três métodos (`query`, `get`, `count`) e o
encoder por um (`encode`). Uma dupla de dublês de dez linhas cobre a interface
inteira, e o que se testa passa a ser a LÓGICA, que é local e determinística:
nenhuma dessas camadas chama LLM.

O que os dublês NÃO cobrem, e é bom deixar explícito: o comportamento real do
ChromaDB (case-sensitivity do `$contains`, semântica do `$and`) e a qualidade
dos embeddings. Isso é E3 de infraestrutura — precisa do serviço de verdade.
Aqui se verifica que, DADO o que o banco devolve, a fusão e o corte fazem o que
prometem.
"""

from __future__ import annotations

import pytest

# O módulo de recuperação deve ser uma unidade importável por si só. O PR #102
# dependia de importar `agente` antes; essa ordem acidental agora tem regressão
# dedicada também em `test_imports_modulos_extraidos.py`.
from src.conhecimento.agente_recuperacao import (
    _busca_hibrida,
    _diversificar_por_fonte,
    _rerankar,
    eh_query_de_revisao,
)


# ── dublês: a interface inteira que o código usa ───────────────────────────

class ColecaoFalsa:
    """Devolve os documentos na ordem em que foram registrados por query."""

    def __init__(self, por_query=None, por_get=None):
        self._por_query = por_query or []
        self._por_get = por_get or []
        self.chamadas_query = 0

    def query(self, query_embeddings=None, n_results=10, **_):
        self.chamadas_query += 1
        ids, docs, metas = zip(*self._por_query) if self._por_query else ((), (), ())
        corte = slice(0, n_results)
        return {"ids": [list(ids)[corte]], "documents": [list(docs)[corte]],
                "metadatas": [list(metas)[corte]]}

    def get(self, **_):
        if not self._por_get:
            return {"ids": [], "documents": [], "metadatas": []}
        ids, docs, metas = zip(*self._por_get)
        return {"ids": list(ids), "documents": list(docs), "metadatas": list(metas)}

    def count(self):
        return len(self._por_query) + len(self._por_get)


class EncoderFalso:
    """`encode` devolve um vetor por variação — o conteúdo não importa aqui."""

    def encode(self, textos):
        import numpy as np
        return np.zeros((len(textos), 3), dtype=float)


class ItemLexical:
    def __init__(self, chunk_id, documento, metadata, rank):
        self.chunk_id, self.documento = chunk_id, documento
        self.metadata, self.rank = metadata, rank


class IndiceLexicalFalso:
    disponivel = True

    def __init__(self, itens):
        self._itens = itens

    def buscar(self, variacoes, termos=None, limite=60):
        return self._itens


def _chunk(i, arquivo="a.pdf", texto="texto"):
    return (f"id{i}", texto, {"arquivo": arquivo, "citacao": f"Autor {i}"})


# ── Camada 2: a fusão RRF ──────────────────────────────────────────────────

def test_busca_semantica_sozinha_preserva_a_ordem_do_banco():
    colecao = ColecaoFalsa(por_query=[_chunk(i) for i in range(1, 4)])
    saida = _busca_hibrida(["pergunta"], [], colecao, EncoderFalso())
    assert [m["_rrf_score"] for _, m in saida] == sorted(
        (m["_rrf_score"] for _, m in saida), reverse=True)
    assert len(saida) == 3


def test_chunk_achado_pelas_duas_buscas_sobe_acima_dos_dois_primeiros():
    """É a razão de ser da RRF: concordância entre semântica e BM25 pesa mais
    que uma boa posição numa lista só.

    O chunk `id3` é 3º na semântica e 1º no BM25 — soma 1/63 + 1/61 = 0,0323,
    contra 1/61 = 0,0164 do 1º colocado semântico.
    """
    colecao = ColecaoFalsa(por_query=[_chunk(i) for i in (1, 2, 3)])
    lexical = IndiceLexicalFalso([ItemLexical("id3", "texto", {"arquivo": "a.pdf"}, 1)])

    saida = _busca_hibrida(["p"], ["termo"], colecao, EncoderFalso(),
                           indice_lexical=lexical)
    assert saida[0][1]["citacao"] == "Autor 3"


def test_o_mesmo_chunk_nas_duas_listas_nao_duplica():
    colecao = ColecaoFalsa(por_query=[_chunk(1)])
    lexical = IndiceLexicalFalso([ItemLexical("id1", "texto", {"arquivo": "a.pdf"}, 1)])
    saida = _busca_hibrida(["p"], ["t"], colecao, EncoderFalso(),
                           indice_lexical=lexical)
    assert len(saida) == 1, "o pool deve ser deduplicado por chunk_id"


def test_encoder_que_falha_nao_derruba_a_busca():
    """Degradação honesta: sem embeddings, o BM25 ainda responde."""

    class EncoderQuebrado:
        def encode(self, _):
            raise RuntimeError("modelo indisponível")

    lexical = IndiceLexicalFalso([ItemLexical("id9", "texto", {"arquivo": "b.pdf"}, 1)])
    saida = _busca_hibrida(["p"], ["t"], ColecaoFalsa(), EncoderQuebrado(),
                           indice_lexical=lexical)
    assert len(saida) == 1


def test_indice_lexical_indisponivel_cai_no_fallback_sem_estourar():
    class LexicalMorto:
        disponivel = False

    saida = _busca_hibrida(["p"], ["termo"], ColecaoFalsa(por_query=[_chunk(1)]),
                           EncoderFalso(), indice_lexical=LexicalMorto())
    assert len(saida) == 1


def test_banco_vazio_devolve_lista_vazia_em_vez_de_erro():
    assert _busca_hibrida(["p"], ["t"], ColecaoFalsa(), EncoderFalso()) == []


# ── Camada 3: diversificação por fonte ─────────────────────────────────────

def _pontuado(score, arquivo, i):
    return (score, i, f"trecho {i}", {"arquivo": arquivo, "citacao": f"c{i}"})


def test_teto_por_fonte_impede_um_pdf_de_ocupar_o_topo_inteiro():
    pontuados = [_pontuado(10 - i, "dominante.pdf", i) for i in range(6)]
    pontuados += [_pontuado(1, "outro.pdf", 100), _pontuado(0.9, "terceiro.pdf", 101)]
    sel = _diversificar_por_fonte(pontuados, n_final=4, max_por_fonte=2)

    fontes = [m["arquivo"] for _, m in sel]
    assert fontes.count("dominante.pdf") == 2
    assert {"outro.pdf", "terceiro.pdf"} <= set(fontes)


def test_teto_relaxa_quando_nao_ha_fontes_suficientes():
    """Melhor devolver mais chunks da mesma fonte que devolver menos que o
    pedido — o orçamento do prompt ficaria ocioso."""
    pontuados = [_pontuado(10 - i, "unica.pdf", i) for i in range(6)]
    assert len(_diversificar_por_fonte(pontuados, n_final=5, max_por_fonte=2)) == 5


def test_diversificacao_respeita_a_ordem_de_score():
    pontuados = [_pontuado(9, "a.pdf", 1), _pontuado(8, "b.pdf", 2),
                 _pontuado(7, "c.pdf", 3)]
    sel = _diversificar_por_fonte(pontuados, n_final=3, max_por_fonte=1)
    assert [m["citacao"] for _, m in sel] == ["c1", "c2", "c3"]


def test_lista_vazia_nao_quebra():
    assert _diversificar_por_fonte([], n_final=5) == []


# ── Camada 3: reranking ────────────────────────────────────────────────────

def test_rerank_prefere_o_chunk_com_sobreposicao_lexical():
    candidatos = [
        ("Nada a ver com o assunto perguntado.", {"arquivo": "ruido.pdf"}),
        ("O autoencoder detecta anomalia no inversor fotovoltaico.",
         {"arquivo": "ml-inversores/alvo.pdf"}),
    ]
    saida = _rerankar(candidatos, "autoencoder anomalia inversor", n_final=1)
    assert saida[0][1]["arquivo"] == "ml-inversores/alvo.pdf"


def test_rerank_devolve_no_maximo_n_final():
    candidatos = [(f"doc {i}", {"arquivo": f"{i}.pdf"}) for i in range(20)]
    assert len(_rerankar(candidatos, "doc", n_final=5)) == 5


def test_rerank_sem_candidatos_devolve_vazio():
    assert _rerankar([], "qualquer coisa", n_final=5) == []


def test_termos_extra_resgatam_fonte_que_a_pergunta_nao_nomeia():
    """A pergunta diz 'Paderborn'; o arquivo é 'stender...'. Sem os termos da
    expansão, o boost por autor não teria como funcionar."""
    candidatos = [
        ("conteúdo genérico sobre bancada", {"arquivo": "outro.pdf"}),
        ("descrição do conjunto de dados", {"arquivo": "stender-2020.pdf"}),
    ]
    com_extra = _rerankar(candidatos, "dataset de Paderborn", n_final=1,
                          termos_extra=["stender"])
    assert com_extra[0][1]["arquivo"] == "stender-2020.pdf"


# ── Camada 1: o gatilho de revisão bibliográfica ───────────────────────────

@pytest.mark.parametrize("pergunta", [
    "faça uma revisão bibliográfica sobre detecção de anomalia",
    "qual o estado da arte em RUL de inversores?",
])
def test_pergunta_de_revisao_e_reconhecida(pergunta):
    assert eh_query_de_revisao(pergunta) is True


def test_pergunta_pontual_nao_dispara_modo_revisao():
    """O modo revisão amplia o orçamento de busca; disparar à toa custa
    latência e polui o contexto."""
    assert eh_query_de_revisao("qual o NPR do contator?") is False
