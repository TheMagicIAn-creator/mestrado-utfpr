"""Perguntas de CONTAGEM não podem ser respondidas por amostra de busca.

Caso real (29/07/2026): perguntado "quais foram as 10 últimas memórias
consolidadas?", o agente respondeu com 4 — o número de chunks que a busca
semântica devolveu — e, ao ser contestado ("só tem 4 aí"), CONFIRMOU:
"o diretório possui apenas 4 arquivos consolidados". O vault tinha 26.

A amostra virou população, e a contestação do pesquisador virou confirmação.
Estes testes fixam a correção: contagem e lista saem da varredura dos
metadados do índice, sem embeddings e sem LLM.
"""

from __future__ import annotations

from src.conhecimento.obsidian import (
    inventario_por_classe,
    responder_inventario_vault,
)


class _ColecaoFalsa:
    """Índice mínimo: só o que `inventario_por_classe` lê (ids + metadados)."""

    def __init__(self, metas):
        self._metas = list(metas)
        self.buscas_semanticas = 0

    def get(self, include=None, **_kw):
        del include
        return {
            "ids": [m["caminho_obsidian"] for m in self._metas],
            "metadatas": list(self._metas),
        }

    def query(self, *_a, **_k):  # pragma: no cover - não deve ser chamado
        self.buscas_semanticas += 1
        raise AssertionError("inventário não pode usar busca semântica")


def _memorias(n: int, chunks_por_nota: int = 3):
    """n memórias consolidadas, cada uma quebrada em vários chunks."""
    metas = []
    for i in range(n):
        caminho = f"memorias/2026-07-{(i % 28) + 1:02d}_{i:02d}-00-00_consolidado.md"
        for c in range(chunks_por_nota):
            metas.append({
                "caminho_obsidian": caminho,
                "classe_fonte": "memoria_consolidada",
                "chunk_index": c,
                "titulo": f"Consolidado {i}",
                "data_registro": f"2026-07-{(i % 28) + 1:02d}",
            })
    return metas


# ── o caso que falhou ────────────────────────────────────────────────────────

def test_conta_arquivos_e_nao_chunks():
    """26 notas em 78 chunks continuam sendo 26 — foi aqui que errou."""
    colecao = _ColecaoFalsa(_memorias(26))
    itens = inventario_por_classe(colecao, {"memoria_consolidada"})
    assert len(itens) == 26


def test_resposta_declara_o_total_mesmo_listando_menos():
    """Listar 10 sem dizer o total foi o que permitiu ler '4' como 'só 4'."""
    colecao = _ColecaoFalsa(_memorias(26))
    r = responder_inventario_vault(
        colecao, "quais foram as 10 últimas memórias consolidadas?"
    )
    assert "26" in r, "o total precisa aparecer explicitamente"
    assert r.count("\n| 1 ") or "| 1 |" in r
    linhas = [ln for ln in r.splitlines() if ln.startswith("| ") and "|" in ln]
    # cabeçalho + separador + 10 registros
    assert len(linhas) == 12, linhas
    assert "outras 16" in r


def test_nao_afirma_amostra_como_total():
    colecao = _ColecaoFalsa(_memorias(26))
    r = responder_inventario_vault(colecao, "quantas memórias consolidadas existem?")
    assert "26" in r
    assert "metadados do índice" in r
    assert "amostra" in r.lower()


def test_pede_menos_do_que_existe_respeita_o_pedido():
    colecao = _ColecaoFalsa(_memorias(26))
    r = responder_inventario_vault(colecao, "me dê as 3 últimas memórias consolidadas")
    linhas = [ln for ln in r.splitlines() if ln.startswith("| ")]
    assert len(linhas) == 5  # cabeçalho + separador + 3


def test_pede_mais_do_que_existe_nao_inventa():
    colecao = _ColecaoFalsa(_memorias(4))
    r = responder_inventario_vault(colecao, "as 10 últimas memórias consolidadas")
    linhas = [ln for ln in r.splitlines() if ln.startswith("| ")]
    assert len(linhas) == 6  # cabeçalho + separador + 4
    assert "4" in r and "Todas" in r


# ── ordenação ────────────────────────────────────────────────────────────────

def test_ordena_da_mais_recente_para_a_mais_antiga():
    colecao = _ColecaoFalsa(_memorias(5))
    itens = inventario_por_classe(colecao, {"memoria_consolidada"})
    nomes = [i["nome"] for i in itens]
    assert nomes == sorted(nomes, reverse=True)


def test_pedido_das_primeiras_inverte_a_ordem():
    colecao = _ColecaoFalsa(_memorias(6))
    recentes = responder_inventario_vault(colecao, "as 2 últimas memórias consolidadas")
    antigas = responder_inventario_vault(colecao, "as 2 primeiras memórias consolidadas")
    assert recentes != antigas
    assert "mais recentes" in recentes and "mais antigas" in antigas


# ── escopo: não sequestrar perguntas que pedem conteúdo ──────────────────────

def test_pergunta_de_conteudo_segue_para_o_rag():
    colecao = _ColecaoFalsa(_memorias(26))
    for pergunta in (
        "resuma a última memória consolidada",
        "sobre o que falamos na última sessão?",
        "qual foi o assunto das memórias consolidadas?",
    ):
        assert responder_inventario_vault(colecao, pergunta) is None, pergunta


def test_pergunta_fora_do_tema_segue_para_o_rag():
    colecao = _ColecaoFalsa(_memorias(26))
    for pergunta in (
        "quais são os resultados do autoencoder?",
        "liste os artigos sobre IGBT",
        "quantas falhas a FMECA tem?",
    ):
        assert responder_inventario_vault(colecao, pergunta) is None, pergunta


def test_classe_errada_nao_e_contada():
    """Sessões não podem inflar a contagem de memórias consolidadas."""
    metas = _memorias(3) + [
        {"caminho_obsidian": "sessoes/2026-07-29_10-00_sessao_web.md",
         "classe_fonte": "sessao_atual", "chunk_index": 0,
         "titulo": "Sessão", "data_registro": "2026-07-29"},
    ]
    colecao = _ColecaoFalsa(metas)
    assert len(inventario_por_classe(colecao, {"memoria_consolidada"})) == 3
    r = responder_inventario_vault(colecao, "quantas memórias consolidadas há?")
    assert "**3 memórias consolidadas**" in r


def test_sessoes_tem_inventario_proprio():
    metas = [
        {"caminho_obsidian": f"sessoes_arquivadas/2026-07-{d:02d}_10-00_sessao_web.md",
         "classe_fonte": "sessao_arquivada", "chunk_index": 0,
         "titulo": "Sessão", "data_registro": f"2026-07-{d:02d}"}
        for d in range(1, 15)
    ]
    r = responder_inventario_vault(_ColecaoFalsa(metas), "liste as últimas sessões")
    assert "**14 sessões**" in r


def test_indice_vazio_nao_responde_nada():
    assert responder_inventario_vault(_ColecaoFalsa([]),
                                      "quantas memórias consolidadas?") is None


def test_data_legivel_vem_do_nome_do_arquivo():
    colecao = _ColecaoFalsa([
        {"caminho_obsidian": "memorias/2026-07-29_13-33-49_consolidado.md",
         "classe_fonte": "memoria_consolidada", "chunk_index": 0,
         "titulo": "C", "data_registro": "2026-07-29"},
    ])
    r = responder_inventario_vault(colecao, "quais as memórias consolidadas?")
    assert "29/07/2026 13:33" in r
