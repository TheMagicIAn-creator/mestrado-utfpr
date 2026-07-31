"""Perguntas de CONTAGEM não podem ser respondidas por amostra de busca.

Dois casos reais, em dias seguidos:

1. (29/07) "quais foram as 10 últimas memórias consolidadas?" → o agente
   respondeu 4, o número de chunks que a busca semântica devolveu, e ao ser
   contestado CONFIRMOU que só existiam 4. O vault tinha 26.

2. (30/07, na nuvem) a mesma pergunta → respondeu "15 indexadas", contando
   corretamente o ÍNDICE. Mas o índice da nuvem vem de um snapshot congelado
   em 20/07: 12 consolidações existiam no vault e não estavam nele. E entrou
   na conta `resultados-fase5-ml.md`, que não é consolidação — a classe é
   atribuída pela PASTA, então qualquer .md em `memorias/` virava uma.

Daí as duas fontes: o DISCO responde "o que existe", o ÍNDICE responde "o que
eu consigo buscar", e a diferença tem que aparecer na resposta.
"""

from __future__ import annotations

import pytest

import src.conhecimento.obsidian as ob
from src.conhecimento.obsidian import (
    inventario_por_classe,
    responder_inventario_vault,
)


class _ColecaoFalsa:
    """Índice mínimo: só o que `inventario_por_classe` lê (ids + metadados)."""

    def __init__(self, metas):
        self._metas = list(metas)

    def get(self, include=None, **_kw):
        del include
        return {
            "ids": [m["caminho_obsidian"] for m in self._metas],
            "metadatas": list(self._metas),
        }

    def query(self, *_a, **_k):  # pragma: no cover - não deve ser chamado
        raise AssertionError("inventário não pode usar busca semântica")


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Vault isolado: sem isto o teste contaria as notas reais do projeto."""
    monkeypatch.setattr(ob, "PASTA_VAULT_OBSIDIAN", tmp_path)
    return tmp_path


def _criar(vault, subpasta, nomes):
    destino = vault / subpasta
    destino.mkdir(parents=True, exist_ok=True)
    for nome in nomes:
        (destino / nome).write_text("# nota\n\ncorpo", encoding="utf-8")


def _nomes_consolidados(n):
    return [f"2026-07-{(i % 28) + 1:02d}_{i:02d}-00-00_consolidado.md"
            for i in range(n)]


def _metas(nomes, classe="memoria_consolidada", subpasta="memorias", chunks=3):
    """Metadados de índice, com a nota quebrada em vários chunks."""
    return [
        {"caminho_obsidian": f"{subpasta}/{nome}", "classe_fonte": classe,
         "chunk_index": c, "titulo": nome, "data_registro": nome[:10]}
        for nome in nomes for c in range(chunks)
    ]


# ── caso 1: amostra virando população ────────────────────────────────────────

def test_conta_arquivos_e_nao_chunks(vault):
    nomes = _nomes_consolidados(26)
    _criar(vault, "memorias", nomes)
    itens = inventario_por_classe(_ColecaoFalsa(_metas(nomes)),
                                  {"memoria_consolidada"})
    assert len(itens) == 26


def test_resposta_declara_o_total_mesmo_listando_menos(vault):
    nomes = _nomes_consolidados(26)
    _criar(vault, "memorias", nomes)
    r = responder_inventario_vault(_ColecaoFalsa(_metas(nomes)),
                                   "quais foram as 10 últimas memórias consolidadas?")
    assert "26" in r
    linhas = [ln for ln in r.splitlines() if ln.startswith("| ")]
    assert len(linhas) == 12          # cabeçalho + separador + 10
    assert "outras 16" in r


def test_nao_afirma_amostra_como_total(vault):
    nomes = _nomes_consolidados(26)
    _criar(vault, "memorias", nomes)
    r = responder_inventario_vault(_ColecaoFalsa(_metas(nomes)),
                                   "quantas memórias consolidadas existem?")
    assert "26" in r and "busca semântica" in r


# ── caso 2: índice defasado e classe atribuída pela pasta ────────────────────

def test_arquivo_fora_do_padrao_nao_conta_como_consolidacao(vault):
    """`resultados-fase5-ml.md` mora em memorias/ mas não é consolidação."""
    nomes = _nomes_consolidados(3)
    _criar(vault, "memorias", nomes + ["resultados-fase5-ml.md"])
    metas = _metas(nomes + ["resultados-fase5-ml.md"])
    itens = inventario_por_classe(_ColecaoFalsa(metas), {"memoria_consolidada"})
    assert len(itens) == 3
    assert all("consolidado" in i["nome"] for i in itens)


def test_indice_defasado_e_denunciado_na_resposta(vault):
    """O caso da nuvem: 26 no disco, 14 no snapshot."""
    nomes = _nomes_consolidados(26)
    _criar(vault, "memorias", nomes)
    indexados = sorted(nomes)[:14]          # snapshot congelado
    r = responder_inventario_vault(_ColecaoFalsa(_metas(indexados)),
                                   "liste todas as memórias consolidadas")
    assert "26" in r, "o total tem de vir do disco"
    assert "12 de 26" in r, "a defasagem tem de ser explicitada"
    assert "reconstruir_cerebro_obsidian" in r, "e dizer como resolver"
    assert "⚠️" in r


def test_marca_por_item_o_que_nao_esta_indexado(vault):
    nomes = _nomes_consolidados(4)
    _criar(vault, "memorias", nomes)
    itens = inventario_por_classe(_ColecaoFalsa(_metas(sorted(nomes)[:2])),
                                  {"memoria_consolidada"})
    assert sum(1 for i in itens if i["indexado"]) == 2
    assert sum(1 for i in itens if not i["indexado"]) == 2


def test_indice_vazio_ainda_conta_pelo_disco(vault):
    """Na nuvem sem snapshot, o vault versionado continua sendo a verdade."""
    _criar(vault, "memorias", _nomes_consolidados(5))
    r = responder_inventario_vault(_ColecaoFalsa([]),
                                   "quantas memórias consolidadas?")
    assert "5" in r and "5 de 5" in r


def test_sem_vault_e_sem_indice_nao_responde(vault):
    assert responder_inventario_vault(_ColecaoFalsa([]),
                                      "quantas memórias consolidadas?") is None


# ── ordenação e recorte ──────────────────────────────────────────────────────

def test_ordena_da_mais_recente_para_a_mais_antiga(vault):
    nomes = _nomes_consolidados(5)
    _criar(vault, "memorias", nomes)
    itens = inventario_por_classe(_ColecaoFalsa(_metas(nomes)),
                                  {"memoria_consolidada"})
    ordem = [i["nome"] for i in itens]
    assert ordem == sorted(ordem, reverse=True)


def test_pedido_das_primeiras_inverte_a_ordem(vault):
    nomes = _nomes_consolidados(6)
    _criar(vault, "memorias", nomes)
    col = _ColecaoFalsa(_metas(nomes))
    recentes = responder_inventario_vault(col, "as 2 últimas memórias consolidadas")
    antigas = responder_inventario_vault(col, "as 2 primeiras memórias consolidadas")
    assert "mais recentes" in recentes and "mais antigas" in antigas
    assert recentes != antigas


def test_pede_mais_do_que_existe_nao_inventa(vault):
    nomes = _nomes_consolidados(4)
    _criar(vault, "memorias", nomes)
    r = responder_inventario_vault(_ColecaoFalsa(_metas(nomes)),
                                   "as 10 últimas memórias consolidadas")
    linhas = [ln for ln in r.splitlines() if ln.startswith("| ")]
    assert len(linhas) == 6 and "Todas" in r


# ── classes distintas ────────────────────────────────────────────────────────

def test_classe_errada_nao_e_contada(vault):
    _criar(vault, "memorias", _nomes_consolidados(3))
    _criar(vault, "sessoes", ["2026-07-29_10-00_sessao_web.md"])
    r = responder_inventario_vault(_ColecaoFalsa([]),
                                   "quantas memórias consolidadas há?")
    assert "**3 memórias consolidadas**" in r


def test_sessoes_tem_inventario_proprio(vault):
    _criar(vault, "sessoes_arquivadas",
           [f"2026-07-{d:02d}_10-00_sessao_web.md" for d in range(1, 15)])
    r = responder_inventario_vault(_ColecaoFalsa([]), "liste as últimas sessões")
    assert "**14 sessões**" in r


# ── escopo: não sequestrar perguntas que pedem conteúdo ──────────────────────

def test_pergunta_de_conteudo_segue_para_o_rag(vault):
    _criar(vault, "memorias", _nomes_consolidados(26))
    col = _ColecaoFalsa([])
    for pergunta in (
        "resuma a última memória consolidada",
        "sobre o que falamos na última sessão?",
        "qual foi o assunto das memórias consolidadas?",
    ):
        assert responder_inventario_vault(col, pergunta) is None, pergunta


def test_pergunta_fora_do_tema_segue_para_o_rag(vault):
    _criar(vault, "memorias", _nomes_consolidados(26))
    col = _ColecaoFalsa([])
    for pergunta in (
        "quais são os resultados do autoencoder?",
        "liste os artigos sobre IGBT",
        "quantas falhas a FMECA tem?",
    ):
        assert responder_inventario_vault(col, pergunta) is None, pergunta


def test_data_legivel_vem_do_nome_do_arquivo(vault):
    _criar(vault, "memorias", ["2026-07-29_13-33-49_consolidado.md"])
    r = responder_inventario_vault(_ColecaoFalsa([]), "quais as memórias consolidadas?")
    assert "29/07/2026 13:33" in r
