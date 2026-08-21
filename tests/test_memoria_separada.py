"""
Sprint 4 — separação de memória de AVALIAÇÃO vs PRODUÇÃO (8.2 / critério #18).

Avaliações automatizadas não podem contaminar a memória de produção
(`sessoes_pv`): vão para a coleção separada `avaliacoes_agente`.
"""

from pathlib import Path

from src.core.config import (
    NOME_COLECAO_AVALIACOES,
    NOME_COLECAO_SESSOES,
    RAIZ_PROJETO,
)


def test_colecao_avaliacao_e_separada():
    assert NOME_COLECAO_AVALIACOES
    assert NOME_COLECAO_AVALIACOES != NOME_COLECAO_SESSOES


def test_avaliador_canonico_e_offline_e_nao_escreve_memoria():
    script = Path(RAIZ_PROJETO) / "scripts" / "avaliar_agente.py"
    txt = script.read_text(encoding="utf-8")
    assert "Avaliação offline" in txt
    assert "indexar_sessao" not in txt
    assert "get_or_create_collection" not in txt
    assert "NOME_COLECAO_SESSOES" not in txt


def test_avaliador_nao_expoe_opt_in_de_memoria():
    txt = (Path(RAIZ_PROJETO) / "scripts" / "avaliar_agente.py").read_text(encoding="utf-8")
    assert "--com-memoria" not in txt
    assert "NOME_COLECAO_AVALIACOES" not in txt
