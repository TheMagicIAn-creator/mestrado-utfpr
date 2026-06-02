"""
Sprint 4 — separação de memória de AVALIAÇÃO vs PRODUÇÃO (8.2 / critério #18).

Avaliações automatizadas não podem contaminar a memória de produção
(`sessoes_pv`): vão para a coleção separada `avaliacoes_agente`.
"""

import re
from pathlib import Path

from src.core.config import (
    NOME_COLECAO_AVALIACOES,
    NOME_COLECAO_SESSOES,
    RAIZ_PROJETO,
)


def test_colecao_avaliacao_e_separada():
    assert NOME_COLECAO_AVALIACOES
    assert NOME_COLECAO_AVALIACOES != NOME_COLECAO_SESSOES


def test_avaliadores_gravam_na_colecao_de_avaliacao():
    for script in ("avaliar_agente_100.py", "avaliar_respostas_reais.py"):
        txt = (Path(RAIZ_PROJETO) / "scripts" / script).read_text(encoding="utf-8")
        assert "NOME_COLECAO_AVALIACOES" in txt, script
        m = re.search(
            r"def gravar_memoria.*?get_or_create_collection\(name=(\w+)\)",
            txt, re.S,
        )
        assert m and m.group(1) == "NOME_COLECAO_AVALIACOES", script


def test_bateria_100_nao_grava_memoria_por_padrao():
    txt = (Path(RAIZ_PROJETO) / "scripts" / "avaliar_agente_100.py").read_text(
        encoding="utf-8"
    )
    assert "def main(gravar_memorias: bool = False)" in txt
    assert "main(gravar_memorias=_args.com_memoria)" in txt
