"""Limites estruturais definidos no fechamento da auditoria geral."""

from pathlib import Path


RAIZ_SRC = Path(__file__).resolve().parents[1] / "src"
MAX_LINHAS_MODULO = 1_000


def test_nenhum_modulo_src_ultrapassa_mil_linhas():
    excedentes = {}
    for caminho in RAIZ_SRC.rglob("*.py"):
        linhas = len(caminho.read_text(encoding="utf-8").splitlines())
        if linhas > MAX_LINHAS_MODULO:
            excedentes[caminho.relative_to(RAIZ_SRC).as_posix()] = linhas

    assert not excedentes, f"módulos acima de {MAX_LINHAS_MODULO} linhas: {excedentes}"
