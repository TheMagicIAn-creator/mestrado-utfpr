"""Contratos dos utilitarios textuais compartilhados."""

from pathlib import Path

import pytest

from src.core.texto import normalizar_busca, normalizar_espacos, normalizar_sem_acentos


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("Tensão MÉDIA", "tensao media"),
        (None, ""),
        (123, "123"),
    ],
)
def test_normalizar_sem_acentos(entrada, esperado):
    assert normalizar_sem_acentos(entrada) == esperado


def test_normalizar_espacos_preserva_pontuacao():
    assert normalizar_espacos("  Tensão,\n  MÉDIA! ") == "tensao, media!"


def test_normalizar_busca_remove_pontuacao_e_compacta():
    assert normalizar_busca("  Tensão,\n  MÉDIA! ") == "tensao media"


def test_normalizadores_e_adaptadores_nao_voltam_a_ser_duplicados():
    raiz = Path(__file__).resolve().parents[1] / "src"
    fontes = "\n".join(
        caminho.read_text(encoding="utf-8")
        for caminho in raiz.rglob("*.py")
    )

    assert "def _normalizar(" not in fontes
    assert "def _tokens(" not in fontes
    assert "def _log(" not in fontes
