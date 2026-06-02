"""
Sprint 4 — confiança da busca web (seção 11).

A classificação A>B>C>D evita tratar Wikipedia (C) como fonte normativa.
"""

from src.conhecimento.web_search import _nivel_confianca


def test_nivel_a_norma_doi():
    assert _nivel_confianca("https://doi.org/10.1234/x", "DOI")[0] == "A"
    assert _nivel_confianca("https://www.iso.org/standard/123", "ISO")[0] == "A"
    assert _nivel_confianca("https://www.ieee.org/x", "IEEE")[0] == "A"


def test_nivel_b_universidade():
    assert _nivel_confianca("https://web.mit.edu/page", "MIT")[0] == "B"
    assert _nivel_confianca("https://link.springer.com/x", "Springer")[0] == "B"


def test_nivel_c_wikipedia():
    assert _nivel_confianca("https://pt.wikipedia.org/wiki/X", "Wikipedia (pt)")[0] == "C"


def test_nivel_d_informal():
    assert _nivel_confianca("https://um-blog-qualquer.com/post", "Blog")[0] == "D"
