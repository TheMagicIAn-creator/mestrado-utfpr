"""
Sprint 2 — SMD probabilística (item 4.3).

SMD_95 = menor severidade cuja TAXA DE DETECÇÃO (sobre repetições) atinge o
alvo (0,95). Substitui a SMD por "primeira média acima do limiar".
"""

from src.ml.injecao_falhas import smd_probabilistico


def test_smd_95_usa_taxa_de_deteccao():
    deteccoes = {
        0.1: [False] * 10,
        0.3: [True] * 5 + [False] * 5,   # 50%
        0.5: [True] * 9 + [False],       # 90%
        1.0: [True] * 10,                # 100%
    }
    r = smd_probabilistico(deteccoes, alvo=0.95)
    assert r["smd_pontual"] == 0.3       # primeira com qualquer detecção
    assert r["smd_95"] == 1.0            # primeira com taxa >= 0,95
    assert abs(r["taxa_deteccao"][0.5] - 0.9) < 1e-9
    assert r["n_repeticoes"][1.0] == 10


def test_smd_95_none_quando_nunca_alcanca_alvo():
    deteccoes = {0.5: [True, False], 1.0: [True, False]}  # 50% sempre
    r = smd_probabilistico(deteccoes, alvo=0.95)
    assert r["smd_95"] is None
    assert r["smd_pontual"] == 0.5
