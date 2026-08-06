"""
Testes do diagnóstico de limiar (`scripts/diagnostico_limiar.py`).

As funções de decisão são puras e rodam sem torch e sem o dataset — de
propósito. O que este script mede (o alvo de FP é imposto? os blocos estão no
mesmo regime?) é justamente o tipo de afirmação que não pode depender de rodar
o pipeline inteiro para ser conferida.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnostico_limiar import (  # noqa: E402
    LIMITE_DRIFT_IQR,
    alvo_foi_atingido,
    deslocamento_iqr,
    limiar_fpr_maximo,
    resumo_regime,
)

# ── o corte estrito ─────────────────────────────────────────────────────────


def test_com_91_janelas_alvo_de_1pct_exige_zero_excedencias():
    """A amostra real não pode esconder 1/91 = 1,10% sob o rótulo de 1%."""
    info = limiar_fpr_maximo(np.linspace(0.0, 9.0, 91), 1.0)
    assert info["max_excedencias"] == 0
    assert info["excedencias_observadas"] == 0
    assert info["fpr_observado_pct"] == 0.0
    assert info["percentil_efetivo"] == 100.0
    assert info["resolucao_amostral_pct"] == pytest.approx(100.0 / 91.0)
    assert info["alvo_resolvivel_na_amostra"] is False


def test_amostra_grande_resolve_o_alvo_e_gasta_o_orcamento():
    info = limiar_fpr_maximo(np.arange(1000, dtype=float), 1.0)
    assert info["max_excedencias"] == 10
    assert info["excedencias_observadas"] == 10
    assert info["fpr_observado_pct"] == 1.0
    assert info["alvo_resolvivel_na_amostra"] is True


def test_orcamento_usa_floor_nunca_arredonda_para_cima():
    """`floor` e não `round`: arredondar para cima violaria o próprio alvo."""
    assert limiar_fpr_maximo(np.arange(150, dtype=float), 1.0)["max_excedencias"] == 1
    assert limiar_fpr_maximo(np.arange(199, dtype=float), 1.0)["max_excedencias"] == 1
    assert limiar_fpr_maximo(np.arange(200, dtype=float), 1.0)["max_excedencias"] == 2


def test_alvo_zero_e_valido_e_significa_nenhuma_excedencia():
    info = limiar_fpr_maximo(np.arange(50, dtype=float), 0.0)
    assert info["max_excedencias"] == 0
    assert info["limiar"] == 49.0


def test_entradas_invalidas_sao_recusadas():
    with pytest.raises(ValueError):
        limiar_fpr_maximo([], 1.0)
    with pytest.raises(ValueError):
        limiar_fpr_maximo([1.0, np.nan], 1.0)
    with pytest.raises(ValueError):
        limiar_fpr_maximo([1.0, 2.0], 100.0)
    with pytest.raises(ValueError):
        limiar_fpr_maximo([1.0, 2.0], -1.0)


def test_corte_estrito_nunca_e_menor_que_o_percentil_equivalente():
    """Se fosse menor, a restrição seria violada por construção."""
    rng = np.random.default_rng(3)
    for n in (44, 91, 300):
        s = rng.lognormal(0, 1, n)
        info = limiar_fpr_maximo(s, 1.0)
        assert float((s > info["limiar"]).mean() * 100.0) <= 1.0 + 1e-12


# ── regime e drift ──────────────────────────────────────────────────────────


def test_resumo_regime_descreve_mediana_e_iqr():
    r = resumo_regime([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert r["n"] == 9
    assert r["mediana"] == 5.0
    assert r["iqr"] == pytest.approx(4.0)
    assert (r["min"], r["max"]) == (1.0, 9.0)


def test_regime_vazio_e_recusado():
    with pytest.raises(ValueError):
        resumo_regime([])


def test_deslocamento_e_medido_em_iqr_do_bloco_de_referencia():
    """Em IQRs, não em Hz: o que importa é o salto FRENTE à dispersão."""
    ref = resumo_regime([50, 51, 52, 53, 54])
    igual = resumo_regime([50, 51, 52, 53, 54])
    longe = resumo_regime([100, 101, 102, 103, 104])
    assert deslocamento_iqr(ref, igual) == 0.0
    assert deslocamento_iqr(ref, longe) > LIMITE_DRIFT_IQR


def test_deslocamento_e_simetrico_em_modulo():
    a = resumo_regime([10, 11, 12, 13, 14])
    b = resumo_regime([20, 21, 22, 23, 24])
    assert deslocamento_iqr(a, b) == pytest.approx(deslocamento_iqr(b, a))


def test_iqr_zero_nao_divide_por_zero():
    """Bloco constante é degenerado, mas não pode explodir o diagnóstico."""
    constante = resumo_regime([7.0] * 20)
    outro = resumo_regime([9.0] * 20)
    assert np.isfinite(deslocamento_iqr(constante, outro))


# ── a pergunta central: o alvo descreve o sistema? ──────────────────────────


def test_alvo_declarado_mas_nao_atingido_e_reportado_como_nao_atingido():
    """O caso REAL: alvo de 1%, FPR observado de 10,23% no teste."""
    assert alvo_foi_atingido(10.227272727272728, 1.0) is False
    assert alvo_foi_atingido(0.0, 1.0) is True
    assert alvo_foi_atingido(1.0, 1.0) is True       # igual satisfaz


def test_igualdade_no_limite_nao_falha_por_ponto_flutuante():
    assert alvo_foi_atingido(0.1 + 0.2, 0.3) is True
