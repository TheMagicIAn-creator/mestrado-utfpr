"""
Testes das funções fechadas da Weibull — identidades matemáticas, não regressão.

O valor destes testes é que eles não conferem "o número de ontem": conferem
**identidades da distribuição**. Se `R + F ≠ 1` ou `h ≠ f/R`, a implementação
está errada mesmo que o resultado pareça plausível no gráfico.

Rodam sem torch e sem dataset.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.ml.confiabilidade import (
    acumulada,
    classificar_forma,
    confiabilidade,
    curvas,
    densidade,
    diagnostico_papel_weibull,
    eixos_papel_weibull,
    intensidade_weibull,
    marcos,
    mediana_de_posto,
    posicoes_probabilidade_censuradas,
    quantil,
    taxa_acumulada,
    taxa_falha,
    vida_media,
)

# Cobre mortalidade infantil, aleatório e desgaste.
FORMAS = [0.5, 1.0, 1.5, 2.5, 4.4, 5.4]
ESCALAS = [1.0, 39.3, 97.1]


# ── identidades que definem a distribuição ─────────────────────────────────

@pytest.mark.parametrize("beta", FORMAS)
@pytest.mark.parametrize("eta", ESCALAS)
def test_confiabilidade_e_acumulada_somam_um(beta, eta):
    t = np.linspace(0.01, 3 * eta, 50)
    np.testing.assert_allclose(confiabilidade(t, beta, eta) + acumulada(t, beta, eta),
                               1.0, rtol=1e-12)


@pytest.mark.parametrize("beta", FORMAS)
def test_taxa_de_falha_e_densidade_sobre_confiabilidade(beta):
    """h(t) = f(t)/R(t) — a definição de taxa de falha."""
    eta = 39.3
    t = np.linspace(0.5, 2 * eta, 60)
    np.testing.assert_allclose(taxa_falha(t, beta, eta),
                               densidade(t, beta, eta) / confiabilidade(t, beta, eta),
                               rtol=1e-10)


@pytest.mark.parametrize("beta", FORMAS)
def test_intensidade_canonica_preserva_alias_legado(beta):
    eixo = np.linspace(0.01, 1.0, 50)
    np.testing.assert_allclose(
        intensidade_weibull(eixo, beta, 0.4),
        taxa_falha(eixo, beta, 0.4),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("beta", FORMAS)
def test_taxa_acumulada_e_menos_log_da_confiabilidade(beta):
    """H(t) = −ln R(t), onde a comparação é numericamente bem-condicionada.

    A identidade é exata, mas `-ln R` NÃO é forma estável de calcular H: para
    H pequeno, R ≈ 1 e o logaritmo perde precisão. Aqui a comparação fica na
    faixa em que ela é significativa (R < 0,99); a estabilidade na cauda
    inferior é conferida pelo teste seguinte.
    """
    eta = 12.0
    t = np.linspace(0.5, 2 * eta, 60)
    R = confiabilidade(t, beta, eta)
    ok = R < 0.99
    np.testing.assert_allclose(taxa_acumulada(t[ok], beta, eta),
                               -np.log(R[ok]), rtol=1e-9)


@pytest.mark.parametrize("beta", FORMAS)
def test_acumulada_e_estavel_para_risco_muito_pequeno(beta):
    """`F = 1 - exp(-H)` sofre cancelamento catastrófico quando H → 0.

    Este teste é o que motivou trocar a implementação por `-expm1(-H)`. Com a
    forma ingênua, β = 5,4 e t = η/24 devolvem erro de ordens de grandeza — e
    F é a base de todo o papel de Weibull.
    """
    eta = 12.0
    t = np.array([eta / 24, eta / 12, eta / 6])
    H = taxa_acumulada(t, beta, eta)
    np.testing.assert_allclose(acumulada(t, beta, eta), -np.expm1(-H), rtol=1e-14)
    # e continua batendo com a expansão de primeira ordem F ≈ H para H pequeno
    pequeno = H < 1e-6
    if pequeno.any():
        np.testing.assert_allclose(acumulada(t, beta, eta)[pequeno],
                                   H[pequeno], rtol=1e-6)


@pytest.mark.parametrize("beta", FORMAS)
@pytest.mark.parametrize("eta", ESCALAS)
def test_eta_e_a_vida_caracteristica(beta, eta):
    """R(η) = e⁻¹ ≈ 0,368 para QUALQUER β — é o que define η."""
    assert confiabilidade(eta, beta, eta) == pytest.approx(math.exp(-1.0), rel=1e-12)


@pytest.mark.parametrize("beta", FORMAS)
def test_vida_mediana_fechada(beta):
    """B50 = η·(ln 2)^(1/β)."""
    eta = 85.4
    esperado = eta * math.log(2.0) ** (1.0 / beta)
    assert quantil(0.50, beta, eta) == pytest.approx(esperado, rel=1e-12)


@pytest.mark.parametrize("beta", FORMAS)
def test_quantil_inverte_a_acumulada(beta):
    """F(B_p) = p, por construção."""
    eta = 39.3
    for p in (0.01, 0.10, 0.50, 0.90):
        assert float(acumulada(quantil(p, beta, eta), beta, eta)) == pytest.approx(p, rel=1e-10)


def test_beta_igual_um_e_exponencial():
    """Com β = 1 a Weibull é exponencial: h constante = 1/η, MTTF = η."""
    eta = 50.0
    t = np.linspace(0.1, 200, 40)
    np.testing.assert_allclose(taxa_falha(t, 1.0, eta), 1.0 / eta, rtol=1e-12)
    assert vida_media(1.0, eta) == pytest.approx(eta, rel=1e-12)


# ── monotonia: é o que dá sentido de engenharia a β ────────────────────────

def test_taxa_de_falha_cresce_quando_beta_maior_que_um():
    h = taxa_falha(np.linspace(1, 100, 50), 2.5, 40.0)
    assert np.all(np.diff(h) > 0), "β > 1 deve dar h(t) crescente (desgaste)"


def test_taxa_de_falha_decresce_quando_beta_menor_que_um():
    h = taxa_falha(np.linspace(1, 100, 50), 0.5, 40.0)
    assert np.all(np.diff(h) < 0), "β < 1 deve dar h(t) decrescente"


@pytest.mark.parametrize("beta", FORMAS)
def test_confiabilidade_e_sempre_decrescente(beta):
    R = confiabilidade(np.linspace(0.01, 300, 120), beta, 40.0)
    assert np.all(np.diff(R) <= 0)
    assert R[0] <= 1.0 and R[-1] >= 0.0


# ── a ressalva do IC: onde estava a afirmação sem lastro ───────────────────

def test_ic_que_cruza_um_torna_a_leitura_nao_conclusiva():
    """O log dizia "β > 1 → desgaste" sem olhar o intervalo.

    Se o IC95 de β cruza 1, o dado não distingue desgaste de falha aleatória.
    """
    r = classificar_forma(1.4, ic_beta=(0.8, 2.1))
    assert r["conclusivo"] is False
    assert "CRUZA 1" in r["leitura"]
    assert "não afirmar" in r["leitura"].lower()


def test_ic_que_nao_cruza_um_sustenta_a_leitura_de_desgaste():
    r = classificar_forma(4.39, ic_beta=(3.1, 5.8))
    assert r["conclusivo"] is True
    assert r["regime"] == "desgaste"
    assert "crescente" in r["leitura"]


def test_beta_abaixo_de_um_com_ic_limpo_e_mortalidade_infantil():
    r = classificar_forma(0.6, ic_beta=(0.4, 0.85))
    assert r["conclusivo"] is True
    assert r["regime"] == "mortalidade_infantil"


def test_sem_ic_a_leitura_sai_mas_como_pontual():
    r = classificar_forma(2.3)
    assert r["conclusivo"] is True
    assert r["regime"] == "desgaste"


def test_eixo_de_magnitude_nao_autoriza_desgaste_nem_manutencao():
    r = classificar_forma(4.39, ic_beta=(3.1, 5.8), eixo_tempo=False)
    assert r["conclusivo"] is True
    assert r["regime"] == "intensidade_deteccao_crescente"
    assert r["inferencia_manutencao_autorizada"] is False
    assert "nao significa desgaste" in r["leitura"].lower()


# ── papel de Weibull ───────────────────────────────────────────────────────

def test_papel_de_weibull_lineariza_a_distribuicao():
    """Na escala ln t × ln(−ln(1−F)), a Weibull é reta de inclinação β."""
    beta, eta = 2.5, 40.0
    t = np.linspace(5, 120, 60)
    x, y = eixos_papel_weibull(t, acumulada(t, beta, eta))
    inclinacao = np.polyfit(x, y, 1)[0]
    assert inclinacao == pytest.approx(beta, rel=1e-6)


def test_mediana_de_posto_e_crescente_e_interna():
    p = mediana_de_posto(38)
    assert p.shape == (38,)
    assert np.all(np.diff(p) > 0)
    assert 0.0 < p[0] and p[-1] < 1.0


def test_posicoes_censuradas_usam_o_tamanho_total_da_amostra():
    tempos = np.concatenate([np.linspace(0.5, 0.9, 12), np.ones(19)])
    eventos = np.concatenate([np.ones(12, dtype=bool), np.zeros(19, dtype=bool)])
    t, f, metodo = posicoes_probabilidade_censuradas(tempos, eventos)

    assert len(t) == len(f) == 12
    assert f[-1] == pytest.approx((12 - 0.3) / (31 + 0.4))
    assert f[-1] < 0.40
    assert "Kaplan-Meier" in metodo


def test_posicoes_censuradas_agrupam_empates_da_grade():
    tempos = np.array([0.2, 0.2, 0.2, 0.4, 0.4, 1.0])
    eventos = np.array([True, True, True, True, True, False])
    t, f, metodo = posicoes_probabilidade_censuradas(tempos, eventos)

    np.testing.assert_allclose(t, [0.2, 0.4])
    assert len(f) == 2
    assert np.all(np.diff(f) > 0)
    assert "empates agrupados" in metodo


def test_diagnostico_papel_detecta_ajuste_incompativel():
    tempos = np.array([0.20, *np.linspace(0.70, 0.95, 29), 1.0])
    eventos = np.array([True] * 30 + [False])
    d = diagnostico_papel_weibull(tempos, eventos, beta=8.6, eta=0.90)

    assert d["n_pontos"] == 30
    assert d["r2"] < 0.90


def test_relatorio_respeita_metodo_e_limiar_operacionais_canonicos():
    from scripts.relatorio_confiabilidade import _limiar_operacional

    metodo, valor = _limiar_operacional({
        "score_method": "mse",
        "score_threshold": 2.58,
        "limiar_localizado": 16.53,
        "limiar": 2.58,
    })
    assert metodo == "mse"
    assert valor == pytest.approx(2.58)


# ── blocos de saída ────────────────────────────────────────────────────────

def test_curvas_tem_todas_as_funcoes_e_mesmo_comprimento():
    c = curvas(4.39, 39.3, t_max=60.0, n=50)
    for chave in ("t", "R", "F", "f", "h", "H"):
        assert len(c[chave]) == 50, chave
    assert all(np.isfinite(c[k]).all() for k in ("R", "F", "H"))


def test_marcos_ordenam_b1_b10_mediana():
    m = marcos(4.39, 39.3)
    assert m["b1"] < m["b10"] < m["vida_mediana"]
    assert m["R_em_eta"] == pytest.approx(math.exp(-1.0))


def test_b10_e_menor_que_mttf_em_distribuicao_assimetrica():
    """O motivo de B10 ser melhor indicador de manutenção que o MTTF."""
    m = marcos(1.5, 100.0)
    assert m["b10"] < m["mttf"]


# ── entradas inválidas ─────────────────────────────────────────────────────

@pytest.mark.parametrize("beta,eta", [(0, 1), (-1, 1), (1, 0), (1, -5),
                                      (float("nan"), 1), (1, float("inf"))])
def test_parametros_invalidos_sao_recusados(beta, eta):
    with pytest.raises(ValueError):
        confiabilidade(1.0, beta, eta)


def test_tempo_negativo_e_recusado():
    with pytest.raises(ValueError):
        confiabilidade(-1.0, 2.0, 10.0)


def test_quantil_fora_do_intervalo_aberto():
    for p in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            quantil(p, 2.0, 10.0)
