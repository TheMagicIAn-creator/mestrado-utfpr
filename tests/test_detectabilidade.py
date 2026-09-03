"""`a_det`, censura e o portão que impede publicar ajuste rejeitado.

POR QUE ESTE TESTE EXISTE
=========================
Três defeitos desta etapa já chegaram a um artefato publicado, e todos os três
produziam número plausível:

1. **Percentil paramétrico de ajuste rejeitado.** A tabela trazia o `a10` da
   Weibull 2P na coluna vizinha a "2P adotada: não". O número existia; o
   ajuste que o produziu tinha sido reprovado.

2. **Censura tratada como detecção.** Trajetória que nunca cruza o limiar não
   tem `a_det = 1,0` — ela não tem `a_det`. Somá-la como detectada em 1,0
   puxa toda estatística para baixo e inventa detecções.

3. **Ajuste sobre grade quantizada.** Com uma grade de 50 pontos, se as
   detecções caem em dois valores, a Weibull ajustada descreve a GRADE e não
   a dispersão. Aderência alta nesse caso é artefato.

Nada aqui precisa de torch, dataset ou checkpoint: o scorer é uma função com
comportamento conhecido, e o `a_det` de cada caso é conferível à mão.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.detectabilidade import (
    MINIMO_R2_PAPEL,
    ajustar_weibull_2p,
    curva_pod,
    detectabilidade_do_modelo,
    percentil_parametrico,
    percentis_empiricos,
    resumo,
    varrer_trajetoria,
)

GRADE = tuple(np.round(np.linspace(0.02, 1.0, 50), 4).tolist())


def _injetar(janela, a):
    """A "janela" é só o rótulo da trajetória; a injeção devolve o par."""
    return (janela, float(a))


def _scorer_com_gatilho(gatilho: float):
    """Escore que cruza 1,0 exatamente quando `a >= gatilho` da trajetória.

    Cada trajetória carrega seu próprio gatilho, então `a_det` é previsível.
    """
    def scorer(itens):
        return np.asarray(
            [2.0 if a >= (janela + gatilho) else 0.5 for janela, a in itens],
            dtype=float,
        )
    return scorer


def _varrer(gatilhos, *, confirmacoes=3):
    return detectabilidade_do_modelo(
        "modelo",
        "igbt",
        _scorer_com_gatilho(0.0),
        list(gatilhos),
        _injetar,
        limiar=1.0,
        grade=GRADE,
        confirmacoes=confirmacoes,
    )


# ── a confirmação ──────────────────────────────────────────────────────────

@pytest.mark.leve
def test_a_det_e_o_primeiro_ponto_da_sequencia_confirmada():
    """Não o ponto em que a confirmação se completa — o em que ela começa."""
    def scorer(itens):
        # cruza a partir de a >= 0.30 e nunca mais volta
        return np.asarray([2.0 if a >= 0.30 else 0.5 for _, a in itens], dtype=float)

    a_det = varrer_trajetoria(scorer, 0.0, _injetar, 1.0, GRADE, confirmacoes=3)

    primeiros = [a for a in GRADE if a >= 0.30]
    assert a_det == pytest.approx(primeiros[0])


@pytest.mark.leve
def test_cruzamento_isolado_nao_conta_como_deteccao():
    """O caso que a confirmação existe para rejeitar."""
    def scorer(itens):
        # um único ponto acima, no meio da grade
        return np.asarray(
            [2.0 if abs(a - GRADE[20]) < 1e-9 else 0.5 for _, a in itens], dtype=float
        )

    assert varrer_trajetoria(scorer, 0.0, _injetar, 1.0, GRADE, confirmacoes=3) is None


@pytest.mark.leve
def test_dois_cruzamentos_isolados_tambem_nao_bastam():
    def scorer(itens):
        alvos = {GRADE[10], GRADE[30]}
        return np.asarray(
            [2.0 if any(abs(a - x) < 1e-9 for x in alvos) else 0.5 for _, a in itens],
            dtype=float,
        )

    assert varrer_trajetoria(scorer, 0.0, _injetar, 1.0, GRADE, confirmacoes=3) is None


@pytest.mark.leve
def test_confirmacao_maior_que_a_grade_estoura():
    with pytest.raises(ValueError, match="confirmação"):
        varrer_trajetoria(
            _scorer_com_gatilho(0.0), 0.0, _injetar, 1.0, GRADE[:2], confirmacoes=5
        )


@pytest.mark.leve
def test_scorer_que_devolve_tamanho_errado_estoura():
    def scorer(itens):
        return np.zeros(len(itens) - 1)

    with pytest.raises(ValueError, match="scorer"):
        varrer_trajetoria(scorer, 0.0, _injetar, 1.0, GRADE)


# ── censura ────────────────────────────────────────────────────────────────

@pytest.mark.leve
def test_trajetoria_que_nunca_cruza_e_censurada_e_nao_detectada_em_1():
    resultado = _varrer([0.1, 0.2, 5.0])   # a terceira nunca cruza

    assert resultado.n_trajetorias == 3
    assert resultado.n_detectadas == 2
    assert resultado.censura_presente is True
    assert bool(resultado.eventos[2]) is False


@pytest.mark.leve
def test_sem_censura_a_marca_some():
    resultado = _varrer([0.1, 0.2, 0.3])

    assert resultado.n_detectadas == 3
    assert resultado.censura_presente is False
    assert resultado.fracao_detectada == 1.0


@pytest.mark.leve
def test_a_censurada_nao_entra_nos_percentis_empiricos():
    """Se entrasse como 1,0, o a90 apareceria onde não há observação."""
    poucos = _varrer([0.1, 0.2] + [5.0] * 8)   # 20% detectadas

    percentis = percentis_empiricos(poucos)

    assert percentis["a10_empirico"] is not None
    assert percentis["a50_empirico"] is None, (
        "com 20% detectadas não existe mediana observada"
    )
    assert percentis["a90_empirico"] is None


@pytest.mark.leve
def test_percentil_alem_da_fracao_detectada_e_none():
    meio = _varrer([0.05, 0.1, 0.15, 0.2, 0.25, 5.0, 5.0, 5.0, 5.0, 5.0])

    assert meio.fracao_detectada == pytest.approx(0.5)
    assert percentis_empiricos(meio)["a50_empirico"] is not None
    assert percentis_empiricos(meio)["a90_empirico"] is None


# ── POD ────────────────────────────────────────────────────────────────────

@pytest.mark.leve
def test_a_pod_e_monotona_e_termina_na_fracao_detectada():
    resultado = _varrer([0.1, 0.3, 0.5, 5.0])
    curva = curva_pod(resultado)

    pods = curva["pod"]
    assert all(b >= a for a, b in zip(pods, pods[1:])), "POD não pode decrescer"
    assert pods[-1] == pytest.approx(resultado.fracao_detectada)
    assert len(curva["a"]) == len(pods) == len(GRADE)


# ── o portão do resumo paramétrico ─────────────────────────────────────────

@pytest.mark.leve
def test_ajuste_rejeitado_nao_produz_percentil():
    """O defeito exato que foi publicado uma vez."""
    poucas = _varrer([0.1, 0.2, 0.3])          # 3 detecções, abaixo do mínimo
    ajuste = ajustar_weibull_2p(poucas)

    assert ajuste["adotado"] is False
    assert percentil_parametrico(ajuste, 0.10) is None
    assert percentil_parametrico(ajuste, 0.50) is None


@pytest.mark.leve
def test_o_resumo_nunca_traz_parametrico_com_2p_reprovada():
    """Guarda sobre a LINHA publicada, não sobre a função interna."""
    linha = resumo(_varrer([0.1, 0.2, 0.3]))

    assert linha["weibull_2p_adotada"] is False
    assert linha["a10_parametrico"] is None
    assert linha["a50_parametrico"] is None
    assert linha["weibull_2p_motivo_da_rejeicao"]


@pytest.mark.leve
def test_a_rejeicao_diz_o_motivo():
    ajuste = ajustar_weibull_2p(_varrer([0.1, 0.2, 0.3]))
    assert "detec" in ajuste["motivo_da_rejeicao"]


@pytest.mark.leve
def test_grade_quantizada_reprova_mesmo_com_muitas_deteccoes():
    """Vinte trajetórias, mas `a_det` cai em dois valores só.

    A Weibull ajustada descreveria a grade. Aderência alta aqui é artefato, e
    era o caso que passava despercebido.
    """
    gatilhos = [0.1] * 10 + [0.5] * 10
    ajuste = ajustar_weibull_2p(_varrer(gatilhos))

    assert ajuste["n_detectadas"] == 20
    assert ajuste["valores_distintos"] == 2
    assert ajuste["adotado"] is False
    assert "distintos" in ajuste["motivo_da_rejeicao"]


@pytest.mark.leve
def test_amostra_boa_e_adotada_e_ai_o_parametrico_aparece():
    """O outro lado do portão: com dispersão real, o ajuste vale."""
    rng = np.random.default_rng(7)
    gatilhos = np.clip(rng.weibull(2.0, size=60) * 0.30, 0.02, 0.95).tolist()
    resultado = _varrer(gatilhos)
    ajuste = ajustar_weibull_2p(resultado)

    assert ajuste["convergiu"] is True
    assert ajuste["adotado"] is True, ajuste["motivo_da_rejeicao"]
    assert ajuste["r2_papel"] >= MINIMO_R2_PAPEL
    assert percentil_parametrico(ajuste, 0.10) > 0.0


@pytest.mark.leve
def test_o_beta_estimado_recupera_a_forma_geradora():
    """Sem isto, "convergiu" não diz que convergiu para o lugar certo."""
    rng = np.random.default_rng(11)
    gatilhos = np.clip(rng.weibull(2.0, size=400) * 0.25, 0.02, 0.99).tolist()

    ajuste = ajustar_weibull_2p(_varrer(gatilhos))

    assert ajuste["beta"] == pytest.approx(2.0, rel=0.25), (
        f"beta veio {ajuste['beta']}, longe da forma 2,0 que gerou a amostra"
    )


@pytest.mark.leve
def test_o_ajuste_respeita_a_censura():
    """Ignorar a censura enviesaria eta para baixo.

    Mesma amostra detectada, mas com trajetórias censuradas somadas: o eta tem
    de subir, porque há massa observada além do último ponto detectado.
    """
    rng = np.random.default_rng(3)
    base = np.clip(rng.weibull(2.0, size=40) * 0.25, 0.02, 0.90).tolist()

    sem_censura = ajustar_weibull_2p(_varrer(base))
    com_censura = ajustar_weibull_2p(_varrer(base + [5.0] * 20))

    assert com_censura["n_censuradas"] == 20
    assert com_censura["eta"] > sem_censura["eta"], (
        "censura à direita tem de empurrar eta para cima"
    )


# ── o contrato da linha publicada ──────────────────────────────────────────

@pytest.mark.leve
def test_a_linha_declara_que_o_eixo_nao_e_tempo():
    """A colisão de símbolos com a confiabilidade física é o risco permanente."""
    linha = resumo(_varrer([0.1, 0.2, 0.3]))

    assert linha["eixo_nao_e_tempo"] is True
    assert linha["evidence_level"] == "E2"


@pytest.mark.leve
def test_a_linha_traz_o_empirico_antes_do_parametrico():
    rng = np.random.default_rng(5)
    gatilhos = np.clip(rng.weibull(2.0, size=60) * 0.30, 0.02, 0.95).tolist()
    chaves = list(resumo(_varrer(gatilhos)))

    assert chaves.index("a10_empirico") < chaves.index("a10_parametrico")
