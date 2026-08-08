"""
Testes do arcabouço POD — contra os valores TABELADOS da fonte, não contra
o resultado de ontem.

O `k1` é o coração do método: qualquer erro nele desloca silenciosamente todos
os limites de tolerância. A NASA/TM-20210018515 publica a tabela; estes testes
a reproduzem.

Rodam sem torch e sem dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.pod_curva import (
    MARGEM_DERIVA,
    checar_normalidade,
    deriva_de_campo,
    fator_k1,
    limite_pod,
    limite_pof,
    limite_pof_empirico,
    verificar_hipoteses,
    viabilidade,
)

# ── k1 contra a tabela publicada (Apêndices C e D da fonte) ────────────────

@pytest.mark.parametrize("m,esperado", [(10, 2.355), (15, 2.068),
                                        (20, 1.926), (30, 1.777)])
def test_k1_do_lado_da_falha_reproduz_a_tabela(m, esperado):
    """k1F para POD 90/95."""
    assert fator_k1(m, p=0.90) == pytest.approx(esperado, abs=0.001)


@pytest.mark.parametrize("n,esperado", [(40, 2.941), (50, 2.862), (60, 2.807)])
def test_k1_do_lado_do_ruido_reproduz_a_tabela(n, esperado):
    """k1N para POF de 1% com 95% de confiança."""
    assert fator_k1(n, p=0.99) == pytest.approx(esperado, abs=0.001)


@pytest.mark.parametrize("n,alvo_pof,esperado", [
    (40, 0.001, 3.865), (40, 0.01, 2.941), (40, 0.02, 2.613),
    (40, 0.05, 2.125), (40, 0.10, 1.697),
])
def test_k1_varia_com_o_alvo_de_pof(n, alvo_pof, esperado):
    """Tabela da fonte para n = 40, variando o POF alvo."""
    assert fator_k1(n, p=1.0 - alvo_pof) == pytest.approx(esperado, abs=0.001)


def test_k1_decresce_com_a_amostra():
    """Amostra maior → limite menos conservador. É o que torna o LS-POD
    conservador frente a um estudo completo: m menor ⇒ k1 maior."""
    ks = [fator_k1(m, p=0.90) for m in (10, 15, 20, 30, 60)]
    assert all(a > b for a, b in zip(ks, ks[1:]))


def test_k1_recusa_amostra_pequena_demais():
    with pytest.raises(ValueError):
        fator_k1(1)


# ── os dois limites ────────────────────────────────────────────────────────

def test_limite_de_pof_fica_acima_da_media_do_ruido():
    r = limite_pof(np.random.default_rng(0).normal(5, 1, 60))
    assert r["limite"] > r["media"]


def test_limite_de_pod_fica_abaixo_da_media_da_falha():
    r = limite_pod(np.random.default_rng(1).normal(20, 2, 20))
    assert r["limite"] < r["media"]


def test_limites_recusam_amostra_invalida():
    for ruim in ([1.0], [1.0, np.nan], []):
        with pytest.raises(ValueError):
            limite_pof(ruim)


# ── o critério de viabilidade e seus três desfechos ────────────────────────

def test_distribuicoes_bem_separadas_dao_ensaio_viavel():
    rng = np.random.default_rng(2)
    sau = rng.normal(1.0, 0.3, 60)
    falha = rng.normal(20.0, 0.8, 20)
    r = viabilidade(sau, falha, limiar=8.0)
    assert r["veredito"] == "viavel"
    assert r["faixa_admissivel_existe"] and r["limiar_dentro_da_faixa"]


def test_distribuicoes_sobrepostas_caracterizam_FALHA_DE_ENSAIO():
    """O desfecho que importa: não existe limiar que sirva.

    É a formalização do que `docs/decisao_fpr_1pct.md` concluiu ao rejeitar o
    corte de FPR ≤ 1% — resultado do ensaio, não defeito de execução.
    """
    rng = np.random.default_rng(3)
    sau = rng.normal(5.0, 2.0, 60)
    falha = rng.normal(6.0, 2.0, 20)      # quase indistinguíveis
    r = viabilidade(sau, falha, limiar=5.5)
    assert r["veredito"] == "ensaio_falhou"
    assert r["faixa_admissivel_existe"] is False
    assert "NENHUM limiar" in r["leitura"]


def test_faixa_existe_mas_limiar_esta_fora():
    rng = np.random.default_rng(4)
    sau = rng.normal(1.0, 0.3, 60)
    falha = rng.normal(20.0, 0.8, 20)
    r = viabilidade(sau, falha, limiar=0.5)    # abaixo do piso de POF
    assert r["veredito"] == "limiar_fora_da_faixa"
    assert r["faixa_admissivel_existe"] is True
    assert "abaixo do piso" in r["leitura"]


def test_resultado_carrega_evidencia_e_fonte():
    rng = np.random.default_rng(5)
    r = viabilidade(rng.normal(1, .3, 60), rng.normal(20, .8, 20), 8.0)
    assert r["evidence_level"] == "E2"
    assert "MIL-HDBK-1823A" in r["fonte"]


# ── deriva de campo: o achado dos regimes de F0, com fórmula ───────────────

def test_ruido_estavel_nao_dispara_gatilho():
    rng = np.random.default_rng(6)
    r = deriva_de_campo(rng.normal(5, 1, 60), rng.normal(5, 1, 60), limiar=20.0)
    assert r["derivou"] is False and r["invalidou"] is False


def test_ruido_maior_em_campo_dispara_deriva():
    rng = np.random.default_rng(7)
    r = deriva_de_campo(rng.normal(5, 1, 60), rng.normal(9, 1, 60), limiar=50.0)
    assert r["derivou"] is True
    assert "DERIVA" in r["leitura"]


def test_piso_de_campo_acima_do_limiar_invalida_a_inspecao():
    """O gatilho forte: o requisito de POF não está sendo cumprido."""
    rng = np.random.default_rng(8)
    r = deriva_de_campo(rng.normal(5, 1, 60), rng.normal(30, 2, 60), limiar=12.0)
    assert r["invalidou"] is True
    assert "INVALIDAÇÃO" in r["leitura"]
    assert "não reapertar o limiar" in r["leitura"]


def test_margem_de_deriva_e_a_da_fonte():
    assert MARGEM_DERIVA == pytest.approx(0.10)


# ── hipóteses: o método é inválido se não valerem ─────────────────────────

def test_escore_monotono_na_magnitude_e_aplicavel():
    rng = np.random.default_rng(9)
    por_mag = {a: rng.normal(1 + 10 * a, 0.5, 30) for a in (0.1, 0.3, 0.5, 1.0)}
    r = verificar_hipoteses(por_mag, rng.normal(1, 0.3, 60))
    assert r["monotonicidade"]["vale"] is True
    assert r["aplicavel"] is True


def test_escore_nao_monotono_invalida_o_arcabouco():
    """Sem monotonicidade não existe curva POD(a) a ajustar."""
    rng = np.random.default_rng(10)
    por_mag = {0.1: rng.normal(10, .5, 30), 0.5: rng.normal(3, .5, 30),
               1.0: rng.normal(7, .5, 30)}
    r = verificar_hipoteses(por_mag, rng.normal(1, 0.3, 60))
    assert r["monotonicidade"]["vale"] is False
    assert r["aplicavel"] is False


def test_piso_artificial_em_zero_e_detectado():
    rng = np.random.default_rng(11)
    sau = np.concatenate([np.zeros(20), rng.normal(5, 1, 40)])
    por_mag = {0.5: rng.normal(10, 1, 20), 1.0: rng.normal(20, 1, 20)}
    r = verificar_hipoteses(por_mag, sau)
    assert r["piso_artificial_em_zero"]["detectado"] is True
    assert r["aplicavel"] is False


def test_saturacao_no_topo_e_detectada():
    rng = np.random.default_rng(12)
    por_mag = {0.3: rng.normal(10, .1, 30), 0.7: rng.normal(20, .1, 30),
               1.0: rng.normal(20, .1, 30)}     # empata com a anterior
    r = verificar_hipoteses(por_mag, rng.normal(1, .3, 60))
    assert r["saturacao_no_topo"]["detectada"] is True


# ── normalidade: a hipótese que o LS-POD exige, e a contraprova ────────────

def test_amostra_normal_passa_na_checagem():
    r = checar_normalidade(np.random.default_rng(13).normal(5, 1, 80), "n")
    assert r["vale"] is True and r["melhor_escala"] == "bruto"


def test_amostra_lognormal_e_aceita_na_escala_log():
    r = checar_normalidade(np.random.default_rng(14).lognormal(1, 0.5, 80), "ln")
    assert r["vale"] is True and r["melhor_escala"] == "log"


def test_amostra_muito_assimetrica_reprova_nas_duas_escalas():
    """O caso real do escore do AE: assimetria +2,3 e curtose +9,2."""
    x = np.concatenate([np.full(70, 1.0), np.array([50.0] * 10)])
    r = checar_normalidade(x, "escore")
    assert r["vale"] is False and r["melhor_escala"] == "nenhuma"
    assert r["assimetria"] > 1.0


def test_hipoteses_marcam_limites_como_aproximados_quando_nao_ha_normalidade():
    rng = np.random.default_rng(15)
    sau = np.concatenate([np.full(70, 1.0), np.full(10, 50.0)])
    por_mag = {0.5: rng.normal(60, 1, 20), 1.0: rng.normal(80, 1, 20)}
    r = verificar_hipoteses(por_mag, sau)
    assert r["limites_sao_aproximados"] is True


def test_quantil_empirico_e_a_contraprova_sem_hipotese():
    """Se normal e empírico levam à mesma conclusão, ela é robusta."""
    x = np.random.default_rng(16).normal(5, 1, 500)
    assert limite_pof_empirico(x, p=0.99) == pytest.approx(
        limite_pof(x)["limite"], rel=0.15)
