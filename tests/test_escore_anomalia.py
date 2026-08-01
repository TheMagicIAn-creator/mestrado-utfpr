"""Escore de anomalia (fonte única): MSE médio × localizado (top-k)."""

from __future__ import annotations

import numpy as np

import src.ml.escore_anomalia as ea


def _regua_saudavel(seed: int = 0, n: int = 200, f: int = 109):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 0.1, size=(n, f))
    return R, ea.ajustar_estatistica_residuo(R)


def test_escore_mse_e_a_media_dos_quadrados():
    r = np.array([[3.0, 4.0]])          # média de 9 e 16 = 12.5
    assert ea.escore_mse_medio(r)[0] == 12.5


def test_localizado_detecta_falha_que_o_mse_dilui():
    R_sau, stats = _regua_saudavel()
    lim_loc = np.percentile(ea.escore_localizado(R_sau, stats, k=5), 99)
    # falha LOCALIZADA: só 3 de 109 features estouram
    rng = np.random.default_rng(99)
    r_falha = rng.normal(0.0, 0.1, size=(1, 109))
    r_falha[0, [5, 7, 11]] += 2.0
    assert ea.escore_localizado(r_falha, stats, k=5)[0] > lim_loc


def test_localizado_nao_alarma_saudavel():
    R_sau, stats = _regua_saudavel()
    lim_loc = np.percentile(ea.escore_localizado(R_sau, stats, k=5), 99)
    rng = np.random.default_rng(7)
    r_ok = rng.normal(0.0, 0.1, size=(1, 109))
    assert ea.escore_localizado(r_ok, stats, k=5)[0] < lim_loc


def test_pontuar_sem_regua_cai_para_mse():
    R_sau, stats = _regua_saudavel()
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.1, size=(1, 109))
    # sem stats → MSE (comportamento histórico), nunca quebra
    assert ea.pontuar(r, None, metodo="localizado")[0] == ea.escore_mse_medio(r)[0]
    assert ea.pontuar(r, stats, metodo="mse")[0] == ea.escore_mse_medio(r)[0]


def test_k_e_limitado_ao_numero_de_features():
    R_sau, stats = _regua_saudavel(f=4)
    r = np.array([[0.1, 0.2, 0.3, 0.4]])
    # k acima de F não deve quebrar (é limitado internamente)
    assert np.isfinite(ea.escore_localizado(r, stats, k=99)[0])


def test_persistencia_da_regua_roundtrip(tmp_path):
    _, stats = _regua_saudavel()
    ea.salvar_estatistica(stats, tmp_path)
    lido = ea.carregar_estatistica(tmp_path)
    assert lido is not None
    assert np.allclose(lido["mu"], stats["mu"])
    assert np.allclose(lido["sigma"], stats["sigma"])
    # pasta sem o artefato → None (dispara fallback para MSE no pipeline)
    assert ea.carregar_estatistica(tmp_path, "inexistente.npz") is None


def test_limiar_por_fp_alvo_sobe_percentil_quando_p99_estoura():
    # scores de ajuste: massa em [0,1]. scores_val: com cauda pesada que faz o
    # p99 do ajuste ser ultrapassado por >1% dos val → auto deve subir o percentil.
    rng = np.random.default_rng(0)
    ajuste = rng.uniform(0, 1, size=2000)
    val = np.concatenate([rng.uniform(0, 1, size=200),
                          rng.uniform(1.0, 1.5, size=20)])  # ~9% acima do p99 do ajuste
    lim99 = np.percentile(ajuste, 99)
    fp99 = (val > lim99).mean() * 100
    assert fp99 > 1.0  # confirma o cenário do problema
    lim, perc = ea.limiar_por_fp_alvo(ajuste, val, fp_alvo_pct=1.0)
    assert perc >= 99.0 and lim >= lim99          # subiu (ou manteve) o percentil
    assert (val > lim).mean() * 100 <= (val > lim99).mean() * 100  # FP não piora


def test_limiar_por_fp_alvo_respeita_o_alvo():
    rng = np.random.default_rng(1)
    ajuste = rng.normal(0, 1, size=3000)
    val = rng.normal(0, 1, size=1000)
    lim, perc = ea.limiar_por_fp_alvo(ajuste, val, fp_alvo_pct=1.0)
    fp = (val > lim).mean() * 100
    # invariante: ou atinge o alvo, ou usa o percentil mais conservador (99.9)
    assert fp <= 1.0 or perc == 99.9
    assert 99.0 <= perc <= 99.9


# ── incerteza do limiar, e por que NÃO tentamos "melhorar" o estimador ───────

def test_limiar_continua_sendo_o_percentil_empirico():
    """Regressão do resultado publicado: o estimador não pode ter mudado.

    Bootstrap e ajuste paramétrico foram testados como substitutos e
    rejeitados (ver docstring de `incerteza_do_limiar`). Se alguém trocar o
    estimador, os resultados de `resultados/macro/` deixam de valer.
    """
    import numpy as np

    from src.ml.escore_anomalia import limiar_por_fp_alvo

    rng = np.random.default_rng(7)
    a, v = rng.lognormal(0, 1, 73), rng.lognormal(0, 1, 18)
    limiar, p = limiar_por_fp_alvo(a, v, 1.0)
    assert limiar == float(np.percentile(a, p))


def test_fp_alvo_vem_da_configuracao_quando_nao_informado():
    """macro_comum passava 1.0 literal e ignorava AL_IADO_ESCORE_FP_ALVO."""
    import numpy as np

    from src.ml import escore_anomalia as ea

    rng = np.random.default_rng(3)
    a, v = rng.lognormal(0, 1, 80), rng.lognormal(0, 1, 20)
    assert ea.limiar_por_fp_alvo(a, v) == ea.limiar_por_fp_alvo(a, v, ea.FP_ALVO)


def test_incerteza_devolve_intervalo_que_contem_o_limiar():
    import numpy as np

    from src.ml.escore_anomalia import incerteza_do_limiar

    a = np.random.default_rng(11).lognormal(0, 1, 73)
    inc = incerteza_do_limiar(a, 99.0, n_boot=300)
    assert inc["ic_low"] <= inc["limiar"] <= inc["ic_high"]
    assert inc["n_boot"] == 300
    assert inc["largura_relativa"] > 0


def test_incerteza_e_reprodutivel_por_semente():
    import numpy as np

    from src.ml.escore_anomalia import incerteza_do_limiar

    a = np.random.default_rng(11).lognormal(0, 1, 73)
    assert (incerteza_do_limiar(a, 99.0, n_boot=200, seed=1)
            == incerteza_do_limiar(a, 99.0, n_boot=200, seed=1))


def test_incerteza_degrada_com_amostra_pequena():
    """Com < 8 pontos o bootstrap daria falsa sensação de estabilidade."""
    import numpy as np
    from math import isnan

    from src.ml.escore_anomalia import incerteza_do_limiar

    inc = incerteza_do_limiar(np.array([1.0, 2.0, 3.0]), 99.0)
    assert inc["n_boot"] == 0 and isnan(inc["ic_low"])


def test_bootstrap_NAO_reduz_variancia_entre_amostras():
    """Fixa o resultado NEGATIVO que motivou não usar bootstrap no limiar.

    Se algum dia este teste falhar (isto é: o bootstrap passar a reduzir
    dispersão de forma relevante), a decisão registrada deve ser reavaliada.
    Hoje o ganho medido é ~2% — dentro do ruído.
    """
    import numpy as np

    pop = np.random.default_rng(0).lognormal(0, 1, 50_000)
    simples, boot = [], []
    for s in range(25):
        x = np.random.default_rng(100 + s).choice(pop, 73, replace=False)
        simples.append(float(np.percentile(x, 99.0)))
        r = np.random.default_rng(42)
        boot.append(float(np.median([
            np.percentile(r.choice(x, len(x), replace=True), 99.0)
            for _ in range(100)
        ])))
    ganho = 1.0 - float(np.std(boot)) / float(np.std(simples))
    assert ganho < 0.15, (
        f"bootstrap passou a reduzir dispersão em {ganho:.0%} — reavaliar a "
        "decisão de manter o percentil empírico"
    )
