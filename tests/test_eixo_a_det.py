"""
O eixo do Weibull é magnitude de assinatura, não tempo.

POR QUE ESTE TESTE EXISTE
=========================
`rul_weibull.py` chamava o eixo de **TTF** (time to failure) e a unidade de
"passo sintético de degradação". Os dois nomes prometiam tempo e entregavam
outra coisa: o que a trajetória varre é a MAGNITUDE da assinatura injetada, de
0 a 1,0, e o que se registra é a magnitude em que a detecção se confirma. Não
há taxa de degradação de campo que converta magnitude em hora.

Renomear resolveu mais do que vocabulário: `a_det` e a SMD da injeção passaram
a compartilhar a unidade, e podem ser lidas na mesma régua.

Estes testes travam três coisas que a renomeação poderia ter quebrado em
silêncio:

1. a **escala** — η, MTTF e B10 agora vivem em [0; 1], e o chute inicial do MLE
   estava dimensionado para o eixo antigo (1..120);
2. os **aliases** — quem lê `ttf_unidade` deve receber a unidade NOVA, não a
   antiga, senão o JSON continua mentindo com a chave velha;
3. a separação entre **indetectabilidade no teto** e **censura genuína**.

Rodam sem torch e sem dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.rul_weibull import (
    A_DET_MAX,
    A_DET_MIN,
    A_DET_UNIDADE,
    N_STEPS,
    TTF_UNIDADE,
    a_det_da_grade,
    ajustar_weibull,
    classificar_desfechos,
    metadados_tempo_rul,
    passos_persistencia,
    selecionar_trajetorias_holdout,
)


# ── a conversão passo → magnitude ──────────────────────────────────────────

def test_a_grade_vai_de_zero_a_um():
    assert a_det_da_grade(0) == pytest.approx(A_DET_MIN)
    assert a_det_da_grade(N_STEPS - 1) == pytest.approx(A_DET_MAX)


def test_conversao_bate_com_a_grade_usada_na_trajetoria():
    """A trajetória usa `np.linspace(0, 1, N_STEPS)`; a conversão tem de ser a
    MESMA função, senão a magnitude registrada não é a magnitude aplicada."""
    grade = np.linspace(0.0, 1.0, N_STEPS)
    for passo in (0, 1, 2, 37, N_STEPS - 2, N_STEPS - 1):
        assert a_det_da_grade(passo) == pytest.approx(grade[passo])


def test_passo_fora_da_grade_e_grampeado_no_teto():
    """A versão anterior devolvia `n_steps` (120) para não detecção — um índice
    fora da grade, que vai de 0 a 119. O desfecho ia parar num ponto do eixo
    onde nada foi medido."""
    assert a_det_da_grade(N_STEPS) == pytest.approx(A_DET_MAX)
    assert a_det_da_grade(10 * N_STEPS) == pytest.approx(A_DET_MAX)


def test_persistencia_mantem_a_mesma_largura_fisica_entre_grades():
    for n_steps in (101, 251, 501):
        pontos = passos_persistencia(n_steps, largura_magnitude=0.02)
        largura = (pontos - 1) / (n_steps - 1)
        assert largura >= 0.02
        assert largura < 0.02 + 1 / (n_steps - 1)


def test_selecao_limitada_e_estratificada_por_ensaio():
    import pandas as pd

    janelas = []
    for ensaio, quantidade in (("F0L", 142), ("F0M", 135)):
        for indice in range(quantidade):
            janela = pd.DataFrame({"indice": [indice]})
            janela.attrs["ensaio"] = ensaio
            janelas.append(janela)

    todas = selecionar_trajetorias_holdout(janelas, None)
    limitadas = selecionar_trajetorias_holdout(janelas, 100)
    contagem = {
        ensaio: sum(j.attrs["ensaio"] == ensaio for j in limitadas)
        for ensaio in ("F0L", "F0M")
    }

    assert len(todas) == 277
    assert len(limitadas) == 100
    assert contagem == {"F0L": 51, "F0M": 49}


def test_bootstrap_quantizado_distingue_weibull_de_mistura():
    passo = 0.002
    rng = np.random.default_rng(12)
    weibull = 0.35 * rng.weibull(3.2, size=240)
    mistura = np.concatenate([
        0.20 * np.random.default_rng(13).weibull(6.0, size=120),
        0.52 * np.random.default_rng(14).weibull(8.0, size=120),
    ])

    def ajustar(amostra):
        quantizada = np.clip(
            np.ceil(amostra / passo) * passo, passo, 1.0
        )
        return ajustar_weibull(
            quantizada,
            amostra <= 1.0,
            n_boot=0,
            passo_grade=passo,
            n_boot_aderencia=120,
            seed=91,
        )["teste_aderencia_quantizada"]["p_value"]

    assert ajustar(weibull) > 0.05
    assert ajustar(mistura) < 0.05


# ── escala: o ajuste tem de funcionar em [0; 1] ────────────────────────────

def test_weibull_recupera_parametros_na_escala_da_magnitude():
    """Com o chute antigo (`max(mediana, 1.0)`), η_ini caía no TETO do eixo.

    Amostra gerada de uma Weibull conhecida com η = 0,30 — bem abaixo do antigo
    limite inferior de busca (0,1 não prendia, mas 1,0 como chute prendia).
    """
    rng = np.random.default_rng(3)
    beta_v, eta_v = 2.4, 0.30
    a = eta_v * rng.weibull(beta_v, 400)
    r = ajustar_weibull(a, np.ones(len(a), dtype=bool), n_boot=0)

    assert r["fit_converged"]
    assert r["beta"] == pytest.approx(beta_v, rel=0.20)
    assert r["eta"] == pytest.approx(eta_v, rel=0.20)


def test_ajuste_declara_censura_intervalar_na_grade():
    rng = np.random.default_rng(31)
    passo = 1.0 / (N_STEPS - 1)
    amostra_continua = 0.35 * rng.weibull(3.0, 300)
    a = np.clip(np.ceil(amostra_continua / passo) * passo, passo, 1.0)
    r = ajustar_weibull(
        a, np.ones(len(a), dtype=bool), n_boot=0, passo_grade=passo
    )

    assert r["fit_method"] == "mle_interval_censored_grid_right_censored"
    assert r["event_observation"] == "interval_censored_on_a_det_grid"
    assert r["a_det_grid_step"] == pytest.approx(passo)
    assert r["n_niveis_distintos"] == len(np.unique(a))
    assert r["taxa_empates"] == pytest.approx(1 - len(np.unique(a)) / len(a))


def test_marcos_ficam_dentro_da_faixa_de_magnitude():
    rng = np.random.default_rng(4)
    a = np.clip(0.35 * rng.weibull(3.0, 200), 1e-4, 1.0)
    r = ajustar_weibull(a, np.ones(len(a), dtype=bool), n_boot=0)
    assert 0.0 < r["b10"] < r["mttf"] < 1.0, (
        "B10/MTTF fora de [0; 1] indicam que a escala do eixo se perdeu"
    )


# ── aliases: a chave velha, o valor novo ───────────────────────────────────

def test_alias_aponta_para_a_unidade_nova():
    assert TTF_UNIDADE == A_DET_UNIDADE
    assert "fracao_da_assinatura" in A_DET_UNIDADE
    assert "passo" not in A_DET_UNIDADE


def test_metadados_declaram_que_o_eixo_nao_e_tempo():
    m = metadados_tempo_rul()
    assert m["eixo_nao_e_tempo"] is True
    assert m["ttf_unidade"] == m["a_det_unidade"] == A_DET_UNIDADE
    assert m["tempo_fisico_calibrado"] is False
    assert m["passo_tempo_fisico_horas"] is None
    assert m["a_det_por_passo"] == pytest.approx(1.0 / (N_STEPS - 1))
    assert "não converter" in m["nota"]


def test_resultado_mantem_as_chaves_antigas_como_alias():
    a = np.linspace(0.05, 0.9, 40)
    r = ajustar_weibull(a, np.ones(40, dtype=bool), n_boot=0)
    assert r["ttf_min"] == r["a_det_min"]
    assert r["ttf_max"] == r["a_det_max"]
    assert r["ttf_mean_observado"] == r["a_det_mean_detectadas"]


# ── indetectabilidade no teto ≠ censura genuína ────────────────────────────

def test_nao_deteccao_no_teto_nao_e_chamada_de_censura_generica():
    a = np.array([0.2, 0.3, 0.4, 1.0, 1.0])
    ev = np.array([True, True, True, False, False])
    d = classificar_desfechos(a, ev)
    assert d["n_indetectaveis_no_teto"] == 2
    assert d["n_censura_genuina"] == 0
    assert d["pod_mon_no_teto"] == pytest.approx(3 / 5)


def test_interrupcao_antes_do_teto_conta_como_censura_genuina():
    """Não ocorre no desenho atual, mas o campo distingue os dois casos."""
    a = np.array([0.2, 0.55, 1.0])
    ev = np.array([True, False, False])
    d = classificar_desfechos(a, ev)
    assert d["n_censura_genuina"] == 1
    assert d["n_indetectaveis_no_teto"] == 1


def test_a_hipotese_do_tratamento_como_censura_fica_escrita():
    """Tratar indetectabilidade como censura pressupõe assinatura acima da
    nominal. Hipótese defensável — mas hipótese, e tem de estar declarada."""
    d = classificar_desfechos(np.array([1.0]), np.array([False]))
    assert d["tratamento_no_ajuste"] == "right_censored"
    assert "PRESSUPÕE" in d["hipotese_declarada"]


def test_deteccao_total_zera_a_indetectabilidade():
    a = np.linspace(0.1, 0.8, 12)
    d = classificar_desfechos(a, np.ones(12, dtype=bool))
    assert d["n_indetectaveis_no_teto"] == 0
    assert d["pod_mon_no_teto"] == pytest.approx(1.0)


def test_rotulo_empirico_nao_inventa_censura_quando_todos_sao_detectados():
    from src.ml.graficos_rul import _rotulo_posicoes_empiricas

    rotulo = _rotulo_posicoes_empiricas(np.ones(12, dtype=bool))
    assert "sem indetectabilidade" in rotulo
    assert "com censura" not in rotulo


def test_rotulo_empirico_quantifica_indetectabilidade_no_teto():
    from src.ml.graficos_rul import _rotulo_posicoes_empiricas

    rotulo = _rotulo_posicoes_empiricas(
        np.array([True, True, False, False], dtype=bool)
    )
    assert "n=4" in rotulo
    assert "indetect. no teto=2" in rotulo


def test_limites_adaptativos_dao_escala_legivel_ao_fusivel():
    from src.ml.graficos_rul import _limites_eixo_magnitude

    valores = np.array([0.0336, 0.0420, 0.0504, 0.0756])
    minimo, maximo = _limites_eixo_magnitude(valores)

    assert 0.0 <= minimo < valores.min()
    assert valores.max() < maximo <= 1.0
    assert maximo - minimo < 0.10


def test_limites_adaptativos_preservam_o_teto_nominal():
    from src.ml.graficos_rul import _limites_eixo_magnitude

    minimo, maximo = _limites_eixo_magnitude(np.array([0.35, 0.50, 1.0]))
    assert minimo < 0.35
    assert maximo == pytest.approx(1.01)


def test_histograma_marca_indetectabilidade_no_teto(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from src.ml import graficos_rul

    a = np.concatenate([np.linspace(0.10, 0.65, 20), np.ones(2)])
    eventos = np.concatenate([
        np.ones(20, dtype=bool),
        np.zeros(2, dtype=bool),
    ])
    parametros = ajustar_weibull(a, eventos, n_boot=0)
    a_por_falha = {falha["id"]: a for falha in graficos_rul.FALHAS}
    eventos_por_falha = {
        falha["id"]: eventos for falha in graficos_rul.FALHAS
    }
    parametros_por_falha = {
        falha["id"]: parametros for falha in graficos_rul.FALHAS
    }
    figuras = []

    monkeypatch.setattr(
        graficos_rul,
        "salvar_figura",
        lambda fig, _path, _rodape: figuras.append(fig),
    )
    monkeypatch.setattr(graficos_rul, "_log", lambda _mensagem: None)

    graficos_rul.plotar_ttf_histogramas(
        a_por_falha, eventos_por_falha, parametros_por_falha, tmp_path
    )

    assert len(figuras) == 1
    labels = figuras[0].axes[0].get_legend_handles_labels()[1]
    assert any("Indetectáveis no teto" in label for label in labels)
    plt.close(figuras[0])


def test_grafico_usa_uma_barra_por_magnitude_da_grade(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from src.ml import graficos_rul

    a = np.repeat(np.array([0.0336, 0.0420, 0.0504, 0.0756]), 5)
    eventos = np.ones(len(a), dtype=bool)
    parametros = ajustar_weibull(a, eventos, n_boot=0)
    a_por_falha = {falha["id"]: a for falha in graficos_rul.FALHAS}
    eventos_por_falha = {
        falha["id"]: eventos for falha in graficos_rul.FALHAS
    }
    parametros_por_falha = {
        falha["id"]: parametros for falha in graficos_rul.FALHAS
    }
    figuras = []

    monkeypatch.setattr(
        graficos_rul,
        "salvar_figura",
        lambda fig, _path, _rodape: figuras.append(fig),
    )
    monkeypatch.setattr(graficos_rul, "_log", lambda _mensagem: None)

    graficos_rul.plotar_ttf_histogramas(
        a_por_falha, eventos_por_falha, parametros_por_falha, tmp_path
    )

    eixo = figuras[0].axes[0]
    assert len(eixo.patches) == len(np.unique(a))
    assert eixo.get_xlim()[1] - eixo.get_xlim()[0] < 0.10
    plt.close(figuras[0])


def test_rotulos_visuais_nao_reintroduzem_eixo_temporal():
    import ast
    import inspect

    from src.ml import graficos_rul

    arvore = ast.parse(inspect.getsource(graficos_rul))
    textos = "\n".join(
        no.value
        for no in ast.walk(arvore)
        if isinstance(no, ast.Constant) and isinstance(no.value, str)
    )
    for rotulo_proibido in ("F(t) =", "ln t", "f(t) exige", "posição com censura"):
        assert rotulo_proibido not in textos
    assert "validação sintética e2" in textos.lower()


def test_desfechos_entram_no_resultado_do_ajuste():
    a = np.concatenate([np.linspace(0.1, 0.6, 30), np.full(5, 1.0)])
    ev = np.concatenate([np.ones(30, dtype=bool), np.zeros(5, dtype=bool)])
    r = ajustar_weibull(a, ev, n_boot=0)
    assert r["desfechos"]["n_indetectaveis_no_teto"] == 5
    assert r["desfechos"]["pod_mon_no_teto"] == pytest.approx(30 / 35)
    assert r["eixo_nao_e_tempo"] is True


def test_comprimentos_incompativeis_sao_recusados():
    with pytest.raises(ValueError):
        classificar_desfechos(np.array([0.1, 0.2]), np.array([True]))


# ── a montagem do artefato ─────────────────────────────────────────────────

def test_relatorio_monta_sem_rodar_o_pipeline():
    """`relatorio_weibull.py` nasceu de uma extração de `rul_weibull.py`.

    Extração mecânica é onde erro de nome passa despercebido: o módulo só seria
    exercitado numa execução de 8 minutos com o dataset bruto. Aqui ele roda com
    dados de mentira, em milissegundos.
    """
    import numpy as np

    from src.ml.relatorio_weibull import montar_relatorio
    from src.ml.rul_weibull import (
        A_DET_UNIDADE, N_STEPS, TEMPO_FISICO_NOTA, TTF_UNIDADE,
        _json_seguro, metadados_tempo_rul,
    )

    falhas = [{"id": "igbt", "nome": "IGBT", "npr": 90, "cor": "#333"}]
    a = np.concatenate([np.linspace(0.1, 0.7, 25), np.full(5, 1.0)])
    ev = np.concatenate([np.ones(25, dtype=bool), np.zeros(5, dtype=bool)])
    p = ajustar_weibull(a, ev, n_boot=0)

    rel, linhas = montar_relatorio(
        params={"igbt": p}, a_dets_dict={"igbt": a}, eventos_dict={"igbt": ev},
        falhas=falhas, meta_holdout={"protocolo": "teste"},
        metadados_tempo=metadados_tempo_rul(), limiar=7.83,
        n_traj_max=100, n_traj_real=30, n_steps=N_STEPS,
        a_det_unidade=A_DET_UNIDADE, ttf_unidade=TTF_UNIDADE,
        tempo_fisico_calibrado=False, tempo_fisico_nota=TEMPO_FISICO_NOTA,
        min_eventos_weibull=10, max_censura_rul_pct=50.0,
        min_r2_papel_weibull=0.90,
        persistencia_cruzamento=3, json_seguro=_json_seguro,
    )

    assert rel["__meta__"]["evidence_level"] == "E2"
    assert "não tempo" in rel["__meta__"]["evidence_note"]
    assert rel["parametros_simulacao"]["a_det_unidade"] == A_DET_UNIDADE
    bloco = rel["falhas"]["igbt"]
    assert bloco["a_dets"] == bloco["ttfs"]            # alias, mesma lista
    assert bloco["desfechos"]["n_indetectaveis_no_teto"] == 5
    assert len(linhas) == 1
    assert linhas[0]["pod_mon_no_teto"] == pytest.approx(25 / 30)
    assert linhas[0]["n_censura_genuina"] == 0
    assert linhas[0]["margem_restrita_disponivel"] is True
    assert linhas[0]["margem_restrita_horizonte"] == linhas[0]["rul_restrita_horizonte"]
    assert linhas[0]["margem_restrita_inicial"] == linhas[0]["rul_restrita_inicial"]


def test_relatorio_e_serializavel_em_json():
    """O artefato vai para disco com `json.dump`; NaN quebraria JSON estrito."""
    import json

    import numpy as np

    from src.ml.relatorio_weibull import montar_relatorio
    from src.ml.rul_weibull import (
        A_DET_UNIDADE, N_STEPS, TEMPO_FISICO_NOTA, TTF_UNIDADE,
        _json_seguro, metadados_tempo_rul,
    )

    falhas = [{"id": "fusivel_ac", "nome": "Fusível AC", "npr": 30, "cor": "#333"}]
    a = np.full(8, 1.0)                       # nada detectado: ajuste não converge
    ev = np.zeros(8, dtype=bool)
    p = ajustar_weibull(a, ev, n_boot=0)
    assert not p["fit_converged"]

    rel, _ = montar_relatorio(
        params={"fusivel_ac": p}, a_dets_dict={"fusivel_ac": a},
        eventos_dict={"fusivel_ac": ev}, falhas=falhas, meta_holdout={},
        metadados_tempo=metadados_tempo_rul(), limiar=7.83,
        n_traj_max=100, n_traj_real=8, n_steps=N_STEPS,
        a_det_unidade=A_DET_UNIDADE, ttf_unidade=TTF_UNIDADE,
        tempo_fisico_calibrado=False, tempo_fisico_nota=TEMPO_FISICO_NOTA,
        min_eventos_weibull=10, max_censura_rul_pct=50.0,
        min_r2_papel_weibull=0.90,
        persistencia_cruzamento=3, json_seguro=_json_seguro,
    )
    texto = json.dumps(rel, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in texto
    assert rel["falhas"]["fusivel_ac"]["status_ajuste"] == (
        "nao_estimavel_parametrico_rul_restrita")


# ── o painel que ficava mudo ───────────────────────────────────────────────

def test_motivo_nao_estimavel_diz_quantos_eventos_faltaram():
    """O pesquisador reportou que o IGBT "sumia" dos gráficos.

    Sumia: os painéis ficavam sem β/η e a legenda dizia só "ajuste não
    estimável", sem distinguir "faltou 1 evento" de "quebrou". Um buraco
    silencioso num capítulo é pior que um número ruim.
    """
    from src.ml.rul_weibull import MIN_EVENTOS_WEIBULL, motivo_nao_estimavel

    a = np.concatenate([np.linspace(0.15, 0.9, 9), np.full(12, 1.0)])
    ev = np.concatenate([np.ones(9, dtype=bool), np.zeros(12, dtype=bool)])
    texto = motivo_nao_estimavel(classificar_desfechos(a, ev))

    assert "9 detecções em 21 trajetórias" in texto
    assert f"mínimo de {MIN_EVENTOS_WEIBULL}" in texto
    assert "Faltou 1 evento" in texto
    assert "42.9%" in texto or "42,9%" in texto
    assert "NÃO foi afrouxado" in texto, (
        "precisa dizer que o critério não foi relaxado para gerar curva"
    )
    assert "Kaplan-Meier" in texto, (
        "a leitura NÃO paramétrica sobrevive à falta de eventos e tem de ser "
        "oferecida — senão o modo mais difícil da FMECA vira um vazio"
    )


def test_o_motivo_entra_na_interpretacao_do_artefato():
    a = np.concatenate([np.linspace(0.15, 0.9, 9), np.full(12, 1.0)])
    ev = np.concatenate([np.ones(9, dtype=bool), np.zeros(12, dtype=bool)])
    r = ajustar_weibull(a, ev, n_boot=0)

    assert r["fit_converged"] is False
    assert "9 detecções" in r["interpretacao"]["leitura"]
    assert r["interpretacao"]["km_continua_valida"] is True
    # A RUL restrita por Kaplan-Meier continua disponível: é o que impede o
    # capítulo de ficar sem NENHUM número para esta falha.
    assert r["rul_restrita_disponivel"] is True
    assert np.isfinite(r["rul_restrita_inicial"])


def test_criterio_de_minimo_de_eventos_nao_foi_afrouxado():
    """Guarda contra a tentação de baixar o mínimo para 9 e "ganhar" a curva.

    Ajustar Weibull com 9 eventos e 57% de indetectabilidade produz exatamente
    o tipo de número que não deve entrar numa dissertação. Mexer no critério
    DEPOIS de ver o resultado é o pior momento possível para mexer nele.
    """
    from src.ml.rul_weibull import MIN_EVENTOS_WEIBULL

    assert MIN_EVENTOS_WEIBULL >= 10
