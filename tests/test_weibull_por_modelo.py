"""
Detectabilidade E2 para QUALQUER detector, não só o Autoencoder denso.

POR QUE ESTE TESTE EXISTE
=========================
O pesquisador pediu as curvas de confiabilidade "pertinente a cada modelo",
comparando o AE denso com o AE-LSTM do Ibrahim. Não dava: a cadeia
`rul_weibull_execucao` carrega UM checkpoint fixo e itera sobre `FALHAS` —
o laço é por COMPONENTE, nunca por MODELO. E grep por "weibull" nos quatro
macro-códigos e em `modelos_anomalia` devolvia zero: o AE-LSTM nunca gerou
`a_det`.

`gerar_a_det` ganhou um parâmetro `scorer` com a mesma interface que
`macro_comum` já exige dos dois métodos. Estes testes travam o contrato: um
detector arbitrário entra, `a_det` sai, e o resultado é comparável entre modelos.

Rodam sem torch, sem dataset e sem checkpoint — o scorer é uma função de mentira.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.gpvs_principal import JANELA
from src.ml.gpvs import COLUNAS_PRIMARIAS
from src.ml.rul_weibull import gerar_a_det
from src.ml.weibull_por_modelo import (
    comparar_detectabilidade,
    detectabilidade_do_modelo,
    trajetorias_por_falha,
)


def _janela(seed: int = 0) -> pd.DataFrame:
    """Janela GPVS sintética com as colunas primárias que a injeção exige."""
    rng = np.random.default_rng(seed)
    t = np.arange(JANELA) / JANELA
    dados = {}
    for k, col in enumerate(COLUNAS_PRIMARIAS):
        base = np.sin(2 * np.pi * t + k) + 0.01 * rng.normal(size=JANELA)
        dados[col] = base + 10.0
    df = pd.DataFrame(dados)
    df.attrs["ensaio"] = "F0L" if seed % 2 == 0 else "F0M"
    return df


def _scorer_proporcional_a_magnitude(ganho: float):
    """Detector de mentira: pontua pela dispersão da fase A.

    Como a injeção cresce a perturbação com `a_inj`, o escore cresce junto —
    é o que permite testar a varredura sem torch e sem modelo treinado.
    """
    def scorer(janelas):
        return np.asarray(
            [ganho * float(np.std(j[COLUNAS_PRIMARIAS[0]].to_numpy())) for j in janelas],
            dtype=float,
        )
    return scorer


# ── o contrato do scorer ───────────────────────────────────────────────────

def test_gerar_a_det_aceita_scorer_sem_modelo_nenhum():
    """É a mudança que destrava tudo: sem `scorer`, `gerar_a_det` exigia
    modelo, scaler, device e colunas — tudo específico do AE denso."""
    a_det, detectou = gerar_a_det(
        _janela(), modelo=None, scaler=None, device=None, colunas_feat=None,
        limiar=0.0,                      # limiar no chão: detecta de imediato
        tipo_falha="contator_ac", n_steps=21, seed=1,
        scorer=_scorer_proporcional_a_magnitude(1.0),
    )
    assert detectou is True
    assert 0.0 <= a_det <= 1.0


def test_limiar_inalcancavel_devolve_indetectavel_no_teto():
    a_det, detectou = gerar_a_det(
        _janela(), modelo=None, scaler=None, device=None, colunas_feat=None,
        limiar=1e12,
        tipo_falha="contator_ac", n_steps=21, seed=1,
        scorer=_scorer_proporcional_a_magnitude(1.0),
    )
    assert detectou is False
    assert a_det == pytest.approx(1.0), (
        "não detecção tem de ser carimbada em a_inj = 1,0, a última magnitude "
        "realmente aplicada"
    )


def test_detector_mais_sensivel_detecta_com_menos_assinatura():
    """A propriedade que dá sentido à comparação entre modelos.

    Dois detectores sobre a MESMA janela e a mesma realização de ruído: o de
    ganho maior tem de cruzar o limiar em `a_det` menor ou igual.
    """
    comum = dict(
        modelo=None, scaler=None, device=None, colunas_feat=None,
        tipo_falha="contator_ac", n_steps=101, seed=7,
    )
    janela = _janela(2)
    limiar = 3.0
    sensivel, _ = gerar_a_det(
        janela, limiar=limiar, scorer=_scorer_proporcional_a_magnitude(8.0), **comum)
    surdo, _ = gerar_a_det(
        janela, limiar=limiar, scorer=_scorer_proporcional_a_magnitude(2.0), **comum)

    assert sensivel <= surdo


def test_scorer_recebe_janelas_e_nao_vetores_de_features():
    """O scorer é quem sabe featurizar para o seu próprio modelo.

    Se recebesse vetores prontos, o AE-LSTM teria de usar as features do AE
    denso — e a comparação deixaria de ser entre modelos para virar entre
    cabeçotes sobre a mesma representação.
    """
    vistos = []

    def espiao(janelas):
        vistos.append(janelas)
        return np.zeros(len(janelas))

    gerar_a_det(
        _janela(), modelo=None, scaler=None, device=None, colunas_feat=None,
        limiar=1e9, tipo_falha="igbt", n_steps=9, seed=3, scorer=espiao,
    )
    assert vistos, "o scorer não foi chamado"
    assert all(isinstance(j, pd.DataFrame) for j in vistos[0])


# ── o bloco por modelo ─────────────────────────────────────────────────────

def test_detectabilidade_cobre_as_tres_falhas_da_fmeca():
    bloco = detectabilidade_do_modelo(
        "detector de teste", _scorer_proporcional_a_magnitude(6.0),
        limiar=3.0, janelas=[_janela(i) for i in range(6)], n_steps=21,
    )
    assert set(bloco["falhas"]) == {"contator_ac", "igbt", "fusivel_ac"}
    assert bloco["n_trajetorias"] == 6
    assert bloco["evidence_level"] == "E2"
    assert bloco["eixo_nao_e_tempo"] is True

    for dados in bloco["falhas"].values():
        assert len(dados["a_dets"]) == 6
        assert len(dados["eventos_observados"]) == 6
        assert "pod_mon_no_teto" in dados["desfechos"]
        assert "fit_converged" in dados["weibull"]


def test_o_bloco_declara_que_o_eixo_nao_e_tempo():
    """A ressalva viaja com o dado, não fica só no gráfico."""
    bloco = detectabilidade_do_modelo(
        "x", _scorer_proporcional_a_magnitude(1.0), limiar=1e9,
        janelas=[_janela(0)], n_steps=5,
    )
    assert "não tempo" in bloco["nota"]
    assert "confiabilidade ou taxa de" in bloco["nota"]


def test_passo_da_grade_bate_com_n_steps():
    bloco = detectabilidade_do_modelo(
        "x", _scorer_proporcional_a_magnitude(1.0), limiar=1e9,
        janelas=[_janela(0)], n_steps=101,
    )
    assert bloco["a_det_por_passo"] == pytest.approx(1 / 100)


# ── a comparação entre modelos ─────────────────────────────────────────────

def test_comparacao_alinha_os_modelos_por_falha():
    janelas = [_janela(i) for i in range(4)]
    denso = detectabilidade_do_modelo(
        "AE denso", _scorer_proporcional_a_magnitude(8.0), 3.0, janelas, n_steps=21)
    lstm = detectabilidade_do_modelo(
        "AE-LSTM (Ibrahim)", _scorer_proporcional_a_magnitude(2.0), 3.0, janelas,
        n_steps=21)

    comp = comparar_detectabilidade([denso, lstm])
    assert comp["n_modelos"] == 2
    assert len(comp["linhas"]) == 6          # 2 modelos × 3 falhas
    assert {r["modelo"] for r in comp["linhas"]} == {"AE denso", "AE-LSTM (Ibrahim)"}
    for linha in comp["linhas"]:
        assert linha["evidence_level"] == "E2"
        assert 0.0 <= linha["pod_mon_no_teto"] <= 1.0


def test_comparacao_vazia_e_recusada():
    with pytest.raises(ValueError):
        comparar_detectabilidade([])


def test_as_trajetorias_sao_reprodutiveis_entre_modelos():
    """Semente fixa por índice de janela: os dois modelos veem EXATAMENTE a
    mesma realização de ruído injetado. Sem isso a comparação mediria sorte."""
    janelas = [_janela(i) for i in range(3)]
    a1, e1 = trajetorias_por_falha(
        _scorer_proporcional_a_magnitude(5.0), 3.0, janelas, "contator_ac", n_steps=21)
    a2, e2 = trajetorias_por_falha(
        _scorer_proporcional_a_magnitude(5.0), 3.0, janelas, "contator_ac", n_steps=21)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(e1, e2)


# ── o espelho de constantes não pode divergir ─────────────────────────────

def test_constantes_da_grade_espelhadas_batem_com_a_fonte():
    """`rul_weibull` repete literais que vivem em `varredura_a_det`.

    Não é descuido: o manifesto de proveniência lê `N_STEPS`,
    `BATCH_INFERENCIA` e `PERSISTENCIA_MAGNITUDE` de `rul_weibull` por AST, sem
    importar o módulo, e `literal_eval` não resolve referência a outro nome.
    Escrever `N_STEPS = varredura_a_det.N_STEPS` faria o manifesto gravar nada.

    Espelho que diverge da fonte é pior que espelho nenhum: o manifesto
    registraria uma grade que não foi a executada.
    """
    from src.ml import rul_weibull, varredura_a_det

    for nome in ("N_STEPS", "BATCH_INFERENCIA", "PERSISTENCIA_MAGNITUDE",
                 "A_DET_MIN", "A_DET_MAX"):
        assert getattr(rul_weibull, nome) == getattr(varredura_a_det, nome), (
            f"{nome} divergiu entre rul_weibull e varredura_a_det"
        )


def test_rul_weibull_reexporta_a_varredura():
    """Quem já importava `gerar_a_det` de `rul_weibull` não pode quebrar."""
    from src.ml import rul_weibull, varredura_a_det

    for nome in ("gerar_a_det", "calcular_erros_batch", "a_det_da_grade",
                 "passos_persistencia", "selecionar_janelas_baseline_normais"):
        assert getattr(rul_weibull, nome) is getattr(varredura_a_det, nome)


# ── o número citável não pode vir de um ajuste rejeitado ───────────────────

def test_percentil_empirico_e_exato_sem_censura():
    """Com POD_mon = 1,00 nao ha censura: o quantil e medido, nao modelado."""
    from src.ml.weibull_por_modelo import percentis_empiricos

    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    r = percentis_empiricos(a, np.ones(5, dtype=bool))
    assert r["a50_empirico"] == pytest.approx(0.3)
    assert r["censura_presente"] is False
    assert r["fracao_detectada"] == 1.0


def test_percentil_acima_da_fracao_detectada_e_none():
    """Nao identificavel tem de voltar None, nunca um numero que parece medido."""
    from src.ml.weibull_por_modelo import percentis_empiricos

    a = np.array([0.1, 0.2, 1.0, 1.0])
    eventos = np.array([True, True, False, False])
    r = percentis_empiricos(a, eventos)
    assert r["a10_empirico"] is not None
    assert r["a90_empirico"] is None, "90% nao e identificavel com 50% detectado"
    assert r["censura_presente"] is True


def test_a_comparacao_traz_o_empirico_e_marca_o_parametrico():
    """`a10_2p` tem nome que denuncia a origem; antes se chamava so `a10`."""
    janelas = [_janela(i) for i in range(4)]
    bloco = detectabilidade_do_modelo(
        "x", _scorer_proporcional_a_magnitude(8.0), 3.0, janelas, n_steps=21)
    linha = comparar_detectabilidade([bloco])["linhas"][0]

    assert "a10_empirico" in linha and "a50_empirico" in linha
    assert "a10_2p" in linha
    assert "a10" not in linha, "o nome ambiguo nao pode voltar"


# ── o regime de contexto do AE-LSTM decide o que se está medindo ───────────

def test_os_dois_regimes_de_contexto_existem_e_sao_validados():
    from src.ml.macro_ibrahim import CONTEXTO_NORMAL, CONTEXTO_TRAJETORIA, CONTEXTOS

    assert CONTEXTO_NORMAL in CONTEXTOS and CONTEXTO_TRAJETORIA in CONTEXTOS
    with pytest.raises(ValueError) as erro:
        from src.ml.macro_ibrahim import construir_scorer
        construir_scorer(None, np.zeros((4, 3)), ["a"], None,
                         contexto="degrau_magico")
    assert "normal" in str(erro.value) and "trajetoria" in str(erro.value)


def test_trajetoria_usa_os_proprios_itens_como_historico():
    """A diferença entre 'detecta degrau' e 'detecta degradação'.

    Sob `normal`, o AE-LSTM vê L-1 janelas SAUDÁVEIS antes da injetada — um
    degrau que a varredura fabrica e que o campo não daria. Sob `trajetoria`,
    o histórico são as magnitudes anteriores da mesma janela-base.
    """
    from src.ml.modelos_anomalia import sequencias_com_contexto

    itens = np.arange(12, dtype=np.float32).reshape(6, 2)
    contexto_sao = np.zeros((6, 2), dtype=np.float32)

    com_degrau = sequencias_com_contexto(contexto_sao, itens, 3)
    com_trajetoria = sequencias_com_contexto(itens, itens, 3)

    # Nos dois, o ÚLTIMO passo é sempre o item pontuado.
    np.testing.assert_allclose(com_degrau[:, -1], itens)
    np.testing.assert_allclose(com_trajetoria[:, -1], itens)
    # Mas o histórico difere: saudável contra magnitudes anteriores.
    np.testing.assert_allclose(com_degrau[3, 0], contexto_sao[1])
    np.testing.assert_allclose(com_trajetoria[3, 0], itens[1])
    assert not np.allclose(com_degrau, com_trajetoria)


def test_o_regime_viaja_com_o_resultado():
    """Duas execuções que medem perguntas DIFERENTES não podem ficar
    indistinguíveis no artefato."""
    from pathlib import Path

    fonte = (Path(__file__).resolve().parents[1]
             / "src/ml/macro_weibull.py").read_text(encoding="utf-8")
    assert 'bloco["contexto_lstm"]' in fonte
    assert '"--contexto-lstm"' in fonte
