"""A severidade injetada tem de ser monótona, ou `a_det` não significa nada.

POR QUE ESTE TESTE EXISTE
=========================
`a_det` é definido como o MENOR `a` em que a detecção se confirma. Essa
definição só tem sentido se o efeito da injeção crescer com `a`: se a
assinatura fosse não monótona, "o menor `a` que cruza" dependeria da grade e
não da física, e a Weibull ajustada em cima disso descreveria a grade.

Nada disso precisa do dataset: as propriedades são verificáveis sobre uma
janela sintética com valores conhecidos à mão.

O segundo grupo de testes fixa o que SEPARA as duas assinaturas elétricas. Se
IGBT e sensor movessem as mesmas features na mesma direção, a comparação entre
os `a_det` dos dois mediria a mesma coisa duas vezes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.dados_gpvs import (
    FEATURE_COLUMNS,
    PRIMARY_COLUMNS,
    WINDOW_SAMPLES,
    feature_vector,
)
from src.ml.injecao_e2 import (
    ERRO_NOMINAL_SENSOR,
    ESPECIFICACOES,
    GRADE_SEVERIDADE,
    METODO_ASSINATURA,
    METODO_INTERPOLACAO,
    contrato_da_especificacao,
    distancia_ancora,
    injetar_controle,
    injetar_igbt,
    injetar_sensor_realimentacao,
    injetor_de,
)


def _janela(seed: int = 0, amplitude: float = 1.0) -> pd.DataFrame:
    """Um ciclo trifásico limpo a 50 Hz, 200 amostras — o contrato do GPVS."""
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW_SAMPLES) / WINDOW_SAMPLES
    fases = {"a": 0.0, "b": -2 * np.pi / 3, "c": 2 * np.pi / 3}
    dados = {
        "Ipv": np.full(WINDOW_SAMPLES, 8.0) + 0.01 * rng.normal(size=WINDOW_SAMPLES),
        "Vpv": np.full(WINDOW_SAMPLES, 320.0) + 0.01 * rng.normal(size=WINDOW_SAMPLES),
        "Vdc": np.full(WINDOW_SAMPLES, 400.0) + 0.01 * rng.normal(size=WINDOW_SAMPLES),
    }
    for nome, fase in fases.items():
        dados[f"i{nome}"] = amplitude * 10.0 * np.sin(2 * np.pi * t + fase)
        dados[f"v{nome}"] = 220.0 * np.sin(2 * np.pi * t + fase)
    return pd.DataFrame(dados)[list(PRIMARY_COLUMNS)]


def _feat(janela: pd.DataFrame) -> dict[str, float]:
    return dict(zip(FEATURE_COLUMNS, feature_vector(janela).tolist(), strict=True))


# ── a=0 é sempre a janela intacta ──────────────────────────────────────────

@pytest.mark.leve
def test_severidade_zero_nao_altera_nada():
    """Se `a=0` mexesse no sinal, o eixo teria origem deslocada."""
    janela = _janela()
    for injetar in (injetar_igbt, injetar_sensor_realimentacao):
        np.testing.assert_allclose(
            injetar(janela, 0.0)[list(PRIMARY_COLUMNS)].to_numpy(),
            janela[list(PRIMARY_COLUMNS)].to_numpy(),
        )


@pytest.mark.leve
def test_interpolacao_devolve_os_extremos_exatos():
    saudavel, falha = _janela(0), _janela(1, amplitude=1.6)

    np.testing.assert_allclose(
        injetar_controle(saudavel, 0.0, janela_falha=falha).to_numpy(),
        saudavel.to_numpy(),
    )
    np.testing.assert_allclose(
        injetar_controle(saudavel, 1.0, janela_falha=falha).to_numpy(),
        falha.to_numpy(),
    )


# ── monotonicidade: a propriedade que sustenta a_det ───────────────────────

@pytest.mark.leve
def test_igbt_desbalanceia_de_forma_monotona():
    janela = _janela()
    valores = [
        _feat(injetar_igbt(janela, a))["i_rms_unbalance"]
        for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    assert all(b > a for a, b in zip(valores, valores[1:])), (
        f"o desbalanceamento tem de crescer com a severidade; veio {valores}"
    )


@pytest.mark.leve
def test_igbt_eleva_a_distorcao_da_fase_atingida_de_forma_monotona():
    janela = _janela()
    valores = [
        _feat(injetar_igbt(janela, a))["ia_thd"] for a in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(b > a for a, b in zip(valores, valores[1:])), valores


@pytest.mark.leve
def test_sensor_eleva_o_rms_de_forma_monotona_e_exata():
    """O ganho é conhecido: em `a=1` o RMS tem de ser exatamente 1,20x."""
    janela = _janela()
    base = _feat(janela)["ia_rms"]

    for a in (0.25, 0.5, 0.75, 1.0):
        esperado = base * (1.0 + ERRO_NOMINAL_SENSOR * a)
        obtido = _feat(injetar_sensor_realimentacao(janela, a))["ia_rms"]
        assert obtido == pytest.approx(esperado, rel=1e-6)


@pytest.mark.leve
def test_a_grade_de_severidade_e_crescente_e_fica_no_intervalo():
    assert GRADE_SEVERIDADE[0] > 0.0, "a=0 é a janela intacta, não pertence à grade"
    assert GRADE_SEVERIDADE[-1] == 1.0
    assert all(b > a for a, b in zip(GRADE_SEVERIDADE, GRADE_SEVERIDADE[1:]))


# ── as duas assinaturas elétricas têm de ser DISTINGUÍVEIS ─────────────────

@pytest.mark.leve
def test_o_sensor_nao_mexe_no_desbalanceamento_nem_na_distorcao():
    """É o que o separa do IGBT.

    Erro do sistema de medição atinge as três fases igualmente, e ganho é
    transformação linear: nem o desbalanceamento nem a THD podem se mover.
    """
    janela = _janela()
    antes, depois = _feat(janela), _feat(injetar_sensor_realimentacao(janela, 1.0))

    assert depois["i_rms_unbalance"] == pytest.approx(antes["i_rms_unbalance"], abs=1e-9)
    for fase in ("ia", "ib", "ic"):
        assert depois[f"{fase}_thd"] == pytest.approx(antes[f"{fase}_thd"], rel=1e-6)


@pytest.mark.leve
def test_o_igbt_nao_mexe_na_tensao():
    """Num inversor conectado à rede, quem impõe a tensão é a rede."""
    janela = _janela()
    depois = injetar_igbt(janela, 1.0)

    for coluna in ("va", "vb", "vc"):
        np.testing.assert_allclose(
            depois[coluna].to_numpy(), janela[coluna].to_numpy()
        )


@pytest.mark.leve
def test_o_igbt_so_atinge_uma_fase():
    janela = _janela()
    depois = injetar_igbt(janela, 1.0)

    assert not np.allclose(depois["ia"].to_numpy(), janela["ia"].to_numpy())
    for coluna in ("ib", "ic"):
        np.testing.assert_allclose(
            depois[coluna].to_numpy(), janela[coluna].to_numpy()
        )


@pytest.mark.leve
def test_em_a1_o_semiciclo_positivo_desaparece():
    janela = _janela()
    corrente = injetar_igbt(janela, 1.0)["ia"].to_numpy()

    assert np.all(corrente <= 1e-12), "a=1 é a falha COMPLETA da perna"
    assert corrente.min() < -1.0, "o semiciclo negativo tem de sobreviver"


# ── as guardas de contrato ─────────────────────────────────────────────────

@pytest.mark.leve
@pytest.mark.parametrize("a", [-0.01, 1.01, 2.0, -1.0])
def test_severidade_fora_do_intervalo_estoura(a):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        injetar_igbt(_janela(), a)


@pytest.mark.leve
def test_interpolacao_sem_o_outro_extremo_estoura_antes_de_qualquer_numero():
    from src.ml.injecao_e2 import POR_ID

    with pytest.raises(ValueError, match="janela_falha"):
        injetor_de(POR_ID["controle"])


@pytest.mark.leve
def test_janela_com_tamanho_errado_estoura():
    with pytest.raises(ValueError, match=str(WINDOW_SAMPLES)):
        injetar_igbt(_janela().iloc[:50], 0.5)


# ── o contrato publicado ───────────────────────────────────────────────────

@pytest.mark.leve
@pytest.mark.parametrize("especificacao", ESPECIFICACOES, ids=lambda e: e.id)
def test_o_contrato_bate_com_o_escopo_declarado_no_gpvs(especificacao):
    """Guarda cruzada: a especificação não pode divergir de `FAULT_CONTRACTS`.

    Se alguém trocar os ensaios de referência de um item sem trocar o escopo,
    a montagem do contrato levanta AssertionError aqui, não em produção.
    """
    contrato = contrato_da_especificacao(especificacao)

    assert contrato["evidence_level"] == "E2"
    assert contrato["a_axis"] == "fraction_of_nominal_signature_not_time"
    assert contrato["fmeca_scope"] == especificacao.fmeca_scope


@pytest.mark.leve
def test_o_metodo_de_cada_item_viaja_no_contrato():
    """O leitor tem de saber, sem perguntar, o que é física e o que é interpolação."""
    metodos = {
        especificacao.id: contrato_da_especificacao(especificacao)["injection_method"]
        for especificacao in ESPECIFICACOES
    }

    assert metodos["igbt"] == METODO_ASSINATURA
    assert metodos["sensor_realimentacao"] == METODO_ASSINATURA
    assert metodos["controle"] == METODO_INTERPOLACAO


@pytest.mark.leve
def test_so_a_assinatura_eletrica_se_declara_simulacao_fisica():
    for especificacao in ESPECIFICACOES:
        contrato = contrato_da_especificacao(especificacao)
        assert contrato["physical_simulation"] is (
            especificacao.metodo == METODO_ASSINATURA
        )


@pytest.mark.leve
def test_os_tres_itens_da_fmeca_vigente_estao_cobertos():
    escopos = {especificacao.fmeca_scope for especificacao in ESPECIFICACOES}
    assert escopos == {"igbt", "sensor_feedback_system", "inverter_control_system"}


@pytest.mark.leve
def test_o_recorte_historico_do_tcc_nao_voltou():
    """Contator AC e Fusível AC saíram da FMECA vigente e não podem reaparecer."""
    ids = {especificacao.id for especificacao in ESPECIFICACOES}
    assert not ids & {"contator_ac", "fusivel_ac", "contator", "fusivel"}


# ── a âncora ───────────────────────────────────────────────────────────────

@pytest.mark.leve
def test_ancora_zera_quando_as_duas_nuvens_coincidem():
    janelas = [_janela(i) for i in range(4)]
    escala = np.abs(feature_vector(janelas[0])).astype(float) + 1.0

    resultado = distancia_ancora(janelas, janelas, escala)

    assert resultado["distancia_euclidiana_iqr"] == pytest.approx(0.0, abs=1e-9)
    assert resultado["n_injetadas"] == resultado["n_reais"] == 4


@pytest.mark.leve
def test_ancora_cresce_quando_a_caricatura_se_afasta():
    saudaveis = [_janela(i) for i in range(4)]
    escala = np.abs(feature_vector(saudaveis[0])).astype(float) + 1.0
    perto = [injetar_sensor_realimentacao(j, 0.1) for j in saudaveis]
    longe = [injetar_sensor_realimentacao(j, 1.0) for j in saudaveis]

    d_perto = distancia_ancora(perto, saudaveis, escala)["distancia_euclidiana_iqr"]
    d_longe = distancia_ancora(longe, saudaveis, escala)["distancia_euclidiana_iqr"]

    assert d_longe > d_perto > 0.0


@pytest.mark.leve
def test_ancora_nomeia_a_feature_que_mais_diverge():
    """Sem isso a distância é um número sem diagnóstico."""
    saudaveis = [_janela(i) for i in range(3)]
    escala = np.ones(len(FEATURE_COLUMNS), dtype=float)
    injetadas = [injetar_igbt(j, 1.0) for j in saudaveis]

    resultado = distancia_ancora(injetadas, saudaveis, escala)

    assert resultado["feature_mais_distante"] in FEATURE_COLUMNS
    assert resultado["distancia_maxima_iqr"] >= resultado["distancia_mediana_iqr"]


@pytest.mark.leve
def test_ancora_recusa_escala_invalida():
    janelas = [_janela()]
    with pytest.raises(ValueError, match="positiva"):
        distancia_ancora(janelas, janelas, np.zeros(len(FEATURE_COLUMNS)))
    with pytest.raises(ValueError, match="24"):
        distancia_ancora(janelas, janelas, np.ones(5))
