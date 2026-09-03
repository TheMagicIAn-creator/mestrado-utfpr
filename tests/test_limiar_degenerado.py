"""Um limiar que é o máximo da calibração não sustenta o percentil que declara.

POR QUE ESTE TESTE EXISTE
=========================
A publicação canônica pede o percentil 99,9 do erro de reconstrução saudável.
A calibração do GPVS tem 210 janelas. Com 210 observações, `ceil(209 * 0,999)`
dá 209 — o ÚLTIMO índice. O limiar publicado é, literalmente, o maior escore
visto na calibração, e o relatório registrava `p100,000`, ordem `210/210`.

O contrato antigo era honesto: gravava o percentil efetivo ao lado do pedido.
Mas gravar a divergência não impede publicá-la, e três coisas decorrem dela:

  - a variância do limiar é a de um máximo amostral, não a de um quantil;
  - o Recall de ~0,39 é em parte artefato do ponto mais conservador possível,
    não uma medida da capacidade do detector;
  - escrever "p99,9" no texto da dissertação seria incorreto.

Estes testes fixam a aritmética de quando o pedido degenera, e garantem que a
publicação canônica RECUSE em vez de aceitar em silêncio. A varredura de
sensibilidade continua percorrendo percentis degenerados de propósito — ela
mede justamente o efeito de escolhê-los —, então ela não usa `strict`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.treino_comparacao import (
    DegenerateThresholdError,
    calibrate_threshold,
    minimum_n_for_percentile,
)


# ── a aritmética do tamanho mínimo ─────────────────────────────────────────

@pytest.mark.leve
@pytest.mark.parametrize(
    ("percentil", "minimo"),
    [
        (99.9, 1001),   # o caso publicado: precisa de 1001, tem 210
        (99.5, 201),
        (99.0, 101),
        (95.0, 21),
        (90.0, 11),
        (50.0, 3),
    ],
)
def test_tamanho_minimo_para_o_percentil(percentil, minimo):
    assert minimum_n_for_percentile(percentil) == minimo


@pytest.mark.leve
def test_p100_nao_tem_tamanho_que_o_conserte():
    """p100 é o máximo amostral por definição, em qualquer n."""
    assert minimum_n_for_percentile(100.0) is None


@pytest.mark.leve
@pytest.mark.parametrize("percentil", [99.9, 99.5, 99.0, 95.0, 90.0, 50.0])
def test_o_minimo_declarado_e_de_fato_o_ponto_de_virada(percentil):
    """O contrato só vale se o n devolvido for exatamente onde vira.

    Em `n = minimo` o pedido tem de deixar de ser o máximo; em `n = minimo-1`
    ele ainda tem de ser. Sem esta guarda, um erro de arredondamento de uma
    unidade passaria despercebido — e é justamente uma unidade que separa
    "percentil" de "máximo".
    """
    minimo = minimum_n_for_percentile(percentil)
    assert not calibrate_threshold(
        np.arange(minimo, dtype=float), percentil
    ).is_sample_maximum
    assert calibrate_threshold(
        np.arange(minimo - 1, dtype=float), percentil
    ).is_sample_maximum


# ── o caso real do GPVS ────────────────────────────────────────────────────

@pytest.mark.leve
def test_p999_com_a_calibracao_do_gpvs_e_o_maximo():
    """210 janelas é o tamanho real da calibração saudável F0L+F0M."""
    calibracao = calibrate_threshold(np.arange(210, dtype=float), 99.9)

    assert calibracao.is_sample_maximum
    assert calibracao.value == 209.0                    # o maior escore
    assert calibracao.selected_rank == 210
    assert calibracao.calibration_n == 210
    assert calibracao.effective_percentile == 100.0
    assert calibracao.minimum_n_for_request == 1001


@pytest.mark.leve
def test_p99_com_a_mesma_calibracao_e_representavel():
    """A alternativa que os dados sustentam hoje, sem tocar no dataset."""
    calibracao = calibrate_threshold(np.arange(210, dtype=float), 99.0)

    assert not calibracao.is_sample_maximum
    assert calibracao.selected_rank == 208
    assert calibracao.effective_percentile == pytest.approx(99.0476, abs=1e-4)


# ── a recusa ───────────────────────────────────────────────────────────────

@pytest.mark.leve
def test_strict_recusa_o_percentil_degenerado():
    with pytest.raises(DegenerateThresholdError) as erro:
        calibrate_threshold(np.arange(210, dtype=float), 99.9, strict=True)

    mensagem = str(erro.value)
    assert "210" in mensagem, "a mensagem tem de dizer QUAL n foi usado"
    assert "1001" in mensagem, "e QUAL n resolveria"
    assert "MÁXIMO" in mensagem


@pytest.mark.leve
def test_strict_aceita_o_percentil_representavel():
    calibracao = calibrate_threshold(np.arange(210, dtype=float), 99.0, strict=True)
    assert calibracao.value == 207.0


@pytest.mark.leve
def test_sem_strict_o_comportamento_historico_e_preservado():
    """Reproduzir o ponto operacional publicado continua possível.

    A reprodutibilidade do artefato existente não pode depender de eu concordar
    com ele.
    """
    calibracao = calibrate_threshold(np.arange(210, dtype=float), 99.9)
    assert calibracao.value == 209.0


@pytest.mark.leve
def test_o_erro_e_um_ValueError():
    """Chamadores antigos que capturam ValueError continuam funcionando."""
    assert issubclass(DegenerateThresholdError, ValueError)


# ── o que vai para o artefato ──────────────────────────────────────────────

@pytest.mark.leve
def test_o_contrato_publicado_carrega_a_marca():
    """Sem isto, o CSV e o manifesto não distinguem os dois casos."""
    degenerado = calibrate_threshold(np.arange(210, dtype=float), 99.9).as_dict()
    representavel = calibrate_threshold(np.arange(210, dtype=float), 99.0).as_dict()

    assert degenerado["threshold_is_sample_maximum"] is True
    assert degenerado["threshold_minimum_n_for_request"] == 1001
    assert representavel["threshold_is_sample_maximum"] is False
    assert representavel["threshold_minimum_n_for_request"] == 101


@pytest.mark.leve
def test_a_publicacao_canonica_e_estrita_por_padrao():
    """Guarda estrutural: o padrão do treino canônico não pode voltar a ser
    permissivo sem alguém reprovar aqui."""
    import inspect

    from src.ml import comparacao_autoencoders, treino_comparacao

    for modulo, funcao in (
        (treino_comparacao, "train_models"),
        (comparacao_autoencoders, "run"),
    ):
        alvo = getattr(modulo, funcao)
        parametro = inspect.signature(alvo).parameters.get("strict_threshold")
        assert parametro is not None, f"{funcao} perdeu o parâmetro strict_threshold"
        assert parametro.default is True, f"{funcao} deixou de ser estrita por padrão"
