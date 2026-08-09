"""
O split intercalado cobre a faixa de operação SEM abrir mão do anti-vazamento.

POR QUE ESTE TESTE EXISTE
=========================
O split de três blocos contíguos pressupõe sinal aproximadamente estacionário.
O conjunto Stender é uma bancada que varre rotação em rampa, então fatiar em três deu
três FAIXAS DE VELOCIDADE — medido em 09/08/2026:

    treino       mediana  20,45 Hz   IQR 83,13
    calibração   mediana  51,11 Hz   IQR  1,46   ← um regime só
    teste        mediana 100,08 Hz   IQR 17,84

O limiar era congelado na calibração e aplicado ao dobro da fundamental: FPR de
4,4% ali contra 62,8% no teste.

A troca por blocos intercalados resolve a cobertura. O RISCO da troca é
reintroduzir vazamento temporal — que era exatamente o que o split contíguo
existia para impedir. Estes testes existem para provar que isso não acontece:

**a propriedade que não pode cair, em nenhuma hipótese, é
`test_janelas_vizinhas_nunca_caem_em_conjuntos_diferentes`.**

Rodam sem torch e sem dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.split_temporal import (
    CALIB_RATIO_PADRAO,
    ESTRATEGIA_SPLIT,
    N_BLOCOS_PADRAO,
    PURGA_PADRAO,
    TEST_RATIO_PADRAO,
    TRAIN_RATIO_PADRAO,
    sequencia_de_destinos,
    split_blocos_intercalados,
    split_padrao_paderborn,
)

CONJUNTOS = ("treino", "val", "teste")
N_REAL = 228          # janelas da rodada de 09/08/2026, com JANELA = 2048


def _destino_por_indice(split: dict) -> dict:
    return {int(i): nome for nome in CONJUNTOS for i in split[nome]}


# ── A propriedade inegociável ──────────────────────────────────────────────

@pytest.mark.parametrize("n", [60, 120, N_REAL, 457, 1000])
def test_janelas_vizinhas_nunca_caem_em_conjuntos_diferentes(n):
    """Janelas com 50% de sobreposição compartilham metade das amostras.

    Se `i` e `i+1` caem em conjuntos diferentes, metade do sinal de teste esteve
    no treino. É o vazamento que o split contíguo existia para impedir, e
    intercalar os blocos não pode reintroduzi-lo.
    """
    split = split_blocos_intercalados(n)
    destino = _destino_por_indice(split)
    vizinhos_cruzados = [
        i for i in range(n - 1)
        if i in destino and (i + 1) in destino and destino[i] != destino[i + 1]
    ]
    assert not vizinhos_cruzados, (
        f"vazamento em {len(vizinhos_cruzados)} fronteiras: {vizinhos_cruzados[:5]}"
    )


@pytest.mark.parametrize("purga", [1, 2, 4])
def test_a_distancia_minima_entre_conjuntos_respeita_a_purga(purga):
    """Entre a última janela de um conjunto e a primeira do seguinte tem de
    haver pelo menos `purga` índices descartados."""
    split = split_blocos_intercalados(N_REAL, purge_janelas=purga)
    destino = _destino_por_indice(split)
    usados = sorted(destino)
    for a, b in zip(usados, usados[1:]):
        if destino[a] != destino[b]:
            assert b - a > purga, (
                f"índices {a} e {b} em conjuntos diferentes com folga de "
                f"apenas {b - a - 1}, abaixo da purga de {purga}"
            )


# ── partição: nada duplicado, nada inventado ───────────────────────────────

@pytest.mark.parametrize("n", [60, N_REAL, 457])
def test_conjuntos_sao_disjuntos(n):
    split = split_blocos_intercalados(n)
    todos = np.concatenate([split[c] for c in CONJUNTOS])
    assert len(set(todos.tolist())) == len(todos)


@pytest.mark.parametrize("n", [60, N_REAL, 457])
def test_indices_ficam_dentro_da_serie(n):
    split = split_blocos_intercalados(n)
    for c in CONJUNTOS:
        assert split[c].min() >= 0 and split[c].max() < n


def test_a_perda_por_purga_e_pequena_e_declarada():
    """22 de 228 janelas na configuração real — 9,6%. Vale saber se piorar."""
    split = split_blocos_intercalados(N_REAL)
    usadas = sum(len(split[c]) for c in CONJUNTOS)
    assert usadas == 206
    assert N_REAL - usadas == 22


@pytest.mark.parametrize("n", [60, N_REAL, 457])
def test_indices_de_cada_conjunto_saem_ordenados(n):
    """Consumidores fatiam DataFrame por estes índices e assumem ordem
    temporal; `amostra_inicio` tem de sair crescente."""
    split = split_blocos_intercalados(n)
    for c in CONJUNTOS:
        assert np.all(np.diff(split[c]) > 0), c


# ── cobertura de regime: a razão de existir ────────────────────────────────

def test_cada_conjunto_atravessa_a_maior_parte_da_serie():
    """O defeito que originou a mudança: a calibração ocupava uma fatia só.

    Amplitudes com n = 228: treino 99,6%, calibração 69,7%, teste 84,6%. O
    primeiro e o último bloco são de treino, mantendo as pontas da rampa no
    conjunto que ajusta o modelo.
    """
    split = split_blocos_intercalados(N_REAL)
    for c in CONJUNTOS:
        amplitude = split[c].max() - split[c].min()
        assert amplitude > 0.65 * N_REAL, (
            f"{c} cobre só {amplitude} de {N_REAL} índices — voltou a ser fatia"
        )


def test_o_split_contiguo_NAO_cobria_a_serie_e_esse_era_o_problema():
    """Contraprova. Sem ela, o teste acima poderia passar por acidente.

    Com o split contíguo: treino 59%, calibração 19%, teste 18%. É a razão
    numérica de a calibração ter saído com IQR de F0 de 1,46 Hz.
    """
    from src.ml.split_temporal import split_temporal_com_purga

    antigo = split_temporal_com_purga(
        N_REAL, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )
    for c in ("val", "teste"):
        amplitude = antigo[c].max() - antigo[c].min()
        assert amplitude < 0.25 * N_REAL, (
            f"{c} contíguo deveria ocupar ~20% da série — se não ocupa, a "
            "premissa deste teste mudou"
        )
    novo = split_blocos_intercalados(N_REAL)
    for c in ("val", "teste"):
        assert (novo[c].max() - novo[c].min()) > 3 * (
            antigo[c].max() - antigo[c].min()), f"{c} não ganhou cobertura"


def test_cada_conjunto_recebe_varios_blocos_separados():
    split = split_blocos_intercalados(N_REAL)
    assert len(split["limites"]["treino"]) == 7
    assert len(split["limites"]["val"]) == 3
    assert len(split["limites"]["teste"]) == 4


def test_split_padrao_entrega_amostra_independente_minima():
    from src.ml.dados_avaliacao import _indices_sem_sobreposicao

    inicios = np.arange(N_REAL) * 1024
    import pandas as pd

    df = pd.DataFrame({"amostra_inicio": inicios})
    split = split_padrao_paderborn(N_REAL)
    independentes = _indices_sem_sobreposicao(df, split["teste"], janela=2048)

    assert len(split["treino"]) == 104
    assert len(split["val"]) == 42
    assert len(split["teste"]) == 60
    assert len(independentes) == 32


# ── a sequência de destinos ────────────────────────────────────────────────

def test_sequencia_padrao_alterna_como_documentado():
    seq = sequencia_de_destinos(14, {"treino": 0.5, "val": 0.2, "teste": 0.3})
    assert "".join({"treino": "T", "val": "V", "teste": "E"}[x] for x in seq) == (
        "TEVTTETVETTVET"
    )


def test_sequencia_respeita_as_proporcoes():
    seq = sequencia_de_destinos(14, {"treino": 0.5, "val": 0.2, "teste": 0.3})
    assert seq.count("treino") == 7
    assert seq.count("val") == 3
    assert seq.count("teste") == 4


def test_sequencia_e_deterministica():
    """Sem sorteio: duas chamadas dão exatamente a mesma ordem.

    É a propriedade que o split contíguo tinha e que não podia ser perdida —
    reprodutibilidade por construção, não por semente.
    """
    ratios = {"treino": 0.5, "val": 0.2, "teste": 0.3}
    assert sequencia_de_destinos(14, ratios) == sequencia_de_destinos(14, ratios)


@pytest.mark.parametrize("n_blocos", [3, 6, 9, 12, 15, 21, 30])
def test_todo_conjunto_recebe_ao_menos_um_bloco(n_blocos):
    seq = sequencia_de_destinos(n_blocos, {"treino": 0.6, "val": 0.2, "teste": 0.2})
    assert len(seq) == n_blocos
    for c in CONJUNTOS:
        assert seq.count(c) >= 1, f"{c} ficou sem bloco com n_blocos={n_blocos}"


def test_blocos_de_menos_e_recusado():
    with pytest.raises(ValueError):
        sequencia_de_destinos(2, {"treino": 0.6, "val": 0.2, "teste": 0.2})


# ── o split canônico do pipeline ───────────────────────────────────────────

def test_split_padrao_usa_a_estrategia_intercalada():
    split = split_padrao_paderborn(N_REAL)
    assert split["estrategia"] == ESTRATEGIA_SPLIT == "blocos_intercalados"
    assert split["n_blocos"] == N_BLOCOS_PADRAO


def test_split_padrao_carrega_os_ratios_do_modulo():
    r = split_padrao_paderborn(N_REAL)["ratios"]
    assert r == {"train": TRAIN_RATIO_PADRAO, "val": CALIB_RATIO_PADRAO,
                 "test": TEST_RATIO_PADRAO}


def test_autoencoder_e_split_declaram_os_mesmos_ratios():
    """`autoencoder.py` espelha os ratios para o manifesto lê-los por AST.

    Espelho que diverge da fonte é pior que espelho nenhum: o manifesto
    registraria um split que não foi o executado.
    """
    import ast
    from pathlib import Path

    from src.core.config import RAIZ_PROJETO

    fonte = ast.parse((RAIZ_PROJETO / "src/ml/autoencoder.py").read_text("utf-8"))
    valores = {
        alvo.id: no.value.value
        for no in fonte.body if isinstance(no, ast.Assign)
        for alvo in no.targets
        if isinstance(alvo, ast.Name) and isinstance(no.value, ast.Constant)
    }
    assert valores["TRAIN_RATIO"] == TRAIN_RATIO_PADRAO
    assert valores["CALIB_RATIO"] == CALIB_RATIO_PADRAO
    assert valores["TEST_RATIO"] == TEST_RATIO_PADRAO


def test_purga_padrao_cobre_a_sobreposicao_das_janelas():
    """Sobreposição de 50% ⇒ 1 janela de purga já basta; 2 é a margem."""
    assert PURGA_PADRAO >= 1


# ── entradas inválidas ─────────────────────────────────────────────────────

def test_mais_blocos_que_janelas_e_recusado():
    with pytest.raises(ValueError):
        split_blocos_intercalados(10, n_blocos=15)


def test_ratios_que_nao_somam_um_sao_recusados():
    with pytest.raises(ValueError):
        split_blocos_intercalados(N_REAL, train_ratio=0.7, val_ratio=0.2,
                                  test_ratio=0.2)


def test_purga_negativa_e_recusada():
    with pytest.raises(ValueError):
        split_blocos_intercalados(N_REAL, purge_janelas=-1)


def test_serie_curta_demais_falha_com_mensagem_util():
    with pytest.raises(ValueError, match="Janelas insuficientes|n_blocos"):
        split_blocos_intercalados(4, n_blocos=3, purge_janelas=10)
