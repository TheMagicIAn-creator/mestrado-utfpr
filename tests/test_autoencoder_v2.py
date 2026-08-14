from __future__ import annotations

import numpy as np
import pytest

from src.ml.autoencoder_v2.avaliacao import (
    diferenca_pareada,
    metricas_cenario,
    resumir_macros,
)
from src.ml.autoencoder_v2.experimento import (
    limiar_ordem_finita,
    seed_representativo,
    selecionar_arquitetura,
)
from src.ml.autoencoder_v2.modelo import (
    ARQUITETURAS_CANDIDATAS,
    FAMILIAS_FEATURES,
    Arquitetura,
    pesos_por_familia,
    pontuar_residuos,
)
from src.ml.gpvs import FEATURE_COLUMNS


def test_familias_cobrem_cada_feature_exatamente_uma_vez():
    declaradas = [nome for grupo in FAMILIAS_FEATURES.values() for nome in grupo]
    assert len(declaradas) == len(set(declaradas)) == len(FEATURE_COLUMNS)
    assert set(declaradas) == set(FEATURE_COLUMNS)


def test_pesos_somam_um_e_equilibram_familias():
    pesos = pesos_por_familia(FEATURE_COLUMNS)
    mapa = dict(zip(FEATURE_COLUMNS, pesos, strict=True))
    assert pesos.sum() == pytest.approx(1.0)
    for grupo in FAMILIAS_FEATURES.values():
        assert sum(mapa[nome] for nome in grupo) == pytest.approx(0.25)


def test_pontuacao_balanceada_nao_favorece_familia_maior():
    pesos = pesos_por_familia(FEATURE_COLUMNS)
    resultados = []
    for grupo in FAMILIAS_FEATURES.values():
        residuos = np.zeros((1, len(FEATURE_COLUMNS)))
        for nome in grupo:
            residuos[0, FEATURE_COLUMNS.index(nome)] = 1.0
        resultados.append(float(pontuar_residuos(residuos, pesos)[0]))
    assert resultados == pytest.approx([0.25] * len(FAMILIAS_FEATURES))


def test_arquitetura_serializa_sem_perder_tuplas():
    original = ARQUITETURAS_CANDIDATAS[-1]
    assert Arquitetura.de_dict(original.como_dict()) == original


def test_contrato_recusa_feature_sem_familia():
    with pytest.raises(ValueError, match="Contrato de familias"):
        pesos_por_familia([*FEATURE_COLUMNS[:-1], "inventada"])


def test_limiar_finito_usa_ordem_sem_interpolacao():
    scores = np.arange(1, 211, dtype=float)
    info = limiar_ordem_finita(scores, alpha=0.01)
    assert info["order_one_based"] == 209
    assert info["threshold"] == 209.0
    assert info["n_strictly_above_calibration"] == 1
    assert info["empirical_resolution_pct"] == pytest.approx(100 / 210)


def test_selecao_usa_validacao_e_desempata_por_parcimonia():
    import pandas as pd

    linhas = []
    for seed, perda in enumerate((1.00, 1.01, 0.99), start=1):
        linhas.append({
            "arquitetura": "menor", "seed": seed,
            "perda_validacao": perda * 1.015, "n_parametros": 100,
        })
        linhas.append({
            "arquitetura": "maior", "seed": seed,
            "perda_validacao": perda, "n_parametros": 500,
        })
    escolhida, resumo = selecionar_arquitetura(pd.DataFrame(linhas))
    assert escolhida == "menor"
    assert resumo.loc[resumo["arquitetura"].eq("menor"), "selecionada"].item()


def test_seed_representativo_e_o_mais_proximo_da_mediana():
    import pandas as pd

    execucoes = pd.DataFrame([
        {"arquitetura": "a", "seed": 13, "perda_validacao": 0.9},
        {"arquitetura": "a", "seed": 29, "perda_validacao": 0.5},
        {"arquitetura": "a", "seed": 42, "perda_validacao": 0.7},
    ])
    assert seed_representativo(execucoes, "a") == 42


def test_metricas_cenario_respeitam_contagens_e_atraso_sustentado():
    import pandas as pd

    features = pd.DataFrame({"tempo_inicio_s": np.arange(10, dtype=float) * 0.02})
    indice = np.asarray([0.2, 1.2, 0.3, 0.4, 2.0, 2.1, 2.2, 0.1, 0.2, 0.3])
    metricas = metricas_cenario(
        features,
        indice,
        np.asarray([0, 1, 2, 3]),
        np.asarray([4, 5, 6, 7, 8, 9]),
        persistencia=3,
    )
    assert metricas["true_negatives"] == 3
    assert metricas["false_positives"] == 1
    assert metricas["true_positives"] == 3
    assert metricas["false_negatives"] == 3
    assert metricas["specificity"] == pytest.approx(0.75)
    assert metricas["sensitivity"] == pytest.approx(0.5)
    assert metricas["detection_delay_from_nominal_midpoint_s"] == pytest.approx(0.0)


def test_macro_trata_ensaio_como_unidade():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "method": ["autoencoder_v2", "autoencoder_v2"],
            "auc_roc": [0.6, 1.0],
            "average_precision": [0.5, 0.9],
            "sensitivity": [0.2, 0.8],
            "specificity": [0.9, 0.7],
            "balanced_accuracy": [0.55, 0.75],
            "mcc": [0.1, 0.7],
            "sustained_detection": [False, True],
        }
    )
    resumo = resumir_macros(frame)["autoencoder_v2"]
    assert resumo["auc_roc"]["mean"] == pytest.approx(0.8)
    assert resumo["auc_roc"]["n_experiments"] == 2
    assert resumo["sustained_detection"] == {"n": 1, "total": 2, "rate": 0.5}


def test_comparacao_pareada_preserva_pares_de_ensaio():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "experiment": ["F1L", "F2L", "F1L", "F2L"],
            "method": ["autoencoder_v2", "autoencoder_v2", "pca", "pca"],
            "auc_roc": [0.9, 0.6, 0.8, 0.7],
        }
    )
    resultado = diferenca_pareada(
        frame, "autoencoder_v2", "pca", "auc_roc", n_boot=1_000
    )
    assert resultado["mean_difference"] == pytest.approx(0.0)
    assert resultado["wins_a"] == 1
    assert resultado["wins_b"] == 1
    assert resultado["n_pairs"] == 2


@pytest.mark.integracao
def test_rede_v2_tem_gargalo_linear_e_saida_compativel():
    torch = pytest.importorskip("torch")
    from src.ml.autoencoder_v2.modelo import AutoencoderDenso

    arquitetura = ARQUITETURAS_CANDIDATAS[-1]
    modelo = AutoencoderDenso(len(FEATURE_COLUMNS), arquitetura)
    lote = torch.randn(7, len(FEATURE_COLUMNS))
    assert modelo(lote).shape == lote.shape
    assert modelo.encode(lote).shape == (7, arquitetura.latente)
    assert modelo.n_parametros > 0


@pytest.mark.integracao
def test_seed_e_aplicada_antes_da_inicializacao_dos_pesos():
    pytest.importorskip("torch")
    from src.ml.autoencoder_v2.modelo import AutoencoderDenso, configurar_seed

    arquitetura = ARQUITETURAS_CANDIDATAS[0]
    configurar_seed(42)
    primeiro = AutoencoderDenso(len(FEATURE_COLUMNS), arquitetura)
    configurar_seed(42)
    segundo = AutoencoderDenso(len(FEATURE_COLUMNS), arquitetura)

    for peso_a, peso_b in zip(
        primeiro.parameters(), segundo.parameters(), strict=True
    ):
        assert np.array_equal(
            peso_a.detach().numpy(), peso_b.detach().numpy()
        )
