"""Contratos leves do classificador supervisionado legado."""

import numpy as np

from src.ml import classificador_pv


def test_preprocessar_ajusta_apenas_no_treino():
    treino = np.array([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]])
    teste = np.array([[6.0, 22.0]])

    treino_norm, teste_norm, scaler = classificador_pv.preprocessar(treino, teste)

    assert np.allclose(treino_norm.mean(axis=0), 0.0)
    assert np.allclose(teste_norm, scaler.transform(teste))


def test_criar_modelos_mantem_baselines_obrigatorios():
    modelos = classificador_pv.criar_modelos()
    assert {"Random Forest", "Gradient Boosting", "SVM"} <= set(modelos)


def test_linhas_identicas_recebem_o_mesmo_grupo_na_cv():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0]])

    grupos = classificador_pv._grupos_linhas_identicas(X)

    assert grupos[0] == grupos[2]
    assert grupos[0] != grupos[1]
    assert len(np.unique(grupos)) == 3
