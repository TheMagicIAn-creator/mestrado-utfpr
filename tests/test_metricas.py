"""
Sprint 2 — métricas (itens 4.1 e 4.2).

- specificity binária = TN/(TN+FP), rotulada como tal;
- em multiclasse, specificity é o macro one-vs-rest (rotulado);
- MCC, balanced_accuracy, FPR e FNR presentes.
"""

from src.ml.experimentos_artigos import _metricas_classificacao


def test_specificity_binaria_tn_sobre_tn_fp():
    # real: 3 negativos, 2 positivos | pred: 1 FP, 0 FN
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1]  # TN=2, FP=1, FN=0, TP=2
    m = _metricas_classificacao(y_true, y_pred)

    assert m["specificity_tipo"] == "binaria_TN/(TN+FP)"
    assert abs(m["specificity"] - 2 / 3) < 1e-9
    assert abs(m["false_positive_rate"] - 1 / 3) < 1e-9
    assert m["false_negative_rate"] == 0.0
    assert "mcc" in m and "balanced_accuracy" in m
    assert "specificity_macro_ovr" in m


def test_specificity_multiclasse_macro_ovr():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 1, 1, 2]
    m = _metricas_classificacao(y_true, y_pred)

    assert m["specificity_tipo"] == "macro_one_vs_rest"
    assert m["specificity"] == m["specificity_macro_ovr"]
    assert m["false_positive_rate"] is None  # FPR binário não se aplica
    assert m["n_classes"] == 3


def test_mcc_balanced_presentes_em_ambos():
    for y_true, y_pred in (
        ([0, 1, 0, 1], [0, 1, 0, 1]),
        ([0, 1, 2], [0, 1, 2]),
    ):
        m = _metricas_classificacao(y_true, y_pred)
        assert isinstance(m["mcc"], float)
        assert isinstance(m["balanced_accuracy"], float)
