"""
Contratos da taxonomia de evidencia e dos limiares exploratorios.
"""


def test_metricas_anomalia_limiar_exploratorio():
    from src.ml.experimentos_artigos import _metricas_anomalia

    y_true = [0, 0, 1, 1, 0, 1]
    score = [0.10, 0.20, 0.90, 0.80, 0.15, 0.85]
    m = _metricas_anomalia(y_true, score)
    assert m["threshold_source"] == "exploratorio_no_conjunto_avaliado"
    assert m["metrica_dependente_de_limiar"] == "exploratoria"


def test_metricas_anomalia_decisao_nativa():
    from src.ml.experimentos_artigos import _metricas_anomalia

    y_true = [0, 0, 1, 1, 0, 1]
    score = [0.10, 0.20, 0.90, 0.80, 0.15, 0.85]
    y_pred = [0, 0, 1, 1, 0, 1]
    m = _metricas_anomalia(y_true, score, y_pred=y_pred)
    assert m["threshold_source"] == "decisao_nativa_modelo"


def test_consolidar_marca_evidence_level_e1(monkeypatch):
    from src.ml import experimentos_artigos as E

    monkeypatch.setattr(E, "_salvar_resultado", lambda exp, res: None)
    monkeypatch.setattr(E, "_grafico_comparacao", lambda exp, res: [])

    exp = E.REGISTRO["ibrahim"]
    modelos = {"AE-LSTM": {"auc": 0.80, "disponivel": True}}
    res = E._consolidar(exp, modelos, "auc")
    assert res["evidence_level"] == "E1"
    assert "explorat" in res["evidence_note"].lower()


def test_perfil_conhece_niveis_evidencia():
    from src.conhecimento.agente import PERFIL_COMPACTO

    for marca in ("E0", "E1", "E2", "E3"):
        assert marca in PERFIL_COMPACTO
    assert "industrial" in PERFIL_COMPACTO.lower()
