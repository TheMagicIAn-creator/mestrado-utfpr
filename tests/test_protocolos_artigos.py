"""
Contratos do protocolo comparativo ativo de anomalia.

O nucleo executavel vigente e apenas Ibrahim et al. (2022), restrito ao
AE-LSTM temporal. A injecao usada no banco comum e FMECA no espaco de
features, com split temporal e limiar congelado fora dos rotulos do teste.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import src.ml.protocolos_artigos as P


NOMES = [
    "i_a_rms", "i_a_pico_a_pico", "i_a_desvio", "i_a_kurtosis", "i_a_thd",
    "i_a_harm_5", "i_a_harm_7", "i_a_harm_11", "i_a_largura_banda",
    "i_a_centroide", "i_a_energia_media", "i_a_energia_chaveamento",
    "i_b_rms", "i_b_thd", "i_b_harm_5", "i_b_harm_11", "i_b_energia_media",
    "i_c_thd", "i_c_harm_5", "potencia_a", "desbalanceamento_corrente",
    "u_a_rms", "tensao_dc_media",
]


def _X_sintetico(n=320, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(10.0, 2.0, size=(n, len(NOMES)))
    return np.abs(base) + 0.5


@pytest.fixture()
def features_fake(monkeypatch):
    import src.ml.experimentos_artigos as E

    X = _X_sintetico()
    monkeypatch.setattr(
        E, "_carregar_features_paderborn",
        lambda progresso=None: (X.copy(), list(NOMES)),
    )
    return X


def test_injecao_fmeca_perturba_somente_assinatura():
    X = _X_sintetico(n=200)
    rng = np.random.default_rng(1)
    X_anom, tipos = P.injetar_falhas_fmeca(X, NOMES, rng)

    assert len(tipos) == len(X)
    assert set(tipos) <= {"contator_ac", "igbt", "fusivel_ac"}

    intocaveis = [
        NOMES.index("i_a_kurtosis"),
        NOMES.index("u_a_rms"),
        NOMES.index("tensao_dc_media"),
    ]
    assert np.allclose(X_anom[:, intocaveis], X[:, intocaveis])

    idx_igbt = np.where(tipos == "igbt")[0]
    assert len(idx_igbt) > 0
    j_h5 = NOMES.index("i_a_harm_5")
    j_rms = NOMES.index("i_a_rms")
    assert np.all(X_anom[idx_igbt, j_h5] > X[idx_igbt, j_h5])
    assert np.allclose(X_anom[idx_igbt, j_rms], X[idx_igbt, j_rms])

    idx_fus = np.where(tipos == "fusivel_ac")[0]
    assert len(idx_fus) > 0
    j_rms_b = NOMES.index("i_b_rms")
    assert np.all(X_anom[idx_fus, j_rms] < X[idx_fus, j_rms])
    assert np.all(X_anom[idx_fus, j_rms_b] > X[idx_fus, j_rms_b])


def test_injecao_fmeca_deterministica_com_mesma_semente():
    X = _X_sintetico(n=60)
    a1, t1 = P.injetar_falhas_fmeca(X, NOMES, np.random.default_rng(7))
    a2, t2 = P.injetar_falhas_fmeca(X, NOMES, np.random.default_rng(7))
    assert np.allclose(a1, a2)
    assert list(t1) == list(t2)


def test_deteccao_por_falha():
    y_true = [0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 1, 0, 1, 0]
    tipos = ["normal", "normal", "igbt", "igbt", "contator_ac", "fusivel_ac"]
    d = P.deteccao_por_falha(y_true, y_pred, tipos)
    assert d["igbt"] == pytest.approx(0.5)
    assert d["contator_ac"] == pytest.approx(1.0)
    assert d["fusivel_ac"] == pytest.approx(0.0)


def test_preparar_dados_split_temporal_sem_vazamento(features_fake):
    dados = P.preparar_dados_anomalia()
    sp = dados["split"]
    assert sp["tipo"] == "temporal_com_purga"
    assert sp["treino"] + sp["teste"] < sp["n_janelas"]
    assert dados["injecao"]["tipo"] == "fmeca_espaco_features"

    y = dados["y_te"]
    assert int(y.sum()) == len(y) // 2
    assert len(dados["tipos_te"]) == int(y.sum())
    assert abs(float(dados["Xn_tr"].mean())) < 0.05
    assert "X_val" not in dados


def test_preparar_dados_com_validacao(features_fake):
    dados = P.preparar_dados_anomalia(com_validacao=True)
    assert "X_val" in dados and "y_val" in dados
    assert dados["split"]["val"] > 0


def test_executar_protocolo_desconhecido_retorna_none():
    assert P.executar_protocolo("artigo_inexistente") is None
    assert P.executar_protocolo("francisti") is None


def test_comparar_auc_le_apenas_ibrahim(tmp_path, monkeypatch):
    import src.ml.experimentos_artigos as E

    monkeypatch.setattr(E, "PASTA_EXPERIMENTOS", tmp_path)
    (tmp_path / "ibrahim").mkdir()
    (tmp_path / "ibrahim" / "resultado.json").write_text(json.dumps({
        "referencia": "Ibrahim (2022)",
        "modelos": {
            "AE-LSTM": {"auc": 0.73, "disponivel": True},
            "Modelo indisponivel": {"disponivel": False, "motivo": "requer lib"},
        },
    }), encoding="utf-8")

    cmp = E.comparar_anomalia_por_auc()
    assert cmp["ok"] is True
    assert cmp["dados"] == [("Ibrahim (2022)", "AE-LSTM", 0.73)]
    assert "AUC" in cmp["tabela_md"] and "0.730" in cmp["tabela_md"]


def test_comparar_auc_json_corrompido_retorna_fail_fast(tmp_path, monkeypatch):
    import src.ml.experimentos_artigos as E

    monkeypatch.setattr(E, "PASTA_EXPERIMENTOS", tmp_path)
    (tmp_path / "ibrahim").mkdir()
    (tmp_path / "ibrahim" / "resultado.json").write_text(
        "{ isto nao e JSON valido", encoding="utf-8")

    cmp = E.comparar_anomalia_por_auc()
    assert cmp["ok"] is False
    assert cmp["dados"] == []


def test_executar_protocolo_ibrahim_inclui_split_e_injecao(features_fake, monkeypatch):
    def protocolo_fake(dados, progresso=None):
        return {"AE-LSTM": {"disponivel": True, "auc": 0.7}}, {
            "protocolo": "ibrahim2022_series_temporais",
        }

    monkeypatch.setitem(P.PROTOCOLOS, "ibrahim", (protocolo_fake, False))

    modelos, met = P.executar_protocolo("ibrahim")
    assert met["split"]["tipo"] == "temporal_com_purga"
    assert met["injecao"]["tipo"] == "fmeca_espaco_features"
    assert "AE-LSTM" in modelos


def test_metricas_anomalia_caminhos():
    from src.ml.experimentos_artigos import _metricas_anomalia

    y = [0, 0, 1, 1, 0, 1]
    s = [0.1, 0.2, 0.9, 0.8, 0.15, 0.85]

    m1 = _metricas_anomalia(y, s)
    assert m1["threshold_source"] == "exploratorio_no_conjunto_avaliado"
    assert m1["metrica_dependente_de_limiar"] == "exploratoria"

    m2 = _metricas_anomalia(y, s, y_pred=[0, 0, 1, 1, 0, 1])
    assert m2["threshold_source"] == "decisao_nativa_modelo"

    m3 = _metricas_anomalia(
        y, s, y_pred=[0, 0, 1, 1, 0, 1],
        threshold_source="p99_erro_seq_temporal_calibracao",
        limiar=3.0,
    )
    assert m3["threshold_source"] == "p99_erro_seq_temporal_calibracao"
    assert m3["metrica_dependente_de_limiar"] == "a_priori_ou_congelada"
    assert m3["limiar_score"] == pytest.approx(3.0)
    assert m3["ponto_operacao"] == "protocolo_do_artigo"


def test_metricas_regime_raro_colapsa_precision_com_fpr():
    from src.ml.experimentos_artigos import _metricas_anomalia

    y = [1] * 10 + [0] * 10
    yp = [1] * 8 + [0] * 2 + [1] * 3 + [0] * 7
    s = [0.9] * 10 + [0.1] * 10
    m = _metricas_anomalia(y, s, y_pred=yp, threshold_source="x", limiar=0.5)
    assert m["prevalencia_raro"] == 0.05
    assert m["fpr_operacao"] == pytest.approx(0.3)
    assert m["precision_raro"] < m["precision"]
    assert m["precision_raro"] < 0.2

    yp0 = [1] * 8 + [0] * 2 + [0] * 10
    m0 = _metricas_anomalia(y, s, y_pred=yp0, threshold_source="x", limiar=0.5)
    assert m0["fpr_operacao"] == pytest.approx(0.0)
    assert m0["precision_raro"] == pytest.approx(1.0)


def test_apenas_ibrahim_esta_no_dispatch():
    assert set(P.PROTOCOLOS) == {"ibrahim"}
    assert not hasattr(P, "protocolo_francisti")
    assert not hasattr(P, "protocolo_ahirwar")
    assert P.executar_protocolo("ahirwar") is None
    assert P.executar_protocolo("stender") is None


def test_rodar_modelo_isola_falha_de_runtime():
    saida, preds = {}, {}

    def ok():
        return {"disponivel": True, "auc": 0.7}, "y_pred_fake"

    def quebra():
        raise AttributeError("backend indisponivel")

    P._rodar_modelo("Modelo OK", ok, saida, preds)
    P._rodar_modelo("Modelo quebrado", quebra, saida, preds)

    assert saida["Modelo OK"]["disponivel"] is True
    assert preds["Modelo OK"] == "y_pred_fake"
    assert saida["Modelo quebrado"]["disponivel"] is False
    assert "backend indisponivel" in saida["Modelo quebrado"]["motivo"]
    assert "Modelo quebrado" not in preds
