"""
Reestruturação — protocolos de avaliação POR ARTIGO (anti "erro de simulação").

Garante que:
- a injeção FMEA perturba SOMENTE as features da assinatura física de cada
  família de falha, com ground truth por família;
- o preparo de dados usa split TEMPORAL com purga (sem vazamento) e scaler
  ajustado só no treino;
- Francisti decide por Shewhart 3σ FIXO (nunca limiar-oráculo no teste);
- métricas com threshold_source explícito são marcadas "a_priori_ou_congelada"
  e os caminhos legados continuam intactos;
- artigos sem protocolo caem no harness legado (executar_protocolo → None).

CI-leve: sem torch/prophet/sb3 — só numpy + scikit-learn, com features
sintéticas (monkeypatch do carregador do Paderborn).
"""

from __future__ import annotations

import numpy as np
import pytest

import src.ml.protocolos_artigos as P

# Nomes de features representativos do features_ca.py (cobrem as assinaturas)
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
    # banda positiva e escala variada por coluna — parecido com features reais
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


# ── injeção FMEA ─────────────────────────────────────────────────────────────

def test_injecao_perturba_somente_assinatura():
    X = _X_sintetico(n=200)
    rng = np.random.default_rng(1)
    X_anom, tipos = P.injetar_falhas_fmea(X, NOMES, rng)

    assert len(tipos) == len(X)
    assert set(tipos) <= {"lcl", "desbalanceamento", "sensor"}

    # colunas que NENHUMA assinatura toca devem permanecer idênticas
    # (tensões u_* não são afetadas por nenhuma falha de corrente; kurtosis e
    # tensão CC também não).
    intocaveis = [NOMES.index("i_a_kurtosis"), NOMES.index("u_a_rms"),
                  NOMES.index("tensao_dc_media")]
    assert np.allclose(X_anom[:, intocaveis], X[:, intocaveis])

    # numa janela LCL: harmônicos sobem, rms da fase A fica intacto
    idx_lcl = np.where(tipos == "lcl")[0]
    assert len(idx_lcl) > 0
    j_h5 = NOMES.index("i_a_harm_5")
    j_rms = NOMES.index("i_a_rms")
    assert np.all(X_anom[idx_lcl, j_h5] > X[idx_lcl, j_h5])
    assert np.allclose(X_anom[idx_lcl, j_rms], X[idx_lcl, j_rms])

    # numa janela de desbalanceamento: fase A CAI e fases B/C COMPENSAM (sobem)
    idx_db = np.where(tipos == "desbalanceamento")[0]
    assert len(idx_db) > 0
    j_rms_b = NOMES.index("i_b_rms")
    assert np.all(X_anom[idx_db, j_rms] < X[idx_db, j_rms])
    assert np.all(X_anom[idx_db, j_rms_b] > X[idx_db, j_rms_b])


def test_injecao_deterministica_com_mesma_semente():
    X = _X_sintetico(n=60)
    a1, t1 = P.injetar_falhas_fmea(X, NOMES, np.random.default_rng(7))
    a2, t2 = P.injetar_falhas_fmea(X, NOMES, np.random.default_rng(7))
    assert np.allclose(a1, a2)
    assert list(t1) == list(t2)


def test_deteccao_por_falha():
    y_true = [0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 1, 0, 1, 0]
    tipos = ["normal", "normal", "lcl", "lcl", "sensor", "desbalanceamento"]
    d = P.deteccao_por_falha(y_true, y_pred, tipos)
    assert d["lcl"] == pytest.approx(0.5)
    assert d["sensor"] == pytest.approx(1.0)
    assert d["desbalanceamento"] == pytest.approx(0.0)


# ── preparo de dados (split temporal + scaler honesto) ──────────────────────

def test_preparar_dados_split_temporal_sem_vazamento(features_fake):
    dados = P.preparar_dados_anomalia()
    sp = dados["split"]
    assert sp["tipo"] == "temporal_com_purga"
    assert sp["treino"] + sp["teste"] < sp["n_janelas"]  # purga descartou
    # teste balanceado: metade normal, metade anômala
    y = dados["y_te"]
    assert int(y.sum()) == len(y) // 2
    assert len(dados["tipos_te"]) == int(y.sum())
    # scaler ajustado SÓ no treino → treino ~N(0,1); teste não exatamente
    assert abs(float(dados["Xn_tr"].mean())) < 0.05
    assert "X_val" not in dados


def test_preparar_dados_com_validacao(features_fake):
    dados = P.preparar_dados_anomalia(com_validacao=True)
    assert "X_val" in dados and "y_val" in dados
    assert dados["split"]["val"] > 0


# ── protocolo Francisti (Shewhart 3σ fixo) ───────────────────────────────────

def test_protocolo_francisti_limiar_fixo_sem_oraculo(features_fake):
    dados = P.preparar_dados_anomalia()
    saida, met = P.protocolo_francisti(dados)

    z = saida["Z-score (estatístico)"]
    assert z["threshold_source"] == "shewhart_3sigma_a_priori"
    assert z["limiar_score"] == pytest.approx(P.LIMIAR_SIGMA)
    assert z["metrica_dependente_de_limiar"] == "a_priori_ou_congelada"
    assert "deteccao_por_falha" in z
    assert z["auc"] > 0.5  # injeção FMEA é detectável acima do acaso

    # Curadoria (só núcleo CA): o RF supervisionado foi removido — francisti é
    # agora um detector NÃO-supervisionado puro (SPC/Z-score).
    assert "Random Forest (anomalia)" not in saida
    assert met["protocolo"] == "francisti2025_spc"
    assert "fidelidade" in met


def test_executar_protocolo_desconhecido_retorna_none():
    assert P.executar_protocolo("artigo_inexistente") is None


def test_comparar_auc_le_jsons_salvos(tmp_path, monkeypatch):
    """comparar_anomalia_por_auc lê os resultado.json e ordena por AUC."""
    import json
    import src.ml.experimentos_artigos as E

    monkeypatch.setattr(E, "PASTA_EXPERIMENTOS", tmp_path)
    # cria resultado.json mínimo p/ dois experimentos de anomalia
    for k, ref, auc_rf in [("francisti", "Francisti (2025)", 0.94),
                           ("ibrahim", "Ibrahim (2022)", 0.73)]:
        (tmp_path / k).mkdir()
        (tmp_path / k / "resultado.json").write_text(json.dumps({
            "referencia": ref,
            "modelos": {"M1": {"auc": auc_rf, "disponivel": True},
                        "M2": {"disponivel": False, "motivo": "requer torch"}},
        }), encoding="utf-8")

    cmp = E.comparar_anomalia_por_auc()
    assert cmp["ok"] is True
    # ordenado por AUC desc; modelo indisponível (sem auc) é ignorado
    aucs = [a for _ref, _n, a in cmp["dados"]]
    assert aucs == sorted(aucs, reverse=True)
    assert "AUC" in cmp["tabela_md"] and "0.940" in cmp["tabela_md"]


def test_comparar_auc_tolera_json_corrompido(tmp_path, monkeypatch):
    """JSON corrompido em um experimento não derruba a comparação."""
    import json
    import src.ml.experimentos_artigos as E

    monkeypatch.setattr(E, "PASTA_EXPERIMENTOS", tmp_path)
    (tmp_path / "francisti").mkdir()
    (tmp_path / "francisti" / "resultado.json").write_text(
        json.dumps({"referencia": "Francisti (2025)",
                    "modelos": {"M1": {"auc": 0.9, "disponivel": True}}}),
        encoding="utf-8")
    (tmp_path / "ibrahim").mkdir()
    (tmp_path / "ibrahim" / "resultado.json").write_text(
        "{ isto não é JSON válido", encoding="utf-8")  # corrompido

    cmp = E.comparar_anomalia_por_auc()
    assert cmp["ok"] is True  # ainda funciona com o experimento válido
    assert any("0.900" in cmp["tabela_md"] for _ in [0])


def test_executar_protocolo_francisti_inclui_split_e_injecao(features_fake):
    modelos, met = P.executar_protocolo("francisti")
    assert met["split"]["tipo"] == "temporal_com_purga"
    assert met["injecao"]["tipo"] == "fmea_espaco_features"
    assert "Z-score (estatístico)" in modelos


# ── métricas: caminhos legados intactos + caminho de protocolo ───────────────

def test_metricas_anomalia_caminhos():
    from src.ml.experimentos_artigos import _metricas_anomalia

    y = [0, 0, 1, 1, 0, 1]
    s = [0.1, 0.2, 0.9, 0.8, 0.15, 0.85]

    # legado 1: sem y_pred → exploratório (contrato de test_evidencia)
    m1 = _metricas_anomalia(y, s)
    assert m1["threshold_source"] == "exploratorio_no_conjunto_avaliado"
    assert m1["metrica_dependente_de_limiar"] == "exploratoria"

    # legado 2: y_pred sem fonte → decisão nativa
    m2 = _metricas_anomalia(y, s, y_pred=[0, 0, 1, 1, 0, 1])
    assert m2["threshold_source"] == "decisao_nativa_modelo"

    # protocolo: fonte explícita → a priori/congelada, com limiar registrado
    m3 = _metricas_anomalia(y, s, y_pred=[0, 0, 1, 1, 0, 1],
                            threshold_source="shewhart_3sigma_a_priori",
                            limiar=3.0)
    assert m3["threshold_source"] == "shewhart_3sigma_a_priori"
    assert m3["metrica_dependente_de_limiar"] == "a_priori_ou_congelada"
    assert m3["limiar_score"] == pytest.approx(3.0)
    assert m3["ponto_operacao"] == "protocolo_do_artigo"


def test_metricas_regime_raro_colapsa_precision_com_fpr():
    """Regime raro (prevalência 5%): precisão cai quando há FPR>0; e o recall
    (TPR) e a marcação independem da prevalência. Detector com FPR=0 mantém
    precisão alta a qualquer prevalência."""
    from src.ml.experimentos_artigos import _metricas_anomalia

    # 10 normais, 10 anomalias; y_pred com 8 TP, 2 FN, 3 FP, 7 TN -> FPR=0.3
    y = [1]*10 + [0]*10
    yp = [1]*8 + [0]*2 + [1]*3 + [0]*7
    s = [0.9]*10 + [0.1]*10
    m = _metricas_anomalia(y, s, y_pred=yp, threshold_source="x", limiar=0.5)
    assert m["prevalencia_raro"] == 0.05
    assert m["fpr_operacao"] == pytest.approx(0.3)
    # a 50% precision ~ 8/11=0.73; a 5% deve ser MUITO menor (FPR alto)
    assert m["precision_raro"] < m["precision"]
    assert m["precision_raro"] < 0.2

    # FPR=0 -> precision_raro permanece 1.0 (independe da prevalência)
    yp0 = [1]*8 + [0]*2 + [0]*10  # 0 FP
    m0 = _metricas_anomalia(y, s, y_pred=yp0, threshold_source="x", limiar=0.5)
    assert m0["fpr_operacao"] == pytest.approx(0.0)
    assert m0["precision_raro"] == pytest.approx(1.0)


# ── curadoria: Ahirwar/Stender não são mais protocolos executáveis ───────────

def test_ahirwar_stender_nao_estao_no_dispatch():
    """Cortados do núcleo comparativo; restam só Francisti e Ibrahim."""
    assert set(P.PROTOCOLOS) == {"francisti", "ibrahim"}
    assert not hasattr(P, "protocolo_ahirwar")
    assert P.executar_protocolo("ahirwar") is None
    assert P.executar_protocolo("stender") is None


# ── robustez: um modelo quebrado não derruba os demais ───────────────────────

def test_rodar_modelo_isola_falha_de_runtime():
    """Modelo que estoura em runtime vira indisponível; os outros seguem."""
    saida, preds = {}, {}

    def ok():
        return {"disponivel": True, "auc": 0.7}, "y_pred_fake"

    def quebra():
        raise AttributeError("'Prophet' object has no attribute 'stan_backend'")

    P._rodar_modelo("Isolation Forest", ok, saida, preds)
    P._rodar_modelo("Facebook Prophet", quebra, saida, preds)

    assert saida["Isolation Forest"]["disponivel"] is True
    assert preds["Isolation Forest"] == "y_pred_fake"
    assert saida["Facebook Prophet"]["disponivel"] is False
    assert "stan_backend" in saida["Facebook Prophet"]["motivo"]
    assert "Facebook Prophet" not in preds  # sem predição = fora do voto/ensemble
