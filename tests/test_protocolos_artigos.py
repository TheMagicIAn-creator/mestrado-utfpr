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
- o voto do Ahirwar é majoritário sobre as decisões dos membros;
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
    intocaveis = [NOMES.index("i_a_kurtosis"), NOMES.index("u_a_rms"),
                  NOMES.index("tensao_dc_media"), NOMES.index("i_b_rms")]
    assert np.allclose(X_anom[:, intocaveis], X[:, intocaveis])

    # numa janela LCL: harmônicos sobem, rms da fase A fica intacto
    idx_lcl = np.where(tipos == "lcl")[0]
    assert len(idx_lcl) > 0
    j_h5 = NOMES.index("i_a_harm_5")
    j_rms = NOMES.index("i_a_rms")
    assert np.all(X_anom[idx_lcl, j_h5] > X[idx_lcl, j_h5])
    assert np.allclose(X_anom[idx_lcl, j_rms], X[idx_lcl, j_rms])

    # numa janela de desbalanceamento: rms da fase A CAI
    idx_db = np.where(tipos == "desbalanceamento")[0]
    assert len(idx_db) > 0
    assert np.all(X_anom[idx_db, j_rms] < X[idx_db, j_rms])


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

    rf = saida["Random Forest (anomalia)"]
    assert rf["threshold_source"] == "probabilidade_nativa_0.5"

    assert met["protocolo"] == "francisti2025_spc_rf"
    assert "fidelidade" in met


def test_executar_protocolo_desconhecido_retorna_none():
    assert P.executar_protocolo("artigo_inexistente") is None


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


# ── voto majoritário do Ahirwar ──────────────────────────────────────────────

def test_ahirwar_voto_majoritario(features_fake, monkeypatch):
    dados = P.preparar_dados_anomalia()
    y_te = dados["y_te"]

    # membros fake: 2 acertam tudo, 1 erra tudo → maioria (2/3) acerta tudo.
    # O voto recebe as MESMAS predições devolvidas pelo protocolo do Ibrahim
    # (retornar_predicoes=True) — sem refazer fits.
    perfeito = np.asarray(y_te).astype(int)
    invertido = 1 - perfeito
    fake_preds = {
        "Isolation Forest": perfeito,
        "AE-LSTM": perfeito,
        "Facebook Prophet": invertido,
    }
    base = {"anomalias_detectadas": 1}  # marca membro como disponível

    def fake_ibrahim(dados, progresso=None, retornar_predicoes=False):
        saida = {"Isolation Forest": dict(base), "AE-LSTM": dict(base),
                 "Facebook Prophet": dict(base)}
        if retornar_predicoes:
            return saida, {}, fake_preds
        return saida, {}

    monkeypatch.setattr(P, "protocolo_ibrahim", fake_ibrahim)

    saida, met = P.protocolo_ahirwar(dados)
    h = saida["Híbrido (voto)"]
    assert h["threshold_source"] == "voto_majoritario_2_de_3"
    assert h["recall"] == pytest.approx(1.0)
    assert h["precision"] == pytest.approx(1.0)
    assert 0.0 < h["concordancia_media_membros"] < 1.0
    assert met["protocolo"] == "ahirwar2025_voto_hibrido"
