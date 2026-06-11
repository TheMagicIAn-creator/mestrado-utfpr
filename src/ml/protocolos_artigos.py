"""
protocolos_artigos.py — Al IAdo PV / protocolos de avaliação POR ARTIGO.

Por que este módulo existe
--------------------------
Antes, todos os experimentos de anomalia compartilhavam UM único harness:
split aleatório de janelas temporais sobrepostas (vazamento temporal) e limiar
escolhido maximizando F1 NO PRÓPRIO conjunto de teste (oráculo) para os
modelos sem decisão nativa — exatamente os modelos que definem cada artigo
(Z-score, AE-LSTM, Prophet). Isso é o "erro de simulação": todos os métodos
pareciam iguais porque eram avaliados pela mesma régua artificial.

Aqui cada artigo tem o SEU protocolo de decisão, fiel ao método do paper e à
prática da área — e nenhum limiar enxerga os rótulos do teste:

- Francisti et al. (2025)  → Z-score com regra de Shewhart (|z| > 3σ FIXO, a
  priori, por variável) + Random Forest supervisionado (probabilidade ≥ 0,5).
- Ibrahim et al. (2022)    → Isolation Forest com contaminação A PRIORI;
  AE-LSTM com limiar = percentil 99 do erro de reconstrução NO TREINO
  (congelado antes de ver o teste — a mesma disciplina do pipeline principal);
  Prophet com banda de incerteza de 99% (fora da banda = anomalia).
- Sharma et al. (2026)     → PPO ajusta a contaminação do Isolation Forest em
  VALIDAÇÃO temporal separada; o teste só é tocado com o parâmetro congelado.
  Baselines (KNN/SVM/ANN/RNN/CNN) com decisão nativa 0,5.
- Ahirwar & Nandanwar (2025) → voto MAJORITÁRIO entre membros (IF, AE-LSTM,
  Prophet), cada um decidindo pela SUA regra a priori — não uma média de
  scores normalizados.

Infraestrutura comum (igual para todos, como num benchmark justo):
- split TEMPORAL com purga (src/ml/split_temporal.py) — nunca aleatório;
- injeção sintética ORIENTADA PELO FMEA no espaço de features: cada anomalia
  pertence a uma família de falha do FMECA de Torres (2024) — degradação LCL
  (NPR=210), desbalanceamento de fase (NPR=150), falha de sensor — perturbando
  apenas as features que a física daquela falha afeta. Continua E1 (proxy
  sintético em espaço de features), mas com ground truth fisicamente motivado
  e relatório de detecção POR FALHA.

O AUC permanece comparável entre protocolos (independente de limiar); as
métricas de decisão (F1/recall/precisão) passam a refletir a política REAL de
cada método.
"""

from __future__ import annotations

from src.core.logs import get_logger

log = get_logger("protocolos_artigos")

# ── constantes dos protocolos (a priori, documentadas) ──────────────────────
SEVERIDADE_PADRAO = 1.0          # escala global da injeção (1.0 = moderada)
LIMIAR_SIGMA = 3.0               # regra de Shewhart (Francisti)
CONTAMINACAO_A_PRIORI = 0.05     # Isolation Forest (Ibrahim/Ahirwar)
PERCENTIL_TREINO = 99            # AE-LSTM: limiar congelado no treino
INTERVALO_PROPHET = 0.99         # banda de incerteza do Prophet
PURGA_JANELAS = 2                # janelas com 50% de sobreposição → purga 2

# Pesos de amostragem das famílias de falha (ordem de criticidade do FMECA:
# NPR 210 > NPR 150 > sensor sem NPR mas D=10).
PESOS_FALHAS = {"lcl": 0.40, "desbalanceamento": 0.35, "sensor": 0.25}

# Assinaturas FMEA no ESPAÇO DE FEATURES (features_ca.py):
# cada item: (padrão regex do nome da coluna, modo, intensidade min, max).
# modo "soma_std"  → coluna += U(min,max) · severidade · σ_treino
# modo "mult"      → coluna ·= (1 − U(min,max) · severidade)  [redução]
ASSINATURAS_FMEA = {
    "lcl": [
        # harmônicos 5/7/11 das correntes ↑ (filtro atenua menos) + THD ↑
        (r"^i_[abc]_harm_5$", "soma_std", 1.5, 3.0),
        (r"^i_[abc]_harm_7$", "soma_std", 1.0, 2.0),
        (r"^i_[abc]_harm_11$", "soma_std", 1.5, 3.0),
        (r"^i_[abc]_thd$", "soma_std", 1.0, 2.5),
        (r"^i_[abc]_energia_media$", "soma_std", 0.8, 1.5),
    ],
    "desbalanceamento": [
        # amplitude da fase A ↓ → rms/pico/desvio/potência da fase A caem,
        # desbalanceamento entre correntes sobe
        (r"^i_a_rms$", "mult", 0.15, 0.35),
        (r"^i_a_pico_a_pico$", "mult", 0.15, 0.35),
        (r"^i_a_desvio$", "mult", 0.15, 0.35),
        (r"^potencia_a$", "mult", 0.15, 0.35),
        (r"^desbalanceamento_corrente$", "soma_std", 2.0, 4.0),
    ],
    "sensor": [
        # ruído de medição na fase A → dispersão e conteúdo de alta
        # frequência sobem no canal medido
        (r"^i_a_desvio$", "soma_std", 0.8, 1.5),
        (r"^i_a_largura_banda$", "soma_std", 1.0, 2.0),
        (r"^i_a_energia_chaveamento$", "soma_std", 1.0, 2.5),
        (r"^i_a_centroide$", "soma_std", 0.8, 1.5),
        (r"^i_a_thd$", "soma_std", 0.5, 1.0),
    ],
}


# ============================================================
# INJEÇÃO ORIENTADA PELO FMEA (espaço de features)
# ============================================================

def _colunas_por_padrao(nomes: list[str], padrao: str) -> list[int]:
    import re

    rx = re.compile(padrao)
    return [j for j, n in enumerate(nomes) if rx.match(n)]


def injetar_falhas_fmea(X, nomes: list[str], rng, severidade: float = SEVERIDADE_PADRAO):
    """
    Gera uma cópia anômala de cada janela de ``X``, sorteando UMA família de
    falha do FMEA por janela e perturbando SOMENTE as features que a física
    daquela falha afeta (em unidades do desvio-padrão do próprio conjunto).

    Retorna ``(X_anom, tipos)`` onde ``tipos[i]`` ∈ {"lcl",
    "desbalanceamento", "sensor"} é o ground truth da família injetada.

    E1/proxy: as intensidades são plausíveis (fundamentadas nas fórmulas de
    src/ml/injecao_falhas.py, que opera no sinal bruto), não medidas em
    bancada. A vantagem sobre perturbação genérica: a assinatura de cada
    anomalia é FISICAMENTE coerente e a detecção pode ser reportada POR FALHA.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    std = X.std(axis=0) + 1e-9
    n = len(X)

    familias = list(PESOS_FALHAS)
    pesos = np.array([PESOS_FALHAS[f] for f in familias], dtype=float)
    pesos = pesos / pesos.sum()

    # mapeia uma vez: família → [(cols, modo, lo, hi), ...]
    planos = {}
    for fam, regras in ASSINATURAS_FMEA.items():
        plano = []
        for padrao, modo, lo, hi in regras:
            cols = _colunas_por_padrao(nomes, padrao)
            if cols:
                plano.append((np.array(cols), modo, lo, hi))
        planos[fam] = plano

    X_anom = X.copy()
    tipos = []
    for i in range(n):
        fam = familias[int(rng.choice(len(familias), p=pesos))]
        tipos.append(fam)
        for cols, modo, lo, hi in planos[fam]:
            escala = rng.uniform(lo, hi, size=len(cols)) * severidade
            if modo == "mult":
                X_anom[i, cols] *= np.clip(1.0 - escala, 0.05, 1.0)
            else:  # soma_std
                X_anom[i, cols] += escala * std[cols]
    return X_anom, np.array(tipos)


def deteccao_por_falha(y_true, y_pred, tipos) -> dict:
    """Recall por família de falha FMEA (apenas nas janelas anômalas)."""
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tipos = np.asarray(tipos)
    saida = {}
    for fam in PESOS_FALHAS:
        mask = (y_true == 1) & (tipos == fam)
        if mask.sum() == 0:
            continue
        saida[fam] = float(y_pred[mask].mean())
    return saida


# ============================================================
# PREPARO DOS DADOS — split temporal + injeção FMEA
# ============================================================

def preparar_dados_anomalia(com_validacao: bool = False,
                            severidade: float = SEVERIDADE_PADRAO,
                            seed: int = 42,
                            progresso=None) -> dict:
    """
    Pacote de dados comum aos protocolos:

    - janelas do Paderborn em ordem TEMPORAL, divididas em blocos contíguos
      com purga (treino/teste, ou treino/val/teste p/ Sharma);
    - StandardScaler ajustado SOMENTE no treino normal;
    - anomalias FMEA injetadas em cópias das janelas de teste (e validação),
      com ground truth por família (tipos);
    - pacote supervisionado de treino (normal + anomalias FMEA de treino).

    No conjunto de teste, ``tipos_te`` alinha com a metade anômala
    (X_te = [normais | anômalas]).
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from src.ml.experimentos_artigos import _carregar_features_paderborn
    from src.ml.split_temporal import split_temporal_com_purga, split_treino_val

    rng = np.random.default_rng(seed)
    if progresso:
        progresso("Carregando features de normalidade (Paderborn)...")
    X, nomes = _carregar_features_paderborn(progresso)
    n = len(X)

    if com_validacao:
        sp = split_temporal_com_purga(n, 0.6, 0.2, 0.2, purge_janelas=PURGA_JANELAS)
        idx_tr, idx_val, idx_te = sp["treino"], sp["val"], sp["teste"]
    else:
        idx_tr, idx_te = split_treino_val(n, val_frac=0.4,
                                          purge_janelas=PURGA_JANELAS)
        idx_val = None

    Xn_tr = X[idx_tr]
    Xn_te = X[idx_te]

    # Injeção FMEA — sementes derivadas para independência treino/val/teste.
    Xa_tr, tipos_tr = injetar_falhas_fmea(
        Xn_tr, nomes, np.random.default_rng(seed + 1), severidade)
    Xa_te, tipos_te = injetar_falhas_fmea(
        Xn_te, nomes, np.random.default_rng(seed + 2), severidade)

    scaler = StandardScaler().fit(Xn_tr)

    # Variável monitorada pelo Prophet: a feature MAIS INFORMATIVA entre as
    # que as famílias FMEA afetam (maior coeficiente de variação no treino
    # BRUTO). Equivale a monitorar a série de processo sensível à falha —
    # proxy da série de potência usada no artigo do Ibrahim. Sem isso, o
    # Prophet vigiaria uma feature intocada pelas falhas (cego por projeto).
    cols_afetadas = sorted({
        j for regras in ASSINATURAS_FMEA.values()
        for padrao, _, _, _ in regras
        for j in _colunas_por_padrao(nomes, padrao)
    })
    if cols_afetadas:
        sub = Xn_tr[:, cols_afetadas]
        cv = sub.std(axis=0) / (np.abs(sub.mean(axis=0)) + 1e-9)
        col_prophet = int(cols_afetadas[int(np.argmax(cv))])
    else:
        col_prophet = None

    dados = {
        "nomes": nomes,
        "scaler": scaler,
        "col_prophet": col_prophet,
        "col_prophet_nome": nomes[col_prophet] if col_prophet is not None else None,
        "Xn_tr": scaler.transform(Xn_tr),
        "X_tr_sup": np.vstack([scaler.transform(Xn_tr), scaler.transform(Xa_tr)]),
        "y_tr_sup": np.r_[np.zeros(len(Xn_tr)), np.ones(len(Xa_tr))],
        "X_te": np.vstack([scaler.transform(Xn_te), scaler.transform(Xa_te)]),
        "y_te": np.r_[np.zeros(len(Xn_te)), np.ones(len(Xa_te))],
        "tipos_te": tipos_te,
        "split": {
            "tipo": "temporal_com_purga",
            "purga_janelas": PURGA_JANELAS,
            "n_janelas": int(n),
            "treino": int(len(idx_tr)),
            "teste": int(len(idx_te)),
        },
        "injecao": {
            "tipo": "fmea_espaco_features",
            "falhas": list(PESOS_FALHAS),
            "pesos": dict(PESOS_FALHAS),
            "severidade": float(severidade),
            "fonte": "Torres (2024) — FMECA CEAMAZON (NPR 210/150) via "
                     "assinaturas de src/ml/injecao_falhas.py",
            "nota": "E1 — proxy sintético no espaço de features; ground truth "
                    "por família de falha, não medição de bancada.",
        },
    }

    if idx_val is not None:
        Xn_val = X[idx_val]
        Xa_val, tipos_val = injetar_falhas_fmea(
            Xn_val, nomes, np.random.default_rng(seed + 3), severidade)
        dados["X_val"] = np.vstack(
            [scaler.transform(Xn_val), scaler.transform(Xa_val)])
        dados["y_val"] = np.r_[np.zeros(len(Xn_val)), np.ones(len(Xa_val))]
        dados["tipos_val"] = tipos_val
        dados["split"]["val"] = int(len(idx_val))

    return dados


# ============================================================
# AUXILIARES DE MÉTRICA (limiar nunca vê o y do teste)
# ============================================================

def _metricas(y_te, score, y_pred, threshold_source: str, limiar=None,
              tipos_te=None) -> dict:
    """Métricas no ponto de operação REAL do protocolo + recall por falha."""
    from src.ml.experimentos_artigos import _metricas_anomalia

    m = _metricas_anomalia(
        y_te, score, y_pred=y_pred,
        threshold_source=threshold_source, limiar=limiar,
    )
    if tipos_te is not None and y_pred is not None:
        import numpy as np

        n_norm = int(len(y_te) - len(tipos_te))
        tipos_full = np.r_[np.array(["normal"] * n_norm), tipos_te]
        m["deteccao_por_falha"] = deteccao_por_falha(y_te, y_pred, tipos_full)
    return m


def _indisponivel(motivo: str) -> dict:
    return {"disponivel": False, "motivo": motivo}


# ============================================================
# PROTOCOLO — FRANCISTI et al. (2025)
# ============================================================

def protocolo_francisti(dados, progresso=None):
    """
    Z-score com regra de Shewhart: alarme se QUALQUER feature sai da banda de
    ±3σ do comportamento saudável de treino (controle estatístico de processo
    por variável — limiar FIXO a priori, nunca ajustado no teste).
    Random Forest supervisionado decide pela probabilidade nativa ≥ 0,5.
    """
    import numpy as np

    X_te, y_te = dados["X_te"], dados["y_te"]
    tipos = dados["tipos_te"]
    saida = {}

    if progresso:
        progresso("Francisti: Z-score (Shewhart 3σ)...")
    # Xn_tr/X_te já são z-scores (scaler do treino) → |z| direto.
    score_z = np.max(np.abs(X_te), axis=1)
    y_pred_z = (score_z > LIMIAR_SIGMA).astype(int)
    saida["Z-score (estatístico)"] = _metricas(
        y_te, score_z, y_pred_z,
        threshold_source="shewhart_3sigma_a_priori",
        limiar=LIMIAR_SIGMA, tipos_te=tipos,
    )

    if progresso:
        progresso("Francisti: Random Forest (anomalia)...")
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(dados["X_tr_sup"], dados["y_tr_sup"])
    score_rf = rf.predict_proba(X_te)[:, 1]
    y_pred_rf = (score_rf >= 0.5).astype(int)
    saida["Random Forest (anomalia)"] = _metricas(
        y_te, score_rf, y_pred_rf,
        threshold_source="probabilidade_nativa_0.5",
        limiar=0.5, tipos_te=tipos,
    )

    metodologia = {
        "protocolo": "francisti2025_spc_rf",
        "fonte": "Francisti et al. (2025)",
        "decisoes": {
            "Z-score (estatístico)": f"|z| > {LIMIAR_SIGMA}σ por variável "
                                     "(Shewhart, fixo a priori)",
            "Random Forest (anomalia)": "probabilidade nativa ≥ 0,5",
        },
        "fidelidade": [
            "Segue o artigo: controle estatístico Z-score + RF supervisionado.",
            "Adaptação: alarme por variável (máx |z|) — prática padrão de SPC "
            "multivariável simples; o artigo não detalha a agregação.",
        ],
    }
    return saida, metodologia


# ============================================================
# PROTOCOLO — IBRAHIM et al. (2022)
# ============================================================

def protocolo_ibrahim(dados, progresso=None):
    """
    IF com contaminação A PRIORI; AE-LSTM com limiar p99 do erro de
    reconstrução NO TREINO (congelado); Prophet com banda de 99% (fora da
    banda = anomalia). Nenhuma decisão enxerga os rótulos do teste.
    """
    import numpy as np

    from src.ml.experimentos_artigos import lib_disponivel

    X_te, y_te = dados["X_te"], dados["y_te"]
    tipos = dados["tipos_te"]
    saida = {}

    if progresso:
        progresso("Ibrahim: Isolation Forest (contaminação a priori)...")
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(n_estimators=200, random_state=42,
                          contamination=CONTAMINACAO_A_PRIORI)
    iso.fit(dados["Xn_tr"])
    score_if = -iso.decision_function(X_te)
    y_pred_if = (iso.predict(X_te) == -1).astype(int)
    saida["Isolation Forest"] = _metricas(
        y_te, score_if, y_pred_if,
        threshold_source=f"contaminacao_a_priori_{CONTAMINACAO_A_PRIORI}",
        tipos_te=tipos,
    )

    if lib_disponivel("torch"):
        if progresso:
            progresso("Ibrahim: AE-LSTM (limiar p99 do treino)...")
        from src.ml.experimentos_artigos import _score_ae_lstm

        score_te, score_tr = _score_ae_lstm(dados, retornar_treino=True)
        limiar = float(np.percentile(score_tr, PERCENTIL_TREINO))
        y_pred_ae = (score_te > limiar).astype(int)
        saida["AE-LSTM"] = _metricas(
            y_te, score_te, y_pred_ae,
            threshold_source=f"p{PERCENTIL_TREINO}_erro_reconstrucao_treino",
            limiar=limiar, tipos_te=tipos,
        )
    else:
        saida["AE-LSTM"] = _indisponivel("requer torch")

    if lib_disponivel("prophet"):
        if progresso:
            progresso("Ibrahim: Prophet (banda de 99%)...")
        from src.ml.experimentos_artigos import _score_prophet

        score_p = _score_prophet(dados, interval_width=INTERVALO_PROPHET)
        y_pred_p = (score_p > 1.0).astype(int)  # >1 = fora da banda
        saida["Facebook Prophet"] = _metricas(
            y_te, score_p, y_pred_p,
            threshold_source=f"intervalo_prophet_{INTERVALO_PROPHET}",
            limiar=1.0, tipos_te=tipos,
        )
    else:
        saida["Facebook Prophet"] = _indisponivel("requer prophet")

    metodologia = {
        "protocolo": "ibrahim2022_series_temporais",
        "fonte": "Ibrahim et al. (2022)",
        "decisoes": {
            "Isolation Forest": f"contaminação a priori = {CONTAMINACAO_A_PRIORI}",
            "AE-LSTM": f"limiar = p{PERCENTIL_TREINO} do erro de reconstrução "
                       "no TREINO (congelado antes do teste)",
            "Facebook Prophet": f"fora da banda de incerteza de "
                                f"{INTERVALO_PROPHET:.0%}",
        },
        "fidelidade": [
            "Segue o artigo: trio IF / AE-LSTM / Prophet para anomalia.",
            "AE-LSTM usa a MESMA disciplina de limiar congelado do pipeline "
            "principal da dissertação (percentil no saudável de treino).",
            "Adaptação: Prophet univariado monitorando a feature mais "
            "sensível às famílias FMEA (proxy da série de potência do artigo).",
            "Leitura correta: a contaminação a priori de 5% reflete a "
            "prevalência esperada em operação; no teste BALANCEADO (50% "
            "anômalo) ela limita o recall por construção — compare métodos "
            "pelo AUC e pela precisão, não pelo F1 entre protocolos.",
        ],
    }
    if dados.get("col_prophet_nome"):
        metodologia["decisoes"]["Facebook Prophet"] += (
            f" — monitora '{dados['col_prophet_nome']}'")
    return saida, metodologia


# ============================================================
# PROTOCOLO — SHARMA et al. (2026)
# ============================================================

def protocolo_sharma(dados, progresso=None):
    """
    PPO ajusta a contaminação do Isolation Forest maximizando F1 em VALIDAÇÃO
    temporal separada; o teste só é tocado com o parâmetro congelado.
    Baselines com decisão nativa (0,5) — sem busca de limiar em lugar nenhum.
    """
    import numpy as np

    from src.ml.experimentos_artigos import lib_disponivel

    X_te, y_te = dados["X_te"], dados["y_te"]
    tipos = dados["tipos_te"]
    saida = {}

    # — IF base (mesma regra a priori do Ibrahim, para comparação justa) —
    if progresso:
        progresso("Sharma: Isolation Forest base...")
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(n_estimators=200, random_state=42,
                          contamination=CONTAMINACAO_A_PRIORI)
    iso.fit(dados["Xn_tr"])
    saida["Isolation Forest"] = _metricas(
        y_te, -iso.decision_function(X_te),
        (iso.predict(X_te) == -1).astype(int),
        threshold_source=f"contaminacao_a_priori_{CONTAMINACAO_A_PRIORI}",
        tipos_te=tipos,
    )

    # — supervisionados clássicos (decisão nativa 0,5) —
    def _sup(nome, est, rotulo):
        if progresso:
            progresso(f"Sharma: {nome}...")
        est.fit(dados["X_tr_sup"], dados["y_tr_sup"])
        sc = est.predict_proba(X_te)[:, 1]
        saida[rotulo] = _metricas(
            y_te, sc, (sc >= 0.5).astype(int),
            threshold_source="probabilidade_nativa_0.5",
            limiar=0.5, tipos_te=tipos,
        )

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    _sup("KNN", KNeighborsClassifier(n_neighbors=15), "KNN")
    _sup("SVM", SVC(kernel="rbf", probability=True, random_state=42), "SVM")
    _sup("ANN (MLP)", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                    random_state=42), "ANN (MLP)")

    if lib_disponivel("torch"):
        from src.ml.experimentos_artigos import _score_cnn_torch, _score_rnn_torch

        if progresso:
            progresso("Sharma: RNN...")
        sc = _score_rnn_torch(dados)
        saida["RNN"] = _metricas(
            y_te, sc, (sc >= 0.5).astype(int),
            threshold_source="probabilidade_nativa_0.5", limiar=0.5,
            tipos_te=tipos)
        if progresso:
            progresso("Sharma: CNN...")
        sc = _score_cnn_torch(dados)
        saida["CNN"] = _metricas(
            y_te, sc, (sc >= 0.5).astype(int),
            threshold_source="probabilidade_nativa_0.5", limiar=0.5,
            tipos_te=tipos)
    else:
        saida["RNN"] = _indisponivel("requer torch")
        saida["CNN"] = _indisponivel("requer torch")

    # — IF + PPO: ajuste em validação temporal, teste congelado —
    if lib_disponivel("stable_baselines3") and "X_val" in dados:
        if progresso:
            progresso("Sharma: PPO ajustando contaminação em validação...")
        from src.ml.experimentos_artigos import _ppo_buscar_contaminacao

        melhor_cont = _ppo_buscar_contaminacao(
            dados["Xn_tr"], dados["X_val"], dados["y_val"], metrica="f1")
        iso_ppo = IsolationForest(n_estimators=200, random_state=42,
                                  contamination=melhor_cont)
        iso_ppo.fit(dados["Xn_tr"])
        m = _metricas(
            y_te, -iso_ppo.decision_function(X_te),
            (iso_ppo.predict(X_te) == -1).astype(int),
            threshold_source="ppo_otimizado_em_validacao_temporal",
            limiar=float(melhor_cont), tipos_te=tipos,
        )
        m["contaminacao_ppo"] = float(melhor_cont)
        saida["Isolation Forest + PPO"] = m
    elif "X_val" not in dados:
        saida["Isolation Forest + PPO"] = _indisponivel(
            "protocolo exige validação temporal (com_validacao=True)")
    else:
        saida["Isolation Forest + PPO"] = _indisponivel("requer stable_baselines3")

    metodologia = {
        "protocolo": "sharma2026_rl_self_tuning",
        "fonte": "Sharma et al. (2026)",
        "decisoes": {
            "Isolation Forest + PPO": "contaminação otimizada por PPO em "
                                      "VALIDAÇÃO temporal (F1); teste com "
                                      "parâmetro congelado",
            "baselines": "probabilidade nativa ≥ 0,5 (KNN/SVM/ANN/RNN/CNN); "
                         "IF base com contaminação a priori",
        },
        "fidelidade": [
            "Segue o artigo: IF auto-ajustável por RL contra baselines.",
            "Adaptação: ambiente PPO de 1 passo (bandit) — o ajuste é de um "
            "hiperparâmetro contínuo, não um MDP sequencial; documentado.",
            "Split 60/20/20 temporal com purga: o teste nunca participa do "
            "ajuste.",
        ],
    }
    return saida, metodologia


# ============================================================
# PROTOCOLO — AHIRWAR & NANDANWAR (2025)
# ============================================================

def protocolo_ahirwar(dados, progresso=None):
    """
    Híbrido por VOTO MAJORITÁRIO: cada membro (IF, AE-LSTM, Prophet) decide
    pela SUA regra a priori e o ensemble rotula anomalia quando a maioria dos
    membros disponíveis concorda. Fiel à ideia central do artigo (combinação
    de detectores heterogêneos), sem média artificial de scores.
    """
    import numpy as np

    # Reusa o protocolo do Ibrahim para os MEMBROS (mesmas regras a priori).
    saida, _ = protocolo_ibrahim(dados, progresso=progresso)

    y_te = dados["y_te"]
    tipos = dados["tipos_te"]

    membros = {}
    for nome in ("Isolation Forest", "AE-LSTM", "Facebook Prophet"):
        m = saida.get(nome, {})
        if m.get("disponivel", True) and "anomalias_detectadas" in m:
            membros[nome] = m

    if len(membros) >= 2:
        if progresso:
            progresso("Ahirwar: voto majoritário dos membros...")
        # Reconstrói os y_pred dos membros pelas mesmas regras (refazer é
        # barato e evita carregar vetores nos dicts de métricas).
        preds = _predicoes_membros(dados, list(membros))
        matriz = np.vstack(list(preds.values()))
        votos = matriz.sum(axis=0)
        maioria = len(preds) // 2 + 1
        y_pred_h = (votos >= maioria).astype(int)
        score_h = votos / len(preds)

        m_h = _metricas(
            y_te, score_h, y_pred_h,
            threshold_source=f"voto_majoritario_{maioria}_de_{len(preds)}",
            tipos_te=tipos,
        )
        # concordância média entre pares de membros (estatística do ensemble)
        n_m = len(preds)
        if n_m > 1:
            pares = [
                float((matriz[a] == matriz[b]).mean())
                for a in range(n_m) for b in range(a + 1, n_m)
            ]
            m_h["concordancia_media_membros"] = float(np.mean(pares))
        m_h["membros"] = list(preds)
        saida["Híbrido (voto)"] = m_h
    else:
        saida["Híbrido (voto)"] = _indisponivel(
            "precisa de ≥2 membros disponíveis (IF/AE-LSTM/Prophet)")

    metodologia = {
        "protocolo": "ahirwar2025_voto_hibrido",
        "fonte": "Ahirwar & Nandanwar (2025)",
        "decisoes": {
            "membros": "cada um pela própria regra a priori (IF contaminação; "
                       f"AE-LSTM p{PERCENTIL_TREINO} do treino; Prophet banda "
                       f"{INTERVALO_PROPHET:.0%})",
            "Híbrido (voto)": "anomalia quando a MAIORIA dos membros "
                              "disponíveis vota anomalia",
        },
        "fidelidade": [
            "Segue o artigo: ensemble heterogêneo AE-LSTM + Prophet + IF.",
            "Adaptação: voto majoritário simples no lugar da otimização "
            "bayesiana de hiperparâmetros (documentado como simplificação).",
            "Membros em pontos de operação conservadores tornam o voto "
            "majoritário ainda mais conservador (recall baixo por construção) "
            "— achado metodológico a discutir, não defeito de implementação.",
        ],
    }
    if dados.get("col_prophet_nome"):
        metodologia["decisoes"]["membros"] += (
            f"; Prophet monitora '{dados['col_prophet_nome']}'")
    return saida, metodologia


def _predicoes_membros(dados, nomes_membros):
    """Recalcula o y_pred de cada membro pela mesma regra a priori."""
    import numpy as np

    preds = {}
    X_te = dados["X_te"]

    if "Isolation Forest" in nomes_membros:
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(n_estimators=200, random_state=42,
                              contamination=CONTAMINACAO_A_PRIORI)
        iso.fit(dados["Xn_tr"])
        preds["Isolation Forest"] = (iso.predict(X_te) == -1).astype(int)

    if "AE-LSTM" in nomes_membros:
        from src.ml.experimentos_artigos import _score_ae_lstm

        score_te, score_tr = _score_ae_lstm(dados, retornar_treino=True)
        limiar = float(np.percentile(score_tr, PERCENTIL_TREINO))
        preds["AE-LSTM"] = (score_te > limiar).astype(int)

    if "Facebook Prophet" in nomes_membros:
        from src.ml.experimentos_artigos import _score_prophet

        score_p = _score_prophet(dados, interval_width=INTERVALO_PROPHET)
        preds["Facebook Prophet"] = (score_p > 1.0).astype(int)

    return preds


# ============================================================
# DISPATCH
# ============================================================

# key → (função de protocolo, exige validação temporal?)
PROTOCOLOS = {
    "francisti": (protocolo_francisti, False),
    "ibrahim": (protocolo_ibrahim, False),
    "sharma": (protocolo_sharma, True),
    "ahirwar": (protocolo_ahirwar, False),
}


def executar_protocolo(key: str, progresso=None):
    """
    Executa o protocolo do artigo ``key``. Retorna ``(modelos_out,
    metodologia)`` ou ``None`` se o artigo não tem protocolo dedicado
    (o chamador cai no harness genérico legado).
    """
    item = PROTOCOLOS.get(key)
    if item is None:
        return None
    protocolo, com_val = item
    dados = preparar_dados_anomalia(com_validacao=com_val, progresso=progresso)
    modelos_out, metodologia = protocolo(dados, progresso=progresso)
    metodologia["split"] = dados["split"]
    metodologia["injecao"] = dados["injecao"]
    return modelos_out, metodologia
