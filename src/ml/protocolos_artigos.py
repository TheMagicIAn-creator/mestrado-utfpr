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
    # HIPÓTESE (E1, não verdade universal): modelamos perda parcial/assimetria
    # na fase A em que o CONTROLE do inversor (malha de potência) redistribui
    # corrente para B/C, mantendo a potência total — cenário comum em inversores
    # com controle de tensão/corrente. NÃO cobre o caso em que A cai E B/C
    # também caem (perda de linha, carga severamente desbalanceada sem
    # compensação). A assinatura central (sempre presente) é a fase A enfraquecer
    # e a métrica de desbalanceamento subir; a compensação B/C é a parte
    # dependente do controle. Magnitudes plausíveis, não medidas em bancada.
    "desbalanceamento": [
        # Fase A ENFRAQUECE (rms, pico, desvio, energia da fundamental, potência)
        (r"^i_a_rms$", "mult", 0.15, 0.35),
        (r"^i_a_pico_a_pico$", "mult", 0.15, 0.35),
        (r"^i_a_desvio$", "mult", 0.15, 0.35),
        (r"^i_a_energia_baixa$", "mult", 0.15, 0.35),
        (r"^potencia_a$", "mult", 0.15, 0.35),
        # Fases B e C COMPENSAM parcialmente (hipótese dependente do controle)
        (r"^i_[bc]_rms$", "soma_std", 0.6, 1.2),
        (r"^i_[bc]_energia_baixa$", "soma_std", 0.6, 1.2),
        (r"^potencia_[bc]$", "soma_std", 0.6, 1.2),
        # Métrica de desbalanceamento de corrente — assinatura central.
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

    Mantém apenas o detector NÃO-supervisionado do artigo (SPC/Z-score). O
    Random Forest supervisionado foi removido na curadoria do mestrado: por
    treinar nos rótulos da injeção sintética, superestima o desempenho que se
    obteria em operação real, onde NÃO há rótulos de falha (a tese é detecção
    por modelagem de normalidade).
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

    metodologia = {
        "protocolo": "francisti2025_spc",
        "fonte": "Francisti et al. (2025)",
        "decisoes": {
            "Z-score (estatístico)": f"|z| > {LIMIAR_SIGMA}σ por variável "
                                     "(Shewhart, fixo a priori)",
        },
        "fidelidade": [
            "Segue o artigo no detector estatístico (Z-score / SPC).",
            "O Random Forest supervisionado do artigo foi removido: treina nos "
            "rótulos da injeção sintética e superestima o desempenho real "
            "(em operação não há rótulos de falha).",
            "Adaptação: alarme por variável (máx |z|) — prática padrão de SPC "
            "multivariável simples; o artigo não detalha a agregação.",
        ],
    }
    return saida, metodologia


# ============================================================
# PROTOCOLO — IBRAHIM et al. (2022)
# ============================================================

def protocolo_ibrahim(dados, progresso=None, retornar_predicoes: bool = False):
    """
    IF com contaminação A PRIORI; AE-LSTM com limiar = percentil do erro de
    reconstrução numa fatia de CALIBRAÇÃO temporal do treino (o AE não vê a
    calibração no ajuste — evita o limiar otimista do erro de treino);
    Prophet com banda de 99% (fora da banda = anomalia). Nenhuma decisão
    enxerga os rótulos do teste.

    Com ``retornar_predicoes=True`` devolve também ``{modelo: y_pred}`` para
    o ensemble do Ahirwar REUTILIZAR as mesmas decisões (sem refazer fits).
    """
    import numpy as np

    from src.ml.experimentos_artigos import lib_disponivel

    X_te, y_te = dados["X_te"], dados["y_te"]
    tipos = dados["tipos_te"]
    saida = {}
    preds: dict = {}

    if progresso:
        progresso("Ibrahim: Isolation Forest (contaminação a priori)...")
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(n_estimators=200, random_state=42,
                          contamination=CONTAMINACAO_A_PRIORI)
    iso.fit(dados["Xn_tr"])
    score_if = -iso.decision_function(X_te)
    y_pred_if = (iso.predict(X_te) == -1).astype(int)
    preds["Isolation Forest"] = y_pred_if
    saida["Isolation Forest"] = _metricas(
        y_te, score_if, y_pred_if,
        threshold_source=f"contaminacao_a_priori_{CONTAMINACAO_A_PRIORI}",
        tipos_te=tipos,
    )

    if lib_disponivel("torch"):
        if progresso:
            progresso("Ibrahim: AE-LSTM (calibração temporal do limiar)...")
        from src.ml.modelos_anomalia import _score_ae_lstm

        # Fatia de CALIBRAÇÃO: bloco final do treino normal (com purga) fica
        # FORA do ajuste do AE e fornece o erro "não visto" para o percentil.
        # Calibrar no erro de treino subestimaria o erro real (modelo decora).
        Xn_tr = dados["Xn_tr"]
        corte = max(10, int(len(Xn_tr) * 0.8))
        Xn_fit = Xn_tr[:max(1, corte - PURGA_JANELAS)]
        X_calib = Xn_tr[corte:]
        dados_ae = {**dados, "Xn_tr": Xn_fit,
                    "X_te": np.vstack([X_calib, X_te])}
        score_all = _score_ae_lstm(dados_ae)
        score_calib = score_all[:len(X_calib)]
        score_te = score_all[len(X_calib):]
        limiar = float(np.percentile(score_calib, PERCENTIL_TREINO))
        y_pred_ae = (score_te > limiar).astype(int)
        preds["AE-LSTM"] = y_pred_ae
        saida["AE-LSTM"] = _metricas(
            y_te, score_te, y_pred_ae,
            threshold_source=f"p{PERCENTIL_TREINO}_erro_em_calibracao_temporal",
            limiar=limiar, tipos_te=tipos,
        )
    else:
        saida["AE-LSTM"] = _indisponivel("requer torch")

    if lib_disponivel("prophet"):
        if progresso:
            progresso("Ibrahim: Prophet (banda de 99%)...")
        from src.ml.modelos_anomalia import _score_prophet

        score_p = _score_prophet(dados, interval_width=INTERVALO_PROPHET)
        y_pred_p = (score_p > 1.0).astype(int)  # >1 = fora da banda
        preds["Facebook Prophet"] = y_pred_p
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
            "AE-LSTM": f"limiar = p{PERCENTIL_TREINO} do erro numa fatia de "
                       "CALIBRAÇÃO temporal do treino (fora do ajuste do AE; "
                       "congelado antes do teste)",
            "Facebook Prophet": f"fora da banda de incerteza de "
                                f"{INTERVALO_PROPHET:.0%}",
        },
        "fidelidade": [
            "Segue o artigo: trio IF / AE-LSTM / Prophet para anomalia.",
            "AE-LSTM usa a disciplina de limiar congelado do pipeline "
            "principal, com calibração em bloco temporal NÃO visto no ajuste "
            "(o erro de treino subestimaria o erro real).",
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
    if retornar_predicoes:
        return saida, metodologia, preds
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

    # Reusa o protocolo do Ibrahim para os MEMBROS, recebendo as MESMAS
    # predições que geraram as métricas individuais — o voto decide sobre
    # exatamente o que foi reportado (sem refazer fits).
    saida, _met_ibrahim, preds = protocolo_ibrahim(
        dados, progresso=progresso, retornar_predicoes=True)

    y_te = dados["y_te"]
    tipos = dados["tipos_te"]

    if len(preds) >= 2:
        if progresso:
            progresso("Ahirwar: voto majoritário dos membros...")
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
                       f"AE-LSTM p{PERCENTIL_TREINO} em calibração temporal; "
                       f"Prophet banda {INTERVALO_PROPHET:.0%}) — o voto usa "
                       "as MESMAS predições das métricas individuais",
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


# ============================================================
# DISPATCH
# ============================================================

# key → (função de protocolo, exige validação temporal?)
PROTOCOLOS = {
    "francisti": (protocolo_francisti, False),
    "ibrahim": (protocolo_ibrahim, False),
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
