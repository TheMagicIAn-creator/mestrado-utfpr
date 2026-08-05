"""
protocolos_artigos.py — Al IAdo PV / protocolos de avaliação POR ARTIGO.

Por que este módulo existe
--------------------------
Antes, todos os experimentos de anomalia compartilhavam UM único harness:
split aleatório de janelas temporais sobrepostas (vazamento temporal) e limiar
escolhido maximizando F1 NO PRÓPRIO conjunto de teste (oráculo) para modelos
sem decisão nativa. Isso é o "erro de simulação": todos os métodos
pareciam iguais porque eram avaliados pela mesma régua artificial.

Aqui o único protocolo comparativo ativo é Ibrahim et al. (2022), restrito ao
AE-LSTM temporal. A comparação metodológica da dissertação é o Autoencoder denso
proposto contra o AE-LSTM do artigo, ambos avaliados no mesmo problema CA.

Os demais protocolos/modelos permanecem apenas como literatura citável; não
fundamentam a comparação quantitativa vigente do AE denso.

Infraestrutura comum (igual para todos, como num benchmark justo):
- split TEMPORAL com purga (src/ml/split_temporal.py) — nunca aleatório;
- injeção sintética ORIENTADA PELA FMECA no espaço de features: cada anomalia
  pertence a uma família de falha da FMECA de Torres (2024) — Contator AC
  (NPR=315), IGBT (NPR=90), Fusível AC (NPR=30) — perturbando
  apenas as features que a física daquela falha afeta. Continua E1 (proxy
  sintético em espaço de features), mas com ground truth fisicamente motivado
  e relatório de detecção POR FALHA.

O AUC permanece comparável entre protocolos (independente de limiar); as
métricas de decisão (F1/recall/precisão) passam a refletir a política REAL de
cada método.
"""

from __future__ import annotations

import os

from src.core.logs import get_logger

log = get_logger("protocolos_artigos")

# ── constantes do protocolo Ibrahim/AE-LSTM (a priori, documentadas) ─────────
SEVERIDADE_PADRAO = 1.0          # escala global da injeção (1.0 = moderada)
PERCENTIL_TREINO = 99            # AE-LSTM: limiar congelado no treino
PURGA_JANELAS = 2                # janelas com 50% de sobreposição → purga 2
SEQ_LEN = int(os.getenv("AL_IADO_AELSTM_SEQ_LEN", "8"))  # passos temporais do AE-LSTM

# Pesos de amostragem das famílias de falha (ordem de criticidade da FMECA —
# docs/fmeca.md: NPR Contator AC 315 > IGBT 90 > Fusível AC 30).
PESOS_FALHAS = {"contator_ac": 0.40, "igbt": 0.35, "fusivel_ac": 0.25}

# Assinaturas FMECA no ESPAÇO DE FEATURES (features_ca.py) — mesma física da
# injeção no sinal (src/ml/injecao_falhas.py), no domínio das features:
# cada item: (padrão regex do nome da coluna, modo, intensidade min, max).
# modo "soma_std"  → coluna += U(min,max) · severidade · σ_treino
# modo "mult"      → coluna ·= (1 − U(min,max) · severidade)  [redução]
ASSINATURAS_FMECA = {
    # Contator AC (NPR=315): transiente/ruído de comutação → dispersão e
    # conteúdo de alta frequência sobem no canal medido.
    "contator_ac": [
        (r"^i_a_desvio$", "soma_std", 0.8, 1.5),
        (r"^i_a_largura_banda$", "soma_std", 1.0, 2.0),
        (r"^i_a_energia_chaveamento$", "soma_std", 1.0, 2.5),
        (r"^i_a_centroide$", "soma_std", 0.8, 1.5),
        (r"^i_a_thd$", "soma_std", 0.5, 1.0),
    ],
    # IGBT (NPR=90): chaveamento imperfeito → harmônicos 5/7/11 e THD ↑.
    "igbt": [
        (r"^i_[abc]_harm_5$", "soma_std", 1.5, 3.0),
        (r"^i_[abc]_harm_7$", "soma_std", 1.0, 2.0),
        (r"^i_[abc]_harm_11$", "soma_std", 1.5, 3.0),
        (r"^i_[abc]_thd$", "soma_std", 1.0, 2.5),
        (r"^i_[abc]_energia_media$", "soma_std", 0.8, 1.5),
    ],
    # Fusível AC (NPR=30): perda parcial de fase.
    # HIPÓTESE (E1, não verdade universal): a fase A enfraquece e o CONTROLE do
    # inversor redistribui corrente para B/C, mantendo a potência total —
    # comum em inversores com controle de tensão/corrente. NÃO cobre o caso em
    # que A cai E B/C também caem. Assinatura central: fase A enfraquece e a
    # métrica de desbalanceamento sobe; a compensação B/C depende do controle.
    # Magnitudes plausíveis, não medidas em bancada.
    "fusivel_ac": [
        (r"^i_a_rms$", "mult", 0.15, 0.35),
        (r"^i_a_pico_a_pico$", "mult", 0.15, 0.35),
        (r"^i_a_desvio$", "mult", 0.15, 0.35),
        (r"^i_a_energia_baixa$", "mult", 0.15, 0.35),
        (r"^potencia_a$", "mult", 0.15, 0.35),
        (r"^i_[bc]_rms$", "soma_std", 0.6, 1.2),
        (r"^i_[bc]_energia_baixa$", "soma_std", 0.6, 1.2),
        (r"^potencia_[bc]$", "soma_std", 0.6, 1.2),
        (r"^desbalanceamento_corrente$", "soma_std", 2.0, 4.0),
    ],
}


# ============================================================
# INJEÇÃO ORIENTADA PELA FMECA (espaço de features)
# ============================================================

def _colunas_por_padrao(nomes: list[str], padrao: str) -> list[int]:
    import re

    rx = re.compile(padrao)
    return [j for j, n in enumerate(nomes) if rx.match(n)]


def injetar_falhas_fmeca(X, nomes: list[str], rng, severidade: float = SEVERIDADE_PADRAO):
    """
    Gera uma cópia anômala de cada janela de ``X``, sorteando UMA família de
    falha da FMECA por janela e perturbando SOMENTE as features que a física
    daquela falha afeta (em unidades do desvio-padrão do próprio conjunto).

    Retorna ``(X_anom, tipos)`` onde ``tipos[i]`` ∈ {"contator_ac",
    "igbt", "fusivel_ac"} é o ground truth da família injetada.

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
    for fam, regras in ASSINATURAS_FMECA.items():
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
    """Recall por família de falha FMECA (apenas nas janelas anômalas)."""
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
# PREPARO DOS DADOS — split temporal + injeção FMECA
# ============================================================

def preparar_dados_anomalia(com_validacao: bool = False,
                            severidade: float = SEVERIDADE_PADRAO,
                            seed: int = 42,
                            progresso=None) -> dict:
    """
    Pacote de dados comum aos protocolos:

    - janelas do Paderborn em ordem TEMPORAL, divididas em blocos contíguos
      com purga (treino/teste; o split treino/val/teste é suportado mas
      nenhum protocolo ativo o utiliza);
    - StandardScaler ajustado SOMENTE no treino normal;
    - anomalias FMECA injetadas em cópias das janelas de teste (e validação),
      com ground truth por família (tipos);
    - pacote supervisionado de treino (normal + anomalias FMECA de treino).

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

    # Injeção FMECA — sementes derivadas para independência treino/val/teste.
    Xa_tr, tipos_tr = injetar_falhas_fmeca(
        Xn_tr, nomes, np.random.default_rng(seed + 1), severidade)
    Xa_te, tipos_te = injetar_falhas_fmeca(
        Xn_te, nomes, np.random.default_rng(seed + 2), severidade)

    scaler = StandardScaler().fit(Xn_tr)

    dados = {
        "nomes": nomes,
        "scaler": scaler,
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
            "tipo": "fmeca_espaco_features",
            "falhas": list(PESOS_FALHAS),
            "pesos": dict(PESOS_FALHAS),
            "severidade": float(severidade),
            "fonte": "Torres (2024) — FMECA consolidada (docs/fmeca.md): "
                     "Contator AC/IGBT/Fusível AC, via assinaturas de "
                     "src/ml/injecao_falhas.py",
            "nota": "E1 — proxy sintético no espaço de features; ground truth "
                    "por família de falha, não medição de bancada.",
        },
    }

    if idx_val is not None:
        Xn_val = X[idx_val]
        Xa_val, tipos_val = injetar_falhas_fmeca(
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


def _rodar_modelo(nome: str, fn, saida: dict, preds: dict | None = None):
    """
    Executa o scoring de UM modelo com isolamento de falha. Se ``fn`` estourar
    em runtime (lib instalada mas quebrada — ex.: torch sem backend), o modelo
    degrada para indisponível com o motivo do erro, sem
    derrubar os demais modelos do experimento. ``fn`` deve devolver
    ``(metricas, y_pred | None)``.
    """
    try:
        metricas, y_pred = fn()
        saida[nome] = metricas
        if preds is not None and y_pred is not None:
            preds[nome] = y_pred
    except Exception as exc:  # noqa: BLE001 — robustez: um modelo não derruba o resto
        from src.core.logs import get_logger

        get_logger("protocolos_artigos").warning(
            "Modelo '%s' falhou em runtime: %s", nome, exc)
        saida[nome] = _indisponivel(f"erro de execução: {type(exc).__name__}: {exc}")


# ============================================================
# PROTOCOLO — IBRAHIM et al. (2022)
# ============================================================

def protocolo_ibrahim(dados, progresso=None, retornar_predicoes: bool = False):
    """
    AE-LSTM com limiar = percentil do erro de reconstrução numa fatia de
    CALIBRAÇÃO temporal do treino (o AE-LSTM não vê a calibração no ajuste —
    evita o limiar otimista do erro de treino).
    Nenhuma decisão enxerga os rótulos do teste.

    Com ``retornar_predicoes=True`` devolve também ``{modelo: y_pred}`` (as
    decisões por membro, sem refazer fits) — gancho para eventual ensemble.
    """
    import numpy as np

    from src.ml.experimentos_artigos import lib_disponivel

    X_te, y_te = dados["X_te"], dados["y_te"]
    tipos = dados["tipos_te"]
    saida = {}
    preds: dict = {}

    if lib_disponivel("torch"):
        if progresso:
            progresso("Ibrahim: AE-LSTM (calibração temporal do limiar)...")

        def _rodar_ae():
            from src.ml.modelos_anomalia import (
                _score_ae_lstm, sequencias_com_contexto, sequencias_deslizantes,
            )

            # AE-LSTM TEMPORAL (Ibrahim): a LSTM percorre o TEMPO — sequências de
            # janelas consecutivas —, não o eixo das features. Cada item do teste
            # é pontuado como "a janela ATUAL dado o histórico normal precedente"
            # (erro no último passo). Banco de teste idêntico ao dos outros
            # modelos → comparável por AUC. Ver modelos_anomalia._score_ae_lstm.
            L = SEQ_LEN
            Xn_tr = dados["Xn_tr"]
            n_te = int((np.asarray(y_te) == 0).sum())
            Xn_te, Xa_te = X_te[:n_te], X_te[n_te:]   # normais | injetadas (mesma posição)

            seq_tr = sequencias_deslizantes(Xn_tr, L)
            # Fatia de CALIBRAÇÃO temporal: cauda das sequências de treino, fora
            # do ajuste (com purga) → limiar p99 congelado antes do teste.
            corte = max(1, int(len(seq_tr) * 0.8))
            seq_fit = seq_tr[:max(1, corte - PURGA_JANELAS)]
            seq_calib = seq_tr[corte:] if corte < len(seq_tr) else seq_tr[-1:]
            seq_te = np.vstack([
                sequencias_com_contexto(Xn_te, Xn_te, L),   # itens normais
                sequencias_com_contexto(Xn_te, Xa_te, L),   # itens injetados
            ])
            # Ajusta UMA vez em seq_fit; pontua calibração e teste juntos.
            score_all = _score_ae_lstm(seq_fit, np.vstack([seq_calib, seq_te]))
            score_calib = score_all[:len(seq_calib)]
            score_te = score_all[len(seq_calib):]
            limiar = float(np.percentile(score_calib, PERCENTIL_TREINO))
            y_pred_ae = (score_te > limiar).astype(int)
            return _metricas(
                y_te, score_te, y_pred_ae,
                threshold_source=f"p{PERCENTIL_TREINO}_erro_seq_temporal_calibracao",
                limiar=limiar, tipos_te=tipos,
            ), y_pred_ae

        _rodar_modelo("AE-LSTM", _rodar_ae, saida, preds)
    else:
        saida["AE-LSTM"] = _indisponivel("requer torch")

    metodologia = {
        "protocolo": "ibrahim2022_series_temporais",
        "fonte": "Ibrahim et al. (2022)",
        "decisoes": {
            "AE-LSTM": f"sequências de {SEQ_LEN} janelas no TEMPO; limiar = "
                       f"p{PERCENTIL_TREINO} do erro numa fatia de CALIBRAÇÃO "
                       "temporal do treino (fora do ajuste; congelado antes do teste)",
        },
        "fidelidade": [
            "Usa apenas o AE-LSTM do artigo, pois a comparação vigente da "
            "dissertação é AE denso proposto versus AE-LSTM temporal.",
            f"AE-LSTM agora é TEMPORAL de verdade: a LSTM percorre uma sequência "
            f"de {SEQ_LEN} janelas no tempo (a 'correlação na série temporal' do "
            "Ibrahim), não mais o eixo das features. Cada item é a janela ATUAL "
            "dado o histórico normal precedente (erro no último passo).",
            "AE-LSTM usa a disciplina de limiar congelado do pipeline "
            "principal, com calibração em bloco temporal NÃO visto no ajuste "
            "(o erro de treino subestimaria o erro real).",
            "Curadoria: os demais modelos do artigo não entram nesta campanha, "
            "para não desviar a pergunta comparativa central.",
        ],
    }
    if retornar_predicoes:
        return saida, metodologia, preds
    return saida, metodologia


# ============================================================
# DISPATCH
# ============================================================

# key → (função de protocolo, exige validação temporal?)
PROTOCOLOS = {
    "ibrahim": (protocolo_ibrahim, False),
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
