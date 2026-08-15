"""
macro_ibrahim.py — Al IAdo PV / MACRO-CÓDIGO 2: o método do IBRAHIM (2022)

Referência: Ibrahim, M.; Alsheikh, A.; Awaysheh, F. M.; Alshehri, M. D.
"Machine Learning Schemes for Anomaly Detection in Solar Power Plants".
Energies, v. 15, n. 1082, 2022.

MÉTODO (do artigo, fielmente):
  1. Autoencoder-LSTM: encoder/decoder recorrentes que capturam a "correlação
     entre variáveis E a correlação na SÉRIE TEMPORAL" (§3.1 do artigo). A LSTM
     percorre o eixo do TEMPO — uma sequência de janelas consecutivas.
  2. Treino NÃO-supervisionado, só em operação normal (modelagem de normalidade).
  3. Erro de reconstrução como escore de anomalia — L(X,X̂)=‖X̂−X‖² (eq. 3).
  4. Limiar por percentil do erro em bloco normal não visto (auto-calibrado
     para ~1% de FP, igual ao macro proposto — para a comparação ser justa).

O que é NOSSO neste script (declarado, não escondido): a AVALIAÇÃO. O artigo
usa séries de potência de usinas (15 min × 34 dias) com 13 anomalias reais;
aqui o método dele é aplicado ao NOSSO problema — sinal CA do Paderborn com
injeção FMECA por severidade — para que os dois macros sejam comparáveis
maçã-com-maçã. As features de entrada também são as nossas (espectrais FMECA).
O que muda em relação ao macro_proposto é a ARQUITETURA e a organização
temporal (LSTM sobre sequências vs. autoencoder denso por janela); ambos são
comparados pelo erro de reconstrução MSE sob o mesmo protocolo de avaliação.

Este script IMPORTA e ORQUESTRA — a mesma interface de scorer do macro_proposto,
mesma avaliação, mesmos gráficos (src/ml/macro_comum.py).

Uso:
  python src/ml/macro_ibrahim.py

Saídas: resultados/macro/ibrahim_*.{json,md,csv,png}
Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("macro_ibrahim")
_log = _adaptar_log(_logger)


import os
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).parent.parent.parent
PASTA_AE = RAIZ / "resultados" / "autoencoder"
PASTA_SAIDA = RAIZ / "resultados" / "macro"
NOME = "Ibrahim 2022 (AE-LSTM temporal)"

SEQ_LEN = int(os.getenv("AL_IADO_AELSTM_SEQ_LEN", "8"))   # passos temporais
EPOCHS = int(os.getenv("AL_IADO_AELSTM_EPOCHS", "60"))


# ============================================================
# ETAPA 1 — features das janelas (mesma base do macro proposto)
# ============================================================

def features_das_janelas(janelas, colunas, scaler, normalizacao=None) -> np.ndarray:
    """Janelas de sinal → matriz (n, F) normalizada, na ordem do treino.

    Featurização canônica do GPVS, a MESMA do macro proposto — é o que torna a
    comparação legítima: os dois modelos veem exatamente o mesmo vetor de
    entrada. Dois defeitos já moraram aqui, ambos por divergir dessa cadeia:
    o extrator do Stender devolvendo 0,0 para as 24 features do GPVS, e a
    ausência da normalização de comissionamento por ensaio, que inflou o limiar
    em 61 mil vezes e zerou a detecção dos dois modelos. Ver
    `gpvs_principal.vetores_de_janelas`.
    """
    from src.ml.gpvs_principal import vetores_de_janelas

    vet = vetores_de_janelas(janelas, colunas, normalizacao)
    return scaler.transform(vet).astype(np.float32)


# ============================================================
# ETAPA 2 — treina o AE-LSTM TEMPORAL no fluxo normal
# ============================================================

def treinar_detector(X_normal: np.ndarray):
    """Treina o AE-LSTM em sequências temporais de janelas NORMAIS."""
    from src.ml.modelos_anomalia import sequencias_deslizantes, treinar_ae_lstm

    seq = sequencias_deslizantes(X_normal, SEQ_LEN)
    _log(f"  Sequências de treino: {seq.shape} (n, L={SEQ_LEN}, F)")
    return treinar_ae_lstm(seq, epochs=EPOCHS, seed=42)


# ============================================================
# ETAPA 3 — SCORER (mesma interface do macro_proposto)
# ============================================================

# Regimes de contexto temporal do AE-LSTM. A escolha NÃO é detalhe de
# implementação: ela decide o que a comparação com o AE denso está medindo.
CONTEXTO_NORMAL = "normal"
CONTEXTO_TRAJETORIA = "trajetoria"
CONTEXTOS = (CONTEXTO_NORMAL, CONTEXTO_TRAJETORIA)


def construir_scorer(model, X_contexto: np.ndarray, colunas, scaler,
                     normalizacao=None, contexto: str = CONTEXTO_NORMAL):
    """callable(list[DataFrame]) -> escores, no último passo da sequência.

    O PARÂMETRO `contexto` EXISTE POR UM MOTIVO CIENTÍFICO
    =====================================================
    `sequencias_com_contexto` monta, para cada janela pontuada, L-1
    predecessores mais a janela no último passo. De onde vêm os predecessores
    muda o que o AE-LSTM enxerga:

    - `normal` (padrão): predecessores do fluxo SAUDÁVEL de calibração. Durante
      a varredura de magnitude o modelo vê `normal, …, normal, INJETADA` — um
      DEGRAU. Ele detecta descontinuidade contra uma linha de base sã.

    - `trajetoria`: predecessores são os próprios itens, isto é, as magnitudes
      ANTERIORES da mesma trajetória. O modelo vê uma série que DEGRADA
      progressivamente, que é o fenômeno físico que a dissertação modela
      (contato que se desgasta, IGBT que envelhece).

    Por que importa: em 15/08/2026 o AE-LSTM apareceu detectando com 1/3 da
    magnitude do AE denso nas três falhas. Sob `normal`, parte desse ganho pode
    vir do contraste que a varredura fabrica — em campo o histórico também
    estaria degradado, e o degrau não existiria. Comparar os dois regimes separa
    ganho de ARQUITETURA de ganho de CONTRASTE.

    Nenhum dos dois é "o certo": são perguntas diferentes. `normal` responde
    "detecta início abrupto?"; `trajetoria` responde "detecta degradação
    progressiva?". A dissertação precisa dizer qual está reportando.
    """
    from src.ml.modelos_anomalia import pontuar_ae_lstm, sequencias_com_contexto

    if contexto not in CONTEXTOS:
        raise ValueError(
            f"contexto do AE-LSTM deve ser um de {CONTEXTOS}; recebido {contexto!r}"
        )

    def scorer(janelas):
        X = features_das_janelas(janelas, colunas, scaler, normalizacao)
        # Com `trajetoria`, os predecessores saem do próprio lote: para o item i
        # no passo t, `sequencias_com_contexto` usa a posição i-(L-1)+t, que são
        # exatamente as magnitudes anteriores da mesma janela-base.
        base = X if contexto == CONTEXTO_TRAJETORIA else X_contexto
        seq = sequencias_com_contexto(base, X, SEQ_LEN)
        return pontuar_ae_lstm(model, seq)

    return scorer


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

def executar(n_janelas: int | None = None) -> dict:
    import torch

    from src.core.seguranca import carregar_pickle_com_sidecar
    from src.ml.gpvs_principal import (
        carregar_normalizacao_baseline, preparar_janelas_holdout,
    )
    from src.ml.injecao_falhas import N_JANELAS_SMD
    from src.ml.macro_comum import (
        avaliar_deteccao, dividir_calibracao_avaliacao, salvar_saidas,
    )

    _log("=" * 60)
    _log("  MACRO-CÓDIGO 2 — MÉTODO DO IBRAHIM (2022)")
    _log("=" * 60)
    _log("  AE-LSTM temporal | erro de reconstrução (eq. 3 do artigo)")

    arq_modelo = PASTA_AE / "modelo_autoencoder.pt"
    if not arq_modelo.exists():
        raise FileNotFoundError(
            "Artefatos de features/scaler ausentes. Rode antes:\n"
            "  python -m src.ml.exec_etapa_isolada features_gpvs && "
            "python -m src.ml.exec_etapa_isolada autoencoder")
    ckpt = torch.load(arq_modelo, map_location="cpu", weights_only=False)
    scaler = carregar_pickle_com_sidecar(PASTA_AE / "scaler.pkl")
    colunas = ckpt["colunas_feat"]
    normalizacao = carregar_normalizacao_baseline(PASTA_AE)

    _log("\n  Carregando holdout F0 do GPVS-Faults (teste isolado)...")
    janelas, _meta = preparar_janelas_holdout(n_max=n_janelas or N_JANELAS_SMD)
    _log(f"  {len(janelas)} janelas não sobrepostas do bloco de teste F0")

    # MESMA divisão do macro proposto: o AE-LSTM é treinado e calibrado no 1º
    # bloco; FP/AUC/injeção saem do 2º, DISJUNTO (com purga). Sem isso o modelo
    # seria treinado nas próprias janelas de avaliação (vazamento) e a
    # comparação com o método proposto seria inválida.
    j_cal, j_aval = dividir_calibracao_avaliacao(janelas)
    _log(f"  calibração={len(j_cal)} | avaliação={len(j_aval)} (disjuntos)")

    _log("\n  Treinando AE-LSTM temporal no fluxo NORMAL (bloco de calibração)...")
    X_cal = features_das_janelas(j_cal, colunas, scaler, normalizacao)
    model = treinar_detector(X_cal)

    _log("\n  Avaliando detecção por severidade (mesma injeção FMECA)...")
    # contexto temporal do scorer = fluxo normal do bloco de calibração
    scorer = construir_scorer(model, X_cal, colunas, scaler, normalizacao)
    resultado = avaliar_deteccao(NOME, "#1baf7a", scorer, j_cal, j_aval)

    _log(f"\n  Limiar auto-calibrado = {resultado['limiar']:.4f} "
         f"(percentil {resultado['percentil']:.1f}) | FP saudável "
         f"{resultado['fp_pct']:.1f}%")
    for fid, f in resultado["falhas"].items():
        det_1 = f["por_sev"][1.0]["taxa"] * 100
        _log(
            f"    {f['nome']:<14} AUC={f['auc']:.3f} | "
            f"detecção@limiar,sev1.0={det_1:.0f}%"
        )

    saidas = salvar_saidas([resultado], PASTA_SAIDA, prefixo="ibrahim")
    _log(f"\n  Saídas em {PASTA_SAIDA}")
    for k, v in saidas.items():
        _log(f"    {k}: {Path(v).name}")
    _log("\n  Comparativo dos dois: python src/ml/macro_comparar.py")
    _log("=" * 60)
    return resultado


if __name__ == "__main__":
    from src.core.logs import habilitar_console

    habilitar_console()
    executar()
