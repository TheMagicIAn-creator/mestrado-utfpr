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
maçã-com-maçã. As features de entrada também são as nossas (espectrais FMECA);
o que muda em relação ao macro_proposto é a ARQUITETURA (LSTM temporal vs.
densa) e o ESCORE (MSE do artigo vs. localizado top-k nosso).

Este script IMPORTA e ORQUESTRA — a mesma interface de scorer do macro_proposto,
mesma avaliação, mesmos gráficos (src/ml/macro_comum.py).

Uso:
  python src/ml/macro_ibrahim.py

Saídas: resultados/macro/ibrahim_*.{json,md,csv,png}
Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("macro_ibrahim")


def _log(*a):
    t = " ".join(str(x) for x in a)
    if t.strip():
        _logger.info(t)


import json
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

def features_das_janelas(janelas, colunas, scaler) -> np.ndarray:
    """Janelas de sinal → matriz (n, F) normalizada, na ordem do treino."""
    from src.ml.features_ca import extrair_janela

    vet = np.asarray([
        [extrair_janela(j).get(c, 0.0) for c in colunas] for j in janelas
    ], dtype=np.float32)
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

def construir_scorer(model, X_contexto: np.ndarray, colunas, scaler):
    """callable(list[DataFrame]) -> escores. Cada janela é pontuada como
    'a janela ATUAL dado o histórico normal precedente' (erro no último passo)."""
    from src.ml.modelos_anomalia import pontuar_ae_lstm, sequencias_com_contexto

    def scorer(janelas):
        X = features_das_janelas(janelas, colunas, scaler)
        seq = sequencias_com_contexto(X_contexto, X, SEQ_LEN)
        return pontuar_ae_lstm(model, seq)

    return scorer


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

def executar(n_janelas: int | None = None) -> dict:
    import torch

    from src.core.seguranca import carregar_pickle_com_sidecar
    from src.ml.dados_avaliacao import carregar_paderborn_compacto, preparar_janelas_holdout
    from src.ml.injecao_falhas import ARQUIVO_CSV, N_JANELAS_SMD
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
            "  python src/ml/features_ca.py && python src/ml/autoencoder.py")
    ckpt = torch.load(arq_modelo, map_location="cpu", weights_only=False)
    scaler = carregar_pickle_com_sidecar(PASTA_AE / "scaler.pkl")
    colunas = ckpt["colunas_feat"]

    _log("\n  Carregando holdout temporal isolado (Paderborn)...")
    df = carregar_paderborn_compacto(ARQUIVO_CSV)
    janelas, _meta = preparar_janelas_holdout(df, n_max=n_janelas or N_JANELAS_SMD)
    del df
    _log(f"  {len(janelas)} janelas não sobrepostas do bloco de teste")

    # MESMA divisão do macro proposto: o AE-LSTM é treinado e calibrado no 1º
    # bloco; FP/AUC/injeção saem do 2º, DISJUNTO (com purga). Sem isso o modelo
    # seria treinado nas próprias janelas de avaliação (vazamento) e a
    # comparação com o método proposto seria inválida.
    j_cal, j_aval = dividir_calibracao_avaliacao(janelas)
    _log(f"  calibração={len(j_cal)} | avaliação={len(j_aval)} (disjuntos)")

    _log("\n  Treinando AE-LSTM temporal no fluxo NORMAL (bloco de calibração)...")
    X_cal = features_das_janelas(j_cal, colunas, scaler)
    model = treinar_detector(X_cal)

    _log("\n  Avaliando detecção por severidade (mesma injeção FMECA)...")
    # contexto temporal do scorer = fluxo normal do bloco de calibração
    scorer = construir_scorer(model, X_cal, colunas, scaler)
    resultado = avaliar_deteccao(NOME, "#1baf7a", scorer, j_cal, j_aval)

    _log(f"\n  Limiar auto-calibrado = {resultado['limiar']:.4f} "
         f"(percentil {resultado['percentil']:.1f}) | FP saudável "
         f"{resultado['fp_pct']:.1f}%")
    for fid, f in resultado["falhas"].items():
        det_1 = f["por_sev"][1.0]["taxa"] * 100
        _log(f"    {f['nome']:<14} AUC={f['auc']:.3f} | detecção@sev1.0={det_1:.0f}%")

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
