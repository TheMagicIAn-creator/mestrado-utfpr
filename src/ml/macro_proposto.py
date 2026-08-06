"""
macro_proposto.py — Al IAdo PV / MACRO-CÓDIGO 1: o método PROPOSTO

Pipeline completo do método da dissertação, de ponta a ponta, num único script
legível — para citar trechos no texto e para rodar com UM comando.

MÉTODO (nosso):
  1. Features espectrais orientadas pela FMECA (RMS, THD, harmônicos 5/7/11/13,
     desbalanceamento) por janela de ~102 ms — src/ml/features_ca.py
  2. Autoencoder DENSO treinado só em operação SAUDÁVEL (Paderborn) —
     modelagem de normalidade (Ibrahim, 2022: AE não-supervisionado + erro de
     reconstrução como sinal de anomalia)
  3. ESCORE LOCALIZADO (nossa contribuição): média dos top-k |resíduos|
     PADRONIZADOS por feature — sensível a falha concentrada em poucas features
     (harmônicos do IGBT, perda de fase do Fusível), que o MSE médio diluía.
     Fundamentação: erro de reconstrução como sinal de anomalia em Ibrahim
     (2022) e padronização por feature/top-k como régua operacional interna.
  4. Limiar AUTO-CALIBRADO para ~1% de falso positivo em bloco saudável não visto
  5. Avaliação E2: injeção FMECA no SINAL por severidade (src/ml/macro_comum.py)

Este script IMPORTA as funções já validadas e apenas ORQUESTRA o fluxo — não
duplica lógica. O resultado sai no MESMO formato do macro_ibrahim.py, para
comparação maçã-com-maçã.

Uso:
  python src/ml/macro_proposto.py

Saídas: resultados/macro/proposto_*.{json,md,csv,png}
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

_logger = _get_logger("macro_proposto")
_log = _adaptar_log(_logger)


import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).parent.parent.parent
PASTA_AE = RAIZ / "resultados" / "autoencoder"
PASTA_SAIDA = RAIZ / "resultados" / "macro"
NOME = "Proposto (AE denso + escore localizado)"


# ============================================================
# ETAPA 1 — carrega o modelo de normalidade já treinado
# ============================================================

def carregar_detector():
    """Autoencoder treinado + scaler + régua do escore localizado."""
    import torch

    from src.core.seguranca import carregar_pickle_com_sidecar
    from src.ml.autoencoder import Autoencoder
    from src.ml import escore_anomalia as ea

    arq_modelo = PASTA_AE / "modelo_autoencoder.pt"
    if not arq_modelo.exists():
        raise FileNotFoundError(
            "Autoencoder não treinado. Rode antes:\n"
            "  python src/ml/features_ca.py && python src/ml/autoencoder.py")

    ckpt = torch.load(arq_modelo, map_location="cpu", weights_only=False)
    scaler = carregar_pickle_com_sidecar(PASTA_AE / "scaler.pkl")
    info = json.loads((PASTA_AE / "limiar.json").read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(ckpt["n_features"], ckpt["latente_dim"]).to(device)
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()

    estat = ea.carregar_estatistica(PASTA_AE)
    metodo = info.get("metodo_escore", "mse")
    return {"modelo": modelo, "scaler": scaler, "device": device,
            "colunas": ckpt["colunas_feat"], "estat": estat, "metodo": metodo,
            "k": info.get("k_localizado", ea.K_LOCALIZADO)}


# ============================================================
# ETAPA 2 — SCORER: janelas de sinal → escore de anomalia
# ============================================================

def construir_scorer(det: dict):
    """Fecha o detector num callable(list[DataFrame]) -> np.ndarray de escores.

    É a interface que o macro_comum espera — a MESMA que o macro_ibrahim provê,
    garantindo que os dois sejam avaliados exatamente do mesmo jeito.
    """
    from src.ml.features_ca import extrair_janela
    from src.ml import escore_anomalia as ea

    def scorer(janelas):
        vetores = np.asarray([
            [extrair_janela(j).get(c, 0.0) for c in det["colunas"]]
            for j in janelas
        ], dtype=np.float32)
        vnorm = det["scaler"].transform(vetores).astype(np.float32)
        residuos = ea.residuo_por_feature(det["modelo"], vnorm, det["device"])
        return ea.pontuar(residuos, det["estat"], det["metodo"], det["k"])

    return scorer


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

def executar(n_janelas: int | None = None) -> dict:
    from src.ml.dados_avaliacao import carregar_paderborn_compacto, preparar_janelas_holdout
    from src.ml.injecao_falhas import ARQUIVO_CSV, N_JANELAS_SMD
    from src.ml.macro_comum import (
        avaliar_deteccao, dividir_calibracao_avaliacao, salvar_saidas,
    )
    from src.ml import escore_anomalia as ea

    _log("=" * 60)
    _log("  MACRO-CÓDIGO 1 — MÉTODO PROPOSTO")
    _log("=" * 60)

    det = carregar_detector()
    _log(f"\n  Detector: AE denso | escore = "
         f"{ea.descricao_metodo(det['metodo'], det['k'])}")

    _log("\n  Carregando holdout temporal isolado (Paderborn)...")
    df = carregar_paderborn_compacto(ARQUIVO_CSV)
    janelas, _meta = preparar_janelas_holdout(df, n_max=n_janelas or N_JANELAS_SMD)
    del df
    _log(f"  {len(janelas)} janelas não sobrepostas do bloco de teste")

    # Calibração e avaliação DISJUNTAS (com purga): o limiar sai do 1º bloco;
    # FP/AUC/injeção vêm do 2º, que o detector nunca viu.
    j_cal, j_aval = dividir_calibracao_avaliacao(janelas)
    _log(f"  calibração={len(j_cal)} | avaliação={len(j_aval)} (disjuntos)")

    _log("\n  Avaliando detecção por severidade (injeção FMECA no sinal)...")
    resultado = avaliar_deteccao(NOME, "#2a78d6", construir_scorer(det), j_cal, j_aval)

    _log(f"\n  Limiar auto-calibrado = {resultado['limiar']:.4f} "
         f"(percentil {resultado['percentil']:.1f}) | FP saudável "
         f"{resultado['fp_pct']:.1f}%")
    for fid, f in resultado["falhas"].items():
        det_1 = f["por_sev"][1.0]["taxa"] * 100
        _log(
            f"    {f['nome']:<14} AUC={f['auc']:.3f} | "
            f"detecção@limiar,sev1.0={det_1:.0f}%"
        )

    saidas = salvar_saidas([resultado], PASTA_SAIDA, prefixo="proposto")
    _log(f"\n  Saídas em {PASTA_SAIDA}")
    for k, v in saidas.items():
        _log(f"    {k}: {Path(v).name}")
    _log("\n  Para comparar com o Ibrahim: python src/ml/macro_ibrahim.py")
    _log("=" * 60)
    return resultado


if __name__ == "__main__":
    from src.core.logs import habilitar_console

    habilitar_console()
    executar()
