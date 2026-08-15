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
  3. MSE médio de reconstrução, conforme o princípio de Ibrahim (2022).
     O escore localizado permanece como ablação porque não generalizou o ganho
     anterior no split auditado de 09/08/2026.
  4. Limiar calibrado em bloco saudável não visto
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
NOME = "Proposto (AE denso + MSE p99)"


# ============================================================
# ETAPA 1 — carrega o modelo de normalidade já treinado
# ============================================================

def carregar_detector():
    """Autoencoder treinado + scaler + metadados do escore operacional."""
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
    # Extrator do GPVS — o MESMO que produziu as features de treino do detector.
    #
    # Era `features_ca.extrair_janela` (Stender). Isso passou a ser um bug MUDO
    # depois da migração para o GPVS: `det["colunas"]` são as 24 features do
    # GPVS (`Ipv_median`, `ia_thd`, `p_ac_mean`…), e o extrator do Stender
    # devolve 108 features com nomes OUTROS (`i_a_rms`, `i_a_harm_5`…). Nenhum
    # dos 24 nomes existia no dicionário, então `.get(c, 0.0)` devolvia 0,0 para
    # TODOS eles — um vetor de zeros, sem erro de shape, sem aviso. O
    # autoencoder reconstruía o nada e a comparação publicava esse número.
    #
    # O acesso passa a ser `[c]` e não `.get(c, 0.0)` DE PROPÓSITO: feature que
    # falta é defeito, e tem de estourar alto. O default silencioso foi o que
    # transformou uma incompatibilidade de dataset num resultado plausível.
    from src.ml.gpvs_principal import vetor_de_features
    from src.ml import escore_anomalia as ea

    def scorer(janelas):
        vetores = np.asarray(
            [vetor_de_features(j, det["colunas"]) for j in janelas],
            dtype=np.float32,
        )
        vnorm = det["scaler"].transform(vetores).astype(np.float32)
        residuos = ea.residuo_por_feature(det["modelo"], vnorm, det["device"])
        return ea.pontuar(residuos, det["estat"], det["metodo"], det["k"])

    return scorer


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

def executar(n_janelas: int | None = None) -> dict:
    from src.ml.gpvs_principal import preparar_janelas_holdout
    from src.ml.injecao_falhas import N_JANELAS_SMD
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

    _log("\n  Carregando holdout F0 do GPVS-Faults (teste isolado)...")
    janelas, _meta = preparar_janelas_holdout(n_max=n_janelas or N_JANELAS_SMD)
    _log(f"  {len(janelas)} janelas não sobrepostas do bloco de teste F0")

    # Calibração e avaliação DISJUNTAS (com purga): o limiar sai do 1º bloco;
    # FP/AUC/injeção vêm do 2º, que o detector nunca viu.
    j_cal, j_aval = dividir_calibracao_avaliacao(janelas)
    _log(f"  calibração={len(j_cal)} | avaliação={len(j_aval)} (disjuntos)")

    _log("\n  Avaliando detecção por severidade (injeção FMECA no sinal)...")
    nome_metodo = (
        NOME
        if det["metodo"] == "mse"
        else "Proposto (AE denso + escore localizado experimental)"
    )
    resultado = avaliar_deteccao(
        nome_metodo, "#2a78d6", construir_scorer(det), j_cal, j_aval
    )

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
