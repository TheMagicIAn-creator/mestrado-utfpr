"""
diagnostico_escore.py — Al IAdo PV / diagnóstico (NÃO altera o pipeline)

Compara, LADO A LADO e sem substituir nada, dois escores de anomalia sobre
as MESMAS falhas sintéticas injetadas:

  1. MSE médio  — o escore OPERACIONAL atual (média do erro de reconstrução
     sobre todas as ~109 features). Dilui falhas localizadas.
  2. Localizado — média dos top-k maiores resíduos PADRONIZADOS por feature
     (z do |resíduo| contra a distribuição saudável). Sensível a falha que
     mexe em POUCAS features (harmônicos do IGBT, perda de fase do Fusível).

Motivação (docs/auditoria_pipeline_ml.md §3.1): o detector só enxerga bem o
Contator (banda larga) porque o MSE médio dilui as falhas localizadas. Este
script MEDE se um escore localizado recupera a detecção do IGBT/Fusível —
sem forçar amplitude de injeção (isso seria detecção artificial).

Ambos os limiares alvejam ~1% de falso positivo (percentil 99 do escore no
bloco saudável), então a comparação é justa.

É um DIAGNÓSTICO reversível: lê os artefatos do Autoencoder já treinado,
injeta com as MESMAS funções de src/ml/injecao_falhas.py e escreve apenas
resultados/autoencoder/diagnostico_escore.{png,json}. Nada do pipeline muda.

Uso:
  python src/ml/diagnostico_escore.py
  python src/ml/diagnostico_escore.py --k 5

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta: python src/ml/<arquivo>.py
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("diagnostico_escore")


def _log(*args):
    texto = " ".join(str(a) for a in args)
    if texto.strip():
        _logger.info(texto.rstrip("\n"))


import json
import argparse
import numpy as np


# ============================================================
# NÚCLEO — reusa a fonte única do escore (src/ml/escore_anomalia.py)
# ============================================================
# Sem duplicar lógica: as funções de escore vivem no módulo canônico.
from src.ml.escore_anomalia import (  # noqa: E402
    ajustar_estatistica_residuo, escore_localizado, escore_mse_medio,
)


def _wilson(sucessos: int, n: int) -> tuple[float, float]:
    from src.ml.estatistica import intervalo_wilson

    return intervalo_wilson(int(sucessos), int(n))


# ============================================================
# INTEGRAÇÃO — usa o Autoencoder treinado e a injeção validada
# ============================================================

def _residuo_por_feature(janela_df, modelo, scaler, device, colunas_feat) -> np.ndarray:
    """Resíduo (x - x_rec) por feature de UMA janela, em espaço normalizado."""
    import torch

    from src.ml.features_ca import extrair_janela

    feats = extrair_janela(janela_df)
    vetor = np.array([feats.get(c, 0.0) for c in colunas_feat], dtype=np.float32)
    vetor_norm = scaler.transform(vetor.reshape(1, -1)).astype(np.float32)
    modelo.eval()
    with torch.no_grad():
        x = torch.from_numpy(vetor_norm).to(device)
        x_rec = modelo(x)
        r = (x - x_rec).cpu().numpy().ravel()
    return r


def executar_diagnostico(k: int = 5) -> bool:
    import torch

    from src.ml.autoencoder import Autoencoder
    from src.ml.dados_avaliacao import (
        carregar_paderborn_compacto, preparar_janelas_holdout,
    )
    from src.ml.injecao_falhas import (
        FALHAS, FUNCOES_FALHA, SEVERIDADES, N_JANELAS_SMD, ALVO_SMD,
        ARQUIVO_CSV, PASTA_AE,
    )
    from src.core.seguranca import carregar_pickle_com_sidecar

    _log("=" * 60)
    _log("  DIAGNÓSTICO DE ESCORE — MSE médio × localizado (top-k)")
    _log("=" * 60)

    arq_modelo = PASTA_AE / "modelo_autoencoder.pt"
    arq_scaler = PASTA_AE / "scaler.pkl"
    arq_limiar = PASTA_AE / "limiar.json"
    for arq in (arq_modelo, arq_scaler, arq_limiar):
        if not arq.exists():
            _log(f"   ❌ Não encontrado: {arq.name}. Rode antes: "
                 "python src/ml/autoencoder.py")
            return False

    checkpoint = torch.load(arq_modelo, map_location="cpu", weights_only=False)
    scaler = carregar_pickle_com_sidecar(arq_scaler)
    info_limiar = json.loads(arq_limiar.read_text(encoding="utf-8"))

    n_features = checkpoint["n_features"]
    latente_dim = checkpoint["latente_dim"]
    colunas_feat = checkpoint["colunas_feat"]
    limiar_mse = float(info_limiar["limiar"])   # limiar OPERACIONAL do MSE médio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    _log(f"   ✅ Modelo: {n_features} features → latente {latente_dim} | k={k}")

    df = carregar_paderborn_compacto(ARQUIVO_CSV)
    janelas_holdout, _ = preparar_janelas_holdout(df, n_max=N_JANELAS_SMD)
    del df
    _log(f"   ✅ {len(janelas_holdout)} janelas saudáveis do holdout")

    # ── Resíduos saudáveis → régua por-feature + limiar do localizado ──
    R_sau = np.vstack([
        _residuo_por_feature(j, modelo, scaler, device, colunas_feat)
        for j in janelas_holdout
    ])
    stats = ajustar_estatistica_residuo(R_sau)
    score_loc_sau = escore_localizado(R_sau, stats, k=k)
    score_mse_sau = escore_mse_medio(R_sau)
    limiar_loc = float(np.percentile(score_loc_sau, 99))   # ~1% FP, como o MSE

    fp_mse = float((score_mse_sau > limiar_mse).mean() * 100)
    fp_loc = float((score_loc_sau > limiar_loc).mean() * 100)
    _log(f"   Limiar MSE médio = {limiar_mse:.4f} (FP saudável {fp_mse:.1f}%)")
    _log(f"   Limiar localizado = {limiar_loc:.4f} (FP saudável {fp_loc:.1f}%)")

    # ── Injeção por falha/severidade: detecção nos DOIS escores ──
    saida = {
        "k": k, "limiar_mse": limiar_mse, "limiar_localizado": limiar_loc,
        "fp_saudavel_mse_pct": fp_mse, "fp_saudavel_localizado_pct": fp_loc,
        "alvo_smd": ALVO_SMD, "n_janelas": len(janelas_holdout),
        "falhas": {},
    }
    for falha in FALHAS:
        fid, nome = falha["id"], falha["nome"]
        fn = FUNCOES_FALHA[fid]
        _log(f"\n   🔴 {nome} (NPR={falha['npr']})")
        por_sev = {}
        for sev in SEVERIDADES:
            R = []
            for j, janela in enumerate(janelas_holdout):
                jf = fn(janela, sev, seed=10_000 + j) if fid == "contator_ac" \
                    else fn(janela, sev)
                R.append(_residuo_por_feature(jf, modelo, scaler, device, colunas_feat))
            R = np.vstack(R)
            s_mse = escore_mse_medio(R)
            s_loc = escore_localizado(R, stats, k=k)
            det_mse = s_mse > limiar_mse
            det_loc = s_loc > limiar_loc
            taxa_mse, taxa_loc = float(det_mse.mean()), float(det_loc.mean())
            por_sev[str(sev)] = {
                "taxa_mse": taxa_mse,
                "taxa_localizado": taxa_loc,
                "mse_ci": _wilson(det_mse.sum(), len(det_mse)),
                "loc_ci": _wilson(det_loc.sum(), len(det_loc)),
            }
            _log(f"      sev={sev:>4}: MSE {taxa_mse*100:5.1f}%  |  "
                 f"localizado {taxa_loc*100:5.1f}%")
        saida["falhas"][fid] = {"nome": nome, "npr": falha["npr"], "por_sev": por_sev}

    PASTA_AE.mkdir(parents=True, exist_ok=True)
    arq_json = PASTA_AE / "diagnostico_escore.json"
    arq_json.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"\n   ✅ {arq_json.name}")
    _plotar(saida, FALHAS, PASTA_AE)
    _log("\n  Diagnóstico concluído. Leia docs/auditoria_pipeline_ml.md §3.1.")
    return True


def _plotar(saida: dict, FALHAS, pasta) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.ml.estilo_graficos import (
        COR_ALERTA, COR_METODO, COR_NEUTRA, TAM, aplicar_estilo, salvar_figura,
    )

    aplicar_estilo()
    sevs = [float(s) for s in next(iter(saida["falhas"].values()))["por_sev"]]
    fig, axes = plt.subplots(1, len(FALHAS), figsize=TAM["painel_3"],
                             layout="constrained", sharey=True)
    fig.suptitle("Detectabilidade por severidade — MSE médio × escore localizado")
    for ax, falha in zip(axes, FALHAS):
        info = saida["falhas"][falha["id"]]
        por = info["por_sev"]
        y_mse = [por[str(s)]["taxa_mse"] * 100 for s in sevs]
        y_loc = [por[str(s)]["taxa_localizado"] * 100 for s in sevs]
        # A cor codifica o MÉTODO, não a falha — a falha já está no título do
        # painel. Antes, a linha do escore localizado usava `falha["cor"]`
        # (azul, verde, amarelo), enquanto a legenda — desenhada só no primeiro
        # painel — anunciava "Localizado (top-k)" em azul. Nos painéis do IGBT
        # e do Fusível a legenda ficava simplesmente errada.
        #
        # Convenção de src/ml/estilo_graficos.py: COR_METODO para o método
        # proposto, COR_NEUTRA para o baseline; "a cor segue a entidade, nunca
        # o rank". É a mesma leitura dos gráficos de comparação com o Ibrahim.
        ax.plot(sevs, y_mse, "o-", color=COR_NEUTRA, label="MSE médio (atual)")
        ax.plot(sevs, y_loc, "s-", color=COR_METODO, label="Localizado (top-k)")
        ax.axhline(saida["alvo_smd"] * 100, color=COR_ALERTA, linestyle="--",
                   linewidth=1.5, label=f"Alvo SMD {saida['alvo_smd']*100:.0f}%")
        ax.set_title(f"{falha['nome']} (NPR={falha['npr']})", fontsize=10)
        ax.set_xlabel("Severidade")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Taxa de detecção (%)")
    axes[0].legend(fontsize=8)
    salvar_figura(
        fig, pasta / "diagnostico_escore.png",
        "Diagnóstico E2: ambos os limiares alvejam ~1% de FP no bloco saudável; "
        "não altera o pipeline operacional.",
    )
    _log(f"   📊 diagnostico_escore.png")


if __name__ == "__main__":
    from src.core.logs import habilitar_console

    habilitar_console()
    parser = argparse.ArgumentParser(
        description="Compara MSE médio × escore localizado (top-k) — diagnóstico")
    parser.add_argument("--k", type=int, default=5,
                        help="Nº de features de maior desvio agregadas (padrão: 5)")
    args = parser.parse_args()
    executar_diagnostico(k=args.k)
