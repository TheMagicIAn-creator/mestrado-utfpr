"""
diagnostico_escore.py — Al IAdo PV / diagnóstico comparativo E2 no GPVS

Compara, LADO A LADO e sem substituir nada, dois escores de anomalia sobre
as MESMAS falhas sintéticas injetadas:

  1. MSE médio  — referência HISTÓRICA (média do erro de reconstrução sobre
     todas as ~109 features). Dilui falhas localizadas.
  2. Localizado — ablação: média dos top-k maiores resíduos
     PADRONIZADOS por feature
     (z do |resíduo| contra a distribuição saudável). Sensível a falha que
     mexe em POUCAS features (harmônicos do IGBT, perda de fase do Fusível).

Motivação (docs/auditoria_pipeline_ml.md §3.1): o detector só enxerga bem o
Contator (banda larga) porque o MSE médio dilui as falhas localizadas. Este
script MEDE se um escore localizado recupera a detecção do IGBT/Fusível —
sem forçar amplitude de injeção (isso seria detecção artificial).

Cada escore usa o limiar estimado na calibração e a mesma separação de dados.
O MSE é operacional; o localizado permanece publicado para comparação.

É um DIAGNÓSTICO de ablação: lê o Autoencoder já treinado, usa o mesmo holdout
GPVS-Faults F0 e as mesmas funções de src/ml/injecao_falhas.py, sem misturar o
benchmark Paderborn nem alterar o escore operacional.

Uso:
  python src/ml/diagnostico_escore.py
  python src/ml/diagnostico_escore.py --k 5

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta: python src/ml/<arquivo>.py
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("diagnostico_escore")
_log = _adaptar_log(_logger)


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


def _limiares_comparacao(
    info_limiar: dict,
    score_loc_sau: np.ndarray,
    k: int,
) -> tuple[float, float, float, bool]:
    """Retorna MSE p99, localizado, percentil efetivo e se ele é operacional."""
    limiar_mse = float(
        info_limiar.get("mse_p99", info_limiar.get("limiar_p99"))
    )
    percentil = float(
        info_limiar.get(
            "threshold_effective_percentile",
            info_limiar.get("percentil_limiar", 99.0),
        )
    )
    k_publicado = info_limiar.get("top_k") or info_limiar.get("k_localizado")
    localizado_operacional = (
        info_limiar.get("score_method", info_limiar.get("metodo_escore"))
        == "localizado"
        and k_publicado is not None
        and int(k_publicado) == int(k)
    )
    mesmo_k = k_publicado is not None and int(k_publicado) == int(k)
    if localizado_operacional:
        limiar_loc = float(
            info_limiar.get(
                "score_threshold", info_limiar.get("limiar_localizado")
            )
        )
    elif mesmo_k and info_limiar.get("limiar_localizado") is not None:
        limiar_loc = float(
            info_limiar.get("limiar_localizado")
        )
    else:
        limiar_loc = float(np.percentile(score_loc_sau, percentil))
    return limiar_mse, limiar_loc, percentil, localizado_operacional


# ============================================================
# INTEGRAÇÃO — usa o Autoencoder treinado e a injeção validada
# ============================================================

def _residuo_por_feature(
    janela_df, modelo, scaler, device, colunas_feat,
    normalizacao_baseline: dict | None = None,
) -> np.ndarray:
    """Resíduo (x - x_rec) por feature de UMA janela, em espaço normalizado."""
    import torch

    from src.ml.gpvs_principal import extrair_janela, normalizar_vetores_f0

    feats = extrair_janela(janela_df)
    vetor = np.array([feats.get(c, 0.0) for c in colunas_feat], dtype=np.float32)
    if normalizacao_baseline is not None:
        vetor = normalizar_vetores_f0(
            vetor.reshape(1, -1), [janela_df.attrs.get("ensaio")],
            normalizacao_baseline,
        )[0]
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
    from src.ml.gpvs import DOI_GPVS
    from src.ml.gpvs_principal import (
        carregar_normalizacao_baseline, preparar_janelas_holdout,
    )
    from src.ml.injecao_falhas import (
        FALHAS, FUNCOES_FALHA, SEVERIDADES, N_JANELAS_SMD, ALVO_SMD,
        PASTA_AE,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    _log(f"   ✅ Modelo: {n_features} features → latente {latente_dim} | k={k}")

    janelas_holdout, meta_holdout = preparar_janelas_holdout(
        n_max=N_JANELAS_SMD
    )
    normalizacao_baseline = carregar_normalizacao_baseline(PASTA_AE)
    _log(
        f"   ✅ {len(janelas_holdout)} janelas saudáveis GPVS F0 "
        "do holdout temporal"
    )

    # ── Resíduos saudáveis → comparação com a régua ajustada no treino ──
    R_sau = np.vstack([
        _residuo_por_feature(
            j, modelo, scaler, device, colunas_feat, normalizacao_baseline
        )
        for j in janelas_holdout
    ])
    from src.ml import escore_anomalia as ea

    stats = ea.carregar_estatistica(PASTA_AE)
    if stats is None:
        stats = ajustar_estatistica_residuo(R_sau)
    score_loc_sau = escore_localizado(R_sau, stats, k=k)
    score_mse_sau = escore_mse_medio(R_sau)
    limiar_mse, limiar_loc, percentil_loc, loc_operacional = _limiares_comparacao(
        info_limiar, score_loc_sau, k
    )

    fp_mse = float((score_mse_sau > limiar_mse).mean() * 100)
    fp_loc = float((score_loc_sau > limiar_loc).mean() * 100)
    _log(f"   MSE histórico p99 = {limiar_mse:.4f} (FP saudável {fp_mse:.1f}%)")
    origem_loc = "operacional publicado" if loc_operacional else "recalculado"
    _log(
        f"   Localizado p{percentil_loc:g} = {limiar_loc:.4f} "
        f"({origem_loc}; FP saudável {fp_loc:.1f}%)"
    )

    # ── Injeção por falha/severidade: detecção nos DOIS escores ──
    saida = {
        "k": k, "limiar_mse": limiar_mse, "limiar_localizado": limiar_loc,
        "score_method_operacional": info_limiar.get(
            "score_method", info_limiar.get("metodo_escore")
        ),
        "threshold_effective_percentile": percentil_loc,
        "limiar_localizado_operacional": loc_operacional,
        "fp_saudavel_mse_pct": fp_mse, "fp_saudavel_localizado_pct": fp_loc,
        "dataset": "GPVS-Faults",
        "dataset_doi": DOI_GPVS,
        "protocolo_avaliacao": meta_holdout,
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
                R.append(_residuo_por_feature(
                    jf, modelo, scaler, device, colunas_feat,
                    normalizacao_baseline,
                ))
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
    fig.suptitle(
        "GPVS-Faults F0 — MSE operacional × escore localizado (ablação)"
    )
    for ax, falha in zip(axes, FALHAS):
        info = saida["falhas"][falha["id"]]
        por = info["por_sev"]
        y_mse = [por[str(s)]["taxa_mse"] * 100 for s in sevs]
        y_loc = [por[str(s)]["taxa_localizado"] * 100 for s in sevs]
        mse_low = [por[str(s)]["mse_ci"][0] * 100 for s in sevs]
        mse_high = [por[str(s)]["mse_ci"][1] * 100 for s in sevs]
        loc_low = [por[str(s)]["loc_ci"][0] * 100 for s in sevs]
        loc_high = [por[str(s)]["loc_ci"][1] * 100 for s in sevs]
        # A cor codifica o MÉTODO, não a falha — a falha já está no título do
        # painel. Antes, a linha do escore localizado usava `falha["cor"]`
        # (azul, verde, amarelo), enquanto a legenda — desenhada só no primeiro
        # painel — anunciava "Localizado (top-k)" em azul. Nos painéis do IGBT
        # e do Fusível a legenda ficava simplesmente errada.
        #
        # Convenção de src/ml/estilo_graficos.py: COR_METODO para o método
        # proposto, COR_NEUTRA para o baseline; "a cor segue a entidade, nunca
        # o rank". É a mesma leitura dos gráficos de comparação com o Ibrahim.
        ax.errorbar(
            sevs, y_mse,
            yerr=[
                np.maximum(0.0, np.asarray(y_mse) - mse_low),
                np.maximum(0.0, np.asarray(mse_high) - y_mse),
            ],
            fmt="o-", capsize=2.5, color=COR_NEUTRA,
            label="MSE médio (operacional)",
        )
        rotulo_loc = "Localizado (operacional)" if saida.get(
            "limiar_localizado_operacional"
        ) else "Localizado (ablação)"
        ax.errorbar(
            sevs, y_loc,
            yerr=[
                np.maximum(0.0, np.asarray(y_loc) - loc_low),
                np.maximum(0.0, np.asarray(loc_high) - y_loc),
            ],
            fmt="s-", capsize=2.5, color=COR_METODO, label=rotulo_loc,
        )
        ax.axhline(saida["alvo_smd"] * 100, color=COR_ALERTA, linestyle="--",
                   linewidth=1.5, label=f"Alvo SMD {saida['alvo_smd']*100:.0f}%")
        ax.set_title(f"{falha['nome']} (NPR={falha['npr']})", fontsize=10)
        ax.set_xlabel(r"Magnitude injetada $a_{inj}$ (fração nominal)")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Taxa de detecção (%)")
    axes[0].legend(fontsize=8)
    salvar_figura(
        fig, pasta / "diagnostico_escore.png",
        f"GPVS-Faults F0, n={saida['n_janelas']} por nível; IC95% de Wilson. "
        "MSE usa o limiar operacional; localizado é ablação no percentil "
        f"p{saida['threshold_effective_percentile']:g} e não altera o detector.",
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
