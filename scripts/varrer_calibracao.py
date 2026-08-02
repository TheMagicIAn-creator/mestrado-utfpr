"""
varrer_calibracao.py — Al IAdo PV

Varre `k` (top-k do escore localizado) × `percentil` (limiar) e reporta, para
cada par: falso positivo no bloco saudável, **o intervalo de confiança do
limiar**, e a detecção por falha da FMECA.

POR QUE ESTE SCRIPT EXISTE
==========================
A auditoria recomendava "varrer k × percentil e escolher o par que traz o FP
para ~1–2% mantendo o recall". A recomendação estava no documento havia
semanas, e **o script nunca existiu** — o bloqueio nunca foi o dataset nem o
torch, era não haver código.

POR QUE O IC DO LIMIAR APARECE EM TODA LINHA
============================================
Medido em `src/ml/escore_anomalia.incerteza_do_limiar`: com ~73 janelas de
calibração, o IC95 do limiar tem largura da ordem do próprio limiar. Isso
significa que **duas configurações podem diferir em FP por puro ruído de
estimativa**. Escolher o "melhor par" olhando só o FP pontual é escolher ruído.

A coluna `ic_rel` é o guarda-costas dessa leitura: acima de ~0,3, diferenças
finas de FP entre linhas não sustentam conclusão.

O QUE ESTE SCRIPT **NÃO** FAZ
=============================
Não troca o estimador do limiar. Bootstrap, ajuste paramétrico e EVT foram
testados e rejeitados (ver docstring de `incerteza_do_limiar` e
docs/auditoria_pipeline_ml.md §22): o limite é o TAMANHO DA AMOSTRA, não o
estimador. Este script varre o que de fato dá para mexer — `k` e o percentil.

Uso (na máquina com o dataset e o modelo treinado):

    python scripts/varrer_calibracao.py
    python scripts/varrer_calibracao.py --k 5 10 15 --percentil 99 99.5 99.9

Saída: tabela no terminal + `resultados/autoencoder/varredura_calibracao.csv`
e `.json`. Não sobrescreve nenhum artefato do pipeline.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.utils import configurar_saida_utf8  # noqa: E402

configurar_saida_utf8()

from src.ml.escore_anomalia import (  # noqa: E402
    ajustar_estatistica_residuo,
    escore_localizado,
    incerteza_do_limiar,
)

K_PADRAO = (5, 10, 15)
PERCENTIL_PADRAO = (99.0, 99.5, 99.9)
# Acima disto, o limiar "balança" demais para sustentar comparação fina de FP.
IC_REL_SUSPEITO = 0.30


def _carregar_contexto():
    """Modelo, scaler e janelas saudáveis — mesmo caminho do diagnóstico.

    Reusa `src/ml/diagnostico_escore` em vez de duplicar a carga: se o formato
    do checkpoint mudar, os dois seguem juntos.
    """
    import torch

    from src.core.config import RAIZ_PROJETO
    from src.ml.autoencoder import Autoencoder
    from src.ml.dados_avaliacao import (
        carregar_paderborn_compacto,
        preparar_janelas_holdout,
    )
    from src.ml.diagnostico_escore import _residuo_por_feature
    from src.ml.injecao_falhas import (
        ARQUIVO_CSV,
        FALHAS,
        FUNCOES_FALHA,
        N_JANELAS_SMD,
        SEVERIDADES,
    )
    from src.core.utils import carregar_pickle_com_sidecar

    pasta = Path(RAIZ_PROJETO) / "resultados" / "autoencoder"
    faltando = [n for n in ("modelo_autoencoder.pt", "scaler.pkl")
                if not (pasta / n).exists()]
    if faltando:
        print(f"  ❌ Faltam artefatos do treino: {', '.join(faltando)}")
        print("     Rode antes: python src/ml/autoencoder.py")
        return None

    checkpoint = torch.load(pasta / "modelo_autoencoder.pt",
                            map_location="cpu", weights_only=False)
    scaler = carregar_pickle_com_sidecar(pasta / "scaler.pkl")
    device = torch.device("cpu")
    modelo = Autoencoder(checkpoint["n_features"], checkpoint["latente_dim"]).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()

    df = carregar_paderborn_compacto(ARQUIVO_CSV)
    janelas, _ = preparar_janelas_holdout(df, n_max=N_JANELAS_SMD)
    del df
    print(f"  ✅ {len(janelas)} janelas saudáveis | "
          f"{checkpoint['n_features']} features")

    colunas = checkpoint["colunas_feat"]
    residuos_sau = np.vstack([
        _residuo_por_feature(j, modelo, scaler, device, colunas) for j in janelas
    ])
    return {
        "modelo": modelo, "scaler": scaler, "device": device,
        "colunas": colunas, "janelas": janelas, "residuos_sau": residuos_sau,
        "FALHAS": FALHAS, "FUNCOES_FALHA": FUNCOES_FALHA,
        "SEVERIDADES": SEVERIDADES, "pasta": pasta,
        "_residuo": _residuo_por_feature,
    }


def _residuos_da_falha(ctx, fn, sev):
    """Resíduos das janelas com a falha injetada na severidade `sev`."""
    return np.vstack([
        ctx["_residuo"](fn(j.copy(), sev), ctx["modelo"], ctx["scaler"],
                        ctx["device"], ctx["colunas"])
        for j in ctx["janelas"]
    ])


def varrer(ks=K_PADRAO, percentis=PERCENTIL_PADRAO) -> list[dict]:
    ctx = _carregar_contexto()
    if ctx is None:
        return []

    # Resíduos das falhas são caros e NÃO dependem de k nem do percentil —
    # calculados uma vez só, fora da varredura.
    print("  ⏳ Injetando falhas (uma vez; independe de k e do percentil)...")
    residuos_falha = {}
    for falha in ctx["FALHAS"]:
        fid = falha["id"]
        fn = ctx["FUNCOES_FALHA"][fid]
        for sev in ctx["SEVERIDADES"]:
            residuos_falha[(fid, float(sev))] = _residuos_da_falha(ctx, fn, sev)
    print(f"  ✅ {len(residuos_falha)} combinações falha × severidade")

    linhas = []
    for k in ks:
        stats = ajustar_estatistica_residuo(ctx["residuos_sau"])
        s_sau = escore_localizado(ctx["residuos_sau"], stats, k=k)
        for p in percentis:
            inc = incerteza_do_limiar(s_sau, p)
            limiar = inc["limiar"]
            linha = {
                "k": int(k), "percentil": float(p), "limiar": limiar,
                "fp_pct": float((s_sau > limiar).mean() * 100.0),
                "ic_low": inc["ic_low"], "ic_high": inc["ic_high"],
                "ic_rel": inc["largura_relativa"],
            }
            for falha in ctx["FALHAS"]:
                fid = falha["id"]
                # Detecção na severidade MÁXIMA — o teto do que a configuração
                # alcança. A SMD por severidade fica com os macro-códigos.
                r = residuos_falha[(fid, float(max(ctx["SEVERIDADES"])))]
                s = escore_localizado(r, stats, k=k)
                linha[f"rec_{fid}"] = float((s > limiar).mean() * 100.0)
            linhas.append(linha)
    return linhas


def _imprimir(linhas: list[dict]) -> None:
    if not linhas:
        return
    fids = [c[4:] for c in linhas[0] if c.startswith("rec_")]
    cab = (f"{'k':>3} {'perc':>6} {'limiar':>9} {'FP%':>6} {'ic_rel':>7}  "
           + " ".join(f"{f[:10]:>11}" for f in fids))
    print("\n" + cab)
    print("-" * len(cab))
    for ln in sorted(linhas, key=lambda x: (x["fp_pct"], -min(
            x[f"rec_{f}"] for f in fids))):
        alerta = " ⚠️" if ln["ic_rel"] > IC_REL_SUSPEITO else "  "
        print(f"{ln['k']:>3} {ln['percentil']:>6.1f} {ln['limiar']:>9.4f} "
              f"{ln['fp_pct']:>6.2f} {ln['ic_rel']:>7.2f}{alerta}"
              + " ".join(f"{ln[f'rec_{f}']:>10.1f}%" for f in fids))
    print(f"\n  ⚠️ = IC do limiar acima de {IC_REL_SUSPEITO:.0%} do próprio valor: "
          "diferenças finas de FP entre estas linhas podem ser ruído.")
    print("  Ordenado por FP crescente; empate desfeito pela pior detecção.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--k", type=int, nargs="+", default=list(K_PADRAO))
    ap.add_argument("--percentil", type=float, nargs="+",
                    default=list(PERCENTIL_PADRAO))
    args = ap.parse_args()

    print("AL IADO PV — varredura de calibração (k × percentil)")
    linhas = varrer(args.k, args.percentil)
    if not linhas:
        return 1
    _imprimir(linhas)

    from src.core.config import RAIZ_PROJETO

    pasta = Path(RAIZ_PROJETO) / "resultados" / "autoencoder"
    with (pasta / "varredura_calibracao.csv").open("w", newline="",
                                                   encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    (pasta / "varredura_calibracao.json").write_text(
        json.dumps({"evidence_level": "E2", "linhas": linhas},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  📄 {pasta / 'varredura_calibracao.csv'}")
    print("\n  Escolha o par pelo FP E pela detecção — e reporte o FP com o IC")
    print("  ao lado. Um FP menor com IC largo não é melhora demonstrada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
