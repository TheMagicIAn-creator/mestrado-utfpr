"""
varrer_calibracao.py — Al IAdo PV

Varre `k` (top-k do escore localizado) e reporta a detecção de cada falha da
FMECA num **ponto de operação FIXO** (FPR = 10%), lido da própria ROC.

POR QUE SÓ `k`, E NÃO `k × percentil`
=====================================
A primeira versão varria os dois e ranqueava pela detecção no limiar DE CADA
configuração. Isso é degenerado: baixar o percentil baixa o limiar, o que
sempre aumenta a detecção. Com a coluna de FP removida (ela não é mensurável
com 44 janelas), o topo da tabela era **sempre** o percentil mais permissivo —
a varredura só sabia recomendar mais falso positivo.

O percentil não é propriedade do detector: ele escolhe ONDE sentar na curva
ROC. Quem muda a curva é o `k`, porque muda o escore. Comparar detectores exige
o MESMO ponto de operação — é o que `macro_comum.avaliar_deteccao` já faz com
`corte_fpr = quantile(s_sau, 0.90)`, e é o que se usa aqui.

Resultado: a varredura decide `k` (o que ela pode decidir) e não opina sobre o
percentil (o que ela não pode).

POR QUE ESTE SCRIPT EXISTE
==========================
A auditoria recomendava "varrer k × percentil e escolher o par que traz o FP
para ~1–2% mantendo o recall". A recomendação estava no documento havia
semanas, e **o script nunca existiu** — o bloqueio nunca foi o dataset nem o
torch, era não haver código.

O QUE ESTE SCRIPT **NÃO** FAZ
=============================
Não troca o estimador do limiar. Bootstrap, ajuste paramétrico e EVT foram
testados e rejeitados (ver docstring de `incerteza_do_limiar` e
docs/auditoria_pipeline_ml.md §22): o limite é o TAMANHO DA AMOSTRA, não o
estimador. Este script varre o que de fato muda o detector: o `k`.

Uso (na máquina com o dataset e o modelo treinado):

    python scripts/varrer_calibracao.py
    python scripts/varrer_calibracao.py --k 3 5 8 10

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
)
from src.ml.estatistica import intervalo_wilson  # noqa: E402

K_PADRAO = (5, 10, 15)
# FPR do ponto de operação. MESMO valor de macro_comum.avaliar_deteccao, para a
# varredura e a comparação com a literatura falarem a mesma língua.
FPR_OPERACAO = 0.10


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
    from src.core.seguranca import carregar_pickle_com_sidecar

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


def _residuos_da_falha(ctx, fn, sev, fid):
    """Resíduos das janelas com a falha injetada na severidade `sev`.

    O Contator AC é ruído gaussiano: sem semente POR JANELA, todas recebem a
    mesma realização e a falha de maior NPR (315) acaba medida numa única
    amostra de ruído. `macro_comum.py:97` já passa `seed=20_000 + i`; aqui
    faltava — as janelas eram idênticas entre si.
    """
    saida = []
    for i, j in enumerate(ctx["janelas"]):
        alvo = (fn(j.copy(), sev, seed=20_000 + i) if fid == "contator_ac"
                else fn(j.copy(), sev))
        saida.append(ctx["_residuo"](alvo, ctx["modelo"], ctx["scaler"],
                                     ctx["device"], ctx["colunas"]))
    return np.vstack(saida)


def varrer(ks=K_PADRAO) -> list[dict]:
    ctx = _carregar_contexto()
    if ctx is None:
        return []

    # Só a severidade MÁXIMA é usada. A versão anterior calculava as 7
    # severidades (21 matrizes, ~2100 extrações de feature) e descartava 18.
    sev_max = float(max(ctx["SEVERIDADES"]))
    print(f"  ⏳ Injetando falhas em severidade {sev_max} (não dependem de k)...")
    residuos_falha = {
        falha["id"]: _residuos_da_falha(ctx, ctx["FUNCOES_FALHA"][falha["id"]],
                                        sev_max, falha["id"])
        for falha in ctx["FALHAS"]
    }
    print(f"  ✅ {len(residuos_falha)} falhas")

    # A régua μ/σ não depende de k — calculada uma vez.
    stats = ajustar_estatistica_residuo(ctx["residuos_sau"])

    linhas = []
    for k in ks:
        s_sau = escore_localizado(ctx["residuos_sau"], stats, k=k)
        # Ponto de operação FIXO, lido da ROC do próprio k. É o que torna a
        # comparação entre valores de k justa — e o que impede o ranking de
        # degenerar para "quem tem o limiar mais baixo vence".
        corte = float(np.quantile(s_sau, 1.0 - FPR_OPERACAO))
        linha = {"k": int(k), "corte_fpr": corte, "fpr_alvo": FPR_OPERACAO,
                 "n_saudavel": int(len(s_sau))}
        for falha in ctx["FALHAS"]:
            fid = falha["id"]
            s = escore_localizado(residuos_falha[fid], stats, k=k)
            det = s > corte
            lo, hi = intervalo_wilson(int(det.sum()), len(det))
            linha[f"rec_{fid}"] = float(det.mean() * 100.0)
            linha[f"ic_{fid}"] = (float(lo * 100.0), float(hi * 100.0))
        linhas.append(linha)
    return linhas


def _imprimir(linhas: list[dict]) -> None:
    if not linhas:
        return
    fids = [c[4:] for c in linhas[0] if c.startswith("rec_")]
    n = linhas[0]["n_saudavel"]
    fpr = linhas[0]["fpr_alvo"]

    cab = f"{'k':>3}  " + "  ".join(f"{f[:12]:>22}" for f in fids)
    print(f"\n  Detecção em severidade máxima, FPR fixo em {fpr:.0%} "
          f"(IC95 de Wilson, n={n})\n")
    print("  " + cab)
    print("  " + "-" * len(cab))
    for ln in sorted(linhas, key=lambda x: -min(x[f"rec_{f}"] for f in fids)):
        celulas = []
        for f in fids:
            lo, hi = ln[f"ic_{f}"]
            celulas.append(f"{ln[f'rec_{f}']:>5.1f}% [{lo:>4.0f}–{hi:>3.0f}]")
        print(f"  {ln['k']:>3}  " + "  ".join(f"{c:>22}" for c in celulas))

    print("\n  Ordenado pela PIOR detecção entre as falhas (maior é melhor).")
    print("  O ponto de operação é o MESMO para todo k — lido da ROC de cada")
    print("  um. Sem isso o ranking degenera: limiar menor sempre detecta mais.")
    print("\n  O PERCENTIL não é varrido, e não é omissão: ele escolhe onde")
    print("  sentar na ROC, não muda a curva. Quem muda a curva é o k. O")
    print("  percentil operacional é decidido pelo pipeline, com o alvo de")
    print("  falso positivo verificado em bloco não visto.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compara valores de k do escore localizado em FPR fixo.")
    ap.add_argument("--k", type=int, nargs="+", default=list(K_PADRAO),
                    help="valores de k (top-k) a comparar")
    args = ap.parse_args()

    print("AL IADO PV — varredura de calibração (k × percentil)")
    linhas = varrer(args.k)
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
    print("\n  Escolha o k pela detecção da PIOR falha, checando os IC: com "
          f"{linhas[0]['n_saudavel']} janelas eles são largos, e diferenças")
    print("  pequenas entre valores de k podem não ser reais.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
