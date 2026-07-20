# -*- coding: utf-8 -*-
"""
Verificador dos artefatos pós-realinhamento FMECA (rodar após o pipeline).

Uso:  python scripts/verificar_resultados_fmeca.py

Faz DUAS classes de checagem sobre resultados/:

1. ESTRUTURAIS (reprovam ✗): a taxonomia dos artefatos precisa ser a da FMECA
   consolidada (docs/fmeca.md) — ids {contator_ac, igbt, fusivel_ac}, NPR
   sempre = S×O×D (315/90/30; D isolado nunca é NPR), modo/efeito/causa
   preenchidos, ressalvas KS/SMD presentes, nenhum vestígio dos ids antigos
   (lcl/desbalanceamento/sensor) como falha vigente.

2. CONTINUIDADE (informativas ⚠): os mecanismos elétricos foram PRESERVADOS
   no realinhamento, então os parâmetros Weibull devem ficar PRÓXIMOS da era
   anterior sob o MESMO modelo (referência: execução 2026-07 pré-rename):
       igbt        ≈ antigo "lcl"             (β≈3.45, η≈55)
       fusivel_ac  ≈ antigo "desbalanceamento" (β≈2.30, η≈102)
       contator_ac ≈ antigo "sensor"           (β≈4.63, η≈37)
   Como o Autoencoder é RETREINADO na reexecução (código mudou → limiar novo),
   desvios moderados são esperados e NÃO reprovam — o quadro comparativo sai
   no relatório para você julgar. Divergência GROSSEIRA (ex.: ordem invertida
   entre famílias) merece investigação.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_AE = RAIZ / "resultados" / "autoencoder"
PASTA_EXP = RAIZ / "resultados" / "experimentos"
PASTA_CMP = RAIZ / "resultados" / "comparacao"

ESPERADO = {
    "contator_ac": {"s": 5, "o": 7, "d": 9, "npr": 315},
    "igbt":        {"s": 5, "o": 6, "d": 3, "npr": 90},
    "fusivel_ac":  {"s": 5, "o": 3, "d": 2, "npr": 30},
}
IDS_ANTIGOS = {"lcl", "desbalanceamento", "sensor"}
# era anterior (mesmo mecanismo, rótulo antigo) — só para o comparativo
ERA_ANTERIOR = {
    "igbt":        {"beta": 3.45, "eta": 55.0, "mttf": 49.5, "era": "lcl"},
    "fusivel_ac":  {"beta": 2.30, "eta": 102.0, "mttf": 90.4, "era": "desbalanceamento"},
    "contator_ac": {"beta": 4.63, "eta": 37.3, "mttf": 34.1, "era": "sensor"},
}

falhas_estruturais: list[str] = []
avisos: list[str] = []


def _carrega(caminho: Path):
    if not caminho.exists():
        return None
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        falhas_estruturais.append(f"{caminho.name}: JSON ilegível ({exc})")
        return None


def checa_injecao() -> None:
    d = _carrega(PASTA_AE / "injecao_falhas_report.json")
    if d is None:
        avisos.append("injecao_falhas_report.json ausente — rode o pipeline.")
        return
    fam = d.get("falhas", {})
    ids = set(fam)
    if ids != set(ESPERADO):
        falhas_estruturais.append(
            f"injeção: ids {sorted(ids)} ≠ esperado {sorted(ESPERADO)}")
    for fid, esp in ESPERADO.items():
        f = fam.get(fid) or {}
        for k in ("s", "o", "d", "npr"):
            if f.get(k) != esp[k]:
                falhas_estruturais.append(
                    f"injeção[{fid}].{k} = {f.get(k)!r} ≠ {esp[k]}")
        if f.get("npr") == f.get("d"):
            falhas_estruturais.append(f"injeção[{fid}]: NPR igual ao D (proibido)")
        for campo in ("modo_falha", "efeito", "causa"):
            if not (f.get(campo) or "").strip():
                falhas_estruturais.append(f"injeção[{fid}].{campo} vazio")
    smd = d.get("smd", {})
    if set(smd) and set(smd) != set(ESPERADO):
        falhas_estruturais.append(f"injeção: smd com ids {sorted(smd)}")
    velho = ids & IDS_ANTIGOS
    if velho:
        falhas_estruturais.append(f"injeção: ids ANTIGOS presentes: {velho}")
    print("• injeção: ids/NPR/S-O-D/modo-efeito-causa verificados")
    for fid in ESPERADO:
        v = smd.get(fid)
        print(f"    SMD {fid:12s}: {v if v is not None else '— não detectada'}")


def checa_validacao() -> None:
    d = _carrega(PASTA_AE / "validacao_report.json")
    if d is None:
        avisos.append("validacao_report.json ausente — rode o pipeline.")
        return
    casos = [k for k, v in d.items()
             if isinstance(v, dict) and "auc_roc" in v]
    familias = {c.rsplit("_sev", 1)[0] for c in casos}
    if familias - set(ESPERADO):
        extras = familias - set(ESPERADO)
        if extras & IDS_ANTIGOS:
            falhas_estruturais.append(
                f"validação: famílias ANTIGAS presentes: {extras & IDS_ANTIGOS}")
        else:
            avisos.append(f"validação: famílias inesperadas: {extras}")
    if set(ESPERADO) - familias:
        falhas_estruturais.append(
            f"validação: faltam famílias {set(ESPERADO) - familias}")
    print("• validação: famílias e AUC/recall por severidade")
    for c in sorted(casos):
        v = d[c]
        print(f"    {c:26s} AUC={v.get('auc_roc'):.3f}  "
              f"recall={v.get('recall', float('nan')):.3f}")


def checa_weibull() -> None:
    d = _carrega(PASTA_AE / "weibull_results.json")
    if d is None:
        avisos.append("weibull_results.json ausente — rode o pipeline.")
        return
    fam = d.get("falhas", {})
    if set(fam) != set(ESPERADO):
        falhas_estruturais.append(
            f"weibull: ids {sorted(fam)} ≠ esperado {sorted(ESPERADO)}")
        return
    print("• weibull: continuidade com a era anterior (mesmo mecanismo)")
    print(f"    {'família':12s} {'β novo':>7s} {'β era':>7s} "
          f"{'η novo':>8s} {'η era':>8s}  KS")
    for fid, ref in ERA_ANTERIOR.items():
        f = fam.get(fid, {})
        w = f.get("weibull", {})
        beta, eta = w.get("beta"), w.get("eta")
        ks_ok = f.get("ajuste_weibull_adequado")
        ks_txt = "ok" if ks_ok else "rejeitado"
        if beta is None:
            falhas_estruturais.append(f"weibull[{fid}]: sem parâmetros")
            continue
        print(f"    {fid:12s} {beta:7.2f} {ref['beta']:7.2f} "
              f"{eta:8.1f} {ref['eta']:8.1f}  {ks_txt}")
        # continuidade informativa: desvio relativo grande vira aviso
        for nome, novo, era in (("beta", beta, ref["beta"]),
                                ("eta", eta, ref["eta"])):
            if era and abs(novo - era) / era > 0.35:
                avisos.append(
                    f"weibull[{fid}].{nome} = {novo:.2f} difere >35% da era "
                    f"anterior ({era:.2f}; rótulo antigo '{ref['era']}') — "
                    "esperado se o Autoencoder foi retreinado; confira se a "
                    "ORDEM entre famílias segue coerente.")
        if "ressalva_ajuste" not in f:
            falhas_estruturais.append(f"weibull[{fid}]: sem campo de ressalva KS")


def checa_experimentos() -> None:
    achou = False
    for key in ("francisti", "ibrahim"):
        d = _carrega(PASTA_EXP / key / "resultado.json")
        if d is None:
            avisos.append(f"experimento '{key}' sem resultado — rode-o no chat.")
            continue
        achou = True
        modelos = d.get("modelos", {})
        if key == "ibrahim" and "Facebook Prophet" in {
            m for m, v in modelos.items()
            if isinstance(v, dict) and v.get("disponivel")
        }:
            falhas_estruturais.append("ibrahim: Prophet ainda ativo (foi cortado)")
        for nome, m in modelos.items():
            det = (m or {}).get("deteccao_por_falha") if isinstance(m, dict) else None
            if det and (set(det) & IDS_ANTIGOS):
                falhas_estruturais.append(
                    f"{key}/{nome}: detecção por família com ids ANTIGOS")
    if achou:
        print("• experimentos: taxonomia e curadoria verificadas")


def checa_comparacao() -> None:
    d = _carrega(PASTA_CMP / "comparacao_literatura.json")
    if d is None:
        avisos.append("comparação com a literatura ausente — rode "
                      "'compare meu método com a literatura' no chat.")
        return
    fams = set((d.get("auc_por_falha_metodo") or {}))
    if fams & IDS_ANTIGOS:
        falhas_estruturais.append(f"comparação: ids ANTIGOS: {fams & IDS_ANTIGOS}")
    print("• comparação com a literatura: presente e na taxonomia nova")


def main() -> int:
    print("=" * 62)
    print(" VERIFICAÇÃO DOS RESULTADOS — FMECA consolidada (docs/fmeca.md)")
    print("=" * 62)
    checa_injecao()
    checa_validacao()
    checa_weibull()
    checa_experimentos()
    checa_comparacao()

    print("-" * 62)
    for a in avisos:
        print(f"  ⚠ {a}")
    if falhas_estruturais:
        print("\n✗ REPROVADO — problemas estruturais:")
        for f in falhas_estruturais:
            print(f"    ✗ {f}")
        return 1
    algum_artefato = any([
        (PASTA_AE / "injecao_falhas_report.json").exists(),
        (PASTA_AE / "validacao_report.json").exists(),
        (PASTA_AE / "weibull_results.json").exists(),
    ])
    if not algum_artefato:
        print("\n∅ NADA A VERIFICAR — nenhum artefato do pipeline encontrado. "
              "Rode o pipeline primeiro ('rode o pipeline' no chat).")
        return 1
    print("\n✓ APROVADO — artefatos consistentes com a FMECA consolidada."
          "\n  (Avisos ⚠ acima são informativos: continuidade/pendências.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
