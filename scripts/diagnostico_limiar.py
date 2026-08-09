"""
diagnostico_limiar.py — Al IAdo PV

Diagnóstico do ponto de operação do detector. **Não altera o pipeline**: lê os
artefatos vigentes, mede, e escreve um relatório próprio.

POR QUE ESTE SCRIPT EXISTE (origem: PR #94)
===========================================
A PR #94 propôs impor FPR ≤ 1% no bloco de calibração por ordem estatística. Ela
mediu o custo e o declarou honestamente: em severidade 1,0 o recall cairia para
0,825 (contator), 0,025 (IGBT) e 0,025 (fusível). Isso zeraria a Weibull por
censura e faria o NPR projetado voltar a ser igual ao oficial. A proposta NÃO
foi adotada.

Mas ela produziu dois achados que valem, e que este script preserva:

1. **O regime de F0 muda entre calibração e teste.** Na execução de referência,
   a mediana de F0 vai de ~51 Hz (calibração) para ~100 Hz (teste) — dezenas de
   IQRs de distância. O FPR de 10% no bloco de teste **não é limiar mal
   calibrado; é mudança de regime operacional.** Subir o corte até zerar esses
   alarmes trata um problema de cobertura de dados com um martelo de limiar, e
   é por isso que o recall desaba junto.

2. **O alvo de FP declarado não é imposto.** `AL_IADO_ESCORE_FP_ALVO=1.0`
   escolhe entre cinco percentis candidatos e, se nenhum atinge o alvo, aceita
   o mais conservador **mesmo violando o alvo**. É o que ocorre hoje: p99,9
   escolhido, FPR de 10,2% no teste. O número na configuração não descreve o
   que o sistema faz.

O QUE ESTE SCRIPT **NÃO** FAZ
=============================
Não muda limiar, não regrava `limiar.json`, não toca em nenhum artefato do
pipeline. Ele calcula o que o corte estrito **seria** e quanto custaria — para
que a decisão de não adotá-lo seja verificável em vez de recordada.

Também não vive dentro de `src/ml/`: `escore_anomalia.py` é dependência de
proveniência de três etapas, e acrescentar código lá marcaria autoencoder,
injeção e validação como `stale`, pedindo retreino por um diagnóstico. O custo
não se justifica.

Uso (na máquina com o dataset e o modelo treinado):

    python scripts/diagnostico_limiar.py
    python scripts/diagnostico_limiar.py --alvo 1.0

Saída: relatório no terminal + `resultados/autoencoder/diagnostico_limiar.json`.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.utils import configurar_saida_utf8  # noqa: E402

configurar_saida_utf8()

ALVO_PADRAO_PCT = 1.0
# Deslocamento de mediana, em IQRs do bloco de referência, a partir do qual o
# regime é considerado outro. 1,5 IQR é o critério clássico de outlier de
# Tukey; aqui aplicado à MEDIANA de um bloco inteiro, não a um ponto.
LIMITE_DRIFT_IQR = 1.5
# O teto da busca de F0 NÃO é derivado aqui. Até 09/08/2026 este script
# calculava `F0 + 40 Hz`, replicando a meia-largura que `estimar_f0` usava
# quando a busca era [20, 100] Hz. Depois da PR #107 a faixa é declarada
# explicitamente em `features_ca.F0_MIN/F0_MAX`, e recalcular o teto aqui
# apontaria para 100 Hz enquanto o estimador já vai até 115 — o diagnóstico de
# saturação mediria contra uma régua que não existe mais. Fonte única: o módulo.


# ============================================================
# Funções puras — testáveis sem torch e sem o dataset
# ============================================================

def limiar_fpr_maximo(scores, alvo_pct: float = ALVO_PADRAO_PCT) -> dict:
    """Corte por ordem estatística que limita o FPR empírico ao alvo.

    A decisão do pipeline é ``score > limiar``. Para ``n`` escores saudáveis,
    permite no máximo ``floor(n · alvo/100)`` acima do corte.

    O retorno separa duas ideias que não podem ser confundidas — é o ponto que
    a PR #94 acertou e que precisa sobreviver:

    - ``fpr_observado_pct``: restrição empírica, garantida NO BLOCO usado;
    - ``resolucao_amostral_pct``: menor FPR não nulo que a amostra consegue
      medir (``100/n``). Se ela for MAIOR que o alvo, zero excedências
      **não certifica** o alvo fora da amostra. Com 91 janelas, 1/91 = 1,10%:
      "no máximo 1%" só pode significar "zero eventos observados".
    """
    alvo = float(alvo_pct)
    if not 0.0 <= alvo < 100.0:
        raise ValueError("alvo_pct deve estar em [0, 100)")
    s = np.asarray(scores, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("scores não pode ser vazio")
    if not np.isfinite(s).all():
        raise ValueError("scores deve conter apenas valores finitos")

    n = int(s.size)
    max_exc = min(int(np.floor(n * alvo / 100.0 + 1e-12)), n - 1)
    corte = float(np.sort(s)[n - max_exc - 1])
    excedencias = int(np.count_nonzero(s > corte))
    resolucao = 100.0 / n
    return {
        "limiar": corte,
        "alvo_pct": alvo,
        "n": n,
        "max_excedencias": max_exc,
        "excedencias_observadas": excedencias,
        "fpr_observado_pct": float(excedencias / n * 100.0),
        "percentil_efetivo": float((n - max_exc) / n * 100.0),
        "resolucao_amostral_pct": float(resolucao),
        "alvo_resolvivel_na_amostra": bool(resolucao <= alvo),
    }


def resumo_regime(valores) -> dict:
    """Mediana, IQR e extremos de um bloco — descreve o regime operacional."""
    v = np.asarray(valores, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError("valores não pode ser vazio")
    q25, mediana, q75 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    return {
        "n": int(v.size), "mediana": mediana, "q25": q25, "q75": q75,
        "iqr": q75 - q25, "min": float(v.min()), "max": float(v.max()),
    }


def deslocamento_iqr(referencia: dict, alvo: dict) -> float:
    """Distância entre medianas, medida em IQRs do bloco de REFERÊNCIA.

    Em IQRs, e não em Hz, porque o que importa não é o tamanho absoluto do
    salto: é se ele é grande comparado à dispersão que o bloco de referência
    ensinou ao modelo. IQR (e não desvio-padrão) por ser robusto a cauda.
    """
    iqr = max(float(referencia["iqr"]), 1e-9)
    return abs(float(alvo["mediana"]) - float(referencia["mediana"])) / iqr


def alvo_foi_atingido(fpr_observado_pct: float, alvo_pct: float) -> bool:
    """O alvo declarado descreve o sistema, ou é só aspiração?"""
    return float(fpr_observado_pct) <= float(alvo_pct) + 1e-12


def fracao_no_teto(valores, teto: float, tolerancia: float = 1.0) -> float:
    """Fração de estimativas encostadas no teto da busca — sinal de saturação.

    ``features_ca.estimar_f0`` procura a fundamental em
    ``[F0 − faixa_hz, F0 + faixa_hz] = [20, 100] Hz`` (F0 nominal = 60 Hz, da
    rede brasileira). O Paderborn, porém, é um acionamento de motor de
    velocidade variável: a fundamental acompanha a rotação e passa de 100 Hz.

    Quando a fundamental verdadeira está ACIMA do teto, o estimador não pode
    devolvê-la — devolve o teto (a interpolação parabólica explica os poucos
    décimos acima). Uma MEDIANA de bloco pousada no teto não é um regime
    operacional: é estimador saturado. Esta função mede quanto disso há.
    """
    v = np.asarray(valores, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError("valores não pode ser vazio")
    return float(np.count_nonzero(v >= float(teto) - float(tolerancia)) / v.size)


# ============================================================
# Execução — precisa do modelo treinado e do dataset
# ============================================================

def _contexto():
    """Reusa a carga de `varrer_calibracao` em vez de duplicá-la."""
    from scripts.varrer_calibracao import _carregar_contexto, _residuos_da_falha

    ctx = _carregar_contexto()
    if ctx is not None:
        ctx["_residuos_da_falha"] = _residuos_da_falha
    return ctx


def _regime_por_bloco(pasta: Path) -> dict | None:
    """F0 por bloco temporal, com o MESMO split do autoencoder."""
    import pandas as pd

    from src.core.config import RAIZ_PROJETO
    from src.ml.split_temporal import split_padrao_paderborn

    arq = Path(RAIZ_PROJETO) / "dados" / "processados" / "features_paderborn.parquet"
    if not arq.exists():
        return None
    df = pd.read_parquet(arq)
    coluna = next((c for c in df.columns if c.lower() in
                   {"f0_estimado", "f0", "freq_fundamental"}), None)
    if coluna is None:
        return None

    from src.ml.features_ca import F0_MAX

    teto = float(F0_MAX)
    # O MESMO split do pipeline. Reconstruí-lo aqui com outra função foi o que
    # permitiu os dois bugs anteriores; agora vem da fonte única.
    split = split_padrao_paderborn(len(df))
    blocos, saturacao = {}, {}
    for nome, chave in (("treino", "treino"), ("calibracao", "val"),
                        ("teste", "teste")):
        valores = df.iloc[np.asarray(split[chave], int)][coluna]
        blocos[nome] = resumo_regime(valores)
        saturacao[nome] = fracao_no_teto(valores, teto)
    desloc = deslocamento_iqr(blocos["calibracao"], blocos["teste"])
    return {
        "coluna": coluna, "blocos": blocos,
        "deslocamento_calib_teste_iqr": float(desloc),
        "regimes_distintos": bool(desloc > LIMITE_DRIFT_IQR),
        "teto_busca_f0_hz": teto,
        "fracao_saturada": saturacao,
        "estimador_saturado": bool(max(saturacao.values()) > 0.10),
    }


def executar(alvo_pct: float = ALVO_PADRAO_PCT) -> dict | None:
    from src.core.config import RAIZ_PROJETO
    from src.ml.escore_anomalia import (
        ajustar_estatistica_residuo,
        escore_localizado,
    )

    pasta = Path(RAIZ_PROJETO) / "resultados" / "autoencoder"
    arq_limiar = pasta / "limiar.json"
    if not arq_limiar.exists():
        print(f"  ❌ Não encontrado: {arq_limiar}")
        print("     Rode antes: python src/ml/autoencoder.py")
        return None
    vigente = json.loads(arq_limiar.read_text(encoding="utf-8"))

    ctx = _contexto()
    if ctx is None:
        return None

    stats = ajustar_estatistica_residuo(ctx["residuos_sau"])
    s_sau = escore_localizado(ctx["residuos_sau"], stats, k=5)
    estrito = limiar_fpr_maximo(s_sau, alvo_pct)

    limiar_vigente = float(vigente.get("limiar_localizado", vigente["limiar"]))
    fpr_vigente = float((s_sau > limiar_vigente).mean() * 100.0)

    # Custo do corte estrito: recall por falha em severidade máxima.
    sev_max = float(max(ctx["SEVERIDADES"]))
    custo = {}
    for falha in ctx["FALHAS"]:
        fid = falha["id"]
        R = ctx["_residuos_da_falha"](
            ctx, ctx["FUNCOES_FALHA"][fid], sev_max, fid)
        s = escore_localizado(R, stats, k=5)
        custo[fid] = {
            "nome": falha["nome"], "npr": falha["npr"],
            "recall_vigente": float((s > limiar_vigente).mean()),
            "recall_estrito": float((s > estrito["limiar"]).mean()),
        }

    resultado = {
        "evidence_level": "E2",
        "severidade_referencia": sev_max,
        "alvo_declarado_pct": float(vigente.get("threshold_target_fpr_pct")
                                    or alvo_pct),
        "limiar_vigente": limiar_vigente,
        "percentil_vigente": vigente.get("percentil_limiar"),
        "fpr_vigente_holdout_pct": fpr_vigente,
        "fpr_vigente_teste_pct": vigente.get("fp_test_pct"),
        "alvo_atingido": alvo_foi_atingido(
            float(vigente.get("fp_test_pct") or fpr_vigente), alvo_pct),
        "corte_estrito": estrito,
        "custo_do_corte_estrito": custo,
        "regime_f0": _regime_por_bloco(pasta),
        "nota": (
            "Diagnóstico. NÃO altera limiar.json nem nenhum artefato do "
            "pipeline. O corte estrito é contrafactual: mostra o que seria, "
            "não o que é."
        ),
    }
    (pasta / "diagnostico_limiar.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def _imprimir(r: dict) -> None:
    alvo = r["alvo_declarado_pct"]
    print("\n  ── Ponto de operação vigente ─────────────────────────────")
    print(f"  limiar {r['limiar_vigente']:.4f} (percentil {r['percentil_vigente']})")
    print(f"  FPR no teste: {r['fpr_vigente_teste_pct']:.2f}%  |  alvo declarado: {alvo:.2f}%")
    if not r["alvo_atingido"]:
        print(f"  ⚠️  O alvo de {alvo:.2f}% NÃO é imposto — a busca aceita o")
        print("      percentil mais conservador mesmo violando o alvo.")

    e = r["corte_estrito"]
    print("\n  ── Se o alvo fosse imposto (contrafactual) ───────────────")
    print(f"  corte {e['limiar']:.4f} (percentil {e['percentil_efetivo']:.1f})")
    print(f"  resolução amostral: {e['resolucao_amostral_pct']:.2f}% com n={e['n']}")
    if not e["alvo_resolvivel_na_amostra"]:
        print(f"      → alvo de {alvo:.2f}% está ABAIXO da resolução: zero")
        print("        excedências não certifica a taxa, só a observa.")
    print(f"\n  {'falha':<14}{'recall vigente':>16}{'recall estrito':>16}")
    for fid, c in r["custo_do_corte_estrito"].items():
        print(f"  {fid:<14}{c['recall_vigente']:>15.3f} {c['recall_estrito']:>15.3f}")

    reg = r.get("regime_f0")
    if reg:
        print("\n  ── Regime de F0 por bloco ────────────────────────────────")
        for nome, b in reg["blocos"].items():
            print(f"  {nome:<12} mediana {b['mediana']:8.2f} Hz  "
                  f"IQR {b['iqr']:7.2f}  n={b['n']}")
        print(f"  deslocamento calibração→teste: "
              f"{reg['deslocamento_calib_teste_iqr']:.1f} IQR")
        if reg["regimes_distintos"]:
            print("  ⚠️  Blocos em REGIMES DISTINTOS. O FPR do teste mede")
            print("      cobertura de dados, não erro de calibração — e é por")
            print("      isso que apertar o limiar custa recall em vez de")
            print("      corrigir o problema.")
        teto = reg["teto_busca_f0_hz"]
        print(f"\n  Teto da busca de F0: {teto:.0f} Hz  "
              "(features_ca.estimar_f0)")
        for nome, frac in reg["fracao_saturada"].items():
            print(f"    {nome:<12} {frac*100:5.1f}% das janelas no teto")
        if reg["estimador_saturado"]:
            print("  ⛔ ESTIMADOR SATURADO. F0 encostado no teto não é regime:")
            print("     é a busca não alcançando a fundamental verdadeira. O")
            print("     Paderborn é acionamento de VELOCIDADE VARIÁVEL e passa")
            print("     de 100 Hz; a faixa foi dimensionada para rede de 60 Hz.")
            print("     Com F0 errado, TODA feature harmônica e a THD saem")
            print("     erradas — e o erro de reconstrução sobe sem falha")
            print("     alguma. Ver docs/auditoria_parametros.md §1.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnostica o ponto de operação; não altera nada.")
    ap.add_argument("--alvo", type=float, default=ALVO_PADRAO_PCT,
                    help=f"alvo de FPR em %% (padrão: {ALVO_PADRAO_PCT})")
    args = ap.parse_args()

    print("AL IADO PV — diagnóstico do limiar operacional")
    r = executar(args.alvo)
    if r is None:
        return 1
    _imprimir(r)
    from src.core.config import RAIZ_PROJETO
    print(f"\n  📄 {Path(RAIZ_PROJETO) / 'resultados/autoencoder/diagnostico_limiar.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
