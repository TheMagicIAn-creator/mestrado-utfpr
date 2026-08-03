"""
retroalimentacao_fmeca.py — Al IAdo PV

Converte a **detectabilidade medida** do monitoramento proposto no índice de
detecção da FMECA e calcula o **NPR projetado**. Fecha o ciclo do RCM: o TCC
julgou o D por literatura; a dissertação mede.

NOMENCLATURA (fonte única: docs/nomenclatura_deteccao.md)
=========================================================
- ``D_campo``     — índice FMECA de dificuldade de detecção EM CAMPO, julgado
                    (Torres, 2024, Tab. 4.8). Ordinal 1–10, maior = pior.
- ``POD_mon(s)``  — probabilidade de detecção pelo MONITORAMENTO proposto na
                    severidade ``s``, medida sob E2 no limiar operacional
                    congelado. [0, 1], maior = melhor. Raiz consagrada:
                    MIL-HDBK-1823A (curva POD dos ensaios não destrutivos).
                    O subscrito é OBRIGATÓRIO: em sistemas de potência, POD nu
                    é *Power Oscillation Damping*.
- ``D_mon``       — o mesmo índice da Tab. 4.8, obtido de ``1 − POD_mon``.
- ``D_proj``      — ``min(D_campo, D_mon)``; ver "A emenda min" abaixo.

POR QUE A CONVERSÃO NÃO É UMA RÉGUA QUE NÓS ESCOLHEMOS
======================================================
A objeção que travava a retroalimentação era a circularidade: se as faixas de
conversão fossem escolhidas depois de ver os resultados, a régua teria sido
calibrada para o resultado desejado.

Ela não se aplica. A Tab. 4.8 do TCC define o índice em **percentual de NÃO
detectar** (D=1 → 0–5%; D=10 → 86–100%). Como ``1 − POD_mon`` é exatamente essa
grandeza, a conversão é a LEITURA DA ESCALA — publicada em 2024, antes de
qualquer medição deste projeto. Não há régua a congelar porque não há régua a
escolher.

A EMENDA ``min``
================
O monitoramento proposto é **adicional** ao que já existe em campo. Acrescentar
um detector não torna nenhuma falha mais difícil de detectar; logo o índice só
pode melhorar (diminuir) ou ficar igual. Sem essa emenda, um componente bem
tratado pelo detector poderia sair MAIS crítico do que antes — o oposto do que
retroalimentação deveria produzir. O caso "falha não detectada em severidade
alguma → manter o D original" é caso particular do ``min``, e não exceção.

EVIDÊNCIA
=========
Tudo aqui herda **E2** (validação sintética orientada pela FMECA). O NPR
resultante é "NPR projetado sob validação sintética", NUNCA NPR de campo (E3).
A FMECA oficial permanece a de ``docs/fmeca.md``; isto é análise de
sensibilidade.

Uso:
    python -m src.ml.retroalimentacao_fmeca
    python -m src.ml.retroalimentacao_fmeca --severidade 0.5

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

# ============================================================
# ESCALA Tab. 4.8 — bordas SUPERIORES de "% de não detectar"
# ============================================================
# Lookup por borda superior, e não por par (lo, hi), de propósito: a Tab. 4.8 é
# escrita em percentuais inteiros (0–5, 6–15, ...), o que deixa buracos para
# valores fracionários — 5,5% não cai em faixa nenhuma. Com bordas superiores a
# cobertura é contínua e o comportamento nos limites fica explícito.
#
# ⚠️  A CONFERIR na Tab. 4.8 do TCC. docs/fmeca.md registra apenas os extremos
#     (D=1 → 0–5%; D=10 → 86–100%). As oito faixas intermediárias são
#     RECONSTRUÇÃO ARITMÉTICA forçada por esses extremos: 80 pontos percentuais
#     (6–85) divididos em 8 faixas de 10. Se a tabela do TCC usar outras
#     faixas, é ESTA constante que muda — e só ela.
BORDAS_D: tuple[tuple[int, float], ...] = (
    (1, 5.0), (2, 15.0), (3, 25.0), (4, 35.0), (5, 45.0),
    (6, 55.0), (7, 65.0), (8, 75.0), (9, 85.0), (10, 100.0),
)

FONTE_ESCALA = "Torres (2024), TCC, Tab. 4.8 — índice D em % de não detectar"

# Severidade de referência quando um ESCALAR é inevitável. É a assinatura
# incipiente plenamente desenvolvida — a mais próxima do modo terminal que a
# FMECA classifica. Declarada, nunca implícita; a curva completa acompanha.
SEVERIDADE_REFERENCIA = 1.0

PASTA_PADRAO = Path("resultados") / "autoencoder"
ARQUIVO_VALIDACAO = "validacao_report.json"

# Fonte única dos índices S/O/D_campo. Lida por AST, sem importar o módulo.
FONTE_FALHAS = Path("src") / "ml" / "injecao_falhas.py"
_CAMPOS_FMECA = ("id", "nome", "componente", "s", "o", "d")


def falhas_da_fmeca() -> list[dict]:
    """Lê ``FALHAS`` de ``injecao_falhas.py`` SEM importar o módulo.

    Aquele arquivo importa ``torch`` no topo, então importá-lo arrastaria o
    pipeline pesado só para ler três linhas de tabela — e esta conversão é
    aritmética pura sobre um JSON. Ler por AST mantém a fonte única (nada é
    duplicado aqui) e deixa a ponte FMECA↔detector verificável na CI, que roda
    sem torch. É o mesmo padrão de ``pipeline._parametros_do_fonte``.

    ``ast.literal_eval`` não avalia o dicionário inteiro porque campos como
    ``"npr": 5 * 7 * 9`` são expressões, não literais. Extraímos campo a campo
    e **recalculamos** o NPR — que é a definição, não um valor a copiar.
    """
    from src.core.config import RAIZ_PROJETO

    caminho = Path(RAIZ_PROJETO) / FONTE_FALHAS
    arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    for node in arvore.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "FALHAS" for t in node.targets):
            continue
        if not isinstance(node.value, ast.List):
            break
        falhas = []
        for item in node.value.elts:
            if not isinstance(item, ast.Dict):
                continue
            registro = {}
            for chave, valor in zip(item.keys, item.values):
                if not (isinstance(chave, ast.Constant) and chave.value in _CAMPOS_FMECA):
                    continue
                try:
                    registro[chave.value] = ast.literal_eval(valor)
                except (ValueError, TypeError, SyntaxError):
                    pass
            if {"id", "s", "o", "d"} <= set(registro):
                falhas.append(registro)
        if falhas:
            return falhas
        break
    raise RuntimeError(
        f"não foi possível ler FALHAS de {caminho}. A FMECA é fonte única "
        "(docs/fmeca.md → src/ml/injecao_falhas.py); sem ela não há tabela."
    )


def indice_d(fracao_nao_deteccao: float) -> int:
    """Índice da Tab. 4.8 para uma fração de NÃO detecção em [0, 1].

    >>> indice_d(0.0), indice_d(0.15), indice_d(1.0)
    (1, 2, 10)

    O arredondamento em 6 casas não é preciosismo: ``1 - 0.85`` vale
    ``0.15000000000000002`` em ponto flutuante, o que cairia fora da faixa
    ``≤ 15%`` e devolveria D=3 em vez de D=2.
    """
    if not 0.0 <= fracao_nao_deteccao <= 1.0:
        raise ValueError(
            f"fração de não detecção fora de [0, 1]: {fracao_nao_deteccao!r}. "
            "Esperado 1 − POD_mon."
        )
    pct = round(fracao_nao_deteccao * 100.0, 6)
    for d, borda in BORDAS_D:
        if pct <= borda:
            return d
    return BORDAS_D[-1][0]  # inalcançável: a última borda é 100.0


def d_projetado(d_campo: int, d_mon: int) -> int:
    """``min(D_campo, D_mon)`` — o monitoramento é ADICIONAL ao que há em campo.

    Nunca piora o índice. Ver "A emenda min" no cabeçalho do módulo.
    """
    return int(min(d_campo, d_mon))


def carregar_pod_mon(caminho: Path | str) -> dict:
    """Lê POD_mon por falha e severidade de ``validacao_report.json``.

    Devolve ``{falha_id: {severidade: {"pod": float, "ic": (lo, hi)}}}``.
    O recall do relatório de validação É o POD_mon: mesma quantidade, lida como
    propriedade do método de inspeção em vez de métrica de classificador.
    """
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    meta = dados.get("__meta__", {})
    curvas: dict[str, dict[float, dict]] = {}
    for chave, valores in dados.items():
        if chave == "__meta__" or "_sev" not in chave:
            continue
        falha_id, _, sev_txt = chave.rpartition("_sev")
        try:
            sev = float(sev_txt)
        except ValueError:
            continue
        curvas.setdefault(falha_id, {})[sev] = {
            "pod": float(valores["recall"]),
            "ic": (float(valores.get("recall_ci_low", float("nan"))),
                   float(valores.get("recall_ci_high", float("nan")))),
        }
    return {"curvas": curvas, "meta": meta}


def percentil_efetivo(caminho_validacao: Path | str) -> float | None:
    """Percentil REALMENTE usado no limiar, lido de ``limiar.json`` ao lado.

    ``validacao_report.json`` só carrega ``threshold_method = "p99"``, que é um
    RÓTULO DE MÉTODO — não o percentil. Com a auto-calibração ligada
    (``percentil_auto``), o valor escolhido foi 99,9. Imprimir "(p99)" numa
    tabela destinada à dissertação seria enganoso: são pontos de operação
    diferentes, e a divergência já está documentada (PR #83).

    Devolve ``None`` se ``limiar.json`` não estiver ao lado — o rótulo de
    método continua sendo mostrado, sem inventar precisão que não se tem.
    """
    arq = Path(caminho_validacao).parent / "limiar.json"
    if not arq.exists():
        return None
    try:
        dados = json.loads(arq.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    valor = dados.get("percentil_limiar")
    return float(valor) if isinstance(valor, (int, float)) else None


def tabela_retroalimentacao(caminho_validacao: Path | str,
                            severidade: float = SEVERIDADE_REFERENCIA) -> dict:
    """Monta a tabela de NPR projetado a partir dos artefatos vigentes."""
    lido = carregar_pod_mon(caminho_validacao)
    curvas, meta = lido["curvas"], lido["meta"]
    percentil = percentil_efetivo(caminho_validacao)

    linhas = []
    for falha in falhas_da_fmeca():
        fid = falha["id"]
        curva = curvas.get(fid, {})
        ponto = curva.get(severidade)
        if ponto is None:
            raise KeyError(
                f"severidade {severidade} não consta em {caminho_validacao} "
                f"para '{fid}'. Disponíveis: {sorted(curva)}"
            )
        pod = ponto["pod"]
        d_campo = int(falha["d"])
        d_mon = indice_d(1.0 - pod)
        d_proj = d_projetado(d_campo, d_mon)
        s, o = int(falha["s"]), int(falha["o"])
        linhas.append({
            "id": fid,
            "componente": falha.get("componente") or falha.get("nome") or fid,
            "s": s, "o": o,
            "d_campo": d_campo,
            "npr_oficial": s * o * d_campo,
            "pod_mon": pod,
            "pod_mon_ic": list(ponto["ic"]),
            "nao_deteccao_pct": round((1.0 - pod) * 100.0, 6),
            "d_mon": d_mon,
            "d_projetado": d_proj,
            "npr_projetado": s * o * d_proj,
            "curva_pod_mon": {str(k): v["pod"] for k, v in sorted(curva.items())},
        })

    ordem_oficial = [x["id"] for x in sorted(linhas, key=lambda r: -r["npr_oficial"])]
    ordem_projetada = [x["id"] for x in sorted(linhas, key=lambda r: -r["npr_projetado"])]
    return {
        "evidence_level": "E2",
        "evidence_note": (
            "NPR projetado sob validação sintética orientada pela FMECA (E2). "
            "NÃO é NPR de campo (E3). A FMECA oficial permanece docs/fmeca.md; "
            "isto é análise de sensibilidade."
        ),
        "severidade_referencia": severidade,
        "fonte_escala_d": FONTE_ESCALA,
        "escala_d_a_conferir": (
            "Faixas intermediárias reconstruídas dos extremos registrados em "
            "docs/fmeca.md; conferir na Tab. 4.8 do TCC."
        ),
        "regra": "D_proj = min(D_campo, D_mon); S e O inalterados",
        "limiar_operacional": meta.get("limiar_operacional"),
        "threshold_method": meta.get("threshold_method"),
        "percentil_efetivo": percentil,
        "ordem_oficial": ordem_oficial,
        "ordem_projetada": ordem_projetada,
        "ordem_inverte": ordem_oficial != ordem_projetada,
        "linhas": linhas,
    }


def formatar_markdown(resultado: dict) -> str:
    """Tabela legível para colar na dissertação, com as ressalvas junto."""
    sev = resultado["severidade_referencia"]
    # O percentil EFETIVO, não o rótulo de método: `threshold_method` diz "p99"
    # enquanto a auto-calibração escolheu 99,9. Ver `percentil_efetivo`.
    pct = resultado.get("percentil_efetivo")
    ponto = (f"percentil {pct:g} (auto-calibrado)" if pct is not None
             else f"método {resultado.get('threshold_method')}")
    linhas = [
        "# Retroalimentação da FMECA — NPR projetado (E2)",
        "",
        (f"Severidade de referência: **{sev}** · "
         f"Limiar operacional: **{resultado.get('limiar_operacional')}** — "
         f"{ponto}"),
        "",
        ("| Componente | S | O | D_campo | NPR oficial | POD_mon | "
         "não detecta | D_mon | D_proj | NPR projetado |"),
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in resultado["linhas"]:
        linhas.append(
            f"| {r['componente']} | {r['s']} | {r['o']} | {r['d_campo']} | "
            f"**{r['npr_oficial']}** | {r['pod_mon']:.3f} | "
            f"{r['nao_deteccao_pct']:.1f}% | {r['d_mon']} | {r['d_projetado']} | "
            f"**{r['npr_projetado']}** |"
        )
    linhas += [
        "",
        f"Ordem oficial: {' > '.join(resultado['ordem_oficial'])}",
        f"Ordem projetada: {' > '.join(resultado['ordem_projetada'])}",
        "",
    ]
    if resultado["ordem_inverte"]:
        linhas += [
            ("> **A ordem de criticidade inverte.** Não é artefato: o "
             "monitoramento entrega mais onde a detecção em campo era pior. "
             "Um componente cujo NPR era carregado por D_campo alto cai muito; "
             "um cuja criticidade vem de S×O quase não se move — e passa à "
             "frente."),
            "",
        ]
    linhas += [
        ("> Evidência **E2**. NPR projetado sob validação sintética, não NPR "
         "de campo. A FMECA oficial permanece `docs/fmeca.md`."),
        "",
        ("> As faixas intermediárias da escala D são reconstrução aritmética "
         "— conferir na Tab. 4.8 do TCC (ver `docs/nomenclatura_deteccao.md`)."),
    ]
    return "\n".join(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="NPR projetado a partir do POD_mon medido (E2).")
    ap.add_argument("--severidade", type=float, default=SEVERIDADE_REFERENCIA,
                    help=f"severidade de referência (padrão: {SEVERIDADE_REFERENCIA})")
    ap.add_argument("--pasta", type=Path, default=None,
                    help="pasta dos artefatos do Autoencoder")
    args = ap.parse_args()

    from src.core.config import RAIZ_PROJETO

    pasta = args.pasta or (Path(RAIZ_PROJETO) / PASTA_PADRAO)
    origem = pasta / ARQUIVO_VALIDACAO
    if not origem.exists():
        print(f"  ❌ Não encontrado: {origem}")
        print("\n     Este artefato é VERSIONADO no Git. Se ele sumiu da sua")
        print("     cópia local, restaure — não recalcule:")
        print(f"\n         git restore {PASTA_PADRAO.as_posix()}/{ARQUIVO_VALIDACAO}")
        print("\n     Rodar `python src/ml/validacao.py` sobrescreveria o")
        print("     artefato com uma nova execução, mudando números já")
        print("     publicados. Só faça isso se o Autoencoder foi retreinado.")
        return 1

    resultado = tabela_retroalimentacao(origem, args.severidade)
    (pasta / "retroalimentacao_fmeca.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    md = formatar_markdown(resultado)
    (pasta / "retroalimentacao_fmeca.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n  📄 {pasta / 'retroalimentacao_fmeca.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
