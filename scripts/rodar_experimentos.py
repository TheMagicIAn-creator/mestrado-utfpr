"""
rodar_experimentos.py — Al IAdo PV / runner CLI dos experimentos por artigo.

Dá aos experimentos por artigo (Ghoneim, Francisti, Ibrahim, Sharma, Ahirwar)
a MESMA ergonomia de linha de comando que o `python src/ml/autoencoder.py`
tem para o pipeline principal: rodar standalone, ver a tabela de métricas, o
protocolo de decisão usado e onde o resultado foi salvo — sem abrir o app.

IMPORTANTE — isto NÃO muda a metodologia nem os números:
  - cada experimento JÁ tem o seu protocolo próprio em
    src/ml/protocolos_artigos.py (Shewhart, p99 em calibração, banda do
    congelado; voto majoritário);
  - este arquivo é só um ATALHO de execução/reprodutibilidade. O resultado é
    idêntico ao do chat ("rode o experimento do Ghoneim") ou do
    `executar_experimento(key)`.

Uso:
  python scripts/rodar_experimentos.py                 # lista os experimentos
  python scripts/rodar_experimentos.py francisti       # roda um
  python scripts/rodar_experimentos.py ibrahim sharma  # roda vários
  python scripts/rodar_experimentos.py --todos         # roda todos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite execução direta (python scripts/rodar_experimentos.py) e via -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.logs import habilitar_console  # noqa: E402

habilitar_console()  # UTF-8 seguro + eco de progresso no terminal


def _fmt(v) -> str:
    """Formata número como 3 casas, ou '-' se ausente."""
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def _imprimir_resultado(res: dict) -> None:
    from src.core.logs import limpar_simbolos as _L

    ref = res.get("referencia", res.get("experimento", "?"))
    print("\n" + "=" * 72)
    print(_L(f"{ref} — {res.get('dataset', '?')} ({res.get('tarefa', '?')})"))
    print("=" * 72)

    if not res.get("ok"):
        print("  " + _L(str(res.get("mensagem", "experimento não executável"))))
        return

    modelos = res.get("modelos", {})
    cab = f"{'Modelo':<26} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7}  Decisão"
    print(cab)
    print("-" * len(cab))
    for nome, m in modelos.items():
        if not m.get("disponivel", True):
            print(f"{nome:<26} {'—':>7} {'—':>7} {'—':>7} {'—':>7}  "
                  + _L(f"INDISPONÍVEL: {m.get('motivo', '')}"))
            continue
        decisao = m.get("threshold_source", "—")
        print(f"{nome:<26} {_fmt(m.get('f1')):>7} {_fmt(m.get('auc')):>7} "
              f"{_fmt(m.get('precision')):>7} {_fmt(m.get('recall')):>7}  {decisao}")

    melhor = res.get("melhor_modelo")
    if melhor and melhor != "-":
        print(f"\n  Melhor por {res.get('metrica_principal', 'f1')}: "
              f"{melhor} ({_fmt(res.get('melhor_valor'))})")

    met = res.get("metodologia")
    if met:
        print("\n  Protocolo:", met.get("protocolo", "—"))
        sp = met.get("split", {})
        if sp:
            print(f"    Split: {sp.get('tipo')} (purga={sp.get('purga_janelas')}) "
                  f"treino={sp.get('treino')} teste={sp.get('teste')}")
        inj = met.get("injecao", {})
        if inj:
            print(f"    Injeção: {inj.get('tipo')} — famílias "
                  f"{', '.join(inj.get('falhas', []))}")
        for nome, regra in (met.get("decisoes") or {}).items():
            print(_L(f"    Decisão [{nome}]: {regra}"))

    # detecção por família de falha (quando o protocolo reporta)
    linhas_falha = {
        nome: m["deteccao_por_falha"]
        for nome, m in modelos.items()
        if isinstance(m, dict) and m.get("deteccao_por_falha")
    }
    if linhas_falha:
        print("\n  Detecção por família de falha (recall):")
        familias = sorted({f for d in linhas_falha.values() for f in d})
        print("    " + f"{'Modelo':<26} " + " ".join(f"{f:>16}" for f in familias))
        for nome, d in linhas_falha.items():
            print("    " + f"{nome:<26} "
                  + " ".join(f"{d.get(f, 0.0):>15.0%} " for f in familias))

    pasta = res.get("experimento")
    if pasta:
        print(f"\n  Salvo em: resultados/experimentos/{pasta}/")


def main(argv=None) -> int:
    from src.ml.experimentos_artigos import (
        ORDEM_EXPERIMENTOS, catalogo_experimentos_md, executar_experimento,
    )

    parser = argparse.ArgumentParser(
        description="Roda experimentos de ML por artigo-base (standalone).",
    )
    parser.add_argument("experimentos", nargs="*",
                        help=f"chaves a rodar ({', '.join(ORDEM_EXPERIMENTOS)})")
    parser.add_argument("--todos", action="store_true",
                        help="roda todos os experimentos com modelo treinável")
    parser.add_argument("--auc", action="store_true",
                        help="comparativo por AUC dos experimentos de anomalia "
                             "já salvos (gráfico + tabela; não re-roda)")
    args = parser.parse_args(argv)

    if args.auc:
        from src.ml.experimentos_artigos import comparar_anomalia_por_auc

        cmp = comparar_anomalia_por_auc()
        print("\n" + cmp["mensagem"])
        if cmp["ok"]:
            print("\n" + cmp["tabela_md"])
            if cmp["grafico"]:
                print(f"\nGráfico salvo em: {cmp['grafico']}")
        return 0 if cmp["ok"] else 1

    if args.todos:
        alvos = list(ORDEM_EXPERIMENTOS)
    elif args.experimentos:
        alvos = [a.lower() for a in args.experimentos]
    else:
        print(catalogo_experimentos_md())
        print("\nDica: python scripts/rodar_experimentos.py <chave>  |  --todos")
        return 0

    invalidas = [a for a in alvos if a not in ORDEM_EXPERIMENTOS]
    if invalidas:
        print(f"Chave(s) desconhecida(s): {', '.join(invalidas)}")
        print(f"Disponíveis: {', '.join(ORDEM_EXPERIMENTOS)}")
        return 2

    for key in alvos:
        print(f"\n>>> Rodando: {key}")
        try:
            res = executar_experimento(key, progresso=lambda m: print("   ", m))
        except Exception as exc:  # noqa: BLE001
            res = {"experimento": key, "ok": False, "mensagem": f"erro: {exc}"}
        _imprimir_resultado(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
