"""
macro_comparar.py — Al IAdo PV / comparativo dos dois macro-códigos

Roda (ou reaproveita) os dois métodos sobre o MESMO holdout e a MESMA injeção
FMECA, e emite UMA tabela enxuta + UM gráfico sobreposto:

  - Proposto  (src/ml/macro_proposto.py) — AE denso + escore localizado
  - Ibrahim   (src/ml/macro_ibrahim.py)  — AE-LSTM temporal

Substitui o antigo framework de experimentos por artigo (que produzia tabelas
de 33 colunas e matrizes de confusão enganosas sob limiar de prevalência rara).

Uso:
  python src/ml/macro_comparar.py

Saídas: resultados/macro/comparacao_*.{json,md,csv,png}
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

_logger = _get_logger("macro_comparar")
_log = _adaptar_log(_logger)


from pathlib import Path

RAIZ = Path(__file__).parent.parent.parent
PASTA_SAIDA = RAIZ / "resultados" / "macro"


def executar(n_janelas: int | None = None) -> list[dict]:
    from src.ml import macro_ibrahim, macro_proposto
    from src.ml.macro_comum import salvar_saidas, tabela_enxuta

    _log("=" * 60)
    _log("  COMPARATIVO — método proposto × Ibrahim (2022)")
    _log("=" * 60)

    _log("\n[1/2] Método proposto...")
    r_prop = macro_proposto.executar(n_janelas)

    _log("\n[2/2] Método do Ibrahim...")
    r_ibra = macro_ibrahim.executar(n_janelas)

    resultados = [r_prop, r_ibra]
    saidas = salvar_saidas(resultados, PASTA_SAIDA, prefixo="comparacao")

    _log("\n" + "=" * 60)
    _log("  TABELA COMPARATIVA")
    _log("=" * 60)
    _log("\n" + tabela_enxuta(resultados))
    _log(f"\n  Artefatos em {PASTA_SAIDA}:")
    for k, v in saidas.items():
        _log(f"    {k}: {Path(v).name}")
    _log("\n  Leitura: compare por AUC (independe do limiar). A detecção por")
    _log("  severidade mostra a partir de que intensidade cada método enxerga")
    _log("  cada falha da FMECA. Ambos usam limiar auto-calibrado a ~1% de FP.")
    _log("=" * 60)
    return resultados


if __name__ == "__main__":
    from src.core.logs import habilitar_console

    habilitar_console()
    executar()
