"""Executa a validacao experimental E3 no GPVS-Faults."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.gpvs import (  # noqa: E402
    PASTA_GPVS,
    PASTA_SAIDA,
    executar_validacao_gpvs,
    regenerar_graficos_gpvs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=PASTA_GPVS)
    parser.add_argument("--saida", type=Path, default=PASTA_SAIDA)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patiencia", type=int, default=30)
    parser.add_argument(
        "--somente-graficos", action="store_true",
        help="regenera figuras e manifesto sem retreinar os modelos",
    )
    args = parser.parse_args()
    if args.somente_graficos:
        resultado = regenerar_graficos_gpvs(args.dados, args.saida)
    else:
        resultado = executar_validacao_gpvs(
            diretorio=args.dados,
            pasta_saida=args.saida,
            epochs=args.epochs,
            paciencia=args.patiencia,
        )
    print("Validacao GPVS concluida.")
    for caminho in resultado["outputs"]:
        print(f"  - {caminho}")
    print(f"  - {resultado['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
