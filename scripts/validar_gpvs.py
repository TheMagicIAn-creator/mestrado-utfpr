"""Executa a validacao experimental E3 no GPVS-Faults."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.gpvs import PASTA_GPVS  # noqa: E402
from src.ml.validacao_gpvs_principal import (  # noqa: E402
    PASTA_SAIDA,
    executar_validacao_gpvs_principal,
    regenerar_graficos_gpvs_principal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=PASTA_GPVS)
    parser.add_argument("--saida", type=Path, default=PASTA_SAIDA)
    parser.add_argument(
        "--somente-graficos", action="store_true",
        help="regenera figuras canônicas a partir das tabelas versionadas",
    )
    args = parser.parse_args()
    if args.somente_graficos:
        resultado = regenerar_graficos_gpvs_principal(args.saida)
    else:
        resultado = executar_validacao_gpvs_principal(
            diretorio=args.dados, pasta_saida=args.saida,
        )
    print("Validacao GPVS concluida.")
    for caminho in resultado["outputs"]:
        print(f"  - {caminho}")
    if resultado.get("manifest"):
        print(f"  - {resultado['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
