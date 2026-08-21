"""Verifica ambiente, dataset GPVS e publicacao cientifica canonica."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

DEPENDENCIAS = {
    "dotenv": "python-dotenv",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "starlette": "starlette",
    "uvicorn": "uvicorn",
}
LEGADO_PROIBIDO = (
    "src/ml/autoencoder_v2",
    "src/ml/split_temporal.py",
    "resultados/autoencoder",
    "resultados/gpvs",
    "resultados/macro",
    "resultados/v2",
    ".streamlit",
    "main.py",
    "watcher.py",
)


def verificar(*, auditar_resultados: bool = True) -> dict:
    from src.ml.dados_gpvs import ALL_EXPERIMENTS, DATASET_DIR, dataset_files

    erros: list[str] = []
    avisos: list[str] = []
    dependencias = {
        pacote: importlib.util.find_spec(modulo) is not None
        for modulo, pacote in DEPENDENCIAS.items()
    }
    ausentes = [nome for nome, presente in dependencias.items() if not presente]
    if ausentes:
        erros.append("dependencias ausentes: " + ", ".join(ausentes))

    try:
        arquivos = dataset_files(DATASET_DIR)
        dataset = {"diretorio": str(DATASET_DIR), "ensaios": sorted(arquivos)}
        if tuple(sorted(arquivos)) != tuple(sorted(ALL_EXPERIMENTS)):
            erros.append("inventario GPVS diverge de F0L-F7M")
    except (FileNotFoundError, ValueError) as exc:
        dataset = {"diretorio": str(DATASET_DIR), "erro": str(exc)}
        erros.append(str(exc))

    legado = [relativo for relativo in LEGADO_PROIBIDO if (RAIZ / relativo).exists()]
    if legado:
        erros.append("caminhos legados presentes: " + ", ".join(legado))

    publicacao = None
    if auditar_resultados:
        from scripts.auditar_resultados import auditar_publicacao

        publicacao = auditar_publicacao(RAIZ)
        if not publicacao.get("ok"):
            erros.extend(str(item) for item in publicacao.get("errors", []))
    else:
        avisos.append("auditoria de resultados ignorada por opcao")

    return {
        "ok": not erros,
        "python": sys.version.split()[0],
        "dependencias": dependencias,
        "dataset": dataset,
        "legado": legado,
        "publicacao": publicacao,
        "avisos": avisos,
        "erros": erros,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sem-resultados",
        action="store_true",
        help="nao valida manifestos e hashes publicados",
    )
    args = parser.parse_args(argv)
    resultado = verificar(auditar_resultados=not args.sem_resultados)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return int(not resultado["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
