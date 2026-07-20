"""Executor de uma etapa de ML em subprocesso descartável.

O pipeline completo é acionado pelo Streamlit. Isolar cada etapa impede que
PyTorch, pandas e Matplotlib acumulem memória no processo principal entre
features, treino, validação e prognóstico.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("uso: python -m src.ml.exec_etapa_isolada <etapa>", flush=True)
        return 2
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))

    os.environ["AL_IADO_PIPELINE_CHILD"] = "1"
    etapa = argv[1]
    try:
        from src.ml.pipeline import get_stage

        ok = bool(get_stage(etapa).load_runner()())
        print(f"etapa={etapa} ok={str(ok).lower()}", flush=True)
        return 0 if ok else 1
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
