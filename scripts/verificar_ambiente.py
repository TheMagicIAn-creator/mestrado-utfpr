"""
verificar_ambiente.py — Al IAdo PV / Sprint 4-5 (reprodutibilidade)

Diagnóstico do ambiente local: imports, versões, chaves de API, datasets
(presença + linhas + SHA-256), coleções do ChromaDB, estado das etapas do
pipeline (ready/stale/pending) e bibliotecas opcionais (degradação honesta).

NÃO imprime valores de chaves nem dados sensíveis. Sai com código 0 sempre
(é diagnóstico, não gate); o relatório indica o que está pendente.

Uso:
    python scripts/verificar_ambiente.py
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.utils import configurar_saida_utf8  # noqa: E402

configurar_saida_utf8()

OK, FALTA, AVISO = "✅", "❌", "⚠️"


def _versao(dist: str) -> str | None:
    try:
        return md.version(dist)
    except Exception:
        return None


def secao(titulo: str) -> None:
    print(f"\n{'='*60}\n  {titulo}\n{'='*60}")


def checar_nucleo() -> None:
    secao("Núcleo (obrigatório)")
    for dist in ("numpy", "pandas", "scipy", "scikit-learn", "torch",
                 "chromadb", "streamlit", "sentence-transformers", "pytest"):
        v = _versao(dist)
        print(f"  {OK if v else FALTA} {dist:24s} {v or '(ausente)'}")


def checar_opcionais() -> None:
    secao("Opcionais (degradação honesta)")
    mapa = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }
    for mod, desc in mapa.items():
        try:
            disponivel = importlib.util.find_spec(mod) is not None
        except Exception:
            disponivel = False
        print(f"  {OK if disponivel else AVISO} {mod:20s} {desc}"
              f"{'' if disponivel else '  (recurso fica indisponível)'}")


def checar_chaves() -> None:
    secao("Chaves de API (.env) — só presença, nunca o valor")
    from dotenv import load_dotenv

    load_dotenv()
    for chave in ("GROQ_API_KEY", "GOOGLE_API_KEY"):
        tem = bool(os.getenv(chave))
        print(f"  {OK if tem else AVISO} {chave:18s} {'configurada' if tem else 'ausente'}")


def checar_datasets() -> None:
    secao("Datasets (presença + SHA-256 + linhas)")
    try:
        from scripts.verificar_datasets import verificar as vd

        vd(silencioso=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  {AVISO} não foi possível verificar datasets: {exc}")


def checar_chromadb() -> None:
    secao("ChromaDB")
    try:
        import chromadb

        from src.core.config import NOME_COLECAO, PASTA_CHROMADB

        cli = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        nomes = [c.name for c in cli.list_collections()]
        print(f"  {OK} coleções: {nomes}")
        if NOME_COLECAO in nomes:
            print(f"  {OK} {NOME_COLECAO}: {cli.get_collection(NOME_COLECAO).count()} chunks")
    except Exception as exc:  # noqa: BLE001
        print(f"  {AVISO} ChromaDB indisponível: {exc}")


def checar_pipeline() -> None:
    secao("Pipeline de ML (ready / stale / pending)")
    try:
        from src.ml.pipeline import NOMES_ETAPAS, estado_pipeline

        rotulo = {"ready": OK, "stale": AVISO, "pending": "⬜"}
        for key, info in estado_pipeline().items():
            est = info["estado"]
            extra = f" — {', '.join(info.get('motivos', []))}" if est != "ready" else ""
            print(f"  {rotulo.get(est, '?')} {NOMES_ETAPAS[key]:22s} {est}{extra}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {AVISO} pipeline indisponível: {exc}")


def main() -> int:
    print("AL IADO PV — verificação de ambiente")
    checar_nucleo()
    checar_opcionais()
    checar_chaves()
    checar_datasets()
    checar_chromadb()
    checar_pipeline()
    print("\nDiagnóstico concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
