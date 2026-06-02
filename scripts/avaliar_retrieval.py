"""
avaliar_retrieval.py — Al IAdo PV / Sprint 4 (7.2)

Avalia a RECUPERAÇÃO (separada da geração) sobre um conjunto curado de perguntas
com a fonte esperada. Para cada pergunta, roda a busca real (buscar_contexto) e
mede se a fonte correta foi recuperada e em que posição → Recall@k, MRR, nDCG@k,
taxa de fonte correta e diversidade.

Read-only (não grava memória). Exige ChromaDB indexado + (rerank) chave do LLM.

Uso:
    python scripts/avaliar_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.utils import configurar_saida_utf8  # noqa: E402

configurar_saida_utf8()

# (pergunta, [substrings de autor/arquivo que indicam a fonte correta])
PERGUNTAS = [
    ("Qual o NPR do inversor no FMECA do CEAMAZON?", ["torres"]),
    ("Descreva o dataset de Paderborn do inversor IGBT.", ["stender"]),
    ("O que o guia da NASA recomenda sobre RCM?", ["administration"]),
    ("Cite métodos de detecção de anomalia em usinas solares.",
     ["ibrahim", "sharma", "ahirwar", "francisti"]),
    ("O que a literatura diz sobre FMEA em sistemas fotovoltaicos?",
     ["sakurada", "carpinetti", "torres"]),
    ("Confiabilidade de inversores fotovoltaicos.",
     ["shuttleworth", "dhople", "joshi", "patil", "karim"]),
]
K = 5


def _fontes_recuperadas(citacoes) -> list[str]:
    """Lista ORDENADA (relevância) de rótulos de fonte recuperados."""
    valores = citacoes.values() if isinstance(citacoes, dict) else (citacoes or [])
    return [str(v).lower() for v in valores if v]


def main() -> int:
    from sentence_transformers import SentenceTransformer
    import chromadb

    from src.conhecimento.agente import buscar_contexto
    from src.core.config import (
        MODELO_EMBEDDINGS, NOME_COLECAO, NOME_COLECAO_SESSOES, PASTA_CHROMADB,
    )
    from src.ml.retrieval_metrics import agregimar

    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    cli = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    col = cli.get_or_create_collection(NOME_COLECAO)
    col_s = cli.get_or_create_collection(NOME_COLECAO_SESSOES)

    linhas = []
    print(f"Avaliando retrieval em {len(PERGUNTAS)} perguntas (k={K})\n")
    for pergunta, esperados in PERGUNTAS:
        try:
            _ctx, cit = buscar_contexto(pergunta, modelo, col, col_s,
                                        consultar_literatura=True)
            recuperados = _fontes_recuperadas(cit)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERRO em '{pergunta[:40]}…': {exc}")
            recuperados = []
        relevantes = {s for s in recuperados if any(e in s for e in esperados)}
        rank = next((i for i, s in enumerate(recuperados, 1)
                     if any(e in s for e in esperados)), None)
        linhas.append({"recuperados": recuperados, "relevantes": relevantes})
        print(f"  {'✅' if rank else '❌'} rank={rank or '-'} | {pergunta[:50]}")

    print("\n" + "=" * 50)
    print("Agregado:", agregimar(linhas, k=K))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
