"""
verificar_autores.py — Al IAdo PV

Verificacao pontual de recuperacao: para CADA documento indexado no ChromaDB,
dispara uma pergunta no formato "o que o <Sobrenome> diz?" pela pipeline real
(buscar_contexto) e confirma que o arquivo daquele autor entra nas citacoes.

Serve de rede de seguranca contra regressoes onde o RAG "perde" um paper —
em especial autores com varios arquivos (ex.: Grewal: Kalman + Power
Electronics), onde uma busca por autor pode trazer so o maior arquivo.

Uso:
    python scripts/verificar_autores.py

Saida: total de documentos, quantos foram recuperados (alvo: 39/39) e a lista
dos que falharam, com o top-3 efetivamente recuperado para diagnostico.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import chromadb
from sentence_transformers import SentenceTransformer

from src.conhecimento.agente import buscar_contexto
from src.core.config import MODELO_EMBEDDINGS, NOME_COLECAO, PASTA_CHROMADB


def arquivos_distintos(colecao) -> dict[str, str]:
    """Mapa arquivo → autor (primeiro visto), varrendo toda a colecao."""
    vistos: dict[str, str] = {}
    offset, lote = 0, 500
    while True:
        r = colecao.get(limit=lote, offset=offset, include=["metadatas"])
        metas = r.get("metadatas", [])
        if not metas:
            break
        for m in metas:
            arq = str(m.get("arquivo", "")).strip()
            if arq and arq not in vistos:
                vistos[arq] = str(m.get("autor", "")).strip()
        if len(metas) < lote:
            break
        offset += lote
    return vistos


def main() -> int:
    print("Carregando modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = client.get_or_create_collection(name=NOME_COLECAO)

    arqs = arquivos_distintos(colecao)
    print(f"Documentos distintos indexados: {len(arqs)}\n")

    achou = 0
    falhou: list[tuple[str, str, list[str]]] = []

    for i, (arq, _autor) in enumerate(sorted(arqs.items()), 1):
        sobrenome = arq.split("_", 1)[0].replace("-", " ")
        pergunta = f"o que o {sobrenome} diz sobre o tema da dissertacao?"
        _ctx, cit = buscar_contexto(
            pergunta,
            modelo,
            colecao,
            n_pool=120,
            n_resultados=16,
            n_resultados_revisao=28,
            max_chunks_por_fonte=2,
            contexto_chars=14_000,
            sessao_chars=1_500,
            consultar_literatura=True,
        )
        if arq in cit:
            achou += 1
            print(f"[{i:02d}/{len(arqs)}] OK   {sobrenome}")
        else:
            top = list(cit.keys())[:3]
            falhou.append((arq, sobrenome, top))
            print(f"[{i:02d}/{len(arqs)}] FALHA {sobrenome}  top={top}")

    print(f"\nRecuperados: {achou}/{len(arqs)}")
    if falhou:
        print("\nFALHARAM:")
        for arq, sob, top in falhou:
            print(f"  - {sob}  ({arq})\n      top={top}")
        return 1
    print("Todos os documentos sao recuperaveis pelo nome do autor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
