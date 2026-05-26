"""Reindexa todas as sessões e memórias na base de conhecimento."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from src.core.config import MODELO_EMBEDDINGS, PASTA_CHROMADB
from src.conhecimento.indexador import indexar_sessao

RAIZ = Path(__file__).resolve().parent.parent
modelo = SentenceTransformer(MODELO_EMBEDDINGS)

pastas = [
    RAIZ / "notas" / "sessoes",
    RAIZ / "notas" / "sessoes_arquivadas",
    RAIZ / "notas" / "memorias",
]

total = 0
for pasta in pastas:
    if not pasta.exists():
        continue
    arquivos = sorted(pasta.glob("*.md"))
    print(f"\n{pasta.name}: {len(arquivos)} arquivos")
    for arq in arquivos:
        try:
            n = indexar_sessao(arq, modelo, PASTA_CHROMADB)
            total += n
            print(f"  ✅ {arq.name}: {n} chunks")
        except Exception as e:
            print(f"  ⚠️  {arq.name}: {e}")

print(f"\nTotal: {total} chunks reindexados")