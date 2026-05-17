import sys
from pathlib import Path
sys.path.insert(0, '.')
from sentence_transformers import SentenceTransformer
from src.agente import MODELO_EMBEDDINGS, PASTA_CHROMADB
from src.indexador import indexar_sessao

modelo  = SentenceTransformer(MODELO_EMBEDDINGS)
sessoes = list((Path('notas/sessoes')).glob('*.md'))
print(f'Re-indexando {len(sessoes)} sessões...')
for s in sessoes:
    n = indexar_sessao(s, modelo, PASTA_CHROMADB)
    print(f'  ✅ {s.name} — {n} chunks')
print('Pronto!')