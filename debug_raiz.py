from sentence_transformers import SentenceTransformer
from src.core.config import MODELO_EMBEDDINGS, PASTA_CHROMADB, NOME_COLECAO
from src.conhecimento.agente import _expandir_query, _busca_hibrida, _rerankar, N_RESULTADOS
import chromadb

pergunta = "quais foram os valores de NPR obtidos na FMECA do CEAMAZON"

modelo  = SentenceTransformer(MODELO_EMBEDDINGS)
client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
colecao = client.get_or_create_collection(NOME_COLECAO)

# DIAGNÓSTICO 1 — O chunk NPR existe no ChromaDB?
print("=" * 60)
print("DIAGNÓSTICO 1 — Busca direta por 'NPR' no ChromaDB")
print("=" * 60)
res = colecao.get(
    where_document = {"$contains": "NPR"},
    include        = ["documents", "metadatas"],
    limit          = 5
)
docs  = res.get("documents", [])
metas = res.get("metadatas", [])
print(f"Chunks com 'NPR': {len(docs)}")
for doc, meta in zip(docs, metas):
    print(f"\n  Fonte: {meta.get('citacao','')[:60]}")
    print(f"  Conteúdo: {doc[:200]}")

# DIAGNÓSTICO 2 — A expansão gera bons termos?
print("\n" + "=" * 60)
print("DIAGNÓSTICO 2 — Expansão de query")
print("=" * 60)
expansao  = _expandir_query(pergunta)
variacoes = expansao.get("variacoes", [])
termos    = expansao.get("termos", [])
print(f"Variações: {variacoes}")
print(f"Termos: {termos}")

# DIAGNÓSTICO 3 — A busca híbrida recupera o chunk NPR?
print("\n" + "=" * 60)
print("DIAGNÓSTICO 3 — Busca híbrida")
print("=" * 60)
candidatos = _busca_hibrida(variacoes, termos, colecao, modelo, n_pool=60)
print(f"Total de candidatos: {len(candidatos)}")
chunks_com_npr = [(d, m) for d, m in candidatos if "NPR" in d or "210" in d]
print(f"Candidatos com NPR/210: {len(chunks_com_npr)}")
for doc, meta in chunks_com_npr[:3]:
    print(f"\n  Fonte: {meta.get('citacao','')[:60]}")
    print(f"  Conteúdo: {doc[:300]}")

# DIAGNÓSTICO 4 — O reranker seleciona o chunk NPR?
print("\n" + "=" * 60)
print("DIAGNÓSTICO 4 — Reranking")
print("=" * 60)
melhores = _rerankar(candidatos, pergunta, N_RESULTADOS)
chunks_npm_ranked = [(d, m) for d, m in melhores if "NPR" in d or "210" in d]
print(f"Chunks com NPR/210 nos {N_RESULTADOS} selecionados: {len(chunks_npm_ranked)}")
for doc, meta in chunks_npm_ranked:
    print(f"\n  Fonte: {meta.get('citacao','')[:60]}")
    print(f"  Conteúdo: {doc[:300]}")