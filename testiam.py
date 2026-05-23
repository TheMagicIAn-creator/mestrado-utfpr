from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.core.config import MODELO_EMBEDDINGS, PASTA_CHROMADB, PASTA_LITERATURA
from src.conhecimento.indexador import indexar_pdf_unico
import chromadb

modelo  = SentenceTransformer(MODELO_EMBEDDINGS)
client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
colecao = client.get_or_create_collection("literatura_pv")

pdfs = list(PASTA_LITERATURA.rglob("*torres*aplicacao*"))
if not pdfs:
    print("TCC não encontrado")
else:
    pdf = pdfs[0]
    print(f"Arquivo: {pdf.name}")

    # Remove chunks antigos
    res = colecao.get(where={"arquivo": pdf.name}, include=["metadatas"])
    ids = res.get("ids", [])
    if ids:
        colecao.delete(ids=ids)
        print(f"Removidos {len(ids)} chunks antigos")

    # Reindexa com pipeline novo
    res = indexar_pdf_unico(pdf, modelo, PASTA_CHROMADB)
    print(f"Reindexado: {res['n_chunks']} chunks")
    print(f"Sucesso: {res['sucesso']}")
    if res['erro']:
        print(f"Erro: {res['erro']}")