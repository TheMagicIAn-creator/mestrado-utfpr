import chromadb
from src.core.config import PASTA_CHROMADB, NOME_COLECAO

client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
colecao = client.get_or_create_collection(NOME_COLECAO)

arquivos_vistos = {}
offset = 0
lote   = 200

while True:
    resultado = colecao.get(limit=lote, offset=offset, include=["metadatas"])
    metas = resultado.get("metadatas", [])
    if not metas:
        break
    for m in metas:
        arquivo = m.get("arquivo", "")
        if arquivo not in arquivos_vistos:
            arquivos_vistos[arquivo] = {
                "autor"  : m.get("autor",   ""),
                "titulo" : m.get("titulo",  ""),
                "ano"    : m.get("ano",     ""),
                "citacao": m.get("citacao", ""),
            }
    offset += lote
    if len(metas) < lote:
        break

print(f"Total de documentos: {len(arquivos_vistos)}\n")
print(f"{'ARQUIVO':<60} | {'CITAÇÃO NO CHROMADB'}")
print("-" * 130)
for arquivo, meta in sorted(arquivos_vistos.items()):
    citacao = meta["citacao"] or f"{meta['autor']} ({meta['ano']}) — {meta['titulo']}"
    print(f"{arquivo[:60]:<60} | {citacao[:65]}")