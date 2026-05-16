"""
indexador.py — Al IAdo PV
Lê os PDFs da pasta /literatura e indexa no ChromaDB
para que o agente possa buscar informações por similaridade.

Autor: Rodolfo Torres (UTFPR)
"""

import os
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Caminho para a pasta com os PDFs
PASTA_LITERATURA = Path(__file__).parent.parent / "literatura"

# Caminho para o banco vetorial ChromaDB
PASTA_CHROMADB = Path(__file__).parent.parent / "base_conhecimento"

# Nome da coleção no ChromaDB
NOME_COLECAO = "literatura_pv"

# Modelo de embeddings (transforma texto em vetores numéricos)
# all-MiniLM-L6-v2 é leve, rápido e funciona bem em português técnico
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

# Tamanho dos chunks (pedaços de texto que serão indexados)
TAMANHO_CHUNK = 500   # caracteres por chunk
SOBREPOSICAO  = 50    # sobreposição entre chunks para não perder contexto


# ============================================================
# FUNÇÕES
# ============================================================

def ler_pdf(caminho_pdf: Path) -> str:
    """
    Lê um arquivo PDF e retorna todo o texto como string.
    """
    try:
        reader = PdfReader(str(caminho_pdf))
        texto = ""
        for pagina in reader.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto += texto_pagina + "\n"
        return texto.strip()
    except Exception as e:
        print(f"  ⚠️  Erro ao ler {caminho_pdf.name}: {e}")
        return ""


def dividir_em_chunks(texto: str, tamanho: int, sobreposicao: int) -> list:
    """
    Divide um texto longo em pedaços menores (chunks).

    Por que dividir?
    - O modelo de embeddings tem limite de tamanho de entrada
    - Chunks menores permitem buscas muito mais precisas
    - A sobreposição garante que nenhum contexto seja cortado
    """
    chunks = []
    inicio = 0

    while inicio < len(texto):
        fim = inicio + tamanho
        chunk = texto[inicio:fim]
        if chunk.strip():           # ignora chunks vazios
            chunks.append(chunk)
        inicio = fim - sobreposicao # sobreposição entre chunks

    return chunks


def indexar_literatura():
    """
    Função principal: lê todos os PDFs e indexa no ChromaDB.
    """

    print("=" * 60)
    print("  AL IADO PV — INDEXADOR DE LITERATURA")
    print("=" * 60)

    # ----------------------------------------------------------
    # PASSO 1 — Verifica se a pasta de literatura existe
    # ----------------------------------------------------------
    if not PASTA_LITERATURA.exists():
        print(f"\n❌ Pasta não encontrada: {PASTA_LITERATURA}")
        print("   Verifique se o caminho está correto.")
        return

    # ----------------------------------------------------------
    # PASSO 2 — Encontra todos os PDFs recursivamente
    # ----------------------------------------------------------
    pdfs = list(PASTA_LITERATURA.rglob("*.pdf"))

    if not pdfs:
        print(f"\n❌ Nenhum PDF encontrado em: {PASTA_LITERATURA}")
        return

    print(f"\n📚 PDFs encontrados: {len(pdfs)}")
    for pdf in pdfs:
        print(f"   → {pdf.name}")

    # ----------------------------------------------------------
    # PASSO 3 — Carrega o modelo de embeddings
    # ----------------------------------------------------------
    print(f"\n🔄 Carregando modelo de embeddings: {MODELO_EMBEDDINGS}")
    print("   (Na primeira vez baixa o modelo — pode demorar)")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    print("   ✅ Modelo carregado!")

    # ----------------------------------------------------------
    # PASSO 4 — Conecta ao ChromaDB
    # ----------------------------------------------------------
    print(f"\n🗄️  Conectando ao ChromaDB...")
    PASTA_CHROMADB.mkdir(exist_ok=True)

    client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = client.get_or_create_collection(
        name=NOME_COLECAO,
        metadata={"hnsw:space": "cosine"}  # busca por similaridade de cosseno
    )
    print("   ✅ ChromaDB conectado!")

    # ----------------------------------------------------------
    # PASSO 5 — Indexa cada PDF
    # ----------------------------------------------------------
    total_chunks   = 0
    pdfs_indexados = 0
    pdfs_com_erro  = 0

    print(f"\n📥 Iniciando indexação...\n")

    for i, caminho_pdf in enumerate(pdfs, 1):
        nome_arquivo = caminho_pdf.name
        print(f"  [{i}/{len(pdfs)}] {nome_arquivo}")

        # Lê o texto do PDF
        texto = ler_pdf(caminho_pdf)

        if not texto:
            print(f"         ⚠️  Sem texto extraível — pulando")
            pdfs_com_erro += 1
            continue

        # Divide em chunks
        chunks = dividir_em_chunks(texto, TAMANHO_CHUNK, SOBREPOSICAO)
        print(f"         → {len(chunks)} chunks gerados")

        # Gera embeddings (transforma cada chunk em vetor numérico)
        embeddings = modelo.encode(chunks).tolist()

        # Cria IDs únicos para cada chunk
        ids = [f"{nome_arquivo}__chunk_{j}" for j in range(len(chunks))]

        # Cria metadados para cada chunk
        metadados = [
            {
                "arquivo"     : nome_arquivo,
                "pasta"       : caminho_pdf.parent.name,
                "chunk_index" : j,
                "total_chunks": len(chunks)
            }
            for j in range(len(chunks))
        ]

        # Adiciona ao ChromaDB
        # upsert = insert + update: evita duplicatas se rodar de novo
        colecao.upsert(
            ids       = ids,
            embeddings= embeddings,
            documents = chunks,
            metadatas = metadados
        )

        total_chunks   += len(chunks)
        pdfs_indexados += 1
        print(f"         ✅ Indexado com sucesso!")

    # ----------------------------------------------------------
    # PASSO 6 — Relatório final
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  INDEXAÇÃO CONCLUÍDA!")
    print("=" * 60)
    print(f"  PDFs indexados com sucesso : {pdfs_indexados}")
    print(f"  PDFs com erro              : {pdfs_com_erro}")
    print(f"  Total de chunks no banco   : {total_chunks}")
    print(f"  Coleção ChromaDB           : {NOME_COLECAO}")
    print(f"  Local do banco             : {PASTA_CHROMADB}")
    print("=" * 60)
    print("\n✅ Al IAdo PV está pronto para buscar na literatura!")


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    indexar_literatura()