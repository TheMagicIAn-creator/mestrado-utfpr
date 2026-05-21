"""
indexador.py — Al IAdo PV
Lê os PDFs da pasta /literatura e indexa no ChromaDB
para que o agente possa buscar informações por similaridade.

Autor: Rodolfo Torres (UTFPR)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import re
from pathlib import Path
from pypdf import PdfReader
from src.core.utils import parsear_nome_arquivo
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
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

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

def upsert_em_lotes(colecao, ids, embeddings, documents, metadados, tamanho_lote=500):
    """
    Divide o upsert em lotes para evitar o limite do ChromaDB.
    """
    total = len(ids)
    for inicio in range(0, total, tamanho_lote):
        fim = min(inicio + tamanho_lote, total)
        colecao.upsert(
            ids        = ids[inicio:fim],
            embeddings = embeddings[inicio:fim],
            documents  = documents[inicio:fim],
            metadatas  = metadados[inicio:fim]
        )

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

def indexar_sessao(caminho_md: Path, modelo_embeddings, pasta_chromadb: Path) -> int:
    """
    Indexa uma sessão salva (.md) na coleção de sessões do ChromaDB.
    Chamada automaticamente após cada sessão encerrada.
    Retorna o número de chunks indexados.
    """

    NOME_COLECAO_SESSOES = "sessoes_pv"

    # Lê o arquivo .md
    if not caminho_md.exists():
        print(f"  ⚠️  Arquivo não encontrado: {caminho_md}")
        return 0

    texto = caminho_md.read_text(encoding="utf-8")

    if not texto.strip():
        return 0

    # Remove linhas de erro antes de indexar
    linhas_filtradas = []
    for linha in texto.split("\n"):
        if any(termo in linha.lower() for termo in [
            "rate limit", "429", "resource_exhausted",
            "quota", "error code", "❌ erro"
        ]):
            continue
        linhas_filtradas.append(linha)
    texto = "\n".join(linhas_filtradas)

    # Divide em chunks
    chunks = dividir_em_chunks(texto, TAMANHO_CHUNK, SOBREPOSICAO)

    if not chunks:
        return 0

    # Conecta ao ChromaDB na coleção de sessões
    client = chromadb.PersistentClient(path=str(pasta_chromadb))
    colecao_sessoes = client.get_or_create_collection(
        name     = NOME_COLECAO_SESSOES,
        metadata = {"hnsw:space": "cosine"}
    )

    # Gera embeddings
    embeddings = modelo_embeddings.encode(chunks).tolist()

    # IDs únicos por sessão + chunk
    nome_arquivo = caminho_md.name
    ids = [f"{nome_arquivo}__chunk_{j}" for j in range(len(chunks))]

    # Metadados
    data_sessao = caminho_md.stem[:10]  # YYYY-MM-DD do nome do arquivo
    metadados = [
        {
            "arquivo"     : nome_arquivo,
            "tipo"        : "sessao",
            "data"        : data_sessao,
            "chunk_index" : j,
            "total_chunks": len(chunks)
        }
        for j in range(len(chunks))
    ]

    # Indexa (upsert evita duplicatas)
    upsert_em_lotes(colecao, ids, embeddings, chunks, metadados)

    return len(chunks)

def indexar_literatura():
    """
    Função principal: lê todos os PDFs e indexa no ChromaDB.
    """

    print("=" * 60)
    print("  AL IADO — INDEXADOR DE LITERATURA")
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


        # Cria metadados para cada chunk (com citação acadêmica)
        info_arquivo = parsear_nome_arquivo(nome_arquivo)
        metadados = [
            {
                "arquivo": nome_arquivo,
                "pasta": caminho_pdf.parent.name,
                "chunk_index": j,
                "total_chunks": len(chunks),
                "autor": info_arquivo["autor"],
                "titulo": info_arquivo["titulo"],
                "ano": info_arquivo["ano"],
                "citacao": info_arquivo["citacao"]
            }
            for j in range(len(chunks))
        ]

        # Adiciona ao ChromaDB
        # upsert = insert + update: evita duplicatas se rodar de novo
        upsert_em_lotes(colecao, ids, embeddings, chunks, metadados)

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
    print("\n✅ Al IAdo está pronto para buscar na literatura!")

def indexar_pdf_unico(caminho_pdf: Path, modelo_embeddings, pasta_chromadb: Path) -> dict:
    """
    Indexa um único PDF no ChromaDB.
    Usado pelo upload manual da interface Streamlit.
    Retorna um dicionário com o resultado da operação.
    """

    NOME_COLECAO = "literatura_pv"

    resultado = {
        "sucesso"     : False,
        "nome_arquivo": caminho_pdf.name,
        "n_chunks"    : 0,
        "erro"        : None
    }

    # Lê o PDF
    texto = ler_pdf(caminho_pdf)
    if not texto:
        resultado["erro"] = "Não foi possível extrair texto do PDF."
        return resultado

    # Divide em chunks
    chunks = dividir_em_chunks(texto, TAMANHO_CHUNK, SOBREPOSICAO)
    if not chunks:
        resultado["erro"] = "Nenhum chunk gerado."
        return resultado

    # Conecta ao ChromaDB
    client  = chromadb.PersistentClient(path=str(pasta_chromadb))
    colecao = client.get_or_create_collection(
        name     = NOME_COLECAO,
        metadata = {"hnsw:space": "cosine"}
    )

    # Gera embeddings
    embeddings = modelo_embeddings.encode(chunks).tolist()

    # Metadados com citação acadêmica
    info_arquivo = parsear_nome_arquivo(caminho_pdf.name)
    nome_pasta   = caminho_pdf.parent.name

    ids = [f"{caminho_pdf.name}__chunk_{j}" for j in range(len(chunks))]

    metadados = [
        {
            "arquivo"     : caminho_pdf.name,
            "pasta"       : nome_pasta,
            "chunk_index" : j,
            "total_chunks": len(chunks),
            "autor"       : info_arquivo["autor"],
            "titulo"      : info_arquivo["titulo"],
            "ano"         : info_arquivo["ano"],
            "citacao"     : info_arquivo["citacao"]
        }
        for j in range(len(chunks))
    ]

    # Indexa
    colecao.upsert(
        ids        = ids,
        embeddings = embeddings,
        documents  = chunks,
        metadatas  = metadados
    )

    resultado["sucesso"]  = True
    resultado["n_chunks"] = len(chunks)
    return resultado

# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    indexar_literatura()