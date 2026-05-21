"""
consolidar_memoria.py — Al IAdo PV
Consolida todas as sessões em um único resumo.

Fluxo:
  1. Lê todos os arquivos .md de notas/sessoes/
  2. Envia ao LLM para resumir os principais insights
  3. Salva o resumo em notas/memorias/
  4. Remove chunks antigos de sessoes_pv no ChromaDB
  5. Indexa o resumo consolidado
  6. Arquiva as sessões originais

Pode ser chamado pelo N8N (agendado) ou manualmente:
  python src/consolidar_memoria.py

Autor: Rodolfo Torres (UTFPR)
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_SESSOES  = Path(__file__).parent.parent / "notas" / "sessoes"
PASTA_MEMORIAS = Path(__file__).parent.parent / "notas" / "memorias"
PASTA_ARQUIVO  = Path(__file__).parent.parent / "notas" / "sessoes_arquivadas"
MINIMO_SESSOES = 2   # só consolida se houver pelo menos N sessões


# ============================================================
# LÊ SESSÕES
# ============================================================

def ler_sessoes() -> list:
    """Lê todos os arquivos .md da pasta de sessões."""

    arquivos = sorted(PASTA_SESSOES.glob("*.md"))

    if not arquivos:
        return []

    sessoes = []
    for arquivo in arquivos:
        conteudo = arquivo.read_text(encoding="utf-8")
        # Ignora arquivos muito pequenos (menos de 200 chars = sem conteúdo real)
        if len(conteudo) > 200:
            sessoes.append({
                "arquivo" : arquivo,
                "conteudo": conteudo,
                "data"    : arquivo.stem[:10]
            })

    return sessoes


# ============================================================
# CONSOLIDA COM LLM
# ============================================================

def consolidar_com_llm(sessoes: list) -> str:
    """
    Envia todas as sessões ao LLM e pede um resumo consolidado
    com os principais tópicos, conclusões e insights.
    """

    # Monta o texto de todas as sessões
    texto_sessoes = ""
    for s in sessoes:
        texto_sessoes += f"\n\n{'='*50}\n"
        texto_sessoes += f"SESSÃO: {s['data']}\n"
        texto_sessoes += f"{'='*50}\n"
        texto_sessoes += s["conteudo"][:10000]  # limita por sessão

    # Limita o total para não estourar quota
    texto_sessoes = texto_sessoes[:60000]

    prompt = f"""
Você é um assistente de pesquisa acadêmica. Abaixo estão transcrições de sessões de 
pesquisa do mestrado de Rodolfo Torres (UTFPR) sobre análise preditiva de falhas em 
inversores fotovoltaicos com Machine Learning.

SESSÕES DE PESQUISA:
{texto_sessoes}

Sua tarefa é criar um RESUMO CONSOLIDADO com:

## 1. Principais Tópicos Discutidos
Liste os temas técnicos abordados nas sessões.

## 2. Conclusões e Decisões Tomadas
O que foi decidido ou concluído sobre modelos, metodologias, dados.

## 3. Insights Técnicos Relevantes
Os pontos mais importantes para a dissertação.

## 4. Próximos Passos Identificados
O que ainda precisa ser feito ou investigado.

## 5. Referências Citadas
Quais artigos da literatura foram mais mencionados.

Seja DETALHADO e COMPLETO — preserve todos os detalhes técnicos importantes.
Não resuma demais. É melhor ter mais informação do que perder insights relevantes.
Mantenha equações, nomes de modelos, parâmetros e referências específicas. Este resumo substituirá as sessões 
individuais na memória do agente — deve capturar tudo que é relevante para continuidade da pesquisa.
Responda em português brasileiro.
"""

    # Tenta Groq primeiro, depois Gemini
    resposta = None

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
            llm      = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0.3)
            resposta = llm.invoke([HumanMessage(content=prompt)]).content
            print("   ✅ Resumo gerado pelo Groq")
        except Exception as e:
            print(f"   ⚠️  Groq falhou: {e} — tentando Gemini...")

    if not resposta:
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.messages import HumanMessage
                llm      = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key, temperature=0.3)
                resposta = llm.invoke([HumanMessage(content=prompt)]).content
                print("   ✅ Resumo gerado pelo Gemini")
            except Exception as e:
                print(f"   ❌ Gemini também falhou: {e}")

    return resposta or "Erro: não foi possível gerar o resumo."


# ============================================================
# SALVA MEMÓRIA CONSOLIDADA
# ============================================================

def salvar_consolidado(resumo: str, sessoes: list) -> Path:
    """Salva o resumo consolidado como nota .md."""

    PASTA_MEMORIAS.mkdir(parents=True, exist_ok=True)

    agora        = datetime.now()
    datas        = [s["data"] for s in sessoes]
    nome_arquivo = f"{agora.strftime('%Y-%m-%d')}_consolidado.md"
    caminho      = PASTA_MEMORIAS / nome_arquivo

    conteudo  = f"---\n"
    conteudo += f"data: {agora.strftime('%Y-%m-%d')}\n"
    conteudo += f"tipo: memoria-consolidada\n"
    conteudo += f"sessoes_incluidas: {len(sessoes)}\n"
    conteudo += f"periodo: {datas[0]} a {datas[-1]}\n"
    conteudo += f"tags: [al-iado-pv, memoria, consolidado, mestrado]\n"
    conteudo += f"---\n\n"
    conteudo += f"# Memória Consolidada — {agora.strftime('%d/%m/%Y')}\n\n"
    conteudo += f"> Gerado automaticamente a partir de {len(sessoes)} sessões "
    conteudo += f"({datas[0]} a {datas[-1]})\n\n"
    conteudo += f"---\n\n"
    conteudo += resumo

    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# ============================================================
# ATUALIZA CHROMADB
# ============================================================

def atualizar_chromadb(caminho_consolidado: Path, sessoes: list):
    """
    Remove chunks antigos das sessões e indexa o consolidado.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    from src.agente    import MODELO_EMBEDDINGS, PASTA_CHROMADB
    from src.indexador import dividir_em_chunks, upsert_em_lotes

    NOME_COLECAO_SESSOES = "sessoes_pv"

    print("   🔄 Carregando modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)

    client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = client.get_or_create_collection(name=NOME_COLECAO_SESSOES)

    # Remove chunks das sessões antigas
    print("   🗑️  Removendo chunks das sessões antigas...")
    for sessao in sessoes:
        nome = sessao["arquivo"].name
        try:
            # Busca IDs que pertencem a essa sessão
            resultados = colecao.get(where={"arquivo": nome})
            ids_remover = resultados.get("ids", [])
            if ids_remover:
                colecao.delete(ids=ids_remover)
                print(f"      → {nome}: {len(ids_remover)} chunks removidos")
        except Exception as e:
            print(f"      ⚠️  Erro ao remover {nome}: {e}")

    # Indexa o consolidado
    print("   📥 Indexando memória consolidada...")
    texto  = caminho_consolidado.read_text(encoding="utf-8")
    chunks = dividir_em_chunks(texto, 500, 50)

    if chunks:
        embeddings = modelo.encode(chunks).tolist()
        nome_final = caminho_consolidado.name
        ids        = [f"{nome_final}__chunk_{j}" for j in range(len(chunks))]
        metadados  = [
            {
                "arquivo": nome_final,
                "tipo"   : "memoria-consolidada",
                "data"   : datetime.now().strftime("%Y-%m-%d"),
                "chunk_index"  : j,
                "total_chunks" : len(chunks)
            }
            for j in range(len(chunks))
        ]

        upsert_em_lotes(colecao, ids, embeddings, chunks, metadados)
        print(f"      ✅ {len(chunks)} chunks do consolidado indexados")


# ============================================================
# ARQUIVA SESSÕES ORIGINAIS
# ============================================================

def arquivar_sessoes(sessoes: list):
    """Move sessões processadas para pasta de arquivo."""

    PASTA_ARQUIVO.mkdir(parents=True, exist_ok=True)

    for sessao in sessoes:
        destino = PASTA_ARQUIVO / sessao["arquivo"].name
        sessao["arquivo"].rename(destino)
        print(f"   📦 Arquivado: {sessao['arquivo'].name}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def consolidar():
    """Pipeline completo de consolidação de memória."""

    print("=" * 60)
    print("  AL IADO PV — CONSOLIDAÇÃO DE MEMÓRIA")
    print("=" * 60)

    # 1. Lê sessões
    print("\n📂 Lendo sessões...")
    sessoes = ler_sessoes()

    if len(sessoes) < MINIMO_SESSOES:
        print(f"\n⚠️  Apenas {len(sessoes)} sessão(ões) encontrada(s).")
        print(f"   Mínimo para consolidar: {MINIMO_SESSOES}")
        print(f"   Continue usando o agente e rode novamente depois.")
        return

    print(f"   ✅ {len(sessoes)} sessões encontradas")
    for s in sessoes:
        print(f"      → {s['arquivo'].name}")

    # 2. Consolida com LLM
    print(f"\n🤖 Gerando resumo consolidado com LLM...")
    resumo = consolidar_com_llm(sessoes)

    # 3. Salva o consolidado
    print(f"\n💾 Salvando memória consolidada...")
    caminho = salvar_consolidado(resumo, sessoes)
    print(f"   ✅ Salvo em: {caminho.name}")

    # 4. Atualiza ChromaDB
    print(f"\n🗄️  Atualizando ChromaDB...")
    atualizar_chromadb(caminho, sessoes)

    # 5. Arquiva sessões
    print(f"\n📦 Arquivando sessões originais...")
    arquivar_sessoes(sessoes)

    print(f"\n{'='*60}")
    print(f"  CONSOLIDAÇÃO CONCLUÍDA!")
    print(f"  Sessões processadas : {len(sessoes)}")
    print(f"  Resumo salvo em     : notas/memorias/")
    print(f"  Sessões arquivadas  : notas/sessoes_arquivadas/")
    print(f"{'='*60}")


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    consolidar()