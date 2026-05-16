"""
agente.py — Al IAdo PV
Conecta o Gemini (LLM) ao ChromaDB (memória) usando RAG.

RAG = Retrieval Augmented Generation
      = Geração Aumentada por Recuperação

Fluxo:
  Pergunta → Vetor → ChromaDB → Contexto → Gemini → Resposta

Autor: Rodolfo Torres (UTFPR)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import parsear_nome_arquivo
from langchain_core.messages import HumanMessage


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_CHROMADB    = Path(__file__).parent.parent / "base_conhecimento"
CAMINHO_CLAUDE_MD = Path(__file__).parent.parent / "CLAUDE.md"
NOME_COLECAO      = "literatura_pv"
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
MODELO_GEMINI = "gemini-2.5-flash"
N_RESULTADOS      = 15  # chunks recuperados por busca


# ============================================================
# FUNÇÕES
# ============================================================

def carregar_perfil() -> str:
    """
    Lê o CLAUDE.md e retorna o conteúdo como string.
    Este é o 'sistema de instruções' do agente — quem ele é
    e como deve se comportar.
    """
    if CAMINHO_CLAUDE_MD.exists():
        perfil = CAMINHO_CLAUDE_MD.read_text(encoding="utf-8")
        print("   ✅ Perfil CLAUDE.md carregado!")
        return perfil
    else:
        print("   ⚠️  CLAUDE.md não encontrado — usando perfil padrão")
        return (
            "Você é o Al IAdo PV, assistente especialista em "
            "inversores fotovoltaicos e manutenção preditiva com ML."
        )


def inicializar_agente():
    """
    Inicializa todos os componentes do agente:
      1. Variáveis de ambiente (.env)
      2. Perfil do agente (CLAUDE.md)
      3. Modelo de embeddings (sentence-transformers)
      4. Banco vetorial (ChromaDB)
      5. LLM (Gemini)

    Retorna: (perfil, modelo_embeddings, colecao, llm)
    """

    print("=" * 60)
    print("  AL IADO PV — INICIALIZANDO AGENTE")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Carrega variáveis do .env
    # ----------------------------------------------------------
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "\n❌ GOOGLE_API_KEY não encontrada!\n"
            "   Verifique se o arquivo .env existe e contém:\n"
            "   GOOGLE_API_KEY=sua_chave_aqui"
        )
    print("\n✅ Chave de API carregada do .env")

    # ----------------------------------------------------------
    # 2. Carrega perfil do CLAUDE.md
    # ----------------------------------------------------------
    print("\n📋 Carregando perfil do agente...")
    perfil = carregar_perfil()

    # ----------------------------------------------------------
    # 3. Carrega modelo de embeddings
    # ----------------------------------------------------------
    print("\n🔄 Carregando modelo de embeddings...")
    print("   (Segunda vez em diante: carrega do cache — rápido)")
    modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS)
    print("   ✅ Modelo de embeddings pronto!")

    # ----------------------------------------------------------
    # 4. Conecta ao ChromaDB
    # ----------------------------------------------------------
    print("\n🗄️  Conectando ao ChromaDB...")
    if not PASTA_CHROMADB.exists():
        raise FileNotFoundError(
            "\n❌ Base de conhecimento não encontrada!\n"
            "   Execute primeiro: python src/indexador.py"
        )

    client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = client.get_or_create_collection(name=NOME_COLECAO)

    total = colecao.count()
    print(f"   ✅ ChromaDB conectado! ({total} chunks indexados)")

    # ----------------------------------------------------------
    # 5. Inicializa o Gemini
    # ----------------------------------------------------------
    print("\n🤖 Inicializando Gemini...")
    llm = ChatGoogleGenerativeAI(
        model=MODELO_GEMINI,
        google_api_key=api_key,
        temperature=0.3,  # 0 = determinístico, 1 = criativo
    )
    print("   ✅ Gemini pronto!")

    print("\n" + "=" * 60)
    print("  AL IADO PV ESTÁ ONLINE! 🤖")
    print("=" * 60 + "\n")

    return perfil, modelo_embeddings, colecao, llm


def buscar_contexto(
    pergunta: str,
    modelo_embeddings,
    colecao
) -> tuple:
    """
    Transforma a pergunta em vetor e busca chunks similares no ChromaDB.
    Retorna o contexto formatado e as citações acadêmicas das fontes.
    """

    # Transforma pergunta em vetor
    vetor_pergunta = modelo_embeddings.encode([pergunta]).tolist()

    # Busca os N chunks mais similares
    resultados = colecao.query(
        query_embeddings=vetor_pergunta,
        n_results=N_RESULTADOS
    )

    contexto   = ""
    citacoes   = {}  # arquivo → citação formatada

    documentos = resultados.get("documents", [[]])[0]
    metadados  = resultados.get("metadatas",  [[]])[0]

    for i, (doc, meta) in enumerate(zip(documentos, metadados), 1):
        arquivo = meta.get("arquivo", "fonte desconhecida")
        pasta   = meta.get("pasta",   "")

        # Usa citação do metadado se disponível,
        # senão parseia o nome do arquivo na hora
        if "citacao" in meta:
            citacao = meta["citacao"]
        else:
            citacao = parsear_nome_arquivo(arquivo)["citacao"]

        contexto += (
            f"\n--- Trecho {i} ---\n"
            f"Fonte: {citacao} (tema: {pasta})\n"
            f"{doc}\n"
        )

        if arquivo not in citacoes:
            citacoes[arquivo] = citacao

    return contexto, citacoes

def listar_documentos(colecao) -> str:
    """
    Lista todos os documentos únicos indexados no ChromaDB.
    Usa os metadados — não depende de busca vetorial.
    """
    resultados = colecao.get(include=["metadatas"])
    metadados  = resultados.get("metadatas", [])

    arquivos_vistos = set()
    documentos      = []

    for meta in metadados:
        arquivo = meta.get("arquivo", "desconhecido")
        pasta   = meta.get("pasta",   "desconhecida")
        if arquivo not in arquivos_vistos:
            arquivos_vistos.add(arquivo)
            documentos.append((pasta, arquivo))

    # Ordena por pasta temática
    documentos.sort(key=lambda x: x[0])

    # Formata a saída
    texto  = f"📚 Total de documentos indexados: {len(documentos)}\n\n"
    pasta_atual = ""

    for pasta, arquivo in documentos:
        if pasta != pasta_atual:
            texto      += f"\n📁 {pasta}/\n"
            pasta_atual = pasta
        texto += f"   → {arquivo}\n"

    return texto

def perguntar(
    pergunta: str,
    perfil: str,
    modelo_embeddings,
    colecao,
    llm,
    historico: list = None
) -> str:
    """
    Pipeline RAG completo com memória de conversa.

    O histórico é uma lista de dicionários:
    [
        {"role": "user",      "content": "pergunta anterior"},
        {"role": "assistant", "content": "resposta anterior"},
        ...
    ]
    """

    if historico is None:
        historico = []

    # Busca contexto
    contexto, fontes = buscar_contexto(pergunta, modelo_embeddings, colecao)

    # Formata histórico da conversa
    historico_formatado = ""
    if historico:
        historico_formatado = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        historico_formatado += "HISTÓRICO DA CONVERSA ATUAL:\n"
        historico_formatado += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for turno in historico[-6:]:  # últimos 3 pares de perguntas/respostas
            role    = "Rodolfo" if turno["role"] == "user" else "Al IAdo PV"
            content = turno["content"][:500]  # limita para não estourar quota
            historico_formatado += f"\n{role}:\n{content}\n"

    # Monta prompt com histórico
    prompt = f"""
{perfil}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTO RECUPERADO DA LITERATURA CIENTÍFICA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{contexto}
{historico_formatado}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERGUNTA ATUAL DO PESQUISADOR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pergunta}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES:
- Responda em português brasileiro
- Use o contexto da literatura como base principal
- Considere o histórico da conversa para dar continuidade
- Cite os artigos pelos nomes dos arquivos
- Se o contexto for insuficiente, diga claramente e
  complemente com conhecimento geral, sinalizando
- Seja técnico, preciso e didático
- Profundidade compatível com pós-graduação
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    # Envia ao Gemini
    mensagens = [HumanMessage(content=prompt)]
    resposta  = llm.invoke(mensagens)
    texto     = resposta.content

    if fontes:
        texto += "\n\n---\n📚 *Fontes consultadas nesta resposta:*\n"
        for arquivo, citacao in fontes.items():
            texto += f"  → {citacao}\n"

    return texto