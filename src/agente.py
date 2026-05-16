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


def inicializar_agente(llm_externo=None):
    """
    Inicializa todos os componentes do agente.
    Se llm_externo for fornecido, usa ele.
    Caso contrário, usa Gemini por padrão.
    """

    print("=" * 60)
    print("  AL IADO PV — INICIALIZANDO AGENTE")
    print("=" * 60)

    # Carrega variáveis do .env
    load_dotenv()
    print("\n✅ Variáveis de ambiente carregadas")

    # Perfil
    print("\n📋 Carregando perfil do agente...")
    perfil = carregar_perfil()

    # Embeddings
    print("\n🔄 Carregando modelo de embeddings...")
    modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS)
    print("   ✅ Modelo de embeddings pronto!")

    # ChromaDB
    print("\n🗄️  Conectando ao ChromaDB...")
    if not PASTA_CHROMADB.exists():
        raise FileNotFoundError(
            "\n❌ Base de conhecimento não encontrada!\n"
            "   Execute primeiro: python src/indexador.py"
        )
    client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    colecao = client.get_or_create_collection(name=NOME_COLECAO)
    total   = colecao.count()
    print(f"   ✅ ChromaDB conectado! ({total} chunks indexados)")

    # LLM — usa externo se fornecido, senão inicializa Gemini
    if llm_externo is not None:
        llm = llm_externo
        print("\n🤖 LLM externo recebido!")
    else:
        print("\n🤖 Inicializando Gemini (padrão)...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada no .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model          = MODELO_GEMINI,
            google_api_key = api_key,
            temperature    = 0.3
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
    historico: list = None,
    streaming: bool = True
) -> str:
    """
    Pipeline RAG completo com memória e streaming.
    """

    if historico is None:
        historico = []

    # Busca contexto
    contexto, citacoes = buscar_contexto(pergunta, modelo_embeddings, colecao)

    # Formata histórico
    historico_formatado = ""
    if historico:
        historico_formatado  = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        historico_formatado += "HISTÓRICO DA CONVERSA ATUAL:\n"
        historico_formatado += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for turno in historico[-6:]:
            role    = "Rodolfo" if turno["role"] == "user" else "Al IAdo PV"
            content = turno["content"][:500]
            historico_formatado += f"\n{role}:\n{content}\n"

    # Monta prompt
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

    mensagens = [HumanMessage(content=prompt)]
    texto_completo = ""

    import time

    if streaming:
        # ── MODO STREAMING ──────────────────────────────────────
        max_tentativas = 3
        for tentativa in range(1, max_tentativas + 1):
            try:
                for chunk in llm.stream(mensagens):
                    pedaco = chunk.content
                    print(pedaco, end="", flush=True)
                    texto_completo += pedaco
                print()  # quebra de linha ao terminar
                break
            except Exception as e:
                erro = str(e)
                if "429" in erro and tentativa < max_tentativas:
                    import re
                    match  = re.search(r"retry in (\d+)", erro)
                    espera = int(match.group(1)) + 5 if match else 30
                    print(f"\n  ⏳ Limite atingido. Aguardando {espera}s... ({tentativa}/{max_tentativas-1})")
                    time.sleep(espera)
                else:
                    raise
    else:
        # ── MODO NORMAL (sem streaming) ──────────────────────────
        max_tentativas = 3
        for tentativa in range(1, max_tentativas + 1):
            try:
                resposta       = llm.invoke(mensagens)
                texto_completo = resposta.content
                break
            except Exception as e:
                erro = str(e)
                if "429" in erro and tentativa < max_tentativas:
                    import re
                    match  = re.search(r"retry in (\d+)", erro)
                    espera = int(match.group(1)) + 5 if match else 30
                    print(f"\n  ⏳ Limite atingido. Aguardando {espera}s... ({tentativa}/{max_tentativas-1})")
                    time.sleep(espera)
                else:
                    raise

    # Adiciona fontes ao final
    if citacoes:
        rodape  = "\n\n---\n📚 **Fontes consultadas nesta resposta:**\n"
        for arquivo, citacao in citacoes.items():
            rodape += f"  → {citacao}\n"
        print(rodape)
        texto_completo += rodape

    return texto_completo