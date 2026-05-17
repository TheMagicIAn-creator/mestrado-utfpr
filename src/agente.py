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
NOME_COLECAO_SESSOES = "sessoes_pv"
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
    Retorna: perfil, modelo_embeddings, colecao, colecao_sessoes, llm
    """

    print("=" * 60)
    print("  AL IADO PV — INICIALIZANDO AGENTE")
    print("=" * 60)

    load_dotenv()
    print("\n✅ Variáveis de ambiente carregadas")

    print("\n📋 Carregando perfil do agente...")
    perfil = carregar_perfil()

    print("\n🔄 Carregando modelo de embeddings...")
    modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS)
    print("   ✅ Modelo de embeddings pronto!")

    print("\n🗄️  Conectando ao ChromaDB...")
    if not PASTA_CHROMADB.exists():
        raise FileNotFoundError(
            "\n❌ Base de conhecimento não encontrada!\n"
            "   Execute primeiro: python src/indexador.py"
        )

    client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))

    # Coleção da literatura
    colecao = client.get_or_create_collection(name=NOME_COLECAO)
    print(f"   ✅ Literatura: {colecao.count()} chunks indexados")

    # Coleção de sessões anteriores
    colecao_sessoes = client.get_or_create_collection(
        name     = NOME_COLECAO_SESSOES,
        metadata = {"hnsw:space": "cosine"}
    )
    total_sessoes = colecao_sessoes.count()
    if total_sessoes > 0:
        print(f"   ✅ Sessões anteriores: {total_sessoes} chunks na memória")
    else:
        print(f"   ℹ️  Sessões anteriores: nenhuma ainda (primeira sessão)")

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

    return perfil, modelo_embeddings, colecao, colecao_sessoes, llm


def buscar_contexto(
    pergunta: str,
    modelo_embeddings,
    colecao,
    colecao_sessoes=None
) -> tuple:
    """
    Busca contexto relevante em DUAS fontes:
    1. literatura_pv  — artigos científicos indexados
    2. sessoes_pv     — conversas anteriores (memória persistente)
    """

    vetor_pergunta = modelo_embeddings.encode([pergunta]).tolist()

    contexto = ""
    citacoes = {}

    # ── 1. Busca na literatura ───────────────────────────────────
    resultados_lit = colecao.query(
        query_embeddings = vetor_pergunta,
        n_results        = N_RESULTADOS
    )

    documentos = resultados_lit.get("documents", [[]])[0]
    metadados  = resultados_lit.get("metadatas",  [[]])[0]

    if documentos:
        contexto += "\n📚 DA LITERATURA CIENTÍFICA:\n"
        for i, (doc, meta) in enumerate(zip(documentos, metadados), 1):
            arquivo = meta.get("arquivo", "fonte desconhecida")
            pasta   = meta.get("pasta",   "")
            citacao = meta.get("citacao") or parsear_nome_arquivo(arquivo)["citacao"]
            contexto += (
                f"\n--- Trecho {i} ---\n"
                f"Fonte: {citacao} (tema: {pasta})\n"
                f"{doc}\n"
            )
            if arquivo not in citacoes:
                citacoes[arquivo] = citacao

    # ── 2. Busca nas sessões anteriores ─────────────────────────
    if colecao_sessoes and colecao_sessoes.count() > 0:
        resultados_ses = colecao_sessoes.query(
            query_embeddings = vetor_pergunta,
            n_results        = 3  # menos chunks de sessão que de literatura
        )

        docs_ses = resultados_ses.get("documents", [[]])[0]
        mets_ses = resultados_ses.get("metadatas",  [[]])[0]

        if docs_ses:
            contexto += "\n\n🧠 DE SESSÕES ANTERIORES (memória persistente):\n"
            for i, (doc, meta) in enumerate(zip(docs_ses, mets_ses), 1):
                data = meta.get("data", "data desconhecida")
                contexto += (
                    f"\n--- Memória {i} (sessão de {data}) ---\n"
                    f"{doc}\n"
                )

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

def preparar_prompt(
    pergunta: str,
    perfil: str,
    modelo_embeddings,
    colecao,
    historico: list    = None,
    colecao_sessoes    = None
) -> tuple:
    """
    Prepara o prompt completo sem invocar o LLM.
    Retorna (prompt_str, citacoes_dict).
    Usado pelo Streamlit para fazer streaming separado.
    """

    if historico is None:
        historico = []

    contexto, citacoes = buscar_contexto(
        pergunta, modelo_embeddings, colecao, colecao_sessoes
    )

    historico_formatado = ""
    if historico:
        historico_formatado  = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        historico_formatado += "HISTÓRICO DA CONVERSA ATUAL:\n"
        historico_formatado += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for turno in historico[-6:]:
            role    = "Rodolfo" if turno["role"] == "user" else "Al IAdo PV"
            content = turno["content"][:500]
            historico_formatado += f"\n{role}:\n{content}\n"

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
    INSTRUÇÕES DE CONTEÚDO:
    - Responda em português brasileiro
    - Use o contexto da literatura como base principal
    - Considere o histórico da conversa para dar continuidade
    - Quando o contexto for insuficiente, sinalize claramente e
      complemente com conhecimento geral
    - Profundidade compatível com pós-graduação
    - Seja direto e denso — sem enrolação, sem repetição
    - Aja como co-orientador técnico: quando pertinente,
      faça perguntas de volta ou sugira próximos passos
    - Cite sempre as fontes pelo nome do autor e ano

    INSTRUÇÕES DE FORMATAÇÃO (obrigatório seguir):
    - Use **negrito** para termos técnicos na primeira menção
    - Use ## e ### para organizar respostas longas em seções
    - Use tabelas markdown quando comparar modelos, técnicas
      ou resultados (ex: | Modelo | Vantagem | Limitação |)
    - Use listas numeradas para processos sequenciais
    - Use listas com marcadores para itens sem ordem fixa
    - Use > para citações diretas dos artigos
    - Use `código` para nomes de funções, bibliotecas e parâmetros
    - Use blocos ```python para pseudocódigo e exemplos
    - Destaque conclusões importantes em **negrito**
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    return prompt, citacoes

def perguntar(
    pergunta: str,
    perfil: str,
    modelo_embeddings,
    colecao,
    llm,
    historico: list = None,
    streaming: bool = True,
    colecao_sessoes = None
) -> str:
    """
    Pipeline RAG completo com memória e streaming.
    """

    if historico is None:
        historico = []

    # Busca contexto
    contexto, citacoes = buscar_contexto(pergunta, modelo_embeddings, colecao, colecao_sessoes)

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
    INSTRUÇÕES DE CONTEÚDO:
    - Responda em português brasileiro
    - Use o contexto da literatura como base principal
    - Considere o histórico da conversa para dar continuidade
    - Quando o contexto for insuficiente, sinalize claramente e
      complemente com conhecimento geral
    - Profundidade compatível com pós-graduação
    - Seja direto e denso — sem enrolação, sem repetição
    - Aja como co-orientador técnico: quando pertinente,
      faça perguntas de volta ou sugira próximos passos
    - Cite sempre as fontes pelo nome do autor e ano

    INSTRUÇÕES DE FORMATAÇÃO (obrigatório seguir):
    - Use **negrito** para termos técnicos na primeira menção
    - Use ## e ### para organizar respostas longas em seções
    - Use tabelas markdown quando comparar modelos, técnicas
      ou resultados (ex: | Modelo | Vantagem | Limitação |)
    - Use listas numeradas para processos sequenciais
    - Use listas com marcadores para itens sem ordem fixa
    - Use > para citações diretas dos artigos
    - Use `código` para nomes de funções, bibliotecas e parâmetros
    - Use blocos ```python para pseudocódigo e exemplos
    - Destaque conclusões importantes em **negrito**
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