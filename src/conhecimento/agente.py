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
from src.core.utils import parsear_nome_arquivo
from src.core.config import (
    PASTA_CHROMADB, ARQUIVO_PERFIL, NOME_COLECAO,
    NOME_COLECAO_SESSOES, MODELO_EMBEDDINGS, MODELO_GEMINI,
    N_RESULTADOS,
)
from langchain_core.messages import HumanMessage

# ============================================================
# CONFIGURAÇÕES
# ============================================================

# ============================================================
# FUNÇÕES
# ============================================================

def carregar_perfil() -> str:
    """
    Lê o CLAUDE.md e retorna o conteúdo como string.
    Este é o 'sistema de instruções' do agente — quem ele é
    e como deve se comportar.
    """
    if ARQUIVO_PERFIL.exists():
        perfil = ARQUIVO_PERFIL.read_text(encoding="utf-8")
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

# ============================================================
# RAG AVANÇADO — 3 CAMADAS
# ============================================================

def _expandir_query(pergunta: str) -> dict:
    """
    CAMADA 1 — Expansão de query.
    Usa Groq LLaMA 3.1 8B para gerar variações da pergunta
    e extrair termos-chave para busca por palavras.
    """
    import json as _json
    from src.core.config import GROQ_API_KEY, GOOGLE_API_KEY

    prompt = f"""Você é um sistema de busca especializado em inversores fotovoltaicos e manutenção preditiva.

Dada a pergunta abaixo, gere:
1. Quatro variações usando sinônimos, reformulações e perspectivas diferentes
2. Cinco termos-chave específicos para busca literal (siglas, números, termos técnicos, nomes próprios)

Retorne APENAS um JSON válido neste formato exato, sem explicações:
{{"variacoes": ["...", "...", "...", "..."], "termos": ["...", "...", "...", "...", "..."]}}

Pergunta: {pergunta}"""

    resposta = None

    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
            llm      = ChatGroq(
                model        = "llama-3.1-8b-instant",
                groq_api_key = GROQ_API_KEY,
                temperature  = 0
            )
            resposta = llm.invoke([HumanMessage(content=prompt)]).content
        except Exception:
            pass

    if not resposta and GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            llm      = ChatGoogleGenerativeAI(
                model          = "gemini-2.5-flash",
                google_api_key = GOOGLE_API_KEY,
                temperature    = 0
            )
            resposta = llm.invoke([HumanMessage(content=prompt)]).content
        except Exception:
            pass

    if resposta:
        try:
            import re as _re
            limpo    = _re.sub(r"```json?\n?", "", resposta.strip()).replace("```", "").strip()
            return _json.loads(limpo)
        except Exception:
            pass

    return {"variacoes": [pergunta], "termos": []}


def _busca_hibrida(
    variacoes       : list,
    termos          : list,
    colecao,
    modelo_embeddings,
    n_pool          : int = 60
) -> list:
    """
    CAMADA 2 — Busca híbrida.
    Combina busca semântica (embeddings) com busca por palavras-chave.
    Retorna pool deduplicado de candidatos como lista de (doc, meta).
    """
    pool = {}  # chunk_id → (documento, metadado)

    # Busca semântica para cada variação da query
    n_por_variacao = max(10, n_pool // max(len(variacoes), 1))

    for variacao in variacoes:
        try:
            vetor      = modelo_embeddings.encode([variacao]).tolist()
            resultados = colecao.query(
                query_embeddings = vetor,
                n_results        = min(n_por_variacao, 50)
            )
            docs  = resultados.get("documents", [[]])[0]
            metas = resultados.get("metadatas",  [[]])[0]
            ids   = resultados.get("ids",        [[]])[0]

            for id_, doc, meta in zip(ids, docs, metas):
                if id_ not in pool:
                    pool[id_] = (doc, meta)
        except Exception:
            continue

    # Busca por palavras-chave (keyword search)
    for termo in termos:
        if not termo or len(termo) < 2:
            continue
        try:
            resultados = colecao.get(
                where_document = {"$contains": termo},
                include        = ["documents", "metadatas"],
                limit          = 10
            )
            docs  = resultados.get("documents", [])
            metas = resultados.get("metadatas", [])
            ids   = resultados.get("ids",       [])

            for id_, doc, meta in zip(ids, docs, metas):
                if id_ not in pool:
                    pool[id_] = (doc, meta)
        except Exception:
            continue

    return list(pool.values())


def _rerankar(candidatos: list, pergunta: str, n_final: int) -> list:
    """
    CAMADA 3 — Reranking.
    Usa Groq LLaMA 3.1 8B para avaliar cada chunk candidato
    e selecionar os n_final mais relevantes para a pergunta.
    """
    import json as _json
    from src.core.config import GROQ_API_KEY, GOOGLE_API_KEY

    if not candidatos:
        return []

    if len(candidatos) <= n_final:
        return candidatos

    # Monta lista numerada dos candidatos (limitada para caber no contexto)
    lista_chunks = ""
    for i, (doc, meta) in enumerate(candidatos):
        autor = meta.get("autor", "")
        ano   = meta.get("ano",   "")
        fonte = f"{autor} ({ano})" if autor else "Fonte desconhecida"
        lista_chunks += f"\n[{i}] {fonte}\n{doc[:400]}\n"

    prompt = f"""Você é um sistema de reranking para pesquisa acadêmica sobre inversores fotovoltaicos.

Pergunta do pesquisador: {pergunta}

Abaixo há {len(candidatos)} trechos de documentos científicos numerados de 0 a {len(candidatos)-1}.
Selecione os {n_final} trechos MAIS relevantes para responder a pergunta.

Critérios de relevância:
- Responde direta ou indiretamente à pergunta
- Contém dados, valores, tabelas ou fórmulas mencionadas
- Apresenta definições, métodos ou resultados pertinentes
- É específico, não genérico

Retorne APENAS um JSON com os índices em ordem de relevância (mais relevante primeiro):
{{"selecionados": [índice1, índice2, ...]}}

Trechos:
{lista_chunks}"""

    resposta = None

    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
            llm      = ChatGroq(
                model        = "llama-3.1-8b-instant",
                groq_api_key = GROQ_API_KEY,
                temperature  = 0
            )
            resposta = llm.invoke([HumanMessage(content=prompt)]).content
        except Exception:
            pass

    if not resposta and GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            llm      = ChatGoogleGenerativeAI(
                model          = "gemini-2.5-flash",
                google_api_key = GOOGLE_API_KEY,
                temperature    = 0
            )
            resposta = llm.invoke([HumanMessage(content=prompt)]).content
        except Exception:
            pass

    if resposta:
        try:
            import re as _re
            limpo    = _re.sub(r"```json?\n?", "", resposta.strip()).replace("```", "").strip()
            resultado = _json.loads(limpo)
            indices   = resultado.get("selecionados", [])

            selecionados = [
                candidatos[i]
                for i in indices
                if isinstance(i, int) and 0 <= i < len(candidatos)
            ]

            if selecionados:
                return selecionados[:n_final]
        except Exception:
            pass

    # Fallback: retorna os primeiros n_final sem reranking
    return candidatos[:n_final]


# ============================================================
# BUSCA DE CONTEXTO — PIPELINE RAG 3 CAMADAS
# ============================================================

def buscar_contexto(
    pergunta        : str,
    modelo_embeddings,
    colecao,
    colecao_sessoes = None
) -> tuple:
    """
    Pipeline RAG de 3 camadas:
    1. Expansão: LLM gera variações da query + termos-chave
    2. Busca híbrida: semântica + keyword → pool ~60 chunks
    3. Reranking: LLM seleciona os N mais relevantes do pool
    """
    contexto = ""
    citacoes = {}

    # ── CAMADA 1 — Expansão ──────────────────────────────────
    expansao  = _expandir_query(pergunta)
    variacoes = expansao.get("variacoes", [pergunta])
    termos    = expansao.get("termos",    [])

    if pergunta not in variacoes:
        variacoes.insert(0, pergunta)

    # ── CAMADA 2 — Busca híbrida ─────────────────────────────
    candidatos = _busca_hibrida(
        variacoes, termos, colecao, modelo_embeddings, n_pool=60
    )

    # ── CAMADA 3 — Reranking ─────────────────────────────────
    melhores = _rerankar(candidatos, pergunta, N_RESULTADOS)

    # Monta contexto da literatura
    if melhores:
        contexto += "\n📚 DA LITERATURA CIENTÍFICA:\n"
        for doc, meta in melhores:
            arquivo = meta.get("arquivo", "")
            citacao = meta.get("citacao", arquivo)
            if arquivo and arquivo not in citacoes:
                citacoes[arquivo] = citacao
            contexto += f"\n[Fonte: {citacao}]\n{doc}\n"

    # ── Sessões — busca direta (sem reranking) ───────────────
    if colecao_sessoes:
        try:
            vetor_pergunta = modelo_embeddings.encode([pergunta]).tolist()
            resultados_ses = colecao_sessoes.query(
                query_embeddings = vetor_pergunta,
                n_results        = max(3, N_RESULTADOS // 4)
            )
            docs_ses  = resultados_ses.get("documents", [[]])[0]
            metas_ses = resultados_ses.get("metadatas",  [[]])[0]

            if docs_ses:
                contexto += "\n💭 DA MEMÓRIA DE SESSÕES ANTERIORES:\n"
                for doc, meta in zip(docs_ses, metas_ses):
                    arquivo = meta.get("arquivo", "")
                    contexto += f"\n[Memória: {arquivo}]\n{doc}\n"
        except Exception:
            pass

    return contexto, citacoes

def listar_documentos(colecao) -> str:
    """
    Lista todos os documentos únicos indexados no ChromaDB.
    Usa paginação para evitar o limite de variáveis SQL.
    """

    arquivos_vistos = set()
    documentos      = []
    offset          = 0
    lote            = 200  # busca 200 por vez

    while True:
        try:
            resultados = colecao.get(
                limit   = lote,
                offset  = offset,
                include = ["metadatas"]
            )
        except Exception:
            break

        metadados = resultados.get("metadatas", [])
        if not metadados:
            break

        for meta in metadados:
            arquivo = meta.get("arquivo", "desconhecido")
            pasta   = meta.get("pasta",   "desconhecida")
            citacao = meta.get("citacao", arquivo)
            if arquivo not in arquivos_vistos:
                arquivos_vistos.add(arquivo)
                documentos.append((pasta, citacao))

        offset += lote
        if len(metadados) < lote:
            break

    # Ordena por pasta temática
    documentos.sort(key=lambda x: x[0])

    texto       = f"📚 Total de documentos: {len(documentos)}\n\n"
    pasta_atual = ""

    for pasta, citacao in documentos:
        if pasta != pasta_atual:
            texto      += f"\n📁 {pasta}/\n"
            pasta_atual = pasta
        texto += f"   → {citacao}\n"

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
    - Sobre memória de sessões anteriores: use-a como referência,
    mas nunca afirme com certeza absoluta o que foi ou não dito.
    Se não tiver certeza, diga "não tenho memória clara disso"
    em vez de negar categoricamente.

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
    - Sobre memória de sessões anteriores: use-a como referência,
    mas nunca afirme com certeza absoluta o que foi ou não dito.
    Se não tiver certeza, diga "não tenho memória clara disso"
    em vez de negar categoricamente.

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