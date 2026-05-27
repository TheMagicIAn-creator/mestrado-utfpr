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

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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

ORCAMENTOS_RAG = {
    "groq": {
        "n_pool": 24,
        "n_resultados": 6,
        "contexto_chars": 7_000,
        "sessao_chars": 800,
        "historico_turnos": 10,
        "historico_chars": 900,
        "max_prompt_chars": 28_000,
    },
    "gemini": {
        "n_pool": 36,
        "n_resultados": 10,
        "contexto_chars": 12_000,
        "sessao_chars": 1_500,
        "historico_turnos": 14,
        "historico_chars": 1_400,
        "max_prompt_chars": 48_000,
    },
    "padrao": {
        "n_pool": 30,
        "n_resultados": 8,
        "contexto_chars": 9_000,
        "sessao_chars": 1_100,
        "historico_turnos": 10,
        "historico_chars": 900,
        "max_prompt_chars": 34_000,
    },
}

PERFIL_COMPACTO = """
Você é o Al IAdo PV — pesquisador sênior e coorientador técnico do Rodolfo
Torres no mestrado da UTFPR. Especialista em manutenção preditiva de inversores
fotovoltaicos on-grid, FMEA/FMECA, RCM, confiabilidade, sinais elétricos CA e
Machine Learning aplicado a detecção de anomalias.

══════════════════════════════════════════════════════════════
REGRAS DE CONVERSA (LEIA ANTES DE RESPONDER)
══════════════════════════════════════════════════════════════

1. CUMPRIMENTO — REGRA RÍGIDA
   - Cumprimente APENAS na PRIMEIRA mensagem da conversa (quando não houver
     histórico) ou quando o Rodolfo te cumprimentar primeiro.
   - Se já existe histórico, NÃO comece com "Bom dia/Boa tarde/Boa noite".
     Vá direto ao ponto. Você é um colega numa conversa em andamento, não um
     atendente que toda hora se reapresenta.

2. HISTÓRICO — USE ATIVAMENTE
   - LEIA o histórico antes de responder. Você é responsável pela continuidade
     da conversa.
   - Se você fez uma pergunta no turno anterior e o Rodolfo respondeu "sim",
     "pode seguir", "continue", "ok", "vamos lá", "podemos" — isso é
     CONSENTIMENTO PARA EXECUTAR o que VOCÊ propôs. Execute, não repergunte.
   - Não repita perguntas que já fez. Não gire em círculos. Avance.
   - Quando o Rodolfo for vago, recupere o contexto do histórico e proponha
     o passo concreto que está faltando.

3. INICIATIVA TÉCNICA
   - Você tem capacidade OPERACIONAL de executar etapas do pipeline. Quando o
     Rodolfo pedir para rodar, treinar, recalcular, refazer ou consultar
     resultados, a ferramenta correspondente é acionada automaticamente.
   - NUNCA diga "não tenho capacidade de executar" — você tem.
   - Se a ferramenta acionou e voltou com erro, explique e proponha correção.

4. RIGOR E EVIDÊNCIA
   - Cite autor/ano APENAS quando usar evidência DIRETAMENTE relevante.
   - Se o contexto recuperado for irrelevante para a pergunta, IGNORE-O em
     silêncio. NÃO diga "o contexto trata de X" — só polui a resposta.
   - Se faltar evidência, diga: "isso não está coberto pela base que tenho
     aqui" e siga com conhecimento geral, separando bem os dois.
   - NUNCA invente números, autores, equações ou resultados.

5. VOZ E FORMA
   - Português brasileiro natural, técnico-acadêmico mas humano.
   - Trate o Rodolfo como colega de pesquisa — não como "usuário".
   - Emojis com moderação e propósito (🔬 ⚡ 📊 ✅ 💡 🎯 📈) — complementam,
     nunca substituem o conteúdo. NÃO use 🌃 nem outros emojis "de greeting"
     em mensagens que não são o início da conversa.
   - Pergunta simples → resposta curta. Pergunta profunda → resposta densa
     com tabelas, equações, comparações.
   - Reaja com naturalidade: "Boa pergunta!", "Excelente resultado!",
     "Aqui tem uma sutileza importante...", "Discordo um pouco — veja...".

══════════════════════════════════════════════════════════════
CONTEXTO DO PROJETO (memorize)
══════════════════════════════════════════════════════════════
- Tema: detecção preditiva de falhas em componentes CA de inversor fotovoltaico
  on-grid trifásico via ML, fundamentada em RCM/FMEA.
- TCC base (UFPA, 2024): FMECA do CEAMAZON. Inversor NPR=210 (mais crítico),
  subsistema CA NPR=150 (segundo mais crítico).
- Datasets: Paderborn (inversor SAUDÁVEL, 235k amostras, 10 kHz) para treinar
  o modelo de normalidade; PV Farms (rotulado, falhas CC) para classificação.
- Pipeline: features_ca → autoencoder → injecao_falhas → validacao → rul_weibull.
- Resultados: AE com limiar p99=2,91 (μ+3σ baseline=0,30); injeção LCL com
  AUC=0,935 (sev=1,0); desbalanceamento com AUC=1,000 e Recall=1,0; sensor CA
  com AUC=1,000.
- Orientadora: Profª. Fernanda Cristina Correa. Defesa: março/2027.
""".strip()

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


def _normalizar_texto(texto: str) -> str:
    import re
    import unicodedata

    texto = texto.lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9\s]", " ", texto)


def _saudacao_pelo_horario() -> str:
    """Retorna 'Bom dia', 'Boa tarde' ou 'Boa noite' conforme a hora atual."""
    from datetime import datetime
    hora = datetime.now().hour
    if 5 <= hora < 12:
        return "Bom dia"
    if 12 <= hora < 18:
        return "Boa tarde"
    return "Boa noite"


def resposta_interacao_simples(pergunta: str) -> str | None:
    """
    Responde localmente a mensagens puramente conversacionais sem acionar RAG/LLM.
    Cobre cumprimentos, despedidas, agradecimentos, reações casuais e correções
    de horário — tudo o que não justifica busca na literatura.

    Guards (em ordem):
      1. Se contém '?' → tem pergunta → não intercepta.
      2. Se contém palavra interrogativa (que, qual, onde, cade, quando...) → não intercepta.
      3. Se contém termo técnico do mestrado → não intercepta.
      4. Se tem mais de 14 palavras → não intercepta.
    """
    pergunta_original = pergunta or ""
    txt = _normalizar_texto(pergunta_original).strip()
    termos = [t for t in txt.split() if t]

    if not termos:
        return None

    # Guard 1: ponto de interrogação no texto original → é pergunta de verdade.
    if "?" in pergunta_original:
        return None

    # Guard 2: palavras interrogativas → é pergunta mesmo sem '?'.
    PALAVRAS_INTERROGATIVAS = {
        "que", "qual", "quais", "quanto", "quantos", "quanta", "quantas",
        "onde", "cade", "quando", "como", "por", "porque", "pq", "porquê",
        "poderia", "poderias", "consegue", "consegues", "pode",
    }
    if any(t in PALAVRAS_INTERROGATIVAS for t in termos):
        return None

    # Guard 3: termos técnicos → RAG/ferramentas resolvem.
    TERMOS_PESQUISA = {
        "fmea", "fmeca", "npr", "rpn", "weibull", "rul", "mttf", "b10",
        "autoencoder", "inversor", "inversores", "fotovoltaico", "fotovoltaica",
        "pv", "falha", "falhas", "pipeline", "feature", "features", "validacao",
        "auc", "f1", "recall", "precision", "dataset", "paderborn", "rcm",
        "modelo", "algoritmo", "dissertacao", "mestrado", "metodologia",
        "confiabilidade", "lcl", "igbt", "thd", "fft", "rms", "anomalia",
        "smd", "ceamazon", "filtro", "sensor", "harmonicos", "deteccao",
        "literatura", "artigo", "paper", "tese", "tcc", "metrica", "metricas",
        "resultado", "resultados", "limiar", "baseline", "ml", "imagem",
        "imagens", "grafico", "graficos", "figura", "figuras", "curva",
        "curvas", "plot", "roc", "matriz", "tabela", "internet", "web",
        "wikipedia", "google", "pesquise", "pesquisar", "busque", "buscar",
        "hora", "horas", "data", "dia",
    }
    if any(t in TERMOS_PESQUISA for t in termos):
        return None

    # Guard 4: mensagem comprida geralmente carrega intenção técnica.
    if len(termos) > 12:
        return None

    saudacao_h = _saudacao_pelo_horario()

    palavras_bomdia = {"bom dia", "bomdia"}
    palavras_boatarde = {"boa tarde", "boatarde"}
    palavras_boanoite = {"boa noite", "boanoite"}
    saudacoes_genericas = {
        "oi", "ola", "opa", "salve", "eai", "eae", "hey", "alo",
        "olá", "fala", "fala ai", "fala cara", "tudo bem", "tudo certo",
        "como vai", "como esta",
    }
    agradecimentos = {
        "obrigado", "obrigada", "valeu", "thanks", "grato", "grata",
        "agradeco", "agradeço", "obg", "vlw",
    }
    despedidas = {
        "tchau", "ate", "ateh", "ate mais", "falou", "ate logo",
        "ate amanha", "ate breve", "vou indo", "ate depois",
    }
    reacoes_curtas = {
        "kkk", "kk", "rs", "rsrs", "haha", "hahaha", "hehe", "hehehe",
        "legal", "show", "massa", "top", "bacana", "blz", "beleza",
        "ok", "okay", "certo", "entendi", "entendido", "perfeito",
        "ótimo", "otimo", "boa", "bom", "fechou", "combinado",
    }

    tem_bomdia = "bom dia" in txt or "bomdia" in txt
    tem_boatarde = "boa tarde" in txt or "boatarde" in txt
    tem_boanoite = "boa noite" in txt or "boanoite" in txt
    tem_saudacao_gen = any(t in saudacoes_genericas for t in termos)
    tem_agradecimento = any(t in agradecimentos for t in termos)
    tem_despedida = any(t in despedidas for t in termos) or txt.startswith("ate ")
    tem_reacao = any(t in reacoes_curtas for t in termos)

    # ── Correção de horário (ex.: "Tá de noite cara kkk") ─────
    fala_de_noite = "de noite" in txt or "ta noite" in txt or "esta noite" in txt
    fala_de_tarde = "de tarde" in txt or "ta tarde" in txt or "esta tarde" in txt
    fala_de_dia = "de dia" in txt or "ta dia" in txt or "esta dia" in txt
    if fala_de_noite or fala_de_tarde or fala_de_dia:
        return (
            f"Boa correção! 😅 Eu estava no automático — me perdoe. "
            f"**{saudacao_h}**, Rodolfo. Como posso te ajudar agora?"
        )

    # ── Cumprimentos específicos por período ──────────────────
    if tem_bomdia:
        if saudacao_h == "Bom dia":
            return f"Bom dia, Rodolfo! ☀️ Pronto para mais um dia de pesquisa?"
        return (
            f"Saudação anotada, mas aqui já é **{saudacao_h.lower()}** 🌙. "
            f"De qualquer forma, estou por aqui pronto para o que precisar."
        )
    if tem_boatarde:
        if saudacao_h == "Boa tarde":
            return f"Boa tarde, Rodolfo! 📚 Em que posso ajudar a destravar o trabalho hoje?"
        return f"Aqui na verdade é **{saudacao_h.lower()}**, mas estou à disposição."
    if tem_boanoite:
        if saudacao_h == "Boa noite":
            return (
                f"Boa noite, Rodolfo! 🌙 Estou por aqui — pode pedir para discutir "
                f"literatura, rodar o pipeline ou interpretar resultados."
            )
        return f"Aqui ainda é **{saudacao_h.lower()}**, mas seja bem-vindo!"

    # ── Saudações genéricas ───────────────────────────────────
    if tem_saudacao_gen:
        return (
            f"{saudacao_h}, Rodolfo! 👋 Pode me pedir para revisar literatura, "
            f"rodar etapas do pipeline ou pensar junto sobre a dissertação."
        )

    # ── Despedidas ────────────────────────────────────────────
    if tem_despedida:
        return (
            f"Até mais, Rodolfo! 👋 Quando voltar, é só puxar o assunto onde paramos. "
            f"Boa pesquisa!"
        )

    # ── Agradecimentos ────────────────────────────────────────
    if tem_agradecimento:
        return "Disponha! 🤝 Seguimos lapidando o mestrado com calma e rigor."

    # ── Reações curtas ────────────────────────────────────────
    if tem_reacao and len(termos) <= 5:
        return "Show. 🙂 Quando quiser continuar, é só mandar a próxima pergunta."

    return None


def _orcamento_rag(nome_provedor: str | None = None) -> dict:
    nome = (nome_provedor or "").lower()
    if "groq" in nome or "llama" in nome:
        return ORCAMENTOS_RAG["groq"].copy()
    if "gemini" in nome or "google" in nome:
        return ORCAMENTOS_RAG["gemini"].copy()
    return ORCAMENTOS_RAG["padrao"].copy()


def _limitar_texto(texto: str, limite: int) -> str:
    if limite <= 0 or len(texto) <= limite:
        return texto
    corte = texto[:limite].rsplit(" ", 1)[0].strip()
    return corte + "\n[trecho encurtado para caber no limite do provedor]"


def _tokens_busca(pergunta: str) -> list[str]:
    stopwords = {
        "a", "as", "o", "os", "de", "da", "do", "das", "dos", "e", "em",
        "no", "na", "nos", "nas", "um", "uma", "para", "por", "sobre",
        "fale", "explique", "quais", "qual", "como", "que", "com",
    }
    termos = [
        t for t in _normalizar_texto(pergunta).split()
        if len(t) > 2 and t not in stopwords
    ]
    extras = []
    mapa = {
        "fmea": ["failure", "mode", "effects", "analysis", "fmeca", "npr", "rpn"],
        "fmeca": ["fmea", "criticidade", "criticality", "npr", "rpn"],
        "npr": ["rpn", "fmea", "criticidade"],
        "rpn": ["npr", "fmea", "risk", "priority"],
        "weibull": ["confiabilidade", "rul", "mttf", "b10"],
        "autoencoder": ["anomalia", "reconstrucao", "detector"],
        "inversor": ["fotovoltaico", "pv", "converter", "inverter"],
    }
    for termo in termos:
        extras.extend(mapa.get(termo, []))
    return list(dict.fromkeys(termos + extras))


def _formatar_historico(historico: list, orcamento: dict) -> str:
    if not historico:
        return ""

    linhas = ["\nHISTORICO RECENTE DA CONVERSA:"]
    turnos = historico[-orcamento["historico_turnos"]:]
    for turno in turnos:
        role = "Rodolfo" if turno.get("role") == "user" else "Al IAdo PV"
        content = _limitar_texto(
            str(turno.get("content", "")),
            orcamento["historico_chars"],
        )
        linhas.append(f"\n{role}:\n{content}")
    return "\n".join(linhas)


def _contexto_temporal() -> str:
    """Gera bloco com data, hora e dia da semana atuais."""
    from datetime import datetime
    dias = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
            "sexta-feira", "sábado", "domingo"]
    meses = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
             "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
    agora = datetime.now()
    saudacao = _saudacao_pelo_horario()
    return (
        f"DATA E HORA ATUAL: {dias[agora.weekday()]}, {agora.day} de "
        f"{meses[agora.month - 1]} de {agora.year}, às {agora.strftime('%H:%M')}. "
        f"Período do dia: {saudacao.lower()}."
    )


def _montar_prompt(pergunta: str,
                   contexto: str,
                   historico_formatado: str,
                   orcamento: dict) -> str:
    contexto = _limitar_texto(contexto, orcamento["contexto_chars"])
    bloco_temporal = _contexto_temporal()
    tem_contexto = bool(contexto.strip())
    contexto_bloco = contexto if tem_contexto else "Nenhum trecho relevante recuperado."

    # Marca explicitamente se já existe histórico — chave para impedir
    # o cumprimento repetido a cada mensagem.
    tem_historico = bool(historico_formatado and historico_formatado.strip())
    if tem_historico:
        estado_conversa = (
            "ESTADO DA CONVERSA: em andamento (há histórico anterior). "
            "NÃO cumprimente nesta resposta — vá direto ao conteúdo. "
            "Leia o histórico abaixo para entender o que já foi proposto."
        )
    else:
        estado_conversa = (
            "ESTADO DA CONVERSA: primeira interação. "
            "Você pode cumprimentar pelo período do dia (use a data/hora acima)."
        )

    prompt = f"""
{PERFIL_COMPACTO}

{bloco_temporal}

{estado_conversa}

CONTEXTO RECUPERADO DA LITERATURA E MEMÓRIA:
{contexto_bloco}
{historico_formatado}

PERGUNTA ATUAL DO PESQUISADOR:
{pergunta}

INSTRUCOES OBRIGATÓRIAS DE RESPOSTA:
- Releia as REGRAS DE CONVERSA do perfil — em especial as regras 1 (cumprimento)
  e 2 (uso do histórico).
- Se o ESTADO DA CONVERSA acima for "em andamento", NÃO comece com "Bom dia",
  "Boa tarde", "Boa noite" nem com qualquer saudação.
- Se a pergunta atual for confirmação curta ("sim", "pode seguir", "continue",
  "ok"), interprete como aceite do que VOCÊ propôs no último turno e EXECUTE.
- Se a evidência recuperada não tem relação com a pergunta, IGNORE-A em silêncio.
- Cite autor/ano apenas quando a evidência for de fato usada.
- Tamanho da resposta proporcional ao pedido. Pergunta curta → resposta curta.
- Não invente números, autores, equações ou resultados.
""".strip()

    if len(prompt) > orcamento["max_prompt_chars"]:
        excesso = len(prompt) - orcamento["max_prompt_chars"]
        novo_limite = max(2_000, len(contexto) - excesso - 500)
        contexto = _limitar_texto(contexto, novo_limite)
        contexto_bloco = contexto if contexto.strip() else "Nenhum trecho relevante recuperado."
        prompt = f"""
{PERFIL_COMPACTO}

{bloco_temporal}

CONTEXTO RECUPERADO:
{contexto_bloco}
{historico_formatado}

PERGUNTA ATUAL DO PESQUISADOR:
{pergunta}

INSTRUCOES DE RESPOSTA:
- Português brasileiro, voz natural, precisão técnica.
- Use emojis com moderação (🔬 📊 ✅).
- Cumprimente pelo período do dia quando apropriado.
- Cite autor/ano só quando usar evidência relevante.
- Se a evidência não for relevante, ignore-a sem comentar.
- Ajuste o tamanho ao pedido. Não invente números.
""".strip()

    return prompt

# ============================================================
# RAG AVANÇADO — 3 CAMADAS
# ============================================================

def _expandir_query(pergunta: str) -> dict:
    """
    CAMADA 1 — Expansão de query local.
    Evita chamadas auxiliares ao LLM para nao consumir TPM antes da resposta.
    """
    termos = _tokens_busca(pergunta)
    variacoes = [pergunta]

    txt = _normalizar_texto(pergunta)
    if "fmea" in txt or "fmeca" in txt:
        variacoes.extend([
            "analise de modos e efeitos de falha",
            "failure mode and effects analysis",
            "risk priority number rpn npr criticality",
        ])
    if "weibull" in txt or "rul" in txt:
        variacoes.extend([
            "confiabilidade weibull vida util remanescente",
            "reliability weibull remaining useful life mttf b10",
        ])
    if "autoencoder" in txt or "anomalia" in txt:
        variacoes.extend([
            "detector de anomalias por erro de reconstrucao",
            "autoencoder anomaly detection reconstruction error",
        ])

    return {
        "variacoes": list(dict.fromkeys(variacoes)),
        "termos": termos[:12],
    }


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
                limit          = 60
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
    CAMADA 3 — Reranking local.
    Pontua os chunks por sobreposicao lexical e sinais de dominio.
    Isso reduz latencia e evita gastar TPM com chamadas intermediarias.
    """
    if not candidatos:
        return []

    if len(candidatos) <= n_final:
        return candidatos

    termos = _tokens_busca(pergunta)
    pergunta_norm = _normalizar_texto(pergunta)
    numeros = {t for t in pergunta_norm.split() if any(ch.isdigit() for ch in t)}

    pontuados = []
    for ordem, (doc, meta) in enumerate(candidatos):
        texto = " ".join([
            str(meta.get("citacao", "")),
            str(meta.get("titulo", "")),
            str(meta.get("arquivo", "")),
            doc,
        ])
        texto_norm = _normalizar_texto(texto)
        score = 0.0

        for termo in termos:
            if termo in texto_norm:
                score += 2.0 if len(termo) > 4 else 1.0
                if termo in _normalizar_texto(str(meta.get("citacao", ""))):
                    score += 1.5

        for numero in numeros:
            if numero in texto_norm:
                score += 2.0

        if "tabela" in texto_norm or "table" in texto_norm:
            score += 0.4
        if any(x in texto_norm for x in ("resultado", "metodo", "method", "equacao", "equation")):
            score += 0.3

        pontuados.append((score, -ordem, doc, meta))

    pontuados.sort(reverse=True)
    selecionados = [(doc, meta) for score, _, doc, meta in pontuados if score > 0]
    if not selecionados:
        selecionados = [(doc, meta) for _, _, doc, meta in pontuados]
    return selecionados[:n_final]


# ============================================================
# BUSCA DE CONTEXTO — PIPELINE RAG 3 CAMADAS
# ============================================================

def buscar_contexto(
    pergunta        : str,
    modelo_embeddings,
    colecao,
    colecao_sessoes = None,
    n_pool          : int | None = None,
    n_resultados    : int | None = None,
    contexto_chars  : int | None = None,
    sessao_chars    : int | None = None,
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
        variacoes,
        termos,
        colecao,
        modelo_embeddings,
        n_pool=n_pool or 30,
    )

    # ── CAMADA 3 — Reranking ─────────────────────────────────
    melhores = _rerankar(candidatos, pergunta, n_resultados or min(N_RESULTADOS, 8))

    # Monta contexto da literatura
    if melhores:
        contexto += "\n📚 DA LITERATURA CIENTÍFICA:\n"
        usados = len(contexto)
        limite = contexto_chars or 10_000
        for doc, meta in melhores:
            arquivo = meta.get("arquivo", "")
            citacao = meta.get("citacao", arquivo)
            bloco = f"\n[Fonte: {citacao}]\n{doc}\n"
            if usados + len(bloco) > limite:
                restante = limite - usados - len(f"\n[Fonte: {citacao}]\n")
                if restante <= 300:
                    break
                bloco = f"\n[Fonte: {citacao}]\n{_limitar_texto(doc, restante)}\n"
            if arquivo and arquivo not in citacoes:
                citacoes[arquivo] = citacao
            contexto += bloco
            usados += len(bloco)
            if usados >= limite:
                break

    # ── Sessões — busca direta (sem reranking) ───────────────
    if colecao_sessoes:
        try:
            vetor_pergunta = modelo_embeddings.encode([pergunta]).tolist()
            resultados_ses = colecao_sessoes.query(
                query_embeddings = vetor_pergunta,
                n_results        = max(2, min(4, (n_resultados or 8) // 2))
            )
            docs_ses  = resultados_ses.get("documents", [[]])[0]
            metas_ses = resultados_ses.get("metadatas",  [[]])[0]

            if docs_ses:
                contexto += "\n💭 DA MEMÓRIA DE SESSÕES ANTERIORES:\n"
                usados_ses = 0
                limite_ses = sessao_chars or 1_200
                for doc, meta in zip(docs_ses, metas_ses):
                    arquivo = meta.get("arquivo", "")
                    bloco = f"\n[Memória: {arquivo}]\n{doc}\n"
                    if usados_ses + len(bloco) > limite_ses:
                        restante = limite_ses - usados_ses - len(f"\n[Memória: {arquivo}]\n")
                        if restante <= 200:
                            break
                        bloco = f"\n[Memória: {arquivo}]\n{_limitar_texto(doc, restante)}\n"
                    contexto += bloco
                    usados_ses += len(bloco)
                    if usados_ses >= limite_ses:
                        break
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
    colecao_sessoes    = None,
    nome_provedor: str | None = None,
) -> tuple:
    """
    Prepara o prompt completo sem invocar o LLM.
    Retorna (prompt_str, citacoes_dict).
    Usado pelo Streamlit para fazer streaming separado.
    """

    if historico is None:
        historico = []

    orcamento = _orcamento_rag(nome_provedor)
    contexto, citacoes = buscar_contexto(
        pergunta,
        modelo_embeddings,
        colecao,
        colecao_sessoes,
        n_pool=orcamento["n_pool"],
        n_resultados=orcamento["n_resultados"],
        contexto_chars=orcamento["contexto_chars"],
        sessao_chars=orcamento["sessao_chars"],
    )

    historico_formatado = _formatar_historico(historico, orcamento)
    prompt = _montar_prompt(pergunta, contexto, historico_formatado, orcamento)
    return prompt, citacoes

def perguntar(
    pergunta: str,
    perfil: str,
    modelo_embeddings,
    colecao,
    llm,
    historico: list = None,
    streaming: bool = True,
    colecao_sessoes = None,
    nome_provedor: str | None = None,
) -> str:
    """
    Pipeline RAG completo com memória e streaming.
    """

    if historico is None:
        historico = []

    resposta_simples = resposta_interacao_simples(pergunta)
    if resposta_simples:
        print(resposta_simples)
        return resposta_simples

    prompt, citacoes = preparar_prompt(
        pergunta=pergunta,
        perfil=perfil,
        modelo_embeddings=modelo_embeddings,
        colecao=colecao,
        historico=historico,
        colecao_sessoes=colecao_sessoes,
        nome_provedor=nome_provedor,
    )

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
