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
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# OpenMP duplicado (torch / numpy-MKL / onnxruntime do ChromaDB) ABORTA no
# Windows com access violation (EXIT 139), de forma INTERMITENTE conforme a
# ordem de carga. Precisa ser definido ANTES de importar sentence_transformers
# (que carrega o torch, logo abaixo) — inclusive em scripts que importam este
# módulo antes do config (ex.: a bateria de testes).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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
from src.conhecimento.leitor_anexos import montar_bloco_texto_anexos, tem_imagem
from src.conhecimento.provedores import eh_multimodal

ORCAMENTOS_RAG = {
    "groq": {
        "n_pool": 60,
        "n_resultados": 10,
        "n_resultados_revisao": 16,
        "max_chunks_por_fonte": 2,
        "contexto_chars": 7_000,
        "sessao_chars": 800,
        "historico_turnos": 10,
        "historico_chars": 900,
        "anexos_chars": 6_000,
        "max_prompt_chars": 28_000,
    },
    "gemini": {
        "n_pool": 120,
        "n_resultados": 16,
        "n_resultados_revisao": 28,
        "max_chunks_por_fonte": 2,
        "contexto_chars": 14_000,
        "sessao_chars": 1_500,
        "historico_turnos": 14,
        "historico_chars": 1_400,
        "anexos_chars": 14_000,
        "max_prompt_chars": 48_000,
    },
    "padrao": {
        "n_pool": 80,
        "n_resultados": 12,
        "n_resultados_revisao": 20,
        "max_chunks_por_fonte": 2,
        "contexto_chars": 10_000,
        "sessao_chars": 1_100,
        "historico_turnos": 10,
        "historico_chars": 900,
        "anexos_chars": 9_000,
        "max_prompt_chars": 34_000,
    },
}

# Pesos por pasta tematica — usados no rerank para priorizar literatura
# nuclear do mestrado (PV, ML preditivo, manutencao) sobre material lateral.
PESOS_PASTA = {
    "inversores-pv": 1.4,
    "ml-preditivo": 1.2,
    "manutencao": 1.2,
    "confiabilidade": 1.2,
    "sinais-eletricos": 0.2,
}

# Livros-texto generalistas — uteis em contextos especificos mas ruido em
# perguntas amplas. Recebem penalidade pesada salvo quando a pergunta
# menciona explicitamente o dominio do livro.
TEXTBOOKS_PENALIZADOS = {
    # Gatilhos exigem contexto INEQUIVOCO do dominio do livro — palavras
    # ambiguas como "calculo" (calcular um limiar) ou "imagem" (imagem
    # termografica de PV) NAO devem dar passe livre.
    "stewart_calculo-volume-i_2013.pdf": {
        "stewart", "calculus", "calculo integral", "calculo diferencial",
        "integral indefinida", "integral definida", "derivada parcial",
        "serie de taylor", "convergencia de serie", "limite matematico",
    },
    "gonzalez_digital-image-processing_2008.pdf": {
        "gonzalez", "processamento de imagem", "image processing",
        "filtro espacial", "morfologia matematica", "segmentacao de imagem",
    },
    "tekalp_digital-video-processing_2015.pdf": {
        "tekalp", "processamento de video", "video processing",
        "fluxo otico", "compressao de video",
    },
    "oppenheim_discrete-time-signal-processing_2014.pdf": {
        "oppenheim", "transformada z", "z transform", "fir iir",
    },
    "smith_the-scientist-and-engineer-s-guide-to-digital-signal-process_1999.pdf": {
        "guia dsp", "scientist engineer dsp",
    },
    "diniz_digital-signal-processing-system-analysis-and-design_2021.pdf": {
        "diniz dsp", "design dsp",
    },
    "grewal_kalman-filtering-theory-and-practice-using-matlab_2001.pdf": {
        "kalman", "filtro de kalman", "kalman filter",
    },
    "grewal_power-electronics-chapter-8_2002.pdf": {
        "eletronica de potencia", "power electronics", "snubber",
    },
}

# Indicadores de pergunta de revisao bibliografica / panorama da literatura.
# Disparam expansao agressiva da query nos topicos da dissertacao e aumentam
# o orcamento de resultados.
INDICADORES_REVISAO = (
    "literatura completa", "literatura toda", "revisao bibliografica",
    "revisao da literatura", "estado da arte", "todos os artigos",
    "todas as fontes", "todas as referencias", "todos os autores",
    "panorama", "sintese da literatura", "survey", "review",
    "literatura da dissertacao", "literatura do mestrado",
    "literatura sobre", "cite a literatura", "cite as referencias",
    "cite as fontes", "cite os artigos", "cite os autores",
    "completa da dissertacao", "fundamentacao teorica",
    "referencial teorico", "bibliografia completa",
)

# Variacoes injetadas em queries de revisao — cobrem os pilares teoricos
# da dissertacao para acionar busca semantica em todos os documentos
# relevantes simultaneamente.
TOPICOS_DISSERTACAO = (
    "FMEA FMECA failure mode and effects analysis criticality NPR RPN",
    "RCM reliability centered maintenance manutencao centrada em confiabilidade",
    "inversor fotovoltaico on-grid trifasico PV inverter falhas",
    "autoencoder anomaly detection erro de reconstrucao normalidade",
    "machine learning fault detection PV inverter diagnostico",
    "Weibull RUL remaining useful life MTTF B10 confiabilidade",
    "filtro LCL IGBT capacitor contactor sensor CA inversor",
    "manutencao preditiva monitoramento de condicao prognostico",
    "harmonicos THD sinais eletricos CA inversor fotovoltaico",
    "isolation forest random forest XGBoost classificacao falhas PV",
    "dataset Paderborn IGBT trifasico inversor saudavel benchmark",
    "FMECA NPR criticidade inversor lado CA componente critico",
)

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
   - NÍVEIS DE EVIDÊNCIA — sempre informe ao falar de resultados:
       E0 = hipótese; E1 = benchmark exploratório (perturbação genérica ou
       dataset rotulado, ex.: experimentos por artigo); E2 = validação sintética
       orientada pelo FMEA (injeção/validação do pipeline principal); E3 =
       validação experimental externa em bancada/campo.
     NUNCA trate E1 ou E2 como prova de desempenho industrial. Um limiar
     escolhido no próprio conjunto avaliado é EXPLORATÓRIO (E1), não estimativa
     de generalização.

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

6. IDIOMAS E TRADUÇÃO TÉCNICA
   - Entenda português, inglês, espanhol e francês. Responda em português
     brasileiro por padrão, mas acompanhe o idioma da pergunta se Rodolfo
     escrever claramente em EN/ES/FR ou pedir tradução.
   - Traduza mentalmente termos técnicos para recuperar contexto e resultados:
     fault/falla/faille ↔ falha; anomaly/anomalía/anomalie ↔ anomalia;
     reliability/confiabilidad/fiabilité ↔ confiabilidade; maintenance/
     mantenimiento/maintenance ↔ manutenção; inverter/inversor/onduleur ↔ inversor.
   - Não trate mudança de idioma como assunto paralelo. Use-a só para entender
     melhor a intenção e responder com precisão.

7. RESULTADOS, GRÁFICOS E PARECER
   - Tabelas e gráficos são evidência, não a resposta inteira.
   - Se Rodolfo pedir "mostre", "exiba" ou "matriz", seja direto e traga o
     artefato correto.
   - Se Rodolfo pedir opinião, explicação, apresentação para orientadora,
     escolha metodológica ou implicação para a dissertação, interprete os
     números: priorize modelos, explique trade-offs, aponte ressalvas e diga
     o que isso sustenta academicamente.
   - Sempre diferencie: dados locais do repositório, metodologia inspirada em
     artigo, falhas sintéticas e resultados copiados. Não deixe essa origem
     ambígua.

══════════════════════════════════════════════════════════════
CONTEXTO DO PROJETO (memorize)
══════════════════════════════════════════════════════════════
- Tema: detecção preditiva de falhas em componentes CA de inversor fotovoltaico
  on-grid trifásico via ML, fundamentada em RCM/FMEA.
- TCC base (UFPA, 2024): FMECA do CEAMAZON. Inversor NPR=210 (mais crítico),
  subsistema CA NPR=150 (segundo mais crítico).
- Datasets: Paderborn (inversor SAUDÁVEL, 235k amostras, 10 kHz) para treinar
  o modelo de normalidade; PV Farms (rotulado, falhas CC) para classificação.
- SEPARAÇÃO DE DOMÍNIO (regra rígida): Paderborn → detecção de anomalia CA do
  inversor por modelagem de normalidade; PV Farms → classificação supervisionada
  de falhas CC conhecidas (string, string-terra, string-string). NUNCA afirme
  que o classificador PV Farms diagnostica falhas CA do inversor, nem transfira
  métricas de PV Farms para o pipeline CA. Os dois NÃO se fundem: o uso é
  conceitual/arquitetural, não fusão de dados.
- Experimentos por artigo usam dados locais do repositório. Ghoneim usa
  dados/brutos/train_data.csv e test_data.csv; Francisti, Ibrahim, Sharma e
  Ahirwar usam features locais do Paderborn extraídas de Inverter_Data_Set.csv.
  Como Paderborn é saudável, a validação de anomalia usa falhas sintéticas
  geradas no pipeline para criar ground truth.
- Pipeline: features_ca → autoencoder → injecao_falhas → validacao → rul_weibull.
- Limiar operacional do Autoencoder = percentil 99 do erro de reconstrução
  saudável; μ+3σ é apenas referência comparativa (nunca o limiar em uso).
- NÃO memorize métricas (limiar, AUC, F1, SMD, MTTF). Os números ficam nos
  artefatos (resultados/...) e são carregados dinamicamente pela ferramenta de
  consulta de resultados. Ao falar de desempenho, consulte o artefato ATUAL e
  diga o nível de evidência — nunca cite um número de memória que pode estar
  desatualizado após novo treino ou exclusão de artefato.
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


def pedido_sem_literatura(pergunta: str) -> bool:
    """
    True quando o pesquisador proibe explicitamente literatura/fontes.

    Importante: a palavra "literatura" por si so costuma acionar RAG. Sem este
    guard, prompts como "Nao use literatura; explique FMEA com base no projeto"
    acabam recebendo rodape de fontes, contrariando a instrucao principal.
    """
    txt = _normalizar_texto(pergunta or "")
    alvos = (
        "literatura", "fontes", "fonte", "referencias", "referencia",
        "artigos", "artigo", "papers", "paper", "bibliografia",
        "literature", "sources", "source", "references", "reference",
        "bibliography", "fuentes", "referencias", "bibliografia",
        "litterature", "littérature", "bibliographie",
    )
    negacoes = (
        "nao use", "nao consulte", "nao buscar", "nao busque", "nao cite",
        "sem", "dispense", "ignore",
        "do not use", "dont use", "without", "do not cite",
        "no use", "no consultes", "sin", "ne pas utiliser", "sans",
    )
    if any(n in txt for n in negacoes) and any(a in txt for a in alvos):
        return True
    return any(t in txt for t in (
        "somente com base no projeto",
        "apenas com base no projeto",
        "so com base no projeto",
        "com base no projeto apenas",
        "use somente o projeto",
        "use apenas o projeto",
        "only based on the project",
        "solo con base en el proyecto",
        "seulement sur la base du projet",
    ))


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
        "what", "which", "where", "when", "how", "why", "can", "could",
        "que", "cual", "cuales", "donde", "cuando", "como", "por", "puede",
        "quoi", "quel", "quels", "quelle", "quelles", "ou", "quand",
        "comment", "pourquoi", "peux", "pouvez",
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
        "hora", "horas", "data",
        "fault", "failure", "failures", "anomaly", "anomalies", "reliability",
        "maintenance", "inverter", "photovoltaic", "dataset", "paper",
        "source", "sources", "reference", "references", "metrics", "results",
        "chart", "charts", "figure", "figures", "confusion", "matrix",
        "falla", "fallas", "anomalia", "anomalias", "confiabilidad",
        "mantenimiento", "inversor", "fotovoltaico", "articulo", "fuente",
        "referencia", "metricas", "resultados", "grafico", "matriz",
        "defaillance", "defaillances", "anomalie", "fiabilite", "onduleur",
        "photovoltaique", "article", "source", "reference", "resultats",
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
        "como vai", "como esta", "hi", "hello", "hola", "buenas",
        "bonjour", "salut",
    }
    agradecimentos = {
        "obrigado", "obrigada", "valeu", "thanks", "grato", "grata",
        "agradeco", "agradeço", "obg", "vlw", "thank", "gracias", "merci",
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
                f"Boa noite, Rodolfo! 🌙 Estou por aqui — pode pedir para rodar "
                f"o pipeline, interpretar resultados ou pensar junto na dissertação."
            )
        return f"Aqui ainda é **{saudacao_h.lower()}**, mas seja bem-vindo!"

    # ── Saudações genéricas ───────────────────────────────────
    if tem_saudacao_gen:
        return (
            f"{saudacao_h}, Rodolfo! 👋 Pode me pedir para rodar etapas do "
            f"pipeline, interpretar resultados ou pensar junto sobre a dissertação."
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
        "the", "and", "for", "with", "about", "from", "what", "which", "how",
        "show", "explain", "compare", "give", "me",
        "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "sobre",
        "con", "para", "por", "que", "cual", "cuales", "como", "explique",
        "le", "la", "les", "des", "du", "un", "une", "avec", "pour", "sur",
        "quel", "quels", "quelle", "quelles", "comment", "expliquez",
    }
    termos = [
        t for t in _normalizar_texto(pergunta).split()
        if len(t) > 2 and t not in stopwords
    ]
    extras = []
    # Mapa multilingue PT<->EN<->ES<->FR. Quando a pergunta usa um termo em
    # qualquer idioma suportado, injetamos variantes tecnicas para que papers
    # em ingles e notas em portugues sejam recuperados pelo reranker lexico.
    mapa = {
        "fmea": [
            "failure", "mode", "effects", "analysis", "fmeca", "npr", "rpn",
            "modos", "falhas", "efeitos", "analise", "fallas", "efectos",
            "defaillance", "effets",
        ],
        "fmeca": [
            "fmea", "criticidade", "criticality", "npr", "rpn",
            "modos", "falhas", "efeitos", "analise", "tecnicas",
            "criticidad", "criticite",
        ],
        "npr": ["rpn", "fmea", "criticidade", "criticality"],
        "rpn": ["npr", "fmea", "risk", "priority"],
        "weibull": ["confiabilidade", "reliability", "fiabilidade", "fiabilite", "rul", "mttf", "b10"],
        "autoencoder": ["anomalia", "anomaly", "anomalia", "anomalie", "reconstrucao", "reconstruction", "detector"],
        "inversor": ["fotovoltaico", "pv", "converter", "inverter", "photovoltaic", "ondulador", "onduleur"],
        "inverter": ["inversor", "photovoltaic", "pv", "converter", "ondulador", "onduleur"],
        "ondulador": ["inversor", "inverter", "pv", "fotovoltaico"],
        "onduleur": ["inversor", "inverter", "pv", "photovoltaique"],
        "fotovoltaico": ["pv", "photovoltaic", "solar", "inverter", "fotovoltaica", "photovoltaique"],
        "photovoltaic": ["fotovoltaico", "pv", "solar", "inverter"],
        "manutencao": ["maintenance", "mantenimiento", "preventive", "predictive", "preditiva"],
        "maintenance": ["manutencao", "mantenimiento", "predictive", "reliability"],
        "mantenimiento": ["manutencao", "maintenance", "predictivo", "confiabilidad"],
        "preditiva": ["predictive", "predictivo", "manutencao", "maintenance", "prognosis"],
        "predictive": ["preditiva", "predictivo", "maintenance", "prognosis"],
        "confiabilidade": ["reliability", "confiabilidad", "fiabilite", "rcm", "weibull", "mttf"],
        "reliability": ["confiabilidade", "confiabilidad", "fiabilite", "rcm", "weibull"],
        "confiabilidad": ["confiabilidade", "reliability", "weibull"],
        "fiabilite": ["confiabilidade", "reliability", "weibull"],
        "rcm": ["reliability", "centered", "maintenance", "manutencao", "centrada", "centrado"],
        "anomalia": ["anomaly", "anomalie", "outlier", "detection", "deteccao"],
        "anomaly": ["anomalia", "anomalie", "outlier", "detection"],
        "anomalie": ["anomalia", "anomaly", "detection"],
        "deteccao": ["detection", "deteccion", "detection", "anomalia", "anomaly", "diagnosis"],
        "detection": ["deteccao", "deteccion", "anomalia", "anomaly", "diagnosis"],
        "deteccion": ["deteccao", "detection", "anomalia"],
        "falha": ["failure", "fault", "falla", "defaillance", "defect", "falhas"],
        "falhas": ["failure", "fault", "failures", "fallas", "defaillances", "modes"],
        "fault": ["falha", "failure", "falla", "defaillance", "diagnosis"],
        "failure": ["falha", "fault", "falla", "defaillance", "mode"],
        "falla": ["falha", "fault", "failure", "modo"],
        "defaillance": ["falha", "fault", "failure"],
        "lcl": ["filter", "filtro", "filtre", "passive", "harmonic", "harmonico"],
        "igbt": ["transistor", "switching", "power", "semiconductor"],
        "rul": ["remaining", "useful", "life", "vida", "util", "weibull", "mttf"],
        "ml": ["machine", "learning", "aprendizado", "aprendizaje", "algorithm"],
        "machine": ["learning", "ml", "algorithm", "aprendizado", "aprendizaje"],
        "learning": ["machine", "ml", "aprendizado", "aprendizaje"],
    }
    for termo in termos:
        extras.extend(mapa.get(termo, []))
    return list(dict.fromkeys(termos + extras))


# Autores/instituicoes presentes na base — qualquer mencao a um deles
# forca consulta a literatura, mesmo sem palavras como "fonte" ou
# "artigo". Inclui nomes proprios curtos (NASA) e siglas comuns. A
# lista oficial e materializada via autores_indexados(colecao) na
# primeira chamada, lendo o ChromaDB; este fallback cobre quando a
# colecao nao esta disponivel.
AUTORES_INDEXADOS_FALLBACK = {
    "nasa", "administration",
    "torres",
    "lafraia",
    "carpinetti",
    "sakurada",
    "muqauwim",
    "frontin",
    "moura",
    "eletrica",
    "stewart",
    "gonzalez",
    "tekalp",
    "oppenheim",
    "smith",
    "diniz",
    "grewal",
    "ahirwar",
    "francisti",
    "ghoneim",
    "ibrahim",
    "marangis",
    "narayanan",
    "puc-rio", "pucrio", "puc",
    "risi",
    "sharma",
    "silva",
    "xavier",
    "cristaldi",
    "dhople",
    "joshi",
    "karim",
    "monteiro",
    "pahwa",
    "patil",
    "shuttleworth",
    "stender",
    "voss",
    # Datasets/instituicoes — tambem ativam consulta a literatura
    "paderborn",
    "ceamazon",
    "ufpa",
    "utfpr",
    "ieee",
    "iec",
    "abnt",
    "iso",
    "mil-hdbk", "milhdbk",
}

_AUTORES_CACHE: set[str] = set()
_AUTOR_CANONICO_CACHE: dict[str, set[str]] = {}
# Mapa: autor canonico (ex.: 'Grewal') → lista de arquivos desse autor
# (necessario porque ha autores com varios papers, e where={"autor": X}
# pode trazer todos os chunks de um arquivo e nenhum do outro quando o
# limit nao cobre o primeiro).
_AUTOR_ARQUIVOS_CACHE: dict[str, set[str]] = {}


def autores_indexados(colecao=None) -> set[str]:
    """
    Retorna o conjunto de autores presentes no ChromaDB (campo 'autor'
    do metadado), em minusculas e normalizado. Em caso de erro ou colecao
    nao fornecida, usa o fallback hardcoded.

    Tambem popula _AUTOR_CANONICO_CACHE, que mapeia cada token normalizado
    (incluindo sub-tokens de autores compostos como 'Puc Rio' → 'puc' e
    'rio') para o conjunto de formas canonicas do metadado autor.
    """
    global _AUTORES_CACHE, _AUTOR_CANONICO_CACHE, _AUTOR_ARQUIVOS_CACHE
    if _AUTORES_CACHE:
        return _AUTORES_CACHE
    nomes: set[str] = set(AUTORES_INDEXADOS_FALLBACK)
    canonicos: dict[str, set[str]] = {}
    autor_arquivos: dict[str, set[str]] = {}
    if colecao is not None:
        try:
            offset, lote = 0, 500
            while True:
                r = colecao.get(limit=lote, offset=offset, include=["metadatas"])
                metas = r.get("metadatas", [])
                if not metas:
                    break
                for m in metas:
                    autor_raw = str(m.get("autor", "")).strip()
                    arquivo = str(m.get("arquivo", "")).lower()
                    autor_norm = _normalizar_texto(autor_raw).strip()
                    if autor_raw and arquivo:
                        autor_arquivos.setdefault(autor_raw, set()).add(arquivo)
                    if autor_norm:
                        nomes.add(autor_norm)
                        # Indexa o autor completo e cada sub-token
                        canonicos.setdefault(autor_norm, set()).add(autor_raw)
                        for sub in autor_norm.split():
                            if len(sub) > 2:
                                nomes.add(sub)
                                canonicos.setdefault(sub, set()).add(autor_raw)
                    if "_" in arquivo:
                        primeiro = arquivo.split("_", 1)[0]
                        primeiro_norm = _normalizar_texto(primeiro).strip()
                        if primeiro_norm and len(primeiro_norm) > 2:
                            nomes.add(primeiro_norm)
                            for sub in primeiro_norm.split():
                                if len(sub) > 2:
                                    nomes.add(sub)
                                    if autor_raw:
                                        canonicos.setdefault(sub, set()).add(autor_raw)
                if len(metas) < lote:
                    break
                offset += lote
        except Exception:
            pass
    _AUTORES_CACHE = nomes
    _AUTOR_CANONICO_CACHE = canonicos
    _AUTOR_ARQUIVOS_CACHE = autor_arquivos
    return nomes


def arquivos_do_autor(autor_canonico: str, colecao=None) -> set[str]:
    """Retorna o conjunto de arquivos (filenames) atribuidos a um autor canonico."""
    if not _AUTOR_ARQUIVOS_CACHE:
        autores_indexados(colecao)
    return _AUTOR_ARQUIVOS_CACHE.get(autor_canonico, set())


def autores_canonicos_para(token: str, colecao=None) -> set[str]:
    """
    Dado um token (sobrenome lowercase, ex.: 'puc', 'grewal'), retorna o
    conjunto de formas canonicas do metadado autor que cobrem esse token —
    ex.: 'puc' → {'Puc Rio'}, 'grewal' → {'Grewal'}.
    """
    if not _AUTOR_CANONICO_CACHE:
        autores_indexados(colecao)  # popula cache
    return _AUTOR_CANONICO_CACHE.get(_normalizar_texto(token).strip(), set())


def deve_consultar_literatura(pergunta: str, colecao=None) -> bool:
    """
    Retorna True quando o pedido:
      - menciona explicitamente literatura/artigo/fonte/referencia,
      - OU cita o sobrenome de um autor indexado (Torres, NASA, Ahirwar...),
      - OU pergunta sobre presenca/localizacao de uma fonte
        ("tem o Stender?", "cade o Torres?", "onde esta o paper do Ahirwar?",
         "perdi a indexacao", "indexado", "na base").
    """
    if pedido_sem_literatura(pergunta):
        return False

    txt = _normalizar_texto(pergunta or "")
    frases = (
        "segundo a literatura",
        "com base na literatura",
        "consulte a literatura",
        "na literatura",
        "com referencias",
        "com fontes",
        "cite artigos",
        "citar artigos",
        "cite autores",
        "citar autores",
        "segundo os autores",
        "de acordo com os autores",
        "base indexada",
        "base de conhecimento",
        "nos documentos",
        "documentos indexados",
        "revisao bibliografica",
        "estado da arte",
        "levantamento bibliografico",
        "perdi a indexacao",
        "perdeu a indexacao",
        "na base",
        "no chromadb",
        "indexado",
        "indexada",
        "esta indexado",
        "esta indexada",
        "according to the literature",
        "based on the literature",
        "scientific literature",
        "literature review",
        "state of the art",
        "cite papers",
        "cite sources",
        "indexed documents",
        "knowledge base",
        "segun la literatura",
        "según la literatura",
        "con base en la literatura",
        "revision bibliografica",
        "revisión bibliográfica",
        "estado del arte",
        "citar articulos",
        "citar fuentes",
        "base de conocimiento",
        "selon la litterature",
        "selon la littérature",
        "revue bibliographique",
        "etat de l art",
        "état de l art",
        "citer des articles",
        "citer les sources",
        "base de connaissances",
    )
    if any(frase in txt for frase in frases):
        return True

    gatilhos = {
        "literatura", "artigo", "artigos", "paper", "papers", "fonte",
        "fontes", "referencia", "referencias", "bibliografia",
        "bibliografica", "bibliografico", "citacao", "citacoes",
        "cite", "citar", "autores", "autor", "survey", "review",
        "indexado", "indexada", "indexacao", "indexar",
        "literature", "source", "sources", "reference", "references",
        "citation", "citations", "author", "authors", "bibliography",
        "indexed", "indexing",
        "articulo", "articulos", "fuente", "fuentes", "referencia",
        "referencias", "bibliografia", "autor", "autores", "revision",
        "indexado", "indexada",
        "litterature", "littérature", "article", "articles", "source",
        "sources", "reference", "référence", "references", "références",
        "auteur", "auteurs", "bibliographie", "revue",
    }
    palavras = set(txt.split())
    if palavras & gatilhos:
        return True

    # Mencao a um autor/fonte indexada — "E o da NASA?", "Cade o Torres?"
    nomes = autores_indexados(colecao)
    if palavras & nomes:
        return True

    return False


def formatar_referencias_markdown(citacoes: dict | list | tuple | set) -> str:
    """Formata referencias como lista Markdown, deduplicando e ignorando vazios."""
    valores = citacoes.values() if isinstance(citacoes, dict) else (citacoes or [])
    vistos = []
    for valor in valores:
        if not valor:
            continue
        item = str(valor).strip()
        if item and item not in vistos:
            vistos.append(item)
    return "\n".join(f"- {item}" for item in vistos)


def _formatar_intervalo_paginas(paginas) -> str:
    """
    Comprime um conjunto de números de página em uma string compacta de
    intervalos: {3} → "3"; {3,4,5,8} → "3–5, 8". Usa travessão (en dash) para
    intervalos. Ignora valores nulos/zero/negativos. Retorna "" se vazio.
    """
    try:
        nums = sorted({int(p) for p in paginas if p not in (None, "") and int(p) > 0})
    except (TypeError, ValueError):
        return ""
    if not nums:
        return ""
    grupos: list[tuple[int, int]] = []
    ini = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            grupos.append((ini, prev))
            ini = prev = n
    grupos.append((ini, prev))
    partes = [str(a) if a == b else f"{a}–{b}" for a, b in grupos]
    return ", ".join(partes)


def _paginas_do_intervalo(pagina_inicio, pagina_fim=None) -> list[int]:
    """Normaliza metadados de pagina em uma lista inclusiva e ordenada."""
    try:
        inicio = int(pagina_inicio)
    except (TypeError, ValueError):
        return []
    if inicio <= 0:
        return []
    try:
        fim = int(pagina_fim) if pagina_fim not in (None, "") else inicio
    except (TypeError, ValueError):
        fim = inicio
    if fim < inicio:
        fim = inicio
    return list(range(inicio, fim + 1))


def remover_bloco_fontes_llm(texto: str) -> str:
    """
    Remove qualquer secao terminal de 'Referencias', 'Bibliografia',
    '📚 Fontes' etc. que o LLM tenha gerado por conta propria. Evita o
    duplo bloco quando o Streamlit anexa a lista oficial de citacoes.

    Heuristica anti-falso-positivo: so corta se o cabecalho for SEGUIDO
    por uma lista (linha comecando com '-', '*', '1.', etc.) ou pelo
    fim do texto. Assim, uma menção em prosa do tipo
    "📚 Fontes do paragrafo anterior estavam ok." nao e cortada.

    Detecta cabecalhos como:
      - "## Referencias", "### Referencias"
      - "**Referencias:**", "**Referências bibliográficas**"
      - "📚 Fontes:", "📚 **Fontes consultadas:**"
      - "Referencias:" no inicio de linha
    Apaga do cabecalho ate o final do texto (e separadores '---' que o
    LLM as vezes coloca logo antes).
    """
    if not texto:
        return texto

    import re

    # Palavra-chave do cabecalho — fontes/referencias/bibliografia (com
    # qualificadores opcionais como 'consultadas' ou 'bibliograficas').
    _palavra = (
        r"(?:refer[eê]ncias?(?:\s+bibliogr[áa]ficas?)?"
        r"|bibliografia"
        r"|fontes?(?:\s+consultadas?)?)"
    )
    padroes_cabecalho = [
        # Headers Markdown (##, ###, etc.)
        rf"(?im)^\s*#{{1,6}}\s*{_palavra}\b[^\n]*$",
        # Negrito/italico com colon em qualquer lado: **Refs:**, **Refs**:, **Refs**
        rf"(?im)^\s*\*+\s*{_palavra}\s*:?\s*\*+\s*:?\s*$",
        # Plain text com colon obrigatorio: REFERÊNCIAS:, Bibliografia:
        rf"(?im)^\s*{_palavra}\s*:\s*$",
        # 📚 — regex generoso; _eh_bloco_real filtra os falsos positivos
        # (linhas em prosa que apenas mencionam 'Fontes' sem lista logo abaixo).
        rf"(?im)^\s*📚[^\n]*{_palavra}[^\n]*$",
    ]

    def _eh_bloco_real(start: int, end: int) -> bool:
        """Confirma que o cabecalho e seguido por lista ou fim de texto."""
        apos = texto[end:].lstrip("\n").lstrip(" \t")
        if not apos:
            return True  # fim de texto — header solto conta como bloco
        primeira = apos.split("\n", 1)[0].strip()
        if not primeira:
            return True
        return (
            primeira.startswith(("-", "*", "•"))
            or bool(re.match(r"^\d+[.)]\s", primeira))
        )

    indice_min = len(texto)
    achou = False
    for padrao in padroes_cabecalho:
        for m in re.finditer(padrao, texto):
            if not _eh_bloco_real(m.start(), m.end()):
                continue
            if m.start() < indice_min:
                indice_min = m.start()
                achou = True

    if not achou:
        return texto.rstrip()

    recortado = texto[:indice_min]
    # Engole separadores '---' e linhas em branco logo antes do bloco.
    recortado = re.sub(r"(?:\s*\n\s*-{3,}\s*\n)+\s*$", "\n", recortado)
    return recortado.rstrip()


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


def _bloco_anexos(anexos_texto: str, orcamento: dict) -> str:
    """
    Monta o bloco de ARQUIVOS ANEXADOS para o prompt. Vazio quando nao ha
    anexos de texto. O conteudo ja vem consolidado por
    `montar_bloco_texto_anexos`; aqui so aplicamos o cap de chars do provedor
    e envolvemos com cabecalho + instrucao de uso.
    """
    if not anexos_texto or not anexos_texto.strip():
        return ""
    corpo = _limitar_texto(anexos_texto, orcamento.get("anexos_chars", 9_000))
    return (
        "ARQUIVOS ANEXADOS PELO PESQUISADOR (leia e use quando pertinente):\n"
        f"{corpo}\n"
        "Os arquivos acima foram enviados agora pelo Rodolfo nesta mensagem. "
        "Leia, interprete e use o conteudo quando for pertinente a pergunta. "
        "Trate-os como fonte prioritaria desta resposta; nao invente nada alem "
        "do que o anexo traz. Se a pergunta for sobre o anexo, responda a partir dele."
    )


def _montar_prompt(pergunta: str,
                   contexto: str,
                   historico_formatado: str,
                   orcamento: dict,
                   consultar_literatura: bool = True,
                   anexos_texto: str = "",
                   perfil: str = PERFIL_COMPACTO) -> str:
    # `perfil` é a identidade ESTÁTICA do agente que entra no prompt. Default é
    # o PERFIL_COMPACTO (curado, sem resultados numéricos); o chamador pode
    # injetar outro perfil compacto. Nunca embute métricas — os números vêm dos
    # artefatos via ferramenta de resultados.
    perfil = perfil if (perfil and perfil.strip()) else PERFIL_COMPACTO
    contexto = _limitar_texto(contexto, orcamento["contexto_chars"])
    bloco_temporal = _contexto_temporal()
    bloco_anexos = _bloco_anexos(anexos_texto, orcamento)
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

    rotulo_contexto = (
        "CONTEXTO RECUPERADO DA LITERATURA E MEMORIA"
        if consultar_literatura else
        "CONTEXTO RECUPERADO DA MEMORIA DO PROJETO"
    )
    instrucao_literatura = (
        "- A pergunta pediu literatura/fontes: use evidencias recuperadas quando relevantes e cite autor/ano.\n"
        "- NUNCA escreva uma secao final do tipo 'Referencias', 'Bibliografia', "
        "'Referencias bibliograficas', '## Referencias', '**Referencias:**', "
        "'### Referencias' ou '📚 Fontes'. Apenas cite autor/ano inline no texto. "
        "A lista de fontes consultadas e injetada automaticamente apos sua resposta.\n"
        "- NUNCA afirme que um autor, paper, instituicao ou tema 'nao esta na base', "
        "'nao foi indexado' ou 'minha base nao tem'. Voce so enxerga o CONTEXTO desta "
        "consulta, nao a base inteira. Se um autor citado pelo Rodolfo (NASA, Torres, "
        "Stender, Ahirwar, etc.) nao aparece no contexto recuperado, diga: "
        "'nao veio agora na minha busca para esta pergunta — posso refazer focando "
        "explicitamente no [autor/tema] se voce quiser'. Nunca afirme ausencia total."
        if consultar_literatura else
        "- A pergunta NAO pediu literatura/fontes: responda sem mencionar literatura, artigos, fontes ou referencias.\n"
        "- Use apenas conhecimento do projeto, memoria e raciocinio tecnico geral.\n"
        "- NUNCA escreva secao 'Referencias' ou '📚 Fontes' ao final."
    )

    instrucao_anexos = (
        "- O pesquisador ANEXOU arquivos nesta mensagem (bloco 'ARQUIVOS ANEXADOS'). "
        "Priorize esse conteudo: leia, interprete e responda a partir dele. "
        "Para imagens sem texto, descreva o que a imagem mostra quando o provedor "
        "tiver visao; se nao tiver, avise conforme a nota do anexo.\n"
        if bloco_anexos else ""
    )

    prompt = f"""
{perfil}

{bloco_temporal}

{estado_conversa}

{bloco_anexos}

{rotulo_contexto}:
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
{instrucao_anexos}{instrucao_literatura}
- Se a evidência/memória recuperada não tem relação com a pergunta, IGNORE-A em silêncio.
- Se a pergunta estiver em inglês, espanhol ou francês, entenda naturalmente e
  responda no mesmo idioma quando isso for útil; caso contrário, responda em
  português brasileiro.
- Quando houver resultados do pipeline no contexto, use-os como evidência e
  entregue interpretação técnica quando a pergunta pedir parecer, explicação
  ou recomendação. Não devolva só a tabela nesses casos.
- Tamanho da resposta proporcional ao pedido. Pergunta curta → resposta curta.
- Não invente números, autores, equações ou resultados.
""".strip()

    if len(prompt) > orcamento["max_prompt_chars"]:
        excesso = len(prompt) - orcamento["max_prompt_chars"]
        novo_limite = max(2_000, len(contexto) - excesso - 500)
        contexto = _limitar_texto(contexto, novo_limite)
        contexto_bloco = contexto if contexto.strip() else "Nenhum trecho relevante recuperado."
        prompt = f"""
{perfil}

{bloco_temporal}

{bloco_anexos}

{rotulo_contexto}:
{contexto_bloco}
{historico_formatado}

PERGUNTA ATUAL DO PESQUISADOR:
{pergunta}

INSTRUCOES DE RESPOSTA:
- Português brasileiro, voz natural, precisão técnica.
- Use emojis com moderação (🔬 📊 ✅).
- Cumprimente pelo período do dia quando apropriado.
{("- Priorize os ARQUIVOS ANEXADOS desta mensagem; responda a partir deles.\n" if bloco_anexos else "")}- Se a pergunta NAO pediu literatura/fontes, nao mencione literatura nem referencias.
- Cite autor/ano so quando a pergunta pediu literatura/fontes e a evidencia for relevante.
- Se a evidência não for relevante, ignore-a sem comentar.
- Ajuste o tamanho ao pedido. Não invente números.
""".strip()

    return prompt


def montar_conteudo_humano(prompt: str, anexos: list | None, suporta_imagem: bool):
    """
    Decide o `content` da HumanMessage enviada ao LLM.

    - Provedor multimodal (suporta_imagem=True) COM imagens anexadas → devolve
      uma LISTA de partes: o texto do prompt + uma parte image_url por imagem
      (data URI base64). E o formato que ChatGoogleGenerativeAI entende.
    - Caso contrario → devolve a STRING do prompt. As imagens, quando o provedor
      nao tem visao, ja viraram nota textual em `montar_bloco_texto_anexos`.

    Helper puro: nao chama LLM, so monta a estrutura.
    """
    if not (suporta_imagem and anexos and tem_imagem(anexos)):
        return prompt

    partes: list = [{"type": "text", "text": prompt}]
    for a in anexos:
        if a.get("tipo") == "imagem" and a.get("imagem_b64"):
            mime = a.get("mime") or "image/jpeg"
            partes.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{a['imagem_b64']}"},
            })
    return partes

# ============================================================
# RAG AVANÇADO — 3 CAMADAS
# ============================================================

def eh_query_de_revisao(pergunta: str) -> bool:
    """
    Detecta perguntas tipo 'literatura completa', 'revisao bibliografica',
    'estado da arte', 'cite a literatura', 'panorama'. Sao queries amplas
    que exigem cobertura diversificada da base, nao a melhor passagem unica.
    """
    txt = _normalizar_texto(pergunta or "")
    return any(ind in txt for ind in INDICADORES_REVISAO)


def _expandir_query(pergunta: str) -> dict:
    """
    CAMADA 1 — Expansão de query local.
    Evita chamadas auxiliares ao LLM para nao consumir TPM antes da resposta.

    Para perguntas de revisao bibliografica, injeta variacoes cobrindo TODOS
    os pilares teoricos da dissertacao — assim a busca semantica alcanca os
    artigos especificos de cada area em uma unica passada.
    """
    termos = _tokens_busca(pergunta)
    variacoes = [pergunta]

    txt = _normalizar_texto(pergunta)
    revisao = eh_query_de_revisao(pergunta)

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
    if "inversor" in txt or "fotovoltaic" in txt or "pv" in txt.split():
        variacoes.extend([
            "inversor fotovoltaico falhas componentes",
            "PV inverter failure modes reliability",
        ])
    if "rcm" in txt or "manutencao" in txt:
        variacoes.extend([
            "manutencao centrada em confiabilidade RCM",
            "reliability centered maintenance preventive predictive",
        ])

    # Datasets/instituicoes que mapeiam para um paper especifico
    if "paderborn" in txt:
        variacoes.extend([
            "Paderborn dataset IGBT three phase inverter Stender",
            "data set description three phase IGBT two level inverter",
        ])
        for t in ("stender", "igbt", "data", "set", "description"):
            if t not in termos:
                termos.append(t)
    if "ceamazon" in txt:
        variacoes.extend([
            "CEAMAZON sistema fotovoltaico UFPA Torres RCM FMECA",
        ])
        for t in ("ceamazon", "torres", "ufpa"):
            if t not in termos:
                termos.append(t)

    # Pergunta de revisao → injeta os 12 topicos da dissertacao para puxar
    # documentos diversos. Bonus: termos-chave de cada topico no keyword search.
    if revisao:
        variacoes.extend(TOPICOS_DISSERTACAO)
        for topico in TOPICOS_DISSERTACAO:
            for palavra in topico.split():
                if len(palavra) > 3 and palavra.lower() not in termos:
                    termos.append(palavra.lower())

    return {
        "variacoes": list(dict.fromkeys(variacoes)),
        "termos": termos[:30 if revisao else 12],
        "revisao": revisao,
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

    # Busca por palavras-chave (keyword search).
    # IMPORTANTE: ChromaDB where_document $contains e CASE-SENSITIVE. Como
    # os tokens vem normalizados em minusculas mas o texto dos PDFs tem
    # autores/siglas em title-case ('Karim', 'NASA', 'Torres'), tentamos
    # multiplas variantes de capitalizacao para garantir cobertura.
    autores_conhecidos = autores_indexados(colecao)
    for termo in termos:
        if not termo or len(termo) < 2:
            continue

        variantes = {termo, termo.title(), termo.upper(), termo.capitalize()}
        for variante in variantes:
            try:
                resultados = colecao.get(
                    where_document = {"$contains": variante},
                    include        = ["documents", "metadatas"],
                    limit          = 60,
                )
                docs  = resultados.get("documents", [])
                metas = resultados.get("metadatas", [])
                ids   = resultados.get("ids",       [])

                for id_, doc, meta in zip(ids, docs, metas):
                    if id_ not in pool:
                        pool[id_] = (doc, meta)
            except Exception:
                continue

        # Se o termo bate um autor conhecido, busca pelas formas canonicas
        # do metadado autor — cobre autores compostos (Puc Rio), capitalizacao
        # inconsistente, e multiplos arquivos do mesmo autor (Grewal Kalman
        # + Grewal Power Electronics).
        #
        # IMPORTANTE: quando um autor tem multiplos arquivos com tamanhos
        # muito diferentes (Kalman: 1710 chunks; Power Electronics: 63),
        # uma unica query where={"autor": X} com limit alto so traz o maior.
        # Por isso iteramos por arquivo, garantindo amostra de CADA paper.
        if termo.lower() in autores_conhecidos:
            canonicos = autores_canonicos_para(termo, colecao)
            tentativas = set(canonicos) if canonicos else {
                termo.title(), termo.upper(), termo.capitalize(), termo
            }
            for capit in tentativas:
                arqs = arquivos_do_autor(capit, colecao) or {None}
                for arq in arqs:
                    where_clause: dict = {"autor": capit}
                    if arq:
                        where_clause = {
                            "$and": [
                                {"autor": capit},
                                {"arquivo": arq},
                            ]
                        }
                    try:
                        resultados = colecao.get(
                            where = where_clause,
                            include = ["documents", "metadatas"],
                            limit = 80,
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


def _ajuste_textbook(arquivo: str, texto_pergunta_norm: str) -> float:
    """
    Penaliza livros-texto genericos quando a pergunta NAO entra no dominio
    proprio deles. Stewart so deve aparecer em pergunta de calculo, Gonzalez
    em pergunta de imagem, e assim por diante. Em qualquer outro caso, dao
    ruido em consultas amplas.

    EXCECAO: se a pergunta cita o sobrenome do autor explicitamente
    ("E o do Grewal?", "tem Stewart?"), a penalidade nao se aplica —
    o Rodolfo esta pedindo aquele livro pelo nome.
    """
    if not arquivo or arquivo not in TEXTBOOKS_PENALIZADOS:
        return 0.0
    # Pergunta cita o sobrenome (primeira parte do filename)?
    sobrenome = arquivo.split("_", 1)[0].replace("-", " ")
    if sobrenome and sobrenome in texto_pergunta_norm:
        return 0.0
    gatilhos = TEXTBOOKS_PENALIZADOS[arquivo]
    if any(g in texto_pergunta_norm for g in gatilhos):
        return 0.0
    # Penalidade forte o bastante para dominar os boosts incidentais que um
    # textbook fora de dominio ainda acumula — em especial o match lexical do
    # slug do arquivo (ex.: "calculo" em 'stewart_calculo-volume-i' batendo
    # "calculo do limiar do autoencoder"). Medido: -6 deixava o Stewart vazar
    # nesse caso adversarial; a partir de -10 ele sai sem reduzir a diversidade.
    return -12.0


def _diversificar_por_fonte(
    pontuados: list,
    n_final: int,
    max_por_fonte: int = 2,
) -> list:
    """
    Seleciona ate `n_final` chunks aplicando teto de `max_por_fonte` por
    arquivo. Garante que o top-K cubra mais documentos distintos em vez de
    repetir trechos do mesmo PDF.

    Estrategia: percorre `pontuados` (ja em ordem de score) e aceita chunks
    enquanto a fonte nao bateu o teto. Se faltarem itens ao final, relaxa o
    teto progressivamente ate completar n_final.
    """
    if not pontuados:
        return []

    selecionados: list = []
    contagem: dict[str, int] = {}
    teto = max_por_fonte

    # Varias passadas relaxando o teto, ate completar n_final.
    while len(selecionados) < n_final and teto <= 50:
        progrediu = False
        for _, _, doc, meta in pontuados:
            if len(selecionados) >= n_final:
                break
            fonte = str(meta.get("arquivo", "")) or str(meta.get("citacao", "?"))
            if (doc, meta) in selecionados:
                continue
            if contagem.get(fonte, 0) >= teto:
                continue
            selecionados.append((doc, meta))
            contagem[fonte] = contagem.get(fonte, 0) + 1
            progrediu = True
        if not progrediu:
            break
        teto += 1
    return selecionados


def _rerankar(
    candidatos: list,
    pergunta: str,
    n_final: int,
    max_por_fonte: int = 2,
    termos_extra: list | None = None,
) -> list:
    """
    CAMADA 3 — Reranking local.

    Pontua chunks por sobreposicao lexical, ajusta por pasta tematica
    (PV/ML/manutencao recebem boost, sinais-eletricos atenua), penaliza
    textbooks fora de dominio, e diversifica o top-K aplicando teto de
    chunks por fonte.

    `termos_extra` permite injetar termos da expansao (ex.: 'stender'
    quando a pergunta tem 'paderborn') para que o boost por autor/arquivo
    funcione mesmo quando a pergunta original nao traz o sobrenome.
    """
    if not candidatos:
        return []

    termos = _tokens_busca(pergunta)
    if termos_extra:
        for t in termos_extra:
            if t and t not in termos:
                termos.append(t)
    pergunta_norm = _normalizar_texto(pergunta)
    # Numero so vale se vier acoplado a alguma letra do projeto — assim
    # "+30 artigos" nao premia qualquer trecho que tenha um "30" qualquer.
    numeros_relevantes = {
        t for t in pergunta_norm.split()
        if any(ch.isdigit() for ch in t)
        and any(ch.isalpha() for ch in t)  # ex.: "auc", "npr210", "f1"
    }

    pontuados = []
    termos_norm = {_normalizar_texto(t) for t in termos}
    for ordem, (doc, meta) in enumerate(candidatos):
        citacao = str(meta.get("citacao", ""))
        titulo = str(meta.get("titulo", ""))
        arquivo = str(meta.get("arquivo", ""))
        autor = str(meta.get("autor", ""))
        pasta = str(meta.get("pasta", ""))
        texto = " ".join([citacao, titulo, arquivo, doc])
        texto_norm = _normalizar_texto(texto)
        citacao_norm = _normalizar_texto(citacao)
        autor_norm = _normalizar_texto(autor)
        arquivo_norm = _normalizar_texto(arquivo)

        score = 0.0

        for termo in termos:
            if termo in texto_norm:
                score += 2.0 if len(termo) > 4 else 1.0
                if termo in citacao_norm:
                    score += 1.5

        # Boost forte quando o termo bate o autor do chunk. O nome do
        # arquivo segue o padrao 'autor_titulo-com-hifens_ano.pdf', entao
        # comparo o termo APENAS contra o primeiro segmento (sobrenome) e
        # o ano — assim "calculo" no slug do Stewart nao premia indevidamente
        # (era o caso de Stewart_calculo-volume-i_2013 entrar em queries
        # de "calculo do limiar do autoencoder").
        partes_arquivo = arquivo_norm.replace(".pdf", "").split("_")
        sobrenome_arquivo = partes_arquivo[0] if partes_arquivo else ""
        # Sobrenomes compostos com hifen ('puc-rio') ja vieram normalizados
        # como 'puc rio' (espaco) — quebro para comparar com tokens.
        sobrenome_tokens = sobrenome_arquivo.split()
        ano_arquivo = partes_arquivo[-1] if len(partes_arquivo) > 1 else ""
        for termo_n in termos_norm:
            if not termo_n or len(termo_n) < 3:
                continue
            if termo_n == autor_norm or termo_n in autor_norm.split():
                score += 6.0
            if termo_n == sobrenome_arquivo or termo_n in sobrenome_tokens:
                score += 4.0
            if termo_n == ano_arquivo:
                score += 1.5

        for numero in numeros_relevantes:
            if numero in texto_norm:
                score += 2.0

        if "tabela" in texto_norm or "table" in texto_norm:
            score += 0.3
        if any(x in texto_norm for x in ("resultado", "metodo", "method", "equacao", "equation")):
            score += 0.2

        # Boost por pasta tematica — privilegia o nucleo da dissertacao.
        score += PESOS_PASTA.get(pasta, 0.3)

        # Penalidade para textbooks fora de dominio (Stewart, Gonzalez, etc).
        score += _ajuste_textbook(arquivo, pergunta_norm)

        pontuados.append((score, -ordem, doc, meta))

    pontuados.sort(reverse=True)

    # Se ha candidatos suficientes, aplica diversificacao por fonte.
    if len(pontuados) <= n_final:
        return [(doc, meta) for _, _, doc, meta in pontuados]

    return _diversificar_por_fonte(pontuados, n_final, max_por_fonte=max_por_fonte)


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
    consultar_literatura: bool = True,
    n_resultados_revisao: int | None = None,
    max_chunks_por_fonte: int = 2,
) -> tuple:
    """
    Pipeline RAG de 3 camadas para literatura, mantendo memória sempre ativa.
    Quando consultar_literatura=False, pula expansão/busca/reranking da base
    bibliográfica e usa apenas a memória de sessões.

    Quando a pergunta cheira a revisao bibliografica ("literatura completa",
    "estado da arte", "cite a literatura"), o orcamento sobe para
    `n_resultados_revisao` chunks e cap de `max_chunks_por_fonte=1` para
    maximizar diversidade.
    """
    contexto = ""
    citacoes = {}

    if consultar_literatura:
        # ── CAMADA 1 — Expansão ───────────────────────────────
        expansao  = _expandir_query(pergunta)
        variacoes = expansao.get("variacoes", [pergunta])
        termos    = expansao.get("termos",    [])
        revisao   = expansao.get("revisao",   False)

        if pergunta not in variacoes:
            variacoes.insert(0, pergunta)

        # ── CAMADA 2 — Busca híbrida ─────────────────────────
        candidatos = _busca_hibrida(
            variacoes,
            termos,
            colecao,
            modelo_embeddings,
            n_pool=n_pool or 30,
        )

        # ── CAMADA 3 — Reranking ─────────────────────────────
        if revisao:
            alvo = n_resultados_revisao or (n_resultados or N_RESULTADOS) * 2
            cap = 1  # exige diversidade absoluta em revisao
        else:
            alvo = n_resultados or min(N_RESULTADOS, 8)
            cap = max_chunks_por_fonte
        melhores = _rerankar(
            candidatos,
            pergunta,
            n_final=alvo,
            max_por_fonte=cap,
            termos_extra=termos,
        )
    else:
        melhores = []

    # Monta contexto da literatura em ROUND-ROBIN por fonte: a primeira
    # rodada inclui 1 chunk de cada fonte distinta (em ordem de score), so
    # entao as rodadas seguintes adicionam segundos/terceiros chunks. Assim,
    # mesmo com orcamento de caracteres apertado, o LLM recebe a MAIOR
    # diversidade possivel de fontes — em vez de ficar travado com 2-3 que
    # esgotaram o limite primeiro.
    if consultar_literatura and melhores:
        contexto += "\n📚 DA LITERATURA CIENTÍFICA:\n"
        usados = len(contexto)
        limite = contexto_chars or 10_000

        # Agrupa por fonte preservando ordem (e ordem dos chunks dentro
        # de cada fonte) — assim os chunks de maior score lideram cada rodada.
        por_fonte: dict[str, list] = {}
        ordem_fontes: list[str] = []
        # Páginas efetivamente usadas por fonte → vira "p. X–Y" na citação.
        paginas_por_fonte: dict[str, set] = {}
        for doc, meta in melhores:
            arquivo = meta.get("arquivo", "") or meta.get("citacao", "?")
            if arquivo not in por_fonte:
                por_fonte[arquivo] = []
                ordem_fontes.append(arquivo)
            por_fonte[arquivo].append((doc, meta))

        max_rondas = max((len(c) for c in por_fonte.values()), default=0)
        cheio = False
        for ronda in range(max_rondas):
            if cheio:
                break
            for arquivo in ordem_fontes:
                chunks_fonte = por_fonte[arquivo]
                if ronda >= len(chunks_fonte):
                    continue
                doc, meta = chunks_fonte[ronda]
                citacao = meta.get("citacao", arquivo)
                # Página do chunk (extração page-aware). Chunks antigos sem
                # essa metadado simplesmente não recebem página.
                p_ini, p_fim = meta.get("pagina_inicio"), meta.get("pagina_fim")
                paginas_chunk = _paginas_do_intervalo(p_ini, p_fim)
                pag_chunk = _formatar_intervalo_paginas(paginas_chunk)
                rotulo = citacao + (f" — p. {pag_chunk}" if pag_chunk else "")
                cabecalho = f"\n[Fonte: {rotulo}]\n"
                bloco = f"{cabecalho}{doc}\n"
                if usados + len(bloco) > limite:
                    restante = limite - usados - len(cabecalho)
                    if restante <= 300:
                        cheio = True
                        break
                    bloco = f"{cabecalho}{_limitar_texto(doc, restante)}\n"
                if arquivo and arquivo not in citacoes:
                    citacoes[arquivo] = citacao
                if paginas_chunk:
                    pgs = paginas_por_fonte.setdefault(arquivo, set())
                    pgs.update(paginas_chunk)
                contexto += bloco
                usados += len(bloco)
                if usados >= limite:
                    cheio = True
                    break

        # Enriquece cada citação com as páginas efetivamente usadas no contexto.
        for arquivo, pgs in paginas_por_fonte.items():
            base = citacoes.get(arquivo)
            if not base or "p." in str(base):
                continue
            intervalo = _formatar_intervalo_paginas(pgs)
            if intervalo:
                citacoes[arquivo] = f"{base}, p. {intervalo}"

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


# Nomes amigaveis para os 5 temas (campo `pasta` no metadado do ChromaDB).
_NOMES_TEMAS = {
    "confiabilidade":   "Confiabilidade e FMEA",
    "inversores-pv":    "Inversores PV e modos de falha",
    "manutencao":       "Manutenção preditiva e RCM",
    "ml-preditivo":     "Machine Learning e predição de falhas",
    "sinais-eletricos": "Sinais elétricos e processamento",
}


def catalogo_literatura(colecao) -> str:
    """
    Catálogo COMPLETO e determinístico da literatura indexada.

    Diferente do RAG (que só traz os trechos mais relevantes e leva o LLM a
    truncar/inventar), aqui lemos TODOS os metadados do ChromaDB e devolvemos
    o inventário inteiro, agrupado por tema. A citação é reconstruída a partir
    de autor/ano/título (campos limpos) para não herdar o mojibake do campo
    `citacao`. Use isto sempre que o pesquisador pedir "liste tudo o que você
    tem", "a base bibliográfica completa", "quantos artigos", etc.
    """
    vistos: dict[str, dict] = {}
    offset, lote = 0, 200

    while True:
        try:
            res = colecao.get(limit=lote, offset=offset, include=["metadatas"])
        except Exception:
            break
        metas = res.get("metadatas", []) or []
        if not metas:
            break
        for m in metas:
            arq = m.get("arquivo", "desconhecido")
            if arq not in vistos:
                vistos[arq] = {
                    "pasta":  m.get("pasta", "outros"),
                    "autor":  (m.get("autor") or "Autor desconhecido").strip(),
                    "ano":    str(m.get("ano") or "s.d.").strip(),
                    "titulo": (m.get("titulo") or arq).strip(),
                    "chunks": int(m.get("total_chunks", 0) or 0),
                }
        offset += lote
        if len(metas) < lote:
            break

    if not vistos:
        return (
            "Não encontrei documentos indexados na base de conhecimento. "
            "Verifique se o ChromaDB foi reconstruído (scripts/reconstruir_literatura.py)."
        )

    por_tema: dict[str, list[dict]] = {}
    for info in vistos.values():
        por_tema.setdefault(info["pasta"], []).append(info)

    total = len(vistos)
    linhas = [f"📚 **Base bibliográfica completa — {total} documentos indexados**"]

    # Temas conhecidos primeiro (na ordem do dicionário), depois quaisquer extras.
    ordem_temas = [t for t in _NOMES_TEMAS if t in por_tema]
    ordem_temas += [t for t in sorted(por_tema) if t not in _NOMES_TEMAS]

    for pasta in ordem_temas:
        docs = sorted(por_tema[pasta], key=lambda d: (d["autor"].lower(), d["ano"]))
        nome_tema = _NOMES_TEMAS.get(pasta, pasta)
        linhas.append(f"\n### {nome_tema} ({len(docs)})")
        for d in docs:
            linhas.append(f"- **{d['autor']} ({d['ano']})** — {d['titulo']}")

    linhas.append(
        f"\n_São {total} documentos no total. Posso detalhar qualquer um, "
        "comparar dois trabalhos ou buscar um tema específico — é só pedir._"
    )
    return "\n".join(linhas)


def preparar_prompt(
    pergunta: str,
    perfil: str,
    modelo_embeddings,
    colecao,
    historico: list    = None,
    colecao_sessoes    = None,
    nome_provedor: str | None = None,
    anexos: list | None = None,
) -> tuple:
    """
    Prepara o prompt completo sem invocar o LLM.
    Retorna (prompt_str, citacoes_dict).
    Usado pelo Streamlit para fazer streaming separado.

    `anexos` e a lista de dicts vinda de `leitor_anexos.ler_anexos(...)`: o texto
    extraido (PDF/CSV/Excel/Word/codigo/...) entra no prompt como bloco
    prioritario; imagens viram nota textual aqui (o pixel vai pela via
    multimodal em `montar_conteudo_humano`, chamada pelo invocador do LLM).
    """

    if historico is None:
        historico = []

    orcamento = _orcamento_rag(nome_provedor)
    consultar_literatura = deve_consultar_literatura(pergunta, colecao)
    contexto, citacoes = buscar_contexto(
        pergunta,
        modelo_embeddings,
        colecao,
        colecao_sessoes,
        n_pool=orcamento["n_pool"],
        n_resultados=orcamento["n_resultados"],
        n_resultados_revisao=orcamento.get("n_resultados_revisao"),
        max_chunks_por_fonte=orcamento.get("max_chunks_por_fonte", 2),
        contexto_chars=orcamento["contexto_chars"],
        sessao_chars=orcamento["sessao_chars"],
        consultar_literatura=consultar_literatura,
    )

    suporta_imagem = eh_multimodal(nome_provedor)
    anexos_texto = (
        montar_bloco_texto_anexos(anexos, suporta_imagem=suporta_imagem)
        if anexos else ""
    )

    historico_formatado = _formatar_historico(historico, orcamento)

    # Identidade estática que ENTRA no prompt. O chamador passa `perfil` (ex.:
    # CLAUDE.md). Para não inflar cada prompt com o documento inteiro, usamos o
    # perfil recebido apenas quando é compacto; caso contrário, o PERFIL_COMPACTO
    # curado. Assim o parâmetro deixa de ser ignorado, mas o custo fica contido.
    perfil_prompt = perfil if (perfil and perfil.strip() and len(perfil) <= 6000) else PERFIL_COMPACTO

    prompt = _montar_prompt(
        pergunta,
        contexto,
        historico_formatado,
        orcamento,
        consultar_literatura=consultar_literatura,
        anexos_texto=anexos_texto,
        perfil=perfil_prompt,
    )
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
    anexos: list | None = None,
) -> str:
    """
    Pipeline RAG completo com memória e streaming.

    `anexos` (opcional): lista de dicts de `leitor_anexos.ler_anexos(...)`. O
    texto extraido entra no prompt; imagens vao pela via multimodal quando o
    provedor ativo for multimodal (Gemini). Caso contrario, viram nota textual.
    """

    if historico is None:
        historico = []

    resposta_simples = resposta_interacao_simples(pergunta)
    if resposta_simples:
        print(resposta_simples)
        return resposta_simples

    # Catálogo da literatura: "liste todas as referências", "o que você tem
    # indexado", "quantos artigos". Inventário completo e determinístico, lido
    # direto do ChromaDB — nunca via RAG/LLM (que truncaria/inventaria a lista).
    # Import tardio: ferramentas.py importa este módulo (evita ciclo).
    try:
        from src.conhecimento.ferramentas import _quer_catalogo
        if not anexos and _quer_catalogo(pergunta):
            texto = catalogo_literatura(colecao)
            if streaming:
                print(texto)
            return texto
    except Exception:
        pass

    prompt, citacoes = preparar_prompt(
        pergunta=pergunta,
        perfil=perfil,
        modelo_embeddings=modelo_embeddings,
        colecao=colecao,
        historico=historico,
        colecao_sessoes=colecao_sessoes,
        nome_provedor=nome_provedor,
        anexos=anexos,
    )

    conteudo_humano = montar_conteudo_humano(
        prompt, anexos, eh_multimodal(nome_provedor)
    )
    mensagens = [HumanMessage(content=conteudo_humano)]
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

    refs_md = formatar_referencias_markdown(citacoes)
    if refs_md:
        texto_completo = remover_bloco_fontes_llm(texto_completo)
        rodape = "\n\n---\n📚 **Fontes consultadas nesta resposta:**\n" + refs_md + "\n"
        print(rodape)
        texto_completo += rodape

    return texto_completo
