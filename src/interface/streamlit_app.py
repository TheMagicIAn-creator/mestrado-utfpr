"""
streamlit_app.py - Al IAdo PV
Interface conversacional do agente.

Resultados e execucoes do pipeline de ML aparecem pelo chat, conforme
solicitacao em prompt.

Design (minimalista, 2026-07): a tela e o chat. Tudo que nao e conversa
recua para a barra lateral, e o que na barra nao e status recua para
expanders. Regras que valem para qualquer coisa nova aqui:

  - a tela inicial diz QUEM e o agente em uma linha, nao explica o que ele
    faz em paragrafos (o chat ensina isso na pratica);
  - estado saudavel ocupa uma linha; so a falha ganha espaco e cor;
  - o CSS ajusta escala e respiro usando as variaveis de tema do Streamlit,
    sem fixar cor de fundo — claro e escuro seguem nativos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from src.core.config import RAIZ_PROJETO
from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos
from src.core.tempo import agora_local
from src.core.utils import configurar_saida_utf8

sys.path.insert(0, str(RAIZ_PROJETO))
configurar_saida_utf8()
_logger = get_logger("interface.streamlit")


def _falha_recuperavel(operacao: str, exc: Exception, *, notificar: bool = False) -> None:
    """Registra fallback operacional e, quando necessário, avisa no app."""
    detalhe = mascarar_segredos(str(exc))
    _logger.warning("%s: %s", operacao, detalhe)
    if notificar and hasattr(st, "toast"):
        st.toast(f"{operacao}. Detalhes registrados no log.", icon="⚠️")


st.set_page_config(
    page_title="Al IAdo PV",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Paleta: a MESMA dos graficos (src/ml/estilo_graficos) — a interface e a
# figura falam a mesma lingua visual.
_CORES_ESTADO = {
    "ok": "#1baf7a",
    "alerta": "#eda100",
    "erro": "#e34948",
    "neutro": "#898781",
}


def _html_pensando(rotulo: str = "Pensando") -> str:
    """Texto com brilho pulsante, exibido enquanto a resposta não começa.

    Sempre montado AQUI, a partir de uma string nossa — nunca a partir da
    saída do LLM. É o que permite renderizá-lo com HTML habilitado sem abrir
    o texto do modelo para injeção (o streaming segue em Markdown puro).
    """
    return f'<span class="alp-pensando">{rotulo}…</span>'


def _estado(rotulo: str, nivel: str = "ok") -> str:
    """Indicador de estado: um ponto colorido e uma linha de texto.

    Substitui os blocos st.success/st.warning empilhados na barra lateral —
    saude do sistema nao precisa de caixa colorida de altura cheia.
    """
    cor = _CORES_ESTADO.get(nivel, _CORES_ESTADO["neutro"])
    return (
        f'<div class="alp-estado">'
        f'<span class="alp-ponto" style="background:{cor}"></span>{rotulo}</div>'
    )


# CSS: mantem o menu principal visivel, pois nele fica Settings -> Theme
# para alternar entre claro/escuro nativo do Streamlit. Cores neutras usam
# rgba cinza (legivel nos dois temas); nada de background fixo.
_CSS_MINIMO = """
<style>
:root {
    --alp-acento: #2a78d6;
    --alp-linha: rgba(128, 128, 128, 0.20);
    --alp-linha-forte: rgba(128, 128, 128, 0.34);
    --alp-fraco: rgba(128, 128, 128, 0.95);
}

/* ── chrome do Streamlit ───────────────────────────────────────────── */
.stDeployButton,
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stHeader"] {
    background: transparent;
    height: 2.5rem;
}

/* ── coluna de leitura ─────────────────────────────────────────────── */
/* Estreita de proposito: linha longa cansa e "espalha" o layout. 1160px
   ainda comporta os paineis largos (teto de exibicao das figuras: 1080). */
.block-container {
    max-width: min(1160px, calc(100vw - 3rem));
    padding-top: 2.2rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}
[data-testid="stChatMessage"] {
    max-width: 100%;
    padding: 0.15rem 0;
    background: transparent;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
    font-size: 0.9rem;
}
[data-testid="stBottomBlockContainer"],
[data-testid="stChatInput"] {
    max-width: min(1160px, calc(100vw - 3rem));
}
[data-testid="stBottomBlockContainer"] {
    padding-left: 1.25rem;
    padding-right: 1.25rem;
    padding-bottom: 0.85rem;
}
[data-testid="stChatInput"] {
    border-radius: 14px;
    border: 1px solid var(--alp-linha-forte);
}

/* ── botoes compactos ──────────────────────────────────────────────── */
/* Eram todos "stretch" em altura cheia; dominavam o chat e a barra. */
.stButton > button,
.stDownloadButton > button,
[data-testid="stPopover"] > div > button {
    border-radius: 9px;
    border: 1px solid var(--alp-linha-forte);
    font-size: 0.80rem;
    font-weight: 500;
    line-height: 1.25;
    padding: 0.30rem 0.72rem;
    min-height: 0;
}
.stButton > button p,
.stDownloadButton > button p,
[data-testid="stPopover"] > div > button p {
    font-size: 0.80rem;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stPopover"] > div > button:hover {
    border-color: var(--alp-acento);
    color: var(--alp-acento);
}

/* ── barra lateral ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--alp-linha);
}
[data-testid="stSidebarContent"] {
    padding-top: 1.15rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.75rem;
    line-height: 1.4;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    border: none;
    border-top: 1px solid var(--alp-linha);
    border-radius: 0;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    padding: 0.5rem 0.15rem;
    font-size: 0.80rem;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: var(--alp-acento);
}
[data-testid="stSidebar"] hr { margin: 0.7rem 0; }

.alp-marca {
    font-size: 1.02rem;
    font-weight: 650;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.alp-sub {
    font-size: 0.72rem;
    color: var(--alp-fraco);
    margin: 0.1rem 0 0.85rem;
}
.alp-estado {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    font-size: 0.75rem;
    color: var(--alp-fraco);
}
.alp-ponto {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex: 0 0 auto;
}
/* Numeros da base: linha unica, sem o bloco gigante do st.metric. */
.alp-stats {
    display: flex;
    gap: 0.35rem;
    margin: 0.85rem 0 0.2rem;
}
.alp-stats > div {
    flex: 1;
    padding: 0.42rem 0.1rem;
    border: 1px solid var(--alp-linha);
    border-radius: 9px;
    text-align: center;
    line-height: 1.15;
}
.alp-stats b {
    display: block;
    font-size: 0.95rem;
    font-weight: 600;
}
.alp-stats span {
    font-size: 0.64rem;
    color: var(--alp-fraco);
    letter-spacing: 0.01em;
}

/* ── tela inicial ──────────────────────────────────────────────────── */
.alp-hero {
    text-align: center;
    margin: 4.5rem auto 1.6rem;
    max-width: 34rem;
}
.alp-hero-icone {
    font-size: 1.7rem;
    line-height: 1;
    margin-bottom: 0.7rem;
}
.alp-hero h1 {
    font-size: 1.42rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    margin: 0;
    padding: 0;
}
.alp-hero p {
    font-size: 0.86rem;
    color: var(--alp-fraco);
    margin: 0.45rem 0 0;
}

/* ── espera: brilho pulsante ("shimmer") ───────────────────────────── */
/* Cobre o intervalo entre o Enter e o primeiro token, que antes era um
   spinner generico. O gradiente corre pelo texto via background-clip,
   entao a animacao e so a posicao do fundo — nao repinta layout. */
@keyframes alp-brilho {
    from { background-position: 180% 0; }
    to   { background-position: -80% 0; }
}
.alp-pensando {
    display: inline-block;
    font-size: 0.92rem;
    font-weight: 500;
    background-image: linear-gradient(
        90deg,
        var(--alp-fraco) 0%,
        var(--alp-fraco) 42%,
        var(--alp-acento) 50%,
        var(--alp-fraco) 58%,
        var(--alp-fraco) 100%);
    background-size: 220% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
    animation: alp-brilho 1.9s linear infinite;
}
/* Sem movimento para quem pediu no sistema: vira texto cinza estatico. */
@media (prefers-reduced-motion: reduce) {
    .alp-pensando {
        animation: none;
        background-image: none;
        color: var(--alp-fraco);
        -webkit-text-fill-color: var(--alp-fraco);
    }
}

/* ── conteudo ──────────────────────────────────────────────────────── */
[data-testid="stImage"],
.stImage {
    max-width: 100%;
    overflow: hidden;
}
[data-testid="stImage"] > div,
.stImage > div {
    width: 100% !important;
    max-width: 100% !important;
}
[data-testid="stImage"] img,
.stImage img {
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain;
}
[data-testid="stImageCaption"],
.stImage figcaption {
    white-space: normal;
    overflow-wrap: anywhere;
}
[data-testid="stMarkdownContainer"] {
    overflow-x: auto;
}
[data-testid="stMarkdownContainer"] table {
    width: max-content;
    max-width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    line-height: 1.35;
}
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] td {
    padding: 0.38rem 0.55rem;
    vertical-align: top;
    white-space: nowrap;
}
[data-testid="stMarkdownContainer"] td:first-child,
[data-testid="stMarkdownContainer"] th:first-child {
    white-space: normal;
    min-width: 10rem;
}
[data-testid="stDataFrame"] {
    max-width: 100%;
}
@media (max-width: 900px) {
    .block-container,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInput"] {
        max-width: calc(100vw - 1rem);
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    .block-container {
        padding-top: 1.6rem;
    }
    .alp-hero { margin-top: 2.4rem; }
    [data-testid="stMarkdownContainer"] table {
        font-size: 0.82rem;
    }
    [data-testid="stMarkdownContainer"] th,
    [data-testid="stMarkdownContainer"] td {
        padding: 0.32rem 0.42rem;
    }
}
</style>
"""


# Extensões aceitas como anexo no chat (sem o ponto, como o Streamlit espera).
# Cobre documentos, dados tabulares, código/config e imagens — o leitor_anexos
# decide o extrator por extensão e trata o que não reconhecer.
TIPOS_ANEXO = [
    "pdf", "csv", "tsv", "xlsx", "xls", "xlsm", "docx",
    "txt", "md", "markdown", "rst", "json", "yaml", "yml",
    "toml", "ini", "cfg", "conf", "log", "html", "htm", "xml",
    "py", "js", "ts", "java", "c", "cpp", "cs", "go", "rs", "rb",
    "r", "sql", "sh",
    "png", "jpg", "jpeg", "gif", "webp", "bmp",
]


@st.cache_resource
def carregar_base():
    import chromadb
    from src.conhecimento.agente import carregar_perfil
    from src.conhecimento.embeddings import (
        backend_embeddings,
        criar_modelo_embeddings,
    )
    from src.core.config import (
        NOME_COLECAO,
        NOME_COLECAO_SESSOES,
        NOME_COLECAO_OBSIDIAN,
        ARQUIVO_INDICE_LITERATURA,
        ARQUIVO_INDICE_OBSIDIAN,
        ARQUIVO_MEMORIA_VALIDADA,
        PASTA_CHROMADB,
    )
    from src.ml.pipeline import capacidade_recalculo_pipeline

    with st.spinner("Carregando a base de conhecimento..."):
        relatorio = []
        client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = client.get_or_create_collection(name=NOME_COLECAO)
        colecao_sessoes = client.get_or_create_collection(name=NOME_COLECAO_SESSOES)
        colecao_obsidian = client.get_or_create_collection(
            name=NOME_COLECAO_OBSIDIAN,
            metadata={"hnsw:space": "cosine"},
        )
        if ARQUIVO_INDICE_OBSIDIAN.is_file():
            try:
                from src.conhecimento.indice_portatil import importar_colecao

                restauracao_obsidian = importar_colecao(
                    colecao_obsidian,
                    ARQUIVO_INDICE_OBSIDIAN,
                    mesclar=True,
                )
                relatorio.append(
                    "Obsidian: "
                    f"{restauracao_obsidian['n_chunks']} chunks históricos "
                    f"disponíveis ({restauracao_obsidian['importados']} restaurados; "
                    f"{restauracao_obsidian.get('preservados', 0)} novos preservados)."
                )
            except Exception as exc:
                relatorio.append(f"Obsidian: snapshot inválido - {exc}")
        if colecao.count() == 0 and ARQUIVO_INDICE_LITERATURA.is_file():
            try:
                from src.conhecimento.indice_portatil import importar_colecao

                with st.spinner("Restaurando o índice portátil da literatura..."):
                    restauracao = importar_colecao(
                        colecao, ARQUIVO_INDICE_LITERATURA
                    )
                relatorio.append(
                    "Literatura: "
                    f"{restauracao['n_chunks']} chunks restaurados do snapshot."
                )
            except Exception as exc:
                relatorio.append(f"Literatura: snapshot inválido - {exc}")

        capacidade = capacidade_recalculo_pipeline()
        modo_consulta = not capacidade["disponivel"]
        modelo = criar_modelo_embeddings(modo_consulta=modo_consulta)
        relatorio.append(
            "Embeddings: "
            f"backend {backend_embeddings(modo_consulta=modo_consulta)}."
        )

        try:
            from src.conhecimento.obsidian import (
                espelhar_memoria_validada,
                sincronizar_obsidian,
            )

            espelhar_memoria_validada(ARQUIVO_MEMORIA_VALIDADA)
            if not modo_consulta:
                estado_obsidian = sincronizar_obsidian(colecao_obsidian, modelo)
                relatorio.append(
                    "Obsidian: "
                    f"{estado_obsidian['notas_ativas']} notas do vault prontas."
                )
            else:
                relatorio.append(
                    "Obsidian: notas portáteis prontas; alterações locais "
                    "sincronizadas sob demanda no chat."
                )
        except Exception as exc:
            relatorio.append(f"Obsidian: integração indisponível - {exc}")

        if not modo_consulta:
            try:
                from watcher import iniciar_em_background

                iniciar_em_background(modelo)
            except Exception as exc:
                _falha_recuperavel("Watcher em background indisponível", exc)
                relatorio.append(f"Watcher: indisponível - {mascarar_segredos(str(exc))}")

            try:
                from src.orquestrador import executar_pipeline

                relatorio.extend(executar_pipeline(modelo))
            except Exception as exc:
                print(f"[Orquestrador] Erro: {exc}")
        perfil = carregar_perfil()

        from src.conhecimento.indice_lexical import IndiceLexicalSQLite

        indice_lexical = IndiceLexicalSQLite()
        versao_lexical = f"chroma:{colecao.count()}"
        if modo_consulta and ARQUIVO_INDICE_LITERATURA.is_file():
            try:
                from src.conhecimento.indice_portatil import ler_manifesto

                manifesto = ler_manifesto(ARQUIVO_INDICE_LITERATURA)
                versao_lexical = str(
                    manifesto.get("hash_corpus_sha256") or versao_lexical
                )
            except Exception as exc:
                _falha_recuperavel("Manifesto do índice portátil ilegível", exc)
                relatorio.append("Índice portátil: manifesto ilegível; usando contagem local.")
        elif colecao.count():
            # No PC, o Chroma pode ter sido reindexado sem regenerar ainda o
            # snapshot portatil. IDs amostrados detectam essa troca sem ler o
            # corpus inteiro na inicializacao.
            ids_amostra = []
            for offset in sorted({0, colecao.count() // 2, colecao.count() - 1}):
                try:
                    lote = colecao.get(
                        limit=1,
                        offset=offset,
                        include=["metadatas"],
                    )
                    ids_amostra.extend(lote.get("ids") or [])
                except Exception as exc:
                    _falha_recuperavel("Amostra do índice vetorial indisponível", exc)
            versao_lexical += ":" + ":".join(map(str, ids_amostra))
        try:
            with st.spinner("Preparando a busca lexical BM25..."):
                estado_lexical = indice_lexical.sincronizar(
                    colecao,
                    versao=versao_lexical,
                )
        except Exception as exc:
            estado_lexical = {"reconstruido": False}
            relatorio.append(
                f"Busca lexical indisponivel; busca semantica mantida: {exc}"
            )
        if estado_lexical.get("reconstruido"):
            relatorio.append(
                "Busca lexical: "
                f"{estado_lexical['n_chunks']} chunks preparados em SQLite FTS5."
            )

    return (
        perfil,
        modelo,
        colecao,
        colecao_sessoes,
        colecao_obsidian,
        indice_lexical,
        relatorio,
    )


def inicializar_estado() -> None:
    defaults = {
        "mensagens": [],
        "llm": None,
        "equipe": None,
        "auditor": None,
        "erro_equipe": None,
        "nome_provedor": "Nenhum",
        "caminho_sessao": None,
        "pergunta_pendente": None,
        "anexos_pendentes": [],
        "multimodal": False,
    }
    for chave, valor in defaults.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def conectar_equipe(*, forcar: bool = False) -> bool:
    """Ativa a equipe Gemini (Pro conversa + Flash auditoria) nos papeis fixos."""
    if st.session_state.get("equipe") is not None:
        return True
    if st.session_state.get("erro_equipe") and not forcar:
        return False
    try:
        from src.conhecimento.multiagente import criar_equipe_agentes

        equipe = criar_equipe_agentes()
        st.session_state.equipe = equipe
        st.session_state.llm = equipe.conversa
        st.session_state.auditor = equipe.auditoria
        # O orcamento do prompt e a capacidade multimodal seguem o agente que
        # produz a resposta final; o auditor recebe seu proprio pacote compacto.
        st.session_state.nome_provedor = "Google Gemini"
        st.session_state.multimodal = True
        st.session_state.erro_equipe = None
        return True
    except Exception as exc:
        from src.core.seguranca import mascarar_segredos

        st.session_state.equipe = None
        st.session_state.llm = None
        st.session_state.auditor = None
        st.session_state.nome_provedor = "Nenhum"
        st.session_state.multimodal = False
        st.session_state.erro_equipe = mascarar_segredos(str(exc))
        return False


from src.interface.sidebar import (
    renderizar_pipeline_status,
    renderizar_diagnostico,
    _carregar_metadados_pendentes,
    _estado_persistencia,
    renderizar_sidebar,
)


# Atalhos da tela inicial: rótulo curto no botão, pergunta completa enviada
# ao agente. Servem de exemplo do que ele faz — sem parágrafo explicando.
_SUGESTOES_LOCAL = [
    ("Comparar com a literatura", "Compare meu método com a literatura por AUC."),
    ("Status do pipeline", "Como estão as etapas do pipeline?"),
    ("Weibull e RUL", "Rode a análise de Weibull e interprete MTTF e B10."),
]
_SUGESTOES_CONSULTA = [
    ("Comparar com a literatura", "Compare meu método com a literatura por AUC."),
    ("Resultados publicados", "Mostre os resultados de validação publicados."),
    ("Weibull e RUL", "Interprete os resultados de Weibull, MTTF e B10."),
]


def renderizar_boas_vindas() -> None:
    """Tela inicial: identidade em duas linhas e três atalhos. Nada mais.

    A versão anterior abria com um bloco de instruções e cinco exemplos em
    lista — texto que o pesquisador já sabe e relia a cada sessão.
    """
    from src.conhecimento.agente import _saudacao_pelo_horario
    from src.ml.pipeline import capacidade_recalculo_pipeline

    st.markdown(
        '<div class="alp-hero">'
        '<div class="alp-hero-icone">⚡</div>'
        f'<h1>{_saudacao_pelo_horario()}, Rodolfo.</h1>'
        '<p>Falhas CA em inversores fotovoltaicos — confiabilidade, '
        'detecção de anomalia e Machine Learning.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    sugestoes = (_SUGESTOES_LOCAL if capacidade_recalculo_pipeline()["disponivel"]
                 else _SUGESTOES_CONSULTA)
    # Margens laterais deixam os atalhos alinhados com o texto do herói,
    # em vez de esticados de ponta a ponta da tela.
    colunas = st.columns([1, *([2] * len(sugestoes)), 1], gap="small")[1:-1]
    for coluna, (rotulo, prompt) in zip(colunas, sugestoes):
        if coluna.button(rotulo, key=f"sugestao_{rotulo}", width="stretch"):
            st.session_state.pergunta_pendente = prompt
            st.rerun()


def stream_resposta(prompt: str, llm):
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        yield chunk.content


# ── Cadência de "digitação" do streaming ────────────────────────────────────
# A resposta é revelada palavra a palavra com uma pequena pausa, em vez de
# "estourar" blocos inteiros de texto (efeito comum quando o provedor entrega
# muitos tokens de uma vez). Deixa a leitura mais natural, como se o agente
# estivesse escrevendo na hora.
#   VELOCIDADE_DIGITACAO   = segundos por palavra (MAIOR = mais devagar).
#   ORCAMENTO_DIGITACAO_S  = teto de tempo "datilografado"; passando disso, o
#                            restante aparece direto, para respostas longas não
#                            se arrastarem.
VELOCIDADE_DIGITACAO = 0.05
ORCAMENTO_DIGITACAO_S = 18.0


def stream_resposta_limpa(conteudo, llm, placeholder, refs_md: str) -> str:
    """
    Streama a resposta do LLM dentro de um placeholder e, ao final, substitui
    pelo texto limpo (sem bloco de fontes hallucinado) + lista oficial.

    `conteudo` pode ser uma string (prompt puro) OU uma lista de partes
    multimodais (texto + image_url) quando ha imagem anexada e o provedor e
    multimodal — exatamente o formato aceito por `HumanMessage(content=...)`.

    O efeito visual: o usuario ve a resposta surgir token a token (com um
    cursor '▌'), e o bloco final de fontes aparece apenas UMA vez,
    consolidado, mesmo que o LLM tenha gerado o proprio.
    """
    import re
    import time
    from src.conhecimento.agente import remover_bloco_fontes_llm

    texto = ""
    cursor = "▌"
    gasto = 0.0  # tempo ja "datilografado"; apos o orcamento, revela direto

    # Espera ativa: o brilho ocupa o intervalo (as vezes varios segundos, com
    # RAG + auditoria) entre o pedido e o primeiro token. O primeiro pedaco de
    # texto sobrescreve o placeholder e o faz sumir.
    # A partir daqui NADA usa unsafe_allow_html: o texto do modelo continua
    # sendo renderizado como Markdown puro, com HTML escapado.
    placeholder.markdown(_html_pensando(), unsafe_allow_html=True)

    for chunk in llm.stream([HumanMessage(content=conteudo)]):
        novo = chunk.content or ""
        if not novo:
            continue
        # Revela palavra a palavra com uma pausa curta, para um efeito de
        # digitacao natural. O padrao "\s*\S+\s*" preserva TODOS os caracteres
        # (inclusive espacos no inicio do chunk), evitando colar palavras
        # quando o provedor quebra o texto em " palavra".
        for pedaco in re.findall(r"\s*\S+\s*", novo) or [novo]:
            texto += pedaco
            placeholder.markdown(texto + cursor)
            if gasto < ORCAMENTO_DIGITACAO_S:
                time.sleep(VELOCIDADE_DIGITACAO)
                gasto += VELOCIDADE_DIGITACAO

    texto = remover_bloco_fontes_llm(texto)
    if refs_md:
        final = f"{texto}\n\n---\n📚 **Fontes consultadas:**\n{refs_md}"
    else:
        final = texto
    placeholder.markdown(final)
    return final


from src.interface.renderizacao_imagens import (
    _grupo_imagem,
    _DPI_GERACAO,
    _PX_POR_POLEGADA,
    _TETO_EXIBICAO,
    _LARGURA_PAREAVEL,
    _dimensoes_imagem,
    _polegadas_imagem,
    _largura_exibicao_imagem,
    _imagem_larga,
    _ordem_imagem,
    _DL_KEY,
    _botao_download,
    _botao_download_texto,
    _controles_antevisao,
    _renderizar_imagem_unica,
    _renderizar_lote_regular,
    _renderizar_grupo_imagens,
    renderizar_imagens,
)


def renderizar_mensagem(msg: dict) -> None:
    avatar = "🔬" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Botão de download da conversa exportada — re-renderizado a cada rerun
        # para o download continuar disponível (o Streamlit exige isso).
        exportado = msg.get("export_txt")
        if exportado:
            _botao_download_texto(exportado["data"], exportado["file_name"])
        renderizar_imagens(msg.get("imagens", []))


from src.interface.ciclo_chat import (
    salvar_sessao,
    _cadencia_atingida,
    aprender_da_sessao_web,
    persistir_sessao_web,
    _fechar_turno_simples,
    _contexto_recente,
    responder_com_ferramenta,
    responder_com_rag,
)


def renderizar_chat(
    perfil,
    modelo,
    colecao,
    colecao_sessoes,
    colecao_obsidian,
    indice_lexical=None,
) -> None:
    for msg in st.session_state.mensagens:
        renderizar_mensagem(msg)

    if not st.session_state.mensagens and st.session_state.pergunta_pendente is None:
        renderizar_boas_vindas()

    if st.session_state.pergunta_pendente is None:
        return

    pergunta = st.session_state.pergunta_pendente
    st.session_state.pergunta_pendente = None

    arquivos_pendentes = st.session_state.get("anexos_pendentes", []) or []
    st.session_state.anexos_pendentes = []

    anexos: list = []
    rotulo_anexos = ""
    if arquivos_pendentes:
        from src.conhecimento.leitor_anexos import ler_anexos

        with st.spinner("Lendo anexos..."):
            anexos = ler_anexos(arquivos_pendentes)
        nomes = ", ".join(a.get("nome", "anexo") for a in anexos)
        rotulo_anexos = f"\n\n_📎 Anexos: {nomes}_"

    conteudo_usuario = pergunta + rotulo_anexos

    with st.chat_message("user", avatar="🔬"):
        st.markdown(conteudo_usuario)

    # UM ponto para todos os atalhos determinísticos — exportar conversa, cofre
    # de trechos, inventário do vault, cronologia e saudação. A ordem e o
    # motivo de cada um vivem em src/conhecimento/atalhos.py, não espalhados
    # aqui no meio do render. Com anexo, nenhum se aplica: o pesquisador quer o
    # arquivo lido, e isso é trabalho do RAG.
    if not anexos:
        from src.conhecimento.atalhos import resolver_atalho

        atalho = resolver_atalho(pergunta, {
            "mensagens": st.session_state.mensagens,
            "colecao_obsidian": colecao_obsidian,
        })
        if atalho is not None:
            with st.chat_message("assistant", avatar="⚡"):
                st.markdown(atalho.texto)
                if atalho.anexo_txt:
                    _botao_download_texto(atalho.anexo_txt["data"],
                                          atalho.anexo_txt["file_name"])
            _fechar_turno_simples(conteudo_usuario, atalho.texto, modelo,
                                  anexo_txt=atalho.anexo_txt)
            return

    # Com anexos, ir direto ao RAG (que le o arquivo). Pular o roteador de
    # ferramentas evita misrotear "o que tem nesse arquivo?" para o pipeline ML.
    if anexos:
        resposta = responder_com_rag(
            pergunta,
            perfil,
            modelo,
            colecao,
            colecao_sessoes,
            colecao_obsidian,
            indice_lexical,
            anexos=anexos,
        )
        imagens = []
    else:
        resposta, imagens = responder_com_ferramenta(
            pergunta, perfil, st.session_state.llm
        )
        if not resposta:
            resposta = responder_com_rag(
                pergunta,
                perfil,
                modelo,
                colecao,
                colecao_sessoes,
                colecao_obsidian,
                indice_lexical,
            )
            imagens = []

    st.session_state.mensagens.append({
        "role": "user",
        "content": conteudo_usuario,
        "imagens": [],
    })
    st.session_state.mensagens.append({
        "role": "assistant",
        "content": resposta,
        "imagens": imagens,
    })
    salvar_sessao(
        conteudo_usuario,
        resposta,
        imagens,
        len(st.session_state.mensagens) // 2,
        modelo,
    )
    # Aprendizado automático: extrai decisões duráveis da conversa (a cada N
    # interações), independente do modo_consulta/watcher. É o que faz o agente
    # acumular conhecimento entre sessões.
    aprender_da_sessao_web()
    # Mesmo ritmo: commita o transcrito da sessão no GitHub, para sobreviver
    # a reboots/redeploys do container efêmero da nuvem.
    persistir_sessao_web()

    auditor = st.session_state.get("auditor")
    if auditor is not None:
        aprendizado = auditor.aprender_da_interacao(pergunta, resposta)
        if aprendizado.salvas:
            st.toast(
                f"{aprendizado.salvas} memoria(s) validada(s) para as proximas sessoes."
            )


def main() -> None:
    inicializar_estado()
    st.markdown(_CSS_MINIMO, unsafe_allow_html=True)
    conectar_equipe()

    try:
        (
            perfil,
            modelo,
            colecao,
            colecao_sessoes,
            colecao_obsidian,
            indice_lexical,
            relatorio,
        ) = carregar_base()
    except Exception as exc:
        st.error(f"Erro ao carregar o agente: {exc}")
        return

    # Sem cabeçalho no corpo: a identidade fica na barra lateral e a área
    # principal é só a conversa. O relatório de inicialização virou uma
    # seção do Diagnóstico, em vez de um expander no topo de toda sessão.
    renderizar_sidebar(modelo, colecao, colecao_sessoes, colecao_obsidian, relatorio)

    renderizar_chat(
        perfil,
        modelo,
        colecao,
        colecao_sessoes,
        colecao_obsidian,
        indice_lexical,
    )

    entrada = st.chat_input(
        "Peça uma análise ou anexe um arquivo (PDF, CSV, imagem, código...)",
        accept_file="multiple",
        file_type=TIPOS_ANEXO,
    )
    if entrada:
        # Com accept_file, `entrada` tem .text e .files; sem, e uma string.
        texto = getattr(entrada, "text", None)
        arquivos = getattr(entrada, "files", None)
        if texto is None and isinstance(entrada, str):
            texto = entrada

        anexos_bytes: list[tuple[str, bytes]] = []
        for f in (arquivos or []):
            try:
                anexos_bytes.append((f.name, f.getvalue()))
            except Exception as exc:
                nome = getattr(f, "name", "anexo")
                _falha_recuperavel(
                    f"Não foi possível ler o anexo {nome}", exc, notificar=True
                )

        # Só anexou arquivo, sem texto: damos um pedido padrão de leitura.
        if not (texto or "").strip() and anexos_bytes:
            texto = "Leia o(s) arquivo(s) anexado(s) e me explique o conteúdo."

        if (texto or "").strip() or anexos_bytes:
            st.session_state.pergunta_pendente = texto or ""
            st.session_state.anexos_pendentes = anexos_bytes
            st.rerun()
