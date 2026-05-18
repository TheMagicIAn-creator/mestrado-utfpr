"""
app.py — Al IAdo PV
Interface web com Streamlit.

Como executar:
  streamlit run app.py

Autor: Rodolfo Torres (UTFPR)
"""

from watcher import iniciar_em_background
import sys
import os
from pathlib import Path
from datetime import datetime

# Garante que Python encontra os módulos
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from langchain_core.messages import HumanMessage

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title = "Al IAdo PV ⚡",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)


# ============================================================
# CARREGAMENTO DOS COMPONENTES PESADOS (cache)
# ============================================================

@st.cache_resource
def carregar_base():
    """
    Carrega embeddings e ChromaDB uma única vez.
    O @st.cache_resource evita recarregar a cada interação.
    """
    from sentence_transformers import SentenceTransformer
    import chromadb
    from src.agente import (
        carregar_perfil,
        MODELO_EMBEDDINGS,
        PASTA_CHROMADB,
        NOME_COLECAO,
        NOME_COLECAO_SESSOES
    )

    with st.spinner("🔄 Carregando modelo de embeddings..."):
        modelo = SentenceTransformer(MODELO_EMBEDDINGS)

        # ── Inicia watcher em background ────────────────────────
        try:
            iniciar_em_background(modelo)
        except Exception as e:
            pass  # silencioso — não bloqueia o Streamlit

    with st.spinner("🗄️ Conectando ao ChromaDB..."):
        client          = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao         = client.get_or_create_collection(name=NOME_COLECAO)
        colecao_sessoes = client.get_or_create_collection(name=NOME_COLECAO_SESSOES)

    perfil = carregar_perfil()

    return perfil, modelo, colecao, colecao_sessoes


# ============================================================
# FUNÇÃO DE STREAMING PARA STREAMLIT
# ============================================================

def stream_resposta(prompt: str, llm):
    """
    Generator que faz streaming do LLM para o Streamlit.
    Compatível com st.write_stream().
    """
    mensagens = [HumanMessage(content=prompt)]
    for chunk in llm.stream(mensagens):
        yield chunk.content


# ============================================================
# SIDEBAR
# ============================================================
def _processar_upload(arquivo_pdf, modelo_embeddings):
    """
    Salva o PDF em novos_pdfs/ para ser processado pelo watcher.
    """
    pasta_entrada = Path(__file__).parent / "novos_pdfs"
    pasta_entrada.mkdir(exist_ok=True)

    caminho_destino = pasta_entrada / arquivo_pdf.name

    with open(caminho_destino, "wb") as f:
        f.write(arquivo_pdf.getbuffer())

    st.success(
        f"✅ **PDF enviado para processamento!**\n\n"
        f"**Arquivo:** `{arquivo_pdf.name}`\n\n"
        f"📂 Salvo em `novos_pdfs/`\n\n"
        f"⚙️ O watcher irá processar automaticamente:\n"
        f"renomear → classificar → indexar → nota Obsidian"
    )

    if caminho_destino.exists():
        st.info("💡 Certifique-se que o `watcher.py` está rodando em outro terminal.")

def renderizar_sidebar(perfil, modelo, colecao, colecao_sessoes):
    """Renderiza o painel lateral com controles e estatísticas."""

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/solar-panel.png", width=60)
        st.title("Al IAdo PV ⚡")
        st.caption("Assistente de Mestrado — UTFPR")
        st.divider()

        # ── Seleção de provedor ─────────────────────────────────
        st.subheader("🤖 Provedor de LLM")

        from src.provedores import PROVEDORES
        opcoes = {
            f"{info['emoji']} {info['nome']}": chave
            for chave, info in PROVEDORES.items()
        }

        escolha_label = st.selectbox(
            "Selecione o modelo:",
            options   = list(opcoes.keys()),
            index     = 0,
            key       = "provedor_select"
        )
        escolha = opcoes[escolha_label]
        info    = PROVEDORES[escolha]

        st.caption(f"Limite: {info['limite']}")

        if st.button("🔄 Conectar provedor", use_container_width=True):
            try:
                from src.provedores import inicializar_provedor
                with st.spinner(f"Conectando ao {info['nome']}..."):
                    llm, nome = inicializar_provedor(escolha)
                st.session_state.llm          = llm
                st.session_state.nome_provedor = nome
                st.success(f"✅ {nome} conectado!")
            except Exception as e:
                st.error(f"❌ {e}")

        st.divider()

        # ── Estatísticas ────────────────────────────────────────
        st.subheader("📊 Base de Conhecimento")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Literatura", f"{colecao.count()}", "chunks")
        with col2:
            st.metric("Sessões", f"{colecao_sessoes.count()}", "chunks")

        st.divider()

        # ── Ações ───────────────────────────────────────────────
        st.subheader("⚙️ Ações")

        if st.button("📋 Listar artigos", use_container_width=True):
            from src.agente import listar_documentos
            st.session_state.mostrar_artigos = True

        #if st.button("💾 Salvar sessão", use_container_width=True):
        #    salvar_sessao_streamlit(
        #        st.session_state.get("mensagens", []),
        #        modelo
        #    )

        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.rerun()

        st.divider()

        # ── Upload de PDF ───────────────────────────────────────
        st.subheader("📄 Adicionar PDF")
        st.caption("Tema detectado automaticamente")

        arquivo_pdf = st.file_uploader(
            "Selecione o PDF:",
            type=["pdf"],
            key="pdf_uploader",
            help="O sistema detecta autor, título, ano e tema automaticamente."
        )

        if arquivo_pdf is not None:
            if st.button("📥 Processar PDF", use_container_width=True):
                _processar_upload(arquivo_pdf, modelo)

        # ── Provedor ativo ──────────────────────────────────────
        nome_ativo = st.session_state.get("nome_provedor", "Nenhum")
        st.caption(f"Provedor ativo: **{nome_ativo}**")

        st.divider()

        # ── Provedor ativo ──────────────────────────────────────
        nome_ativo = st.session_state.get("nome_provedor", "Nenhum")
        st.caption(f"Provedor ativo: **{nome_ativo}**")


# ============================================================
# SALVAR SESSÃO
# ============================================================

def salvar_sessao_streamlit(pergunta: str, resposta: str, n: int, modelo_embeddings):
    """
    Cria o arquivo de sessão na primeira interação
    e ADICIONA ao mesmo arquivo nas seguintes.
    """
    from datetime import datetime
    from src.agente import PASTA_CHROMADB

    pasta_sessoes = Path(__file__).parent / "notas" / "sessoes"
    pasta_sessoes.mkdir(parents=True, exist_ok=True)

    # ── Primeira interação — cria o arquivo ─────────────────
    if st.session_state.caminho_sessao is None:
        agora        = datetime.now()
        nome_arquivo = agora.strftime("%Y-%m-%d_%H-%M") + "_sessao_web.md"
        caminho      = pasta_sessoes / nome_arquivo
        data_fmt     = agora.strftime("%d/%m/%Y às %H:%M")

        cabecalho  = f"---\n"
        cabecalho += f"data: {agora.strftime('%Y-%m-%d')}\n"
        cabecalho += f"hora: {agora.strftime('%H:%M')}\n"
        cabecalho += f"tipo: sessao-web\n"
        cabecalho += f"tags: [al-iado-pv, sessao, streamlit, mestrado]\n"
        cabecalho += f"---\n\n"
        cabecalho += f"# Sessão Web Al IAdo PV — {data_fmt}\n\n"

        caminho.write_text(cabecalho, encoding="utf-8")
        st.session_state.caminho_sessao = caminho

    # ── Todas as interações — adiciona ao mesmo arquivo ─────
    caminho = st.session_state.caminho_sessao

    bloco  = f"---\n\n"
    bloco += f"## Interação {n}\n\n"
    bloco += f"**🔬 Você:** {pergunta}\n\n"
    bloco += f"**🤖 Al IAdo PV:**\n\n{resposta}\n\n"

    with open(caminho, "a", encoding="utf-8") as f:
        f.write(bloco)

    # ── Indexa no ChromaDB a cada interação ───────────────
    if n >= 1:
        try:
            from src.indexador import indexar_sessao
            indexar_sessao(caminho, modelo_embeddings, PASTA_CHROMADB)
        except Exception:
            pass  # silencioso — não interrompe o chat


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def main():

    # Inicializa session_state
    if "mensagens"     not in st.session_state:
        st.session_state.mensagens      = []
    if "llm"           not in st.session_state:
        st.session_state.llm            = None
    if "nome_provedor" not in st.session_state:
        st.session_state.nome_provedor  = "Nenhum"
    if "mostrar_artigos" not in st.session_state:
        st.session_state.mostrar_artigos = False
    if "caminho_sessao" not in st.session_state:   # Salva a sessão de forma
        st.session_state.caminho_sessao = None      # simultânea

    # Carrega componentes base
    try:
        perfil, modelo, colecao, colecao_sessoes = carregar_base()
    except Exception as e:
        st.error(f"❌ Erro ao carregar o agente: {e}")
        st.info("Execute `python src/indexador.py` antes de iniciar o Streamlit.")
        return

    # Sidebar
    renderizar_sidebar(perfil, modelo, colecao, colecao_sessoes)

    # Título principal
    st.title("⚡ Al IAdo PV")
    st.caption("Assistente especialista em inversores fotovoltaicos — UTFPR")

    # Listagem de artigos (se solicitada)
    if st.session_state.mostrar_artigos:
        from src.agente import listar_documentos
        with st.expander("📚 Artigos indexados", expanded=True):
            st.text(listar_documentos(colecao))
        st.session_state.mostrar_artigos = False

    # Verifica se o LLM está conectado
    if st.session_state.llm is None:
        st.info("👈 Selecione um provedor de LLM no painel lateral e clique em **Conectar provedor** para começar.")
        return

    # Exibe histórico de mensagens
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"], avatar="🔬" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Input do usuário
    if pergunta := st.chat_input("Digite sua pergunta sobre inversores fotovoltaicos..."):

        # Exibe mensagem do usuário
        with st.chat_message("user", avatar="🔬"):
            st.markdown(pergunta)

        # Prepara o prompt
        from src.agente import preparar_prompt

        historico_para_prompt = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.mensagens
        ]

        with st.spinner("🔍 Buscando na literatura..."):
            prompt, citacoes = preparar_prompt(
                pergunta        = pergunta,
                perfil          = perfil,
                modelo_embeddings = modelo,
                colecao         = colecao,
                historico       = historico_para_prompt,
                colecao_sessoes = colecao_sessoes
            )

        # Gera e exibe resposta com streaming
        with st.chat_message("assistant", avatar="🤖"):
            try:
                resposta_texto = st.write_stream(
                    stream_resposta(prompt, st.session_state.llm)
                )

                # Exibe fontes
                if citacoes:
                    st.divider()
                    st.caption("📚 **Fontes consultadas:**")
                    for arquivo, citacao in citacoes.items():
                        st.caption(f"→ {citacao}")

            except Exception as e:
                erro = str(e)
                if "429" in erro:
                    st.error("⏳ Limite da API atingido. Troque o provedor no painel lateral ou aguarde.")
                else:
                    st.error(f"❌ Erro: {e}")
                resposta_texto = f"[Erro: {e}]"

        # Salva no histórico
        resposta_completa = resposta_texto
        if citacoes:
            resposta_completa += "\n\n**Fontes:** " + ", ".join(citacoes.values())

        st.session_state.mensagens.append({"role": "user",      "content": pergunta})
        st.session_state.mensagens.append({"role": "assistant",  "content": resposta_completa})

        # Auto-salvamento após cada interação
        n_interacoes = len(st.session_state.mensagens) // 2
        salvar_sessao_streamlit(pergunta, resposta_completa, n_interacoes, modelo)

# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()