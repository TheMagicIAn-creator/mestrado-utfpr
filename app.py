"""
app.py — Al IAdo PV
Interface web com Streamlit.

Como executar:
  streamlit run app.py

Autor: Rodolfo Torres (UTFPR)
"""

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

        if st.button("💾 Salvar sessão", use_container_width=True):
            salvar_sessao_streamlit(
                st.session_state.get("mensagens", []),
                modelo
            )

        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.rerun()

        st.divider()

        # ── Provedor ativo ──────────────────────────────────────
        nome_ativo = st.session_state.get("nome_provedor", "Nenhum")
        st.caption(f"Provedor ativo: **{nome_ativo}**")


# ============================================================
# SALVAR SESSÃO
# ============================================================

def salvar_sessao_streamlit(mensagens: list, modelo_embeddings):
    """Salva a sessão atual em .md e indexa no ChromaDB."""

    if not mensagens:
        st.warning("Nenhuma conversa para salvar.")
        return

    pasta_sessoes = Path(__file__).parent / "notas" / "sessoes"
    pasta_sessoes.mkdir(parents=True, exist_ok=True)

    agora        = datetime.now()
    nome_arquivo = agora.strftime("%Y-%m-%d_%H-%M") + "_sessao_web.md"
    caminho      = pasta_sessoes / nome_arquivo

    data_formatada = agora.strftime("%d/%m/%Y às %H:%M")
    conteudo  = f"---\n"
    conteudo += f"data: {agora.strftime('%Y-%m-%d')}\n"
    conteudo += f"hora: {agora.strftime('%H:%M')}\n"
    conteudo += f"tipo: sessao-web\n"
    conteudo += f"tags: [al-iado-pv, sessao, streamlit, mestrado]\n"
    conteudo += f"---\n\n"
    conteudo += f"# Sessão Web Al IAdo PV — {data_formatada}\n\n"

    n_pergunta = 0
    for msg in mensagens:
        if msg["role"] == "user":
            n_pergunta += 1
            conteudo += f"---\n\n## Pergunta {n_pergunta}\n\n"
            conteudo += f"**🔬 Você:** {msg['content']}\n\n"
        else:
            conteudo += f"**🤖 Al IAdo PV:**\n\n{msg['content']}\n\n"

    conteudo += f"\n---\n*Sessão web gerada pelo Al IAdo PV*\n"
    caminho.write_text(conteudo, encoding="utf-8")

    # Indexa no ChromaDB
    try:
        from src.indexador import indexar_sessao
        from src.agente import PASTA_CHROMADB
        n = indexar_sessao(caminho, modelo_embeddings, PASTA_CHROMADB)
        st.success(f"✅ Sessão salva e indexada! ({n} chunks)\n📁 {nome_arquivo}")
    except Exception as e:
        st.success(f"✅ Sessão salva em: {nome_arquivo}")
        st.warning(f"⚠️ Erro ao indexar: {e}")


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


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()