"""
streamlit_app.py - Al IAdo PV
Interface conversacional do agente.

Resultados e execucoes do pipeline de ML aparecem pelo chat, conforme
solicitacao em prompt. A interface nao possui aba separada de resultados.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st
from langchain_core.messages import HumanMessage

from src.core.config import RAIZ_PROJETO
from watcher import iniciar_em_background

sys.path.insert(0, str(RAIZ_PROJETO))


st.set_page_config(
    page_title="Al IAdo PV",
    layout="wide",
    initial_sidebar_state="expanded",
)


def aplicar_estilo() -> None:
    escuro = st.session_state.get("tema_visual", "Claro") == "Escuro"
    cores = {
        "app": "#0f1117" if escuro else "#f6f7fb",
        "sidebar": "#171b24" if escuro else "#eef2f7",
        "panel": "#151a23" if escuro else "#ffffff",
        "panel_soft": "#111827" if escuro else "#ffffff",
        "text": "#e5e7eb" if escuro else "#111827",
        "muted": "#a6adbb" if escuro else "#4b5563",
        "border": "#2a3344" if escuro else "#d7dde8",
        "primary": "#60a5fa" if escuro else "#2563eb",
        "shadow": "rgba(0, 0, 0, 0.24)" if escuro else "rgba(15, 23, 42, 0.04)",
        "input": "#1f2430" if escuro else "#ffffff",
    }
    st.markdown(
        f"""
<style>
:root {{ color-scheme: {"dark" if escuro else "light"}; }}
#MainMenu, footer, .stDeployButton {{ display: none; }}
.stApp {{
    background: {cores["app"]};
    color: {cores["text"]};
}}
.block-container {{
    max-width: 1180px;
    padding-top: 1.1rem;
    padding-bottom: 1.4rem;
}}
[data-testid="stSidebar"] {{
    background: {cores["sidebar"]};
    color: {cores["text"]};
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1rem;
}}
[data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] p {{
    color: {cores["text"]} !important;
}}
[data-testid="stHeader"] {{
    background: transparent;
}}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] strong {{
    color: inherit;
}}
.topbar {{
    border-bottom: 1px solid {cores["border"]};
    padding: 10px 0 14px 0;
    margin-bottom: 18px;
}}
.topbar h1 {{
    margin: 0;
    font-size: 1.7rem;
    line-height: 1.15;
    color: {cores["text"]};
}}
.topbar p {{
    margin: 6px 0 0 0;
    color: {cores["muted"]};
}}
.quiet-panel {{
    border-left: 3px solid {cores["primary"]};
    background: {cores["panel"]};
    border-radius: 8px;
    padding: 13px 16px;
    box-shadow: 0 1px 2px {cores["shadow"]};
}}
.prompt-example {{
    border-left: 3px solid {cores["primary"]};
    padding: 8px 0 8px 12px;
    margin: 6px 0;
    color: {cores["text"]};
}}
.status-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}}
.ok-dot {{ background: #22c55e; }}
.pending-dot {{ background: #f59e0b; }}
[data-testid="stChatMessage"] {{
    border-radius: 8px;
    background: {cores["panel_soft"]};
    border: 1px solid {cores["border"]};
}}
[data-testid="metric-container"] {{
    border: 1px solid {cores["border"]};
    border-radius: 8px;
    padding: 10px 12px;
    background: {cores["panel"]};
}}
div[data-testid="stChatInput"] {{
    max-width: 1180px;
}}
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"] {{
    background: {cores["app"]} !important;
}}
div[data-testid="stChatInput"] > div {{
    background: {cores["input"]} !important;
    border-color: {cores["border"]} !important;
}}
div[data-testid="stChatInput"] textarea {{
    background: {cores["input"]} !important;
    color: {cores["text"]} !important;
}}
div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input {{
    background: {cores["input"]};
    color: {cores["text"]};
    border-color: {cores["border"]};
}}
[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] {{
    background: {cores["panel"]} !important;
    color: {cores["text"]} !important;
    border-color: {cores["border"]} !important;
}}
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpanderDetails"] {{
    background: {cores["panel"]} !important;
    color: {cores["text"]} !important;
    border-color: {cores["border"]} !important;
}}
button[kind="primary"] {{
    background: {cores["primary"]};
}}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def carregar_base():
    from sentence_transformers import SentenceTransformer
    import chromadb
    from src.conhecimento.agente import carregar_perfil
    from src.core.config import (
        MODELO_EMBEDDINGS,
        NOME_COLECAO,
        NOME_COLECAO_SESSOES,
        PASTA_CHROMADB,
    )

    with st.spinner("Carregando embeddings e base de conhecimento..."):
        modelo = SentenceTransformer(MODELO_EMBEDDINGS)
        try:
            iniciar_em_background(modelo)
        except Exception:
            pass

        relatorio = []
        try:
            from src.orquestrador import executar_pipeline

            relatorio = executar_pipeline(modelo)
        except Exception as exc:
            print(f"[Orquestrador] Erro: {exc}")

        client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = client.get_or_create_collection(name=NOME_COLECAO)
        colecao_sessoes = client.get_or_create_collection(name=NOME_COLECAO_SESSOES)
        perfil = carregar_perfil()

    return perfil, modelo, colecao, colecao_sessoes, relatorio


def inicializar_estado() -> None:
    defaults = {
        "mensagens": [],
        "llm": None,
        "nome_provedor": "Nenhum",
        "caminho_sessao": None,
        "pergunta_pendente": None,
        "tema_visual": "Claro",
    }
    for chave, valor in defaults.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def pipeline_status_html() -> str:
    from src.ml.pipeline import NOMES_ETAPAS, pipeline_status

    linhas = []
    for key, pronto in pipeline_status().items():
        cls = "ok-dot" if pronto else "pending-dot"
        estado = "pronto" if pronto else "pendente"
        linhas.append(
            f'<div><span class="status-dot {cls}"></span>'
            f'{NOMES_ETAPAS[key]} <span style="opacity:.65">({estado})</span></div>'
        )
    return "\n".join(linhas)


def renderizar_sidebar(modelo, colecao, colecao_sessoes) -> None:
    with st.sidebar:
        st.markdown("## Al IAdo PV")
        st.caption("Mestrado UTFPR - agente de pesquisa")

        st.radio(
            "Tela",
            options=["Claro", "Escuro"],
            horizontal=True,
            key="tema_visual",
        )

        provedor = st.session_state.get("nome_provedor", "Nenhum")
        if provedor == "Nenhum":
            st.warning("LLM desconectado")
        else:
            st.success(f"LLM ativo: {provedor}")

        st.divider()
        st.markdown("**Provedor**")
        from src.conhecimento.provedores import PROVEDORES, inicializar_provedor

        opcoes = {info["nome"]: chave for chave, info in PROVEDORES.items()}
        escolha_label = st.selectbox(
            "Modelo",
            options=list(opcoes.keys()),
            index=0,
            label_visibility="collapsed",
        )
        escolha = opcoes[escolha_label]
        st.caption(PROVEDORES[escolha]["limite"])
        if st.button("Conectar", use_container_width=True, type="primary"):
            try:
                llm, nome = inicializar_provedor(escolha)
                st.session_state.llm = llm
                st.session_state.nome_provedor = nome
                st.rerun()
            except Exception as exc:
                st.error(f"Erro ao conectar: {exc}")

        st.divider()
        st.markdown("**Conhecimento**")
        c1, c2 = st.columns(2)
        c1.metric("Literatura", colecao.count())
        c2.metric("Sessões", colecao_sessoes.count())

        st.divider()
        st.markdown("**Pipeline ML**")
        st.markdown(pipeline_status_html(), unsafe_allow_html=True)
        st.caption("Para rodar, refazer ou consultar resultados, use o chat.")
        feedback_limpeza = st.session_state.pop("feedback_limpeza_ml", None)
        if feedback_limpeza:
            st.success(feedback_limpeza)

        with st.expander("Resultados e recálculo"):
            from src.ml.pipeline import (
                NOMES_ETAPAS,
                ORDEM_ETAPAS_ML,
                artefatos_a_partir,
                limpar_artefatos,
            )

            labels = {NOMES_ETAPAS[key]: key for key in ORDEM_ETAPAS_ML}
            etapa_label = st.selectbox(
                "Apagar a partir de",
                options=list(labels.keys()),
                help=(
                    "Apaga os artefatos da etapa escolhida e de todas as etapas "
                    "seguintes. Use quando mudar modelo, parâmetros ou dados."
                ),
            )
            etapa = labels[etapa_label]
            existentes = [p for p in artefatos_a_partir(etapa) if p.exists()]
            st.caption(f"{len(existentes)} arquivo(s) existente(s) serão removidos.")
            confirmar = st.checkbox(
                "Confirmo que quero apagar esses resultados",
                key="confirmar_limpeza_ml",
            )
            if st.button(
                "Apagar resultados selecionados",
                use_container_width=True,
                disabled=not confirmar,
            ):
                removidos = limpar_artefatos(etapa)
                st.session_state.feedback_limpeza_ml = (
                    f"{len(removidos)} arquivo(s) removido(s) a partir de {etapa_label}."
                )
                st.rerun()

        st.divider()
        st.markdown("**PDFs**")
        arquivo_pdf = st.file_uploader(
            "Adicionar PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if arquivo_pdf is not None:
            if st.button("Enviar para processamento", use_container_width=True):
                destino = RAIZ_PROJETO / "novos_pdfs" / arquivo_pdf.name
                destino.parent.mkdir(exist_ok=True)
                destino.write_bytes(arquivo_pdf.getbuffer())
                st.success("PDF enviado. O watcher processara automaticamente.")

        st.divider()
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.session_state.caminho_sessao = None
            st.rerun()

        with st.expander("Manutenção"):
            if st.button("Consolidar memória", use_container_width=True):
                try:
                    from src.conhecimento.consolidar_memoria import consolidar

                    ok = consolidar(forcar=True)
                    st.success("Memória consolidada." if ok else "Nada a consolidar.")
                except Exception as exc:
                    st.error(f"Erro: {exc}")

            if st.button("Corrigir metadados ruins", use_container_width=True):
                try:
                    from src.orquestrador import reprocessar_metadados_ruins

                    st.info(reprocessar_metadados_ruins())
                except Exception as exc:
                    st.error(f"Erro: {exc}")


def renderizar_topo(relatorio: list) -> None:
    provedor = st.session_state.get("nome_provedor", "Nenhum")
    status = "Conectado" if provedor != "Nenhum" else "Aguardando conexão"
    st.markdown(
        f"""
<div class="topbar">
  <h1>Al IAdo PV</h1>
  <p>Pesquisa, literatura e Machine Learning para falhas CA em inversores fotovoltaicos. Estado: <strong>{status}</strong>.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    novidades = [
        str(item)
        for item in relatorio
        if item and "nenhum pendente" not in str(item).lower()
    ]
    if novidades:
        with st.expander("Novidades processadas na inicialização", expanded=False):
            for item in novidades:
                st.write(item)


def renderizar_boas_vindas() -> None:
    st.markdown(
        """
<div class="quiet-panel">
  <strong>Como quer trabalhar agora?</strong><br>
  Peça em linguagem natural. Eu posso consultar a literatura, rodar etapas do pipeline,
  explicar métricas, mostrar gráficos ou discutir a dissertação com você.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Exemplos de prompt")
    exemplos = [
        "Explique os resultados de validação e mostre as curvas ROC.",
        "Rode a análise de Weibull e depois interprete MTTF e B10.",
        "Quais falhas tiveram menor severidade mínima detectável?",
        "Compare o papel do FMEA com o Autoencoder na metodologia.",
    ]
    for exemplo in exemplos:
        st.markdown(f'<div class="prompt-example">{exemplo}</div>', unsafe_allow_html=True)


def stream_resposta(prompt: str, llm):
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        yield chunk.content


def renderizar_imagens(imagens: list[dict]) -> None:
    if not imagens:
        return
    cols = st.columns(min(2, len(imagens)))
    for idx, img in enumerate(imagens):
        col = cols[idx % len(cols)]
        col.image(img["path"], caption=img.get("caption", ""), use_container_width=True)


def renderizar_mensagem(msg: dict) -> None:
    avatar = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        renderizar_imagens(msg.get("imagens", []))


def salvar_sessao(pergunta: str, resposta: str, imagens: list[dict], n: int, modelo_embeddings) -> None:
    from src.conhecimento.indexador import indexar_sessao
    from src.core.config import PASTA_CHROMADB

    pasta_sessoes = RAIZ_PROJETO / "notas" / "sessoes"
    pasta_sessoes.mkdir(parents=True, exist_ok=True)

    if st.session_state.caminho_sessao is None:
        agora = datetime.now()
        caminho = pasta_sessoes / f"{agora:%Y-%m-%d_%H-%M}_sessao_web.md"
        caminho.write_text(
            (
                "---\n"
                f"data: {agora:%Y-%m-%d}\n"
                f"hora: {agora:%H:%M}\n"
                "tipo: sessao-web\n"
                "tags: [al-iado-pv, sessao, streamlit, mestrado]\n"
                "---\n\n"
                f"# Sessão Web - {agora:%d/%m/%Y às %H:%M}\n\n"
            ),
            encoding="utf-8",
        )
        st.session_state.caminho_sessao = caminho

    caminhos_img = "\n".join(
        f"- {img.get('caption', 'Imagem')}: {img['path']}" for img in imagens
    )
    bloco = (
        f"---\n\n## Interação {n}\n\n"
        f"**Você:** {pergunta}\n\n"
        f"**Al IAdo PV:**\n\n{resposta}\n\n"
    )
    if caminhos_img:
        bloco += f"**Imagens exibidas:**\n{caminhos_img}\n\n"

    with open(st.session_state.caminho_sessao, "a", encoding="utf-8") as f:
        f.write(bloco)

    try:
        indexar_sessao(st.session_state.caminho_sessao, modelo_embeddings, PASTA_CHROMADB)
    except Exception:
        pass


def responder_com_ferramenta(pergunta: str, perfil: str, llm) -> tuple[str, list[dict]]:
    from src.conhecimento.ferramentas import decidir_acao, processar_com_ferramentas

    with st.spinner("Interpretando o pedido..."):
        decisao = decidir_acao(pergunta, llm)

    if not decisao["usar_ferramenta"]:
        return "", []

    with st.chat_message("assistant", avatar="assistant"):
        with st.status("Executando solicitação...", expanded=True) as status:
            saida = processar_com_ferramentas(
                pergunta=pergunta,
                perfil=perfil,
                llm=llm,
                progresso=status.write,
                decisao=decisao,
            )
            ok = bool(saida["resultado"] and saida["resultado"].get("ok"))
            status.update(
                label="Solicitação concluída" if ok else "Solicitação terminou com erro",
                state="complete" if ok else "error",
            )
        resposta = saida["resposta"] or "Sem resposta."
        imagens = saida["resultado"].get("imagens", []) if saida["resultado"] else []
        st.markdown(resposta)
        renderizar_imagens(imagens)
    return resposta, imagens


def responder_com_rag(pergunta: str,
                      perfil: str,
                      modelo,
                      colecao,
                      colecao_sessoes) -> str:
    from src.conhecimento.agente import preparar_prompt, resposta_interacao_simples

    resposta_simples = resposta_interacao_simples(pergunta)
    if resposta_simples:
        with st.chat_message("assistant", avatar="assistant"):
            st.markdown(resposta_simples)
        return resposta_simples

    if st.session_state.llm is None:
        with st.chat_message("assistant", avatar="assistant"):
            resposta = (
                "Consigo consultar status e resultados do pipeline por aqui, "
                "mas para conversar com a literatura ou interpretar perguntas "
                "abertas preciso que voce conecte um LLM no painel lateral."
            )
            st.info(resposta)
        return resposta

    historico = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.mensagens
    ]
    with st.spinner("Buscando literatura e memória..."):
        prompt, citacoes = preparar_prompt(
            pergunta=pergunta,
            perfil=perfil,
            modelo_embeddings=modelo,
            colecao=colecao,
            historico=historico,
            colecao_sessoes=colecao_sessoes,
            nome_provedor=st.session_state.get("nome_provedor", ""),
        )

    with st.chat_message("assistant", avatar="assistant"):
        try:
            resposta = st.write_stream(stream_resposta(prompt, st.session_state.llm))
            if citacoes:
                st.caption("Fontes consultadas: " + "; ".join(citacoes.values()))
                resposta += "\n\n**Fontes:** " + "; ".join(citacoes.values())
        except Exception as exc:
            erro = str(exc)
            if "413" in erro or "Request too large" in erro:
                st.error(
                    "A solicitação ainda ficou grande demais para o limite do provedor. "
                    "Tente pedir uma resposta mais focada ou troque para Gemini."
                )
            elif "429" in erro:
                st.error("Limite da API atingido. Aguarde ou troque o provedor.")
            else:
                st.error(f"Erro: {exc}")
            resposta = f"[Erro: {exc}]"
    return resposta


def renderizar_chat(perfil, modelo, colecao, colecao_sessoes) -> None:
    for msg in st.session_state.mensagens:
        renderizar_mensagem(msg)

    if not st.session_state.mensagens and not st.session_state.pergunta_pendente:
        renderizar_boas_vindas()

    if not st.session_state.pergunta_pendente:
        return

    pergunta = st.session_state.pergunta_pendente
    st.session_state.pergunta_pendente = None

    with st.chat_message("user", avatar="user"):
        st.markdown(pergunta)

    resposta, imagens = responder_com_ferramenta(
        pergunta, perfil, st.session_state.llm
    )
    if not resposta:
        resposta = responder_com_rag(
            pergunta, perfil, modelo, colecao, colecao_sessoes
        )
        imagens = []

    st.session_state.mensagens.append({
        "role": "user",
        "content": pergunta,
        "imagens": [],
    })
    st.session_state.mensagens.append({
        "role": "assistant",
        "content": resposta,
        "imagens": imagens,
    })
    salvar_sessao(
        pergunta,
        resposta,
        imagens,
        len(st.session_state.mensagens) // 2,
        modelo,
    )


def main() -> None:
    inicializar_estado()
    aplicar_estilo()

    try:
        perfil, modelo, colecao, colecao_sessoes, relatorio = carregar_base()
    except Exception as exc:
        st.error(f"Erro ao carregar o agente: {exc}")
        return

    renderizar_sidebar(modelo, colecao, colecao_sessoes)
    renderizar_topo(relatorio)

    renderizar_chat(perfil, modelo, colecao, colecao_sessoes)

    pergunta = st.chat_input("Peça uma análise, resultado, gráfico ou próxima etapa...")
    if pergunta:
        st.session_state.pergunta_pendente = pergunta
        st.rerun()
