"""
streamlit_app.py - Al IAdo PV
Interface conversacional do agente.

Resultados e execucoes do pipeline de ML aparecem pelo chat, conforme
solicitacao em prompt. A interface usa componentes nativos do Streamlit
para preservar a estetica original — sem overrides de CSS pesados.
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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS mínimo — apenas oculta deploy button e o menu do Streamlit Cloud.
# O tema claro/escuro é gerenciado nativamente pelo Streamlit via Settings (⋮).
_CSS_MINIMO = """
<style>
#MainMenu        { visibility: hidden; }
.stDeployButton  { display: none; }
</style>
"""


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
    }
    for chave, valor in defaults.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def renderizar_pipeline_status() -> None:
    """Mostra status do pipeline no sidebar com elementos nativos do Streamlit."""
    from src.ml.pipeline import NOMES_ETAPAS, pipeline_status

    for key, pronto in pipeline_status().items():
        nome = NOMES_ETAPAS[key]
        if pronto:
            st.markdown(f"✅ {nome}")
        else:
            st.markdown(f"⚪ {nome} _(pendente)_")


def renderizar_sidebar(modelo, colecao, colecao_sessoes) -> None:
    with st.sidebar:
        st.markdown("## ⚡ Al IAdo PV")
        st.caption("Mestrado UTFPR — agente de pesquisa")
        st.caption("💡 Tema claro/escuro: menu ⋮ → Settings → Theme")

        provedor = st.session_state.get("nome_provedor", "Nenhum")
        if provedor == "Nenhum":
            st.warning("LLM desconectado", icon="⚠️")
        else:
            st.success(f"LLM ativo: {provedor}", icon="🟢")

        st.divider()
        st.markdown("**🤖 Provedor**")
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
        st.markdown("**📚 Conhecimento**")
        c1, c2 = st.columns(2)
        c1.metric("Literatura", colecao.count())
        c2.metric("Sessões", colecao_sessoes.count())

        st.divider()
        st.markdown("**🤖 Pipeline ML**")
        renderizar_pipeline_status()
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
        st.markdown("**📄 PDFs**")
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
                st.success("PDF enviado. O watcher processará automaticamente.")

        st.divider()
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.session_state.caminho_sessao = None
            st.rerun()

        with st.expander("⚙️ Manutenção"):
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

    col_titulo, col_status = st.columns([4, 1])
    with col_titulo:
        st.title("⚡ Al IAdo PV")
        st.caption(
            "Pesquisa, literatura e Machine Learning para falhas CA em inversores "
            "fotovoltaicos — Mestrado UTFPR"
        )
    with col_status:
        if provedor != "Nenhum":
            st.success(f"🟢 {provedor}")
        else:
            st.warning("Conecte um LLM →")

    novidades = [
        str(item)
        for item in relatorio
        if item and "nenhum pendente" not in str(item).lower()
    ]
    if novidades:
        with st.expander("📬 Novidades processadas na inicialização", expanded=False):
            for item in novidades:
                st.write(item)


def renderizar_boas_vindas() -> None:
    from src.conhecimento.agente import _saudacao_pelo_horario

    saudacao = _saudacao_pelo_horario()
    st.info(
        f"**{saudacao}, Rodolfo!** 👋\n\n"
        "Como quer trabalhar agora? Peça em linguagem natural — eu consulto a "
        "literatura, rodo etapas do pipeline, explico métricas, mostro gráficos "
        "ou discuto a dissertação com você."
    )

    st.markdown("##### Exemplos de prompt")
    exemplos = [
        "🔬 Explique os resultados de validação e mostre as curvas ROC.",
        "📈 Rode a análise de Weibull e depois interprete MTTF e B10.",
        "🎯 Quais falhas tiveram menor severidade mínima detectável?",
        "📚 Compare o papel do FMEA com o Autoencoder na metodologia.",
        "♻️ Faça o recálculo do pipeline completo.",
    ]
    for exemplo in exemplos:
        st.markdown(f"- _{exemplo}_")


def stream_resposta(prompt: str, llm):
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        yield chunk.content


def renderizar_imagens(imagens: list[dict]) -> None:
    """
    Renderiza imagens, ignorando paths que não existem mais no disco.
    Cenário comum: o usuário apagou artefatos via 'Resultados e recálculo'
    no sidebar e ainda há mensagens antigas com paths inválidos no histórico.
    """
    if not imagens:
        return

    validas = []
    invalidas = 0
    for img in imagens:
        caminho = img.get("path", "")
        if caminho and Path(caminho).is_file():
            validas.append(img)
        else:
            invalidas += 1

    if not validas:
        if invalidas:
            st.caption(
                f"_({invalidas} imagem(ns) referenciada(s) já não existe(m) no disco — "
                "rode o pipeline novamente para regenerá-las.)_"
            )
        return

    cols = st.columns(min(2, len(validas)))
    for idx, img in enumerate(validas):
        col = cols[idx % len(cols)]
        col.image(img["path"], caption=img.get("caption", ""), use_container_width=True)

    if invalidas:
        st.caption(
            f"_({invalidas} imagem(ns) adicional(is) referenciada(s) não está(ão) "
            "mais no disco.)_"
        )


def renderizar_mensagem(msg: dict) -> None:
    avatar = "🔬" if msg["role"] == "user" else "⚡"
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

    if llm is None:
        return "", []

    with st.spinner("Interpretando o pedido..."):
        decisao = decidir_acao(pergunta, llm)

    if not decisao["usar_ferramenta"]:
        return "", []

    with st.chat_message("assistant", avatar="⚡"):
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


def _filtrar_citacoes(citacoes: dict) -> list[str]:
    """Remove citações vazias/None e deduplica preservando ordem."""
    vistos = []
    for v in (citacoes or {}).values():
        if not v:
            continue
        s = str(v).strip()
        if s and s not in vistos:
            vistos.append(s)
    return vistos


def responder_com_rag(pergunta: str,
                      perfil: str,
                      modelo,
                      colecao,
                      colecao_sessoes) -> str:
    from src.conhecimento.agente import preparar_prompt, resposta_interacao_simples

    # ── Atalho: cumprimento/casual responde local sem RAG ────
    resposta_simples = resposta_interacao_simples(pergunta)
    if resposta_simples:
        with st.chat_message("assistant", avatar="⚡"):
            st.markdown(resposta_simples)
        return resposta_simples

    if st.session_state.llm is None:
        with st.chat_message("assistant", avatar="⚡"):
            resposta = (
                "Consigo consultar status e resultados do pipeline por aqui, "
                "mas para conversar com a literatura ou interpretar perguntas "
                "abertas preciso que você conecte um LLM no painel lateral."
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

    with st.chat_message("assistant", avatar="⚡"):
        try:
            resposta = st.write_stream(stream_resposta(prompt, st.session_state.llm))
            fontes = _filtrar_citacoes(citacoes)
            if fontes:
                st.caption("📚 Fontes consultadas: " + " · ".join(fontes))
                resposta = f"{resposta}\n\n---\n📚 **Fontes:** {' · '.join(fontes)}"
        except Exception as exc:
            erro = str(exc)
            if "413" in erro or "Request too large" in erro:
                st.error(
                    "A solicitação ficou grande demais para o limite do provedor. "
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

    with st.chat_message("user", avatar="🔬"):
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
    st.markdown(_CSS_MINIMO, unsafe_allow_html=True)

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
