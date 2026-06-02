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


# CSS minimo: mantem o menu principal visivel, pois nele fica
# Settings -> Theme para alternar entre claro/escuro nativo do Streamlit.
_CSS_MINIMO = """
<style>
.stDeployButton  { display: none; }
.block-container {
    max-width: min(1680px, calc(100vw - 2.5rem));
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}
[data-testid="stChatMessage"] {
    max-width: min(1320px, 100%);
}
[data-testid="stBottomBlockContainer"],
[data-testid="stChatInput"] {
    max-width: min(1680px, calc(100vw - 2.5rem));
}
[data-testid="stBottomBlockContainer"] {
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}
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
        "anexos_pendentes": [],
        "multimodal": False,
    }
    for chave, valor in defaults.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def renderizar_pipeline_status() -> None:
    """Status do pipeline no sidebar: ready / stale / pending."""
    from src.ml.pipeline import NOMES_ETAPAS, estado_pipeline

    for key, info in estado_pipeline().items():
        nome = NOMES_ETAPAS[key]
        estado = info.get("estado")
        if estado == "ready":
            st.markdown(f"✅ {nome}")
        elif estado == "stale":
            motivos = ", ".join(info.get("motivos", [])) or "algo mudou"
            st.markdown(f"⚠️ {nome} _(stale: {motivos})_")
        else:
            st.markdown(f"⚪ {nome} _(pendente)_")


def renderizar_diagnostico(colecao, colecao_sessoes) -> None:
    """Painel de diagnóstico (13.4): ChromaDB, pipeline, libs opcionais, log."""
    import importlib.util

    try:
        st.caption(f"ChromaDB · literatura: {colecao.count()} · "
                   f"sessões: {colecao_sessoes.count()}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"ChromaDB indisponível: {exc}")

    try:
        from src.ml.pipeline import NOMES_ETAPAS, estado_pipeline

        rot = {"ready": "✅", "stale": "⚠️", "pending": "⬜"}
        for key, info in estado_pipeline().items():
            st.caption(f"{rot.get(info['estado'], '?')} {NOMES_ETAPAS[key]} "
                       f"— {info['estado']}")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"pipeline: {exc}")

    libs = {"prophet": "Prophet", "stable_baselines3": "PPO",
            "Orange": "CN2", "torch": "torch"}
    marcas = []
    for mod, desc in libs.items():
        try:
            ok = importlib.util.find_spec(mod) is not None
        except Exception:  # noqa: BLE001
            ok = False
        marcas.append(f"{'✅' if ok else '⛔'}{desc}")
    st.caption("Opcionais: " + " · ".join(marcas))

    try:
        from src.core.logs import ARQUIVO_LOG

        if ARQUIVO_LOG.exists():
            erros = [ln for ln in ARQUIVO_LOG.read_text(encoding="utf-8").splitlines()
                     if "ERROR" in ln]
            st.caption("Último erro: " + erros[-1][-110:] if erros
                       else "Sem erros registrados no log.")
    except Exception:  # noqa: BLE001
        pass


def renderizar_sidebar(modelo, colecao, colecao_sessoes) -> None:
    with st.sidebar:
        st.markdown("## Al IAdo PV")
        st.caption("Assistente de pesquisa | Mestrado UTFPR")

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
                from src.conhecimento.provedores import eh_multimodal

                llm, nome = inicializar_provedor(escolha)
                st.session_state.llm = llm
                st.session_state.nome_provedor = nome
                st.session_state.multimodal = eh_multimodal(nome)
                st.rerun()
            except Exception as exc:
                st.error(f"Erro ao conectar: {exc}")

        st.divider()
        st.markdown("**Base local**")
        c1, c2 = st.columns(2)
        c1.metric("Literatura", colecao.count())
        c2.metric("Sessões", colecao_sessoes.count())
        st.caption("Literatura, memória e resultados são acessados pelo chat.")

        st.divider()
        st.markdown("**Comandos por prompt**")
        st.caption(
            "Use o chat para rodar pipeline, comparar artigos, recalcular, "
            "apagar artefatos, pedir gráficos ou discutir resultados."
        )

        st.divider()
        st.markdown("**Documentos**")
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
        st.markdown("**Sessão**")
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.session_state.caminho_sessao = None
            st.rerun()

        with st.expander("🔧 Diagnóstico"):
            renderizar_diagnostico(colecao, colecao_sessoes)

        with st.expander("Manutenção avançada"):
            st.caption("Use apenas quando quiser forçar tarefas administrativas.")
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

        st.caption("Tema claro/escuro: menu ⋮ → Settings → Theme")


def renderizar_topo(relatorio: list) -> None:
    provedor = st.session_state.get("nome_provedor", "Nenhum")

    col_titulo, col_status = st.columns([4, 1.1])
    with col_titulo:
        st.markdown("## Al IAdo PV")
        st.caption(
            "Pesquisa aplicada, confiabilidade e Machine Learning para falhas CA "
            "em inversores fotovoltaicos | UTFPR"
        )
    with col_status:
        if provedor != "Nenhum":
            st.success(f"{provedor}")
        else:
            st.warning("Conecte um LLM")

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
    from src.conhecimento.agente import _saudacao_pelo_horario

    saudacao = _saudacao_pelo_horario()
    st.info(
        f"**{saudacao}, Rodolfo.**\n\n"
        "Peça em linguagem natural: posso rodar etapas do pipeline, comparar "
        "experimentos, explicar métricas, mostrar gráficos ou discutir decisões "
        "metodológicas da dissertação."
    )

    st.markdown("##### Exemplos de prompt")
    exemplos = [
        "Explique os resultados de validação e mostre as curvas ROC.",
        "Rode a análise de Weibull e depois interprete MTTF e B10.",
        "Compare os modelos de Sharma e Ibrahim com gráficos.",
        "What does the literature say about LCL filter faults?",
        "Explique en español qué modelo parece más confiable.",
    ]
    for exemplo in exemplos:
        st.markdown(f"- _{exemplo}_")


def stream_resposta(prompt: str, llm):
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        yield chunk.content


# ── Cadência de "digitação" do streaming ────────────────────────────────────
# A resposta é revelada palavra a palavra com uma pequena pausa, em vez de
# "estourar" blocos inteiros de texto (efeito comum com o Groq, que entrega
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


def _grupo_imagem(img: dict) -> str:
    grupo = img.get("group")
    if grupo:
        return str(grupo)
    legenda = str(img.get("caption", "Resultados"))
    return legenda.split(" - ", 1)[0] if " - " in legenda else "Resultados"


def _imagem_larga(img: dict) -> bool:
    tipo = str(img.get("kind", "")).lower()
    legenda = str(img.get("caption", "")).lower()
    return (
        tipo in {"comparacao", "wide"}
        or "comparacao" in legenda
        or "anomalias detectadas" in legenda
        or "curvas" in legenda
        or "heatmap" in legenda
    )


def _dimensoes_imagem(path: str) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def _largura_exibicao_imagem(img: dict) -> int | None:
    largura_px, _altura_px = _dimensoes_imagem(img["path"])
    tipo = str(img.get("kind", "")).lower()
    legenda = str(img.get("caption", "")).lower()

    if tipo in {"comparacao", "wide"} or "comparacao" in legenda:
        limite = 1080
    elif tipo == "matriz" or "matriz" in legenda or "confus" in legenda:
        limite = 620
    elif tipo == "modelo":
        limite = 780
    else:
        limite = 860

    if largura_px:
        return min(largura_px, limite)
    return limite


def _ordem_imagem(img: dict, indice: int) -> tuple:
    try:
        ordem_grupo = int(img.get("group_order", 0) or 0)
    except Exception:
        ordem_grupo = 0
    try:
        ordem = int(img.get("order", indice) or indice)
    except Exception:
        ordem = indice
    return ordem_grupo, ordem, indice


def _renderizar_imagem_unica(img: dict, coluna=None) -> None:
    alvo = coluna if coluna is not None else st
    alvo.image(
        img["path"],
        caption=img.get("caption", ""),
        width=_largura_exibicao_imagem(img),
    )


def _renderizar_lote_regular(lote: list[dict]) -> None:
    if not lote:
        return
    if len(lote) == 1:
        _, centro, _ = st.columns([0.12, 0.76, 0.12], gap="small")
        _renderizar_imagem_unica(lote[0], centro)
        return

    for inicio in range(0, len(lote), 2):
        par = lote[inicio:inicio + 2]
        cols = st.columns(len(par), gap="small")
        for col, img in zip(cols, par):
            _renderizar_imagem_unica(img, col)


def _renderizar_grupo_imagens(imagens: list[dict]) -> None:
    pendentes_regulares: list[dict] = []
    for img in imagens:
        if _imagem_larga(img):
            _renderizar_lote_regular(pendentes_regulares)
            pendentes_regulares = []
            _renderizar_imagem_unica(img)
        else:
            pendentes_regulares.append(img)
    _renderizar_lote_regular(pendentes_regulares)


def renderizar_imagens(imagens: list[dict]) -> None:
    """
    Renderiza imagens, ignorando paths que não existem mais no disco.
    Cenário comum: o usuário apagou ou recalculou artefatos pelo chat e ainda
    há mensagens antigas com paths inválidos no histórico.
    """
    if not imagens:
        return

    validas = []
    invalidas = 0
    for idx, img in enumerate(imagens):
        caminho = img.get("path", "")
        if caminho and Path(caminho).is_file():
            img = dict(img)
            img["_idx"] = idx
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

    validas.sort(key=lambda img: _ordem_imagem(img, int(img.get("_idx", 0))))
    grupos: dict[str, list[dict]] = {}
    for img in validas:
        grupos.setdefault(_grupo_imagem(img), []).append(img)

    mostrar_titulos = len(grupos) > 1
    for grupo, itens in grupos.items():
        if mostrar_titulos:
            st.markdown(f"**{grupo}**")
        _renderizar_grupo_imagens(itens)

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


def responder_com_rag(pergunta: str,
                      perfil: str,
                      modelo,
                      colecao,
                      colecao_sessoes,
                      anexos: list | None = None) -> str:
    from src.conhecimento.agente import (
        deve_consultar_literatura,
        formatar_referencias_markdown,
        montar_conteudo_humano,
        preparar_prompt,
        resposta_interacao_simples,
    )

    # ── Atalho: cumprimento/casual responde local sem RAG ────
    # Nunca atalhar quando ha anexos: o pesquisador quer o arquivo lido.
    if not anexos:
        resposta_simples = resposta_interacao_simples(pergunta)
        if resposta_simples:
            with st.chat_message("assistant", avatar="⚡"):
                st.markdown(resposta_simples)
            return resposta_simples

    if st.session_state.llm is None:
        with st.chat_message("assistant", avatar="⚡"):
            resposta = (
                "Consigo consultar status e resultados do pipeline por aqui, "
                "mas para interpretar perguntas abertas preciso que você "
                "conecte um LLM no painel lateral."
            )
            st.info(resposta)
        return resposta

    historico = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.mensagens
    ]
    consultar_literatura = deve_consultar_literatura(pergunta, colecao)
    mensagem_busca = (
        "Buscando literatura e memória..."
        if consultar_literatura else
        "Buscando memória e contexto do projeto..."
    )

    with st.spinner(mensagem_busca):
        prompt, citacoes = preparar_prompt(
            pergunta=pergunta,
            perfil=perfil,
            modelo_embeddings=modelo,
            colecao=colecao,
            historico=historico,
            colecao_sessoes=colecao_sessoes,
            nome_provedor=st.session_state.get("nome_provedor", ""),
            anexos=anexos,
        )

    # Quando ha imagem anexada E o provedor e multimodal, o conteudo vira uma
    # lista (texto + image_url); caso contrario, segue como string.
    conteudo_humano = montar_conteudo_humano(
        prompt, anexos, st.session_state.get("multimodal", False)
    )

    with st.chat_message("assistant", avatar="⚡"):
        try:
            refs_md = formatar_referencias_markdown(citacoes)
            placeholder = st.empty()
            resposta = stream_resposta_limpa(
                conteudo_humano,
                st.session_state.llm,
                placeholder,
                refs_md,
            )
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

    # Com anexos, ir direto ao RAG (que le o arquivo). Pular o roteador de
    # ferramentas evita misrotear "o que tem nesse arquivo?" para o pipeline ML.
    if anexos:
        resposta = responder_com_rag(
            pergunta, perfil, modelo, colecao, colecao_sessoes, anexos=anexos
        )
        imagens = []
    else:
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
            except Exception:
                pass

        # Só anexou arquivo, sem texto: damos um pedido padrão de leitura.
        if not (texto or "").strip() and anexos_bytes:
            texto = "Leia o(s) arquivo(s) anexado(s) e me explique o conteúdo."

        if (texto or "").strip() or anexos_bytes:
            st.session_state.pergunta_pendente = texto or ""
            st.session_state.anexos_pendentes = anexos_bytes
            st.rerun()
