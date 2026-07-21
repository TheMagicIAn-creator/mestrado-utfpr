"""
streamlit_app.py - Al IAdo PV
Interface conversacional do agente.

Resultados e execucoes do pipeline de ML aparecem pelo chat, conforme
solicitacao em prompt. A interface usa componentes nativos do Streamlit
para preservar a estetica original — sem overrides de CSS pesados.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st
from langchain_core.messages import HumanMessage

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.core.conversa_export import (
    montar_transcricao,
    nome_arquivo_conversa,
    quer_exportar_conversa,
)

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
.stDeployButton,
[data-testid="stAppDeployButton"] { display: none; }
.block-container {
    max-width: min(1680px, calc(100vw - 2.5rem));
    padding-top: 4rem;
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
    .block-container {
        padding-top: 3.5rem;
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
            except Exception:
                pass

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
            except Exception:
                pass
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
                except Exception:
                    pass
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


def renderizar_pipeline_status() -> None:
    """Status do pipeline no sidebar: ready / stale / pending."""
    from src.ml.pipeline import (
        NOMES_ETAPAS,
        capacidade_recalculo_pipeline,
        estado_pipeline,
        estado_resultados_publicados,
    )

    if not capacidade_recalculo_pipeline()["disponivel"]:
        for key, info in estado_resultados_publicados().items():
            marcador = "✅" if info["disponivel"] else "⚪"
            st.markdown(f"{marcador} {NOMES_ETAPAS[key]} _(publicado)_")
        return

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


def renderizar_diagnostico(colecao, colecao_sessoes, colecao_obsidian) -> None:
    """Painel de diagnóstico (13.4): ChromaDB, pipeline, libs opcionais, log."""
    import importlib.util

    try:
        from src.conhecimento.obsidian import contar_notas_indexadas

        st.caption(
            f"ChromaDB · literatura: {colecao.count()} · "
            f"sessões: {colecao_sessoes.count()} · "
            f"Obsidian: {contar_notas_indexadas(colecao_obsidian)} notas / "
            f"{colecao_obsidian.count()} chunks"
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"ChromaDB indisponível: {exc}")

    try:
        from src.ml.pipeline import (
            NOMES_ETAPAS,
            capacidade_recalculo_pipeline,
            estado_pipeline,
            estado_resultados_publicados,
        )

        if capacidade_recalculo_pipeline()["disponivel"]:
            rot = {"ready": "✅", "stale": "⚠️", "pending": "⬜"}
            for key, info in estado_pipeline().items():
                st.caption(f"{rot.get(info['estado'], '?')} {NOMES_ETAPAS[key]} "
                           f"— {info['estado']}")
        else:
            st.caption("Modo consulta: cálculo pesado indisponível neste servidor.")
            for key, info in estado_resultados_publicados().items():
                marcador = "✅" if info["disponivel"] else "⬜"
                st.caption(f"{marcador} {NOMES_ETAPAS[key]} — publicado")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"pipeline: {exc}")

    libs = {"torch": "torch"}
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


def _carregar_metadados_pendentes() -> dict:
    """Itens não resolvidos de metadados_pendentes.json ({} se vazio/ausente)."""
    caminho = RAIZ_PROJETO / "metadados_pendentes.json"
    if not caminho.exists():
        return {}
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        nome: info
        for nome, info in dados.items()
        if isinstance(info, dict) and not info.get("resolvido")
    }


def renderizar_sidebar(modelo, colecao, colecao_sessoes, colecao_obsidian) -> None:
    with st.sidebar:
        st.markdown("## Al IAdo PV")
        st.caption("Assistente de pesquisa | Mestrado UTFPR")
        from src.core.config import MARCADOR_BUILD

        st.caption(f"🏷️ build: {MARCADOR_BUILD}")

        equipe = st.session_state.get("equipe")
        if equipe is None:
            st.warning("Equipe de IA desconectada")
            erro = st.session_state.get("erro_equipe")
            if erro:
                st.caption(erro)
            if st.button("Ativar equipe", width="stretch", type="primary"):
                conectar_equipe(forcar=True)
                st.rerun()
        else:
            st.success("Equipe de IA ativa")
            st.caption("Gemini Flash: conversa e síntese (padrão estável)")
            st.caption("Gemini Flash: auditoria de evidências e memória")

        st.divider()
        st.markdown("**Base de conhecimento**")
        c1, c2 = st.columns(2)
        c1.metric("Literatura", colecao.count())
        c2.metric("Sessões", colecao_sessoes.count())
        memorias = equipe.memoria.contar() if equipe is not None else 0
        from src.conhecimento.obsidian import contar_notas_indexadas

        notas_obsidian = contar_notas_indexadas(colecao_obsidian)
        st.caption(
            f"Vault Obsidian: {notas_obsidian} notas pesquisáveis · "
            f"memórias validadas: {memorias}. Literatura, memória e resultados "
            "são acessados pelo chat."
        )

        # Fallback: o caminho normal da nuvem restaura o snapshot portátil no
        # carregamento. O botão só aparece se o snapshot estiver ausente/inválido.
        if colecao.count() == 0:
            st.caption("⚠️ Literatura não indexada (base vazia).")
            if st.button(
                "Indexar literatura",
                icon=":material/sync:",
                width="stretch",
                help="Reconstrói a base a partir dos PDFs de literatura/. "
                     "Use apenas se a restauração automática não estiver disponível.",
            ):
                try:
                    from src.conhecimento.indexador import indexar_literatura
                    with st.spinner("Indexando literatura… alguns minutos."):
                        resumo = indexar_literatura(modelo=modelo)
                    if resumo["erros"]:
                        st.warning(
                            f"Indexação concluída com {resumo['erros']} erro(s)."
                        )
                    st.success("Literatura indexada! Atualizando…")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Falha ao indexar: {exc}")

        st.divider()
        st.markdown("**Comandos por prompt**")
        from src.ml.pipeline import capacidade_recalculo_pipeline

        if capacidade_recalculo_pipeline()["disponivel"]:
            st.caption(
                "Use o chat para rodar pipeline, comparar artigos, recalcular, "
                "apagar artefatos, pedir gráficos ou discutir resultados."
            )
        else:
            st.caption(
                "Modo consulta: peça resultados, tabelas, gráficos, comparações "
                "e interpretação. O recálculo permanece no PC com os datasets."
            )

        st.divider()
        st.markdown("**Documentos**")
        arquivo_pdf = st.file_uploader(
            "Adicionar PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if arquivo_pdf is not None:
            if st.button("Enviar para processamento", width="stretch"):
                # Sanitiza o nome vindo do navegador (anti path-traversal) e
                # confirma que o destino fica DENTRO de novos_pdfs/.
                from src.core.seguranca import (
                    caminho_dentro_do_projeto, nome_arquivo_seguro,
                )

                pasta_novos = RAIZ_PROJETO / "novos_pdfs"
                pasta_novos.mkdir(exist_ok=True)
                nome = nome_arquivo_seguro(arquivo_pdf.name, padrao="upload.pdf")
                destino = caminho_dentro_do_projeto(nome, base=pasta_novos)
                destino.write_bytes(arquivo_pdf.getbuffer())
                st.success("PDF enviado. O watcher processará automaticamente.")

        st.divider()
        st.markdown("**Sessão**")
        if st.button("Limpar conversa", width="stretch"):
            st.session_state.mensagens = []
            st.session_state.caminho_sessao = None
            st.rerun()

        with st.expander("🔧 Diagnóstico"):
            renderizar_diagnostico(colecao, colecao_sessoes, colecao_obsidian)

        metadados_pendentes = _carregar_metadados_pendentes()
        if metadados_pendentes:
            st.caption(
                f"⚠️ {len(metadados_pendentes)} PDF(s) com metadados "
                "pendentes — detalhes em Manutenção avançada."
            )

        with st.expander("Manutenção avançada"):
            st.caption("Use apenas quando quiser forçar tarefas administrativas.")
            if metadados_pendentes:
                st.warning(
                    f"{len(metadados_pendentes)} PDF(s) com autor/ano "
                    "incompletos. Confira na fonte antes de citar."
                )
                for nome, info in metadados_pendentes.items():
                    st.caption(
                        f"• {nome} — {info.get('autor_atual') or 'autor ?'} "
                        f"({info.get('ano_atual') or '????'}) | "
                        f"registrado em {info.get('registrado', '?')}"
                    )
            if st.button("Consolidar memória", width="stretch"):
                try:
                    from src.conhecimento.consolidar_memoria import consolidar

                    ok = consolidar(forcar=True)
                    st.success("Memória consolidada." if ok else "Nada a consolidar.")
                except Exception as exc:
                    st.error(f"Erro: {exc}")

            if st.button("Corrigir metadados ruins", width="stretch"):
                try:
                    from src.orquestrador import reprocessar_metadados_ruins

                    st.info(reprocessar_metadados_ruins())
                except Exception as exc:
                    st.error(f"Erro: {exc}")

        st.caption("Tema claro/escuro: menu ⋮ → Settings → Theme")


def renderizar_topo(relatorio: list) -> None:
    equipe_ativa = st.session_state.get("equipe") is not None

    col_titulo, col_status = st.columns([4, 1.1])
    with col_titulo:
        st.markdown("## Al IAdo PV")
        st.caption(
            "Pesquisa aplicada, confiabilidade e Machine Learning para falhas CA "
            "em inversores fotovoltaicos | UTFPR"
        )
    with col_status:
        if equipe_ativa:
            st.success("Equipe Gemini")
        else:
            st.warning("Ative a equipe")

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
    from src.ml.pipeline import capacidade_recalculo_pipeline

    saudacao = _saudacao_pelo_horario()
    calculo_local = capacidade_recalculo_pipeline()["disponivel"]
    capacidade = (
        "posso rodar etapas do pipeline, comparar experimentos"
        if calculo_local else
        "posso consultar os resultados publicados e comparar experimentos"
    )
    st.info(
        f"**{saudacao}, Rodolfo.**\n\n"
        f"Peça em linguagem natural: {capacidade}, explicar métricas, "
        "mostrar gráficos ou discutir decisões "
        "metodológicas da dissertação."
    )

    st.markdown("##### Exemplos de prompt")
    acao_weibull = (
        "Rode a análise de Weibull e depois interprete MTTF e B10."
        if calculo_local else
        "Interprete os resultados publicados de Weibull, MTTF e B10."
    )
    exemplos = [
        "Explique os resultados de validação e mostre as curvas ROC.",
        acao_weibull,
        "Compare os experimentos de anomalia por AUC.",
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


# Exibição PROPORCIONAL: todos os gráficos são gerados a DPI fixo
# (src/ml/estilo_graficos.DPI), então largura_px/DPI = polegadas físicas.
# Cada polegada vira um nº fixo de pixels na tela — fontes e elementos
# aparecem do MESMO tamanho em todos os gráficos, independente do tipo.
_DPI_GERACAO = 150        # deve casar com src.ml.estilo_graficos.DPI
_PX_POR_POLEGADA = 72     # escala de exibição (12 pol → 864 px)
_TETO_EXIBICAO = 1080     # nunca estoura a largura útil do chat
_LARGURA_PAREAVEL = 560   # só exibe lado a lado o que cabe em meia coluna


def _dimensoes_imagem(path: str) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def _polegadas_imagem(img: dict) -> float | None:
    largura_px, _ = _dimensoes_imagem(img["path"])
    if largura_px:
        return largura_px / _DPI_GERACAO
    return None


def _largura_exibicao_imagem(img: dict) -> int:
    pol = _polegadas_imagem(img)
    if pol is None:
        return 860  # sem PIL/arquivo: largura neutra
    return min(_TETO_EXIBICAO, round(pol * _PX_POR_POLEGADA))


def _imagem_larga(img: dict) -> bool:
    """Painéis (>= 13 pol de largura física) sempre sozinhos, em linha cheia."""
    pol = _polegadas_imagem(img)
    if pol is not None:
        return pol >= 13
    # fallback (imagem ilegível): heurística antiga por legenda
    tipo = str(img.get("kind", "")).lower()
    legenda = str(img.get("caption", "")).lower()
    return (
        tipo in {"comparacao", "wide"}
        or "comparacao" in legenda
        or "anomalias detectadas" in legenda
        or "curvas" in legenda
        or "heatmap" in legenda
    )


def _ordem_imagem(img: dict, indice: int) -> tuple:
    try:
        valor_grupo = img.get("group_order", 0)
        ordem_grupo = int(0 if valor_grupo is None else valor_grupo)
    except Exception:
        ordem_grupo = 0
    try:
        valor_ordem = img.get("order", indice)
        ordem = int(indice if valor_ordem is None else valor_ordem)
    except Exception:
        ordem = indice
    return ordem_grupo, ordem, indice


# Contador monotônico p/ chaves únicas dos download_button (Streamlit exige
# key única por widget; monotônico garante unicidade dentro e entre reruns).
_DL_KEY = [0]


def _botao_download(img: dict, alvo=None, *, compacto: bool = False) -> None:
    """Botão de download da figura (PNG). Não renderiza a imagem, só o botão."""
    destino = alvo if alvo is not None else st
    p = Path(img["path"])
    if not p.is_file():
        return
    try:
        dados = p.read_bytes()
    except OSError:
        return
    _DL_KEY[0] += 1
    legenda = img.get("caption") or p.name
    destino.download_button(
        label="Baixar PNG" if compacto else f"Baixar — {legenda}",
        data=dados,
        file_name=p.name,
        mime="image/png",
        key=f"dl_{_DL_KEY[0]}",
        icon=":material/download:",
        help=f"Salvar {p.name}",
        on_click="ignore",
        width="stretch",
    )


def _botao_download_texto(texto: str, nome: str, alvo=None) -> None:
    """Botão de download de um texto puro (ex.: transcrito da conversa em .txt)."""
    destino = alvo if alvo is not None else st
    _DL_KEY[0] += 1
    destino.download_button(
        label=f"Baixar {nome}",
        data=(texto or "").encode("utf-8"),
        file_name=nome,
        mime="text/plain",
        key=f"dl_txt_{_DL_KEY[0]}",
        icon=":material/download:",
        on_click="ignore",
        width="stretch",
    )


def _controles_antevisao(img: dict, alvo=None) -> None:
    """Antevisão sob demanda: a figura não ocupa o fluxo normal do chat."""
    destino = alvo if alvo is not None else st
    p = Path(img["path"])
    if not p.is_file():
        return

    legenda = img.get("caption") or p.name
    destino.markdown(f"**{legenda}**")
    col_ver, col_baixar = destino.columns(2, gap="small")
    _DL_KEY[0] += 1
    with col_ver.popover(
        "Visualizar",
        icon=":material/visibility:",
        help="Abrir antevisão responsiva sem baixar o arquivo",
        width="stretch",
        key=f"preview_{_DL_KEY[0]}",
    ):
        st.image(
            str(p),
            caption=legenda,
            width="stretch",
        )
        largura, altura = _dimensoes_imagem(str(p))
        tamanho_kb = p.stat().st_size / 1024
        dimensoes = f"{largura} × {altura} px" if largura and altura else "dimensões indisponíveis"
        st.caption(f"{dimensoes} · {tamanho_kb:.0f} KB · PNG")
    _botao_download(img, col_baixar, compacto=True)


def _renderizar_imagem_unica(img: dict, coluna=None) -> None:
    alvo = coluna if coluna is not None else st
    # width="stretch": a figura se ajusta à largura da tela/coluna — nunca
    # estoura nem fica minúscula (substitui a largura fixa em px).
    alvo.image(
        img["path"],
        caption=img.get("caption", ""),
        width="stretch",
    )
    _botao_download(img, alvo)


def _renderizar_lote_regular(lote: list[dict]) -> None:
    """
    Exibe imagens não-panorâmicas. Pareia lado a lado APENAS quando as duas
    cabem em meia coluna (largura de exibição <= _LARGURA_PAREAVEL) — antes,
    gráficos de 12 pol eram espremidos em colunas de ~430 px e o tamanho
    final dependia da paridade do lote.
    """
    fila = list(lote)
    while fila:
        img = fila.pop(0)
        cabe_par = (
            _largura_exibicao_imagem(img) <= _LARGURA_PAREAVEL
            and fila
            and _largura_exibicao_imagem(fila[0]) <= _LARGURA_PAREAVEL
        )
        if cabe_par:
            par = [img, fila.pop(0)]
            cols = st.columns(2, gap="small")
            for col, item in zip(cols, par):
                _renderizar_imagem_unica(item, col)
        else:
            _renderizar_imagem_unica(img)


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

    # inline=True → renderiza na tela (com botão de download embaixo);
    # inline=False → só botão de download (não ocupa a tela com a figura).
    inline = [img for img in validas if img.get("inline", True)]
    download_only = [img for img in validas if not img.get("inline", True)]

    grupos: dict[str, list[dict]] = {}
    for img in inline:
        grupos.setdefault(_grupo_imagem(img), []).append(img)

    mostrar_titulos = len(grupos) > 1
    for grupo, itens in grupos.items():
        if mostrar_titulos:
            st.markdown(f"**{grupo}**")
        _renderizar_grupo_imagens(itens)

    if download_only:
        st.caption(
            "Gráficos disponíveis. Abra a antevisão para inspecionar antes de baixar."
        )
        for inicio in range(0, len(download_only), 2):
            par = download_only[inicio:inicio + 2]
            cols = st.columns(len(par), gap="small")
            for col, img in zip(cols, par):
                _controles_antevisao(img, col)

    if invalidas:
        st.caption(
            f"_({invalidas} imagem(ns) adicional(is) referenciada(s) não está(ão) "
            "mais no disco.)_"
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


def salvar_sessao(pergunta: str, resposta: str, imagens: list[dict], n: int, modelo_embeddings) -> None:
    from src.conhecimento.indexador import indexar_sessao
    from src.core.config import PASTA_CHROMADB

    pasta_sessoes = RAIZ_PROJETO / "notas" / "sessoes"
    pasta_sessoes.mkdir(parents=True, exist_ok=True)

    if st.session_state.caminho_sessao is None:
        agora = agora_local()
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
                      colecao_obsidian,
                      indice_lexical=None,
                      anexos: list | None = None) -> str:
    from src.conhecimento.agente import (
        deve_consultar_literatura,
        formatar_referencias_markdown,
        montar_conteudo_humano,
        preparar_prompt,
        resposta_interacao_simples,
    )

    # Consultas simples de primeiro/último registro são resolvidas pela ordem
    # dos metadados. Isso evita que o LLM troque cronologia por similaridade.
    try:
        from src.conhecimento.obsidian import responder_consulta_cronologica

        resposta_cronologica = responder_consulta_cronologica(
            colecao_obsidian, pergunta
        )
    except Exception:
        resposta_cronologica = None
    if resposta_cronologica:
        with st.chat_message("assistant", avatar="⚡"):
            st.markdown(resposta_cronologica)
        return resposta_cronologica

    # ── Atalho: cumprimento/casual responde local sem RAG/LLM ────
    # Vale MESMO com o LLM conectado: um "olá" não deve acionar o modelo pesado
    # (lento e sujeito a 503). Nunca atalha com anexos: o pesquisador quer o
    # arquivo lido.
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
        # Qualquer Markdown novo ou editado no vault vale no próximo turno.
        # A sincronização é incremental e não recalcula embeddings sem mudanças.
        try:
            from src.conhecimento.obsidian import sincronizar_obsidian

            sincronizar_obsidian(colecao_obsidian, modelo)
        except Exception:
            pass
        prompt, citacoes = preparar_prompt(
            pergunta=pergunta,
            perfil=perfil,
            modelo_embeddings=modelo,
            colecao=colecao,
            historico=historico,
            colecao_sessoes=colecao_sessoes,
            nome_provedor=st.session_state.get("nome_provedor", ""),
            anexos=anexos,
            indice_lexical=indice_lexical,
            colecao_obsidian=colecao_obsidian,
        )

    auditoria = None
    auditor = st.session_state.get("auditor")
    if consultar_literatura and auditor is not None and citacoes:
        with st.spinner("Auditando a cobertura das evidencias..."):
            auditoria = auditor.auditar_evidencias(pergunta, citacoes)
        from src.conhecimento.multiagente import filtrar_citacoes_auditadas

        citacoes = filtrar_citacoes_auditadas(citacoes, auditoria)

    coordenador = st.session_state.llm
    if hasattr(coordenador, "contextualizar_prompt"):
        prompt = coordenador.contextualizar_prompt(
            prompt,
            pergunta,
            auditoria,
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
            # Trava estrutural: sinaliza citações sem lastro (normas fora do
            # rodapé, páginas sem fonte recuperada) que o prompt sozinho não
            # segura. O aviso vira parte da resposta (exibida e exportada).
            if consultar_literatura:
                from src.core.citacao_guarda import alerta_citacao_infundada

                aviso = alerta_citacao_infundada(resposta, citacoes)
                if aviso:
                    placeholder.markdown(resposta + aviso)
                    resposta = resposta + aviso
        except Exception as exc:
            erro = str(exc)
            erro_baixo = erro.lower()
            if "413" in erro or "Request too large" in erro:
                st.error(
                    "A solicitação ficou grande demais para o limite do provedor. "
                    "Tente pedir uma resposta mais focada."
                )
                resposta = f"[Erro: {exc}]"
            elif "503" in erro or "unavailable" in erro_baixo or "high demand" in erro_baixo:
                msg = (
                    "⏳ Os modelos do Gemini estão com alta demanda no momento "
                    "(503). Já tentei repetir e usar um modelo alternativo sem "
                    "sucesso — costuma ser passageiro. Reenvie a pergunta em "
                    "alguns segundos."
                )
                st.warning(msg)
                resposta = msg
            elif "429" in erro or "resource_exhausted" in erro_baixo:
                st.warning("Limite de taxa da API atingido. Aguarde alguns instantes e reenvie.")
                resposta = "[Limite de taxa atingido — reenvie em instantes.]"
            else:
                st.error(f"Erro: {exc}")
                resposta = f"[Erro: {exc}]"
    return resposta


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

    # Atalho: exportar a conversa em .txt. Intercepta ANTES do LLM — o modelo
    # não cria arquivos (só alucinaria "gerei"); aqui montamos o transcrito real
    # e oferecemos o download, que persiste via renderizar_mensagem.
    if not anexos and quer_exportar_conversa(pergunta):
        carimbo = f"{agora_local():%Y-%m-%d_%H-%M}"
        transcricao = montar_transcricao(
            st.session_state.mensagens,
            exportado_em=f"{agora_local():%d/%m/%Y às %H:%M} (America/Sao_Paulo)",
        )
        nome_arq = nome_arquivo_conversa(carimbo)
        trocas = sum(1 for m in st.session_state.mensagens if m.get("role") == "user")
        resposta = (
            f"📄 Preparei o histórico completo desta conversa "
            f"({trocas} {'troca' if trocas == 1 else 'trocas'}) em **{nome_arq}**. "
            "Clique no botão abaixo para baixar — o texto traz cada mensagem na íntegra."
        )
        with st.chat_message("assistant", avatar="⚡"):
            st.markdown(resposta)
            _botao_download_texto(transcricao, nome_arq)
        st.session_state.mensagens.append({
            "role": "user", "content": conteudo_usuario, "imagens": [],
        })
        st.session_state.mensagens.append({
            "role": "assistant", "content": resposta, "imagens": [],
            "export_txt": {"data": transcricao, "file_name": nome_arq},
        })
        salvar_sessao(
            conteudo_usuario, resposta, [],
            len(st.session_state.mensagens) // 2, modelo,
        )
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

    renderizar_sidebar(modelo, colecao, colecao_sessoes, colecao_obsidian)
    renderizar_topo(relatorio)

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
            except Exception:
                pass

        # Só anexou arquivo, sem texto: damos um pedido padrão de leitura.
        if not (texto or "").strip() and anexos_bytes:
            texto = "Leia o(s) arquivo(s) anexado(s) e me explique o conteúdo."

        if (texto or "").strip() or anexos_bytes:
            st.session_state.pergunta_pendente = texto or ""
            st.session_state.anexos_pendentes = anexos_bytes
            st.rerun()
