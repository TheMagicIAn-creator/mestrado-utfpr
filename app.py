"""
app.py — Al IAdo PV
Interface web com Streamlit e orquestração de backend.

Como executar:
  streamlit run app.py

Autor: Rodolfo Torres (UTFPR)
"""

from watcher import iniciar_em_background
import sys
import os
from pathlib import Path
from datetime import datetime

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

RAIZ = Path(__file__).parent
PASTA_AE = RAIZ / "resultados" / "autoencoder"

# ── Mapa de artefatos por etapa (ordem = dependências) ───────
ORDEM_ETAPAS_ML = ["features_ca", "autoencoder", "injecao_falhas",
                   "validacao", "rul_weibull"]

ARTEFATOS_ML = {
    "features_ca": [
        "dados/processados/features_paderborn.parquet",
        "dados/processados/features_paderborn_stats.csv",
    ],
    "autoencoder": [
        "resultados/autoencoder/modelo_autoencoder.pt",
        "resultados/autoencoder/scaler.pkl",
        "resultados/autoencoder/limiar.json",
        "resultados/autoencoder/curva_treino.png",
        "resultados/autoencoder/distribuicao_erro.png",
        "resultados/autoencoder/erro_temporal.png",
    ],
    "injecao_falhas": [
        "resultados/autoencoder/injecao_falhas_resultados.png",
        "resultados/autoencoder/injecao_falhas_comparacao.png",
        "resultados/autoencoder/injecao_falhas_report.json",
    ],
    "validacao": [
        "resultados/autoencoder/validacao_roc.png",
        "resultados/autoencoder/validacao_matriz.png",
        "resultados/autoencoder/validacao_metricas.png",
        "resultados/autoencoder/validacao_tabela.csv",
        "resultados/autoencoder/validacao_report.json",
    ],
    "rul_weibull": [
        "resultados/autoencoder/weibull_ttf.png",
        "resultados/autoencoder/weibull_confiabilidade.png",
        "resultados/autoencoder/weibull_rul.png",
        "resultados/autoencoder/weibull_results.json",
    ],
}

NOMES_ETAPAS = {
    "features_ca"    : "Features CA",
    "autoencoder"    : "Autoencoder",
    "injecao_falhas" : "Injeção de Falhas",
    "validacao"      : "Validação Formal",
    "rul_weibull"    : "RUL / Weibull",
}


def _limpar_artefatos(etapa_inicial: str):
    """Apaga artefatos da etapa e de todas as etapas que dependem dela."""
    idx = ORDEM_ETAPAS_ML.index(etapa_inicial)
    for etapa in ORDEM_ETAPAS_ML[idx:]:
        for rel in ARTEFATOS_ML[etapa]:
            arq = RAIZ / rel
            if arq.exists():
                arq.unlink()


def _regenerar(etapa_inicial: str, status=None) -> list:
    """
    Apaga artefatos da etapa (e downstream) e re-executa DIRETAMENTE
    os módulos de ML em ordem estrita. Para no primeiro erro.
    """
    _limpar_artefatos(etapa_inicial)

    from src.ml.features_ca    import executar_features_ca
    from src.ml.autoencoder    import executar_autoencoder
    from src.ml.injecao_falhas import executar_injecao_falhas
    from src.ml.validacao      import executar_validacao
    from src.ml.rul_weibull    import executar_rul_weibull

    funcs = {
        "features_ca"    : ("Features CA",       executar_features_ca),
        "autoencoder"    : ("Autoencoder",       executar_autoencoder),
        "injecao_falhas" : ("Injeção de Falhas", executar_injecao_falhas),
        "validacao"      : ("Validação Formal",  executar_validacao),
        "rul_weibull"    : ("RUL / Weibull",     executar_rul_weibull),
    }

    idx        = ORDEM_ETAPAS_ML.index(etapa_inicial)
    resultados = []

    for etapa in ORDEM_ETAPAS_ML[idx:]:
        nome, func = funcs[etapa]
        if status:
            status.write(f"⏳ Executando: **{nome}**...")
        try:
            sucesso = func()
            if sucesso:
                msg = f"✅ {nome}: regenerado com sucesso"
                resultados.append(msg)
                if status:
                    status.write(msg)
            else:
                msg = f"❌ {nome}: falhou (retornou False)"
                resultados.append(msg)
                if status:
                    status.write(msg)
                break  # downstream depende desta etapa
        except Exception as e:
            msg = f"❌ {nome}: erro — {e}"
            resultados.append(msg)
            if status:
                status.write(msg)
            break

    return resultados
# ============================================================
# CARREGAMENTO DOS COMPONENTES PESADOS (cache)
# ============================================================

@st.cache_resource
def carregar_base():
    """
    Carrega embeddings e ChromaDB uma única vez.
    Executa também o orquestrador de backend.
    """
    from sentence_transformers import SentenceTransformer
    import chromadb
    from src.conhecimento.agente import carregar_perfil
    from src.core.config import (
        MODELO_EMBEDDINGS,
        PASTA_CHROMADB,
        NOME_COLECAO,
        NOME_COLECAO_SESSOES
    )

    with st.spinner("🔄 Carregando modelo de embeddings..."):
        modelo = SentenceTransformer(MODELO_EMBEDDINGS)
        try:
            iniciar_em_background(modelo)
        except Exception:
            pass

    relatorio_orquestrador = []
    with st.spinner("⚙️ Verificando estado do projeto..."):
        try:
            from src.orquestrador import executar_pipeline
            relatorio_orquestrador = executar_pipeline(modelo)
            print("=" * 60)
            print("  ORQUESTRADOR — RELATÓRIO DE INICIALIZAÇÃO")
            print("=" * 60)
            for linha in relatorio_orquestrador:
                print(f"  {linha}")
            print("=" * 60)
        except Exception as e:
            print(f"[Orquestrador] Erro: {e}")

    with st.spinner("🗄️ Conectando ao ChromaDB..."):
        client          = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao         = client.get_or_create_collection(name=NOME_COLECAO)
        colecao_sessoes = client.get_or_create_collection(name=NOME_COLECAO_SESSOES)

    perfil = carregar_perfil()
    return perfil, modelo, colecao, colecao_sessoes, relatorio_orquestrador


# ============================================================
# FEEDBACK DO ORQUESTRADOR
# ============================================================

def mostrar_novidades_orquestrador(relatorio: list):
    termos_inertes = [
        "nenhum", "sem acúmulo", "já realizada", "já extraídas",
        "já treinado", "já calculado", "já realizado", "aguardando",
        "pendente", "já gerada", "nenhum gatilho"
    ]

    def para_texto(item):
        """Converte item do relatório (str ou dict) em texto legível."""
        if isinstance(item, dict):
            if item.get("executou"):
                return f"Memória consolidada ({item.get('motivo', '')})"
            elif item.get("erro"):
                return f"Erro na consolidação: {item['erro']}"
            else:
                return f"Consolidação: {item.get('motivo', 'sem ação')}"
        return str(item) if item else ""

    linhas_texto = [para_texto(item) for item in relatorio]

    novidades = [
        linha for linha in linhas_texto
        if linha and not any(t in linha.lower() for t in termos_inertes)
    ]

    if novidades:
        with st.expander("⚙️ O sistema processou novidades", expanded=True):
            for linha in novidades:
                if "erro" in linha.lower() or "⚠" in linha:
                    st.warning(f"⚠️ {linha}")
                else:
                    st.success(f"✅ {linha}")

# ============================================================
# STREAMING
# ============================================================

def stream_resposta(prompt: str, llm):
    mensagens = [HumanMessage(content=prompt)]
    for chunk in llm.stream(mensagens):
        yield chunk.content


# ============================================================
# SIDEBAR
# ============================================================

def _processar_upload(arquivo_pdf, modelo_embeddings):
    pasta_entrada = RAIZ / "novos_pdfs"
    pasta_entrada.mkdir(exist_ok=True)
    caminho_destino = pasta_entrada / arquivo_pdf.name
    with open(caminho_destino, "wb") as f:
        f.write(arquivo_pdf.getbuffer())
    st.success(
        f"✅ **PDF enviado!**\n\n"
        f"**Arquivo:** `{arquivo_pdf.name}`\n\n"
        f"⚙️ renomear → classificar → indexar → nota Obsidian"
    )

def renderizar_sidebar(perfil, modelo, colecao, colecao_sessoes):
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/solar-panel.png", width=60)
        st.title("Al IAdo PV ⚡")
        st.caption("Assistente de Mestrado — UTFPR")
        st.divider()

        st.subheader("🤖 Provedor de LLM")
        from src.conhecimento.provedores import PROVEDORES
        opcoes = {
            f"{info['emoji']} {info['nome']}": chave
            for chave, info in PROVEDORES.items()
        }
        escolha_label = st.selectbox(
            "Selecione o modelo:", options=list(opcoes.keys()),
            index=0, key="provedor_select"
        )
        escolha = opcoes[escolha_label]
        info    = PROVEDORES[escolha]
        st.caption(f"Limite: {info['limite']}")

        if st.button("🔄 Conectar provedor", use_container_width=True):
            try:
                from src.conhecimento.provedores import inicializar_provedor
                with st.spinner(f"Conectando ao {info['nome']}..."):
                    llm, nome = inicializar_provedor(escolha)
                st.session_state.llm          = llm
                st.session_state.nome_provedor = nome
                st.success(f"✅ {nome} conectado!")
            except Exception as e:
                st.error(f"❌ {e}")

        st.divider()
        st.subheader("📊 Base de Conhecimento")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Literatura", f"{colecao.count()}", "chunks")
        with col2:
            st.metric("Sessões", f"{colecao_sessoes.count()}", "chunks")

        st.divider()
        st.subheader("⚙️ Ações")
        if st.button("📋 Listar artigos", use_container_width=True):
            st.session_state.mostrar_artigos = True
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.rerun()

        st.divider()
        st.subheader("📄 Adicionar PDF")
        st.caption("Tema detectado automaticamente")
        arquivo_pdf = st.file_uploader(
            "Selecione o PDF:", type=["pdf"], key="pdf_uploader"
        )
        if arquivo_pdf is not None:
            if st.button("📥 Processar PDF", use_container_width=True):
                _processar_upload(arquivo_pdf, modelo)

        nome_ativo = st.session_state.get("nome_provedor", "Nenhum")
        st.caption(f"Provedor ativo: **{nome_ativo}**")


# ============================================================
# SALVAR SESSÃO
# ============================================================

def salvar_sessao_streamlit(pergunta: str, resposta: str,
                             n: int, modelo_embeddings):
    from src.core.config import PASTA_CHROMADB
    pasta_sessoes = RAIZ / "notas" / "sessoes"
    pasta_sessoes.mkdir(parents=True, exist_ok=True)

    if st.session_state.caminho_sessao is None:
        agora        = datetime.now()
        nome_arquivo = agora.strftime("%Y-%m-%d_%H-%M") + "_sessao_web.md"
        caminho      = pasta_sessoes / nome_arquivo
        data_fmt     = agora.strftime("%d/%m/%Y às %H:%M")
        cabecalho    = (f"---\ndata: {agora.strftime('%Y-%m-%d')}\n"
                        f"hora: {agora.strftime('%H:%M')}\ntipo: sessao-web\n"
                        f"tags: [al-iado-pv, sessao, streamlit, mestrado]\n"
                        f"---\n\n# Sessão Web Al IAdo PV — {data_fmt}\n\n")
        caminho.write_text(cabecalho, encoding="utf-8")
        st.session_state.caminho_sessao = caminho

    caminho = st.session_state.caminho_sessao
    bloco   = (f"---\n\n## Interação {n}\n\n"
               f"**🔬 Você:** {pergunta}\n\n"
               f"**🤖 Al IAdo PV:**\n\n{resposta}\n\n")
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(bloco)

    if n >= 1:
        try:
            from src.conhecimento.indexador import indexar_sessao
            indexar_sessao(caminho, modelo_embeddings, PASTA_CHROMADB)
        except Exception:
            pass


# ============================================================
# ABA — RESULTADOS ML
# ============================================================

def renderizar_aba_ml():
    """Dashboard completa dos resultados da Fase 5."""

    import json
    import pandas as pd

    st.header("🔬 Fase 5 — Pipeline de ML")
    st.caption("Análise preditiva de falhas em componentes CA do inversor fotovoltaico")

    # ── Status do pipeline ───────────────────────────────────
    from src.orquestrador import (
        features_ca_pendente, autoencoder_pendente,
        injecao_falhas_pendente, validacao_pendente, rul_weibull_pendente
    )

    etapas = {
        "📐 Features CA"       : not features_ca_pendente(),
        "🧠 Autoencoder"       : not autoencoder_pendente(),
        "💉 Injeção de Falhas" : not injecao_falhas_pendente(),
        "✅ Validação"         : not validacao_pendente(),
        "📈 RUL / Weibull"     : not rul_weibull_pendente(),
    }

    cols = st.columns(5)
    for col, (nome, pronto) in zip(cols, etapas.items()):
        with col:
            if pronto:
                st.success(nome)
            else:
                st.error(f"{nome}\n\n*pendente*")

    st.divider()

    # ── Painel de regeneração ────────────────────────────────
    with st.expander("🔄 Regenerar Resultados", expanded=False):
        st.caption("Apaga os resultados salvos e gera novamente. "
                   "Regenerar uma etapa também refaz todas as que dependem dela.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pipeline completo**")
            if st.button("🔄 Regenerar TUDO", use_container_width=True,
                         type="primary", key="btn_regen_tudo"):
                with st.status("Regenerando pipeline ML completo...",
                               expanded=True) as status:
                    resultados = _regenerar("features_ca", status)
                    ok = all("✅" in r for r in resultados)
                    status.update(
                        label=("Pipeline regenerado!" if ok
                               else "Regeneração interrompida por erro"),
                        state="complete" if ok else "error"
                    )
                st.rerun()

        with col2:
            st.markdown("**A partir de uma etapa**")
            etapa_sel = st.selectbox(
                "Etapa inicial:",
                options=ORDEM_ETAPAS_ML,
                format_func=lambda e: NOMES_ETAPAS[e],
                key="regen_select",
                label_visibility="collapsed"
            )
            if st.button("🔄 Regenerar a partir desta etapa",
                         use_container_width=True, key="btn_regen_etapa"):
                with st.status(f"Regenerando a partir de "
                               f"{NOMES_ETAPAS[etapa_sel]}...",
                               expanded=True) as status:
                    resultados = _regenerar(etapa_sel, status)
                    ok = all("✅" in r for r in resultados)
                    status.update(
                        label=("Regeneração concluída!" if ok
                               else "Regeneração interrompida por erro"),
                        state="complete" if ok else "error"
                    )
                st.rerun()

    st.divider()

    # ── Seção 1: Autoencoder ─────────────────────────────────
    with st.expander("🧠 Autoencoder — Modelagem de Normalidade", expanded=True):
        arq_limiar = PASTA_AE / "limiar.json"
        if arq_limiar.exists():
            with open(arq_limiar) as f:
                limiar_data = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Limiar p99",    f"{limiar_data['limiar']:.4f}")
            c2.metric("μ baseline",    f"{limiar_data['mu']:.4f}")
            c3.metric("σ baseline",    f"{limiar_data['sigma']:.4f}")
            c4.metric("Épocas",        f"{limiar_data.get('epochs_treinadas', '—')}")

            col_a, col_b = st.columns(2)
            with col_a:
                arq = PASTA_AE / "curva_treino.png"
                if arq.exists():
                    st.image(str(arq), caption="Curva de Treinamento", use_container_width=True)
            with col_b:
                arq = PASTA_AE / "distribuicao_erro.png"
                if arq.exists():
                    st.image(str(arq), caption="Distribuição do Erro", use_container_width=True)

            arq = PASTA_AE / "erro_temporal.png"
            if arq.exists():
                st.image(str(arq), caption="Erro de Reconstrução Temporal", use_container_width=True)
        else:
            st.info("Autoencoder ainda não treinado. Execute o pipeline ML.")

    # ── Seção 2: Injeção de Falhas ───────────────────────────
    with st.expander("💉 Injeção de Falhas Sintéticas (FMEA)", expanded=True):
        arq_report = PASTA_AE / "injecao_falhas_report.json"
        if arq_report.exists():
            with open(arq_report) as f:
                report = json.load(f)

            st.caption(f"Limiar: **{report['limiar']:.4f}** | "
                       f"Baseline: **{report['baseline_mean']:.4f}** ± "
                       f"{report['baseline_std']:.4f}")

            # Tabela de SMD
            smd_data = []
            for fid, falha in report["falhas"].items():
                smd = report["smd"].get(fid)
                if smd:
                    erro_smd = falha["resultados"][str(smd)]["erro"]
                    margem   = falha["resultados"][str(smd)]["margem"]
                else:
                    erro_smd = margem = None
                smd_data.append({
                    "Falha"     : falha["nome"],
                    "NPR"       : falha["npr"] or "—",
                    "SMD"       : smd or "não detectada",
                    "Erro SMD"  : f"{erro_smd:.4f}" if erro_smd else "—",
                    "Margem"    : f"{margem:.1f}×" if margem else "—",
                })
            st.dataframe(pd.DataFrame(smd_data), use_container_width=True,
                         hide_index=True)

            col_a, col_b = st.columns(2)
            with col_a:
                arq = PASTA_AE / "injecao_falhas_resultados.png"
                if arq.exists():
                    st.image(str(arq), caption="Erro por Tipo e Severidade",
                             use_container_width=True)
            with col_b:
                arq = PASTA_AE / "injecao_falhas_comparacao.png"
                if arq.exists():
                    st.image(str(arq), caption="Comparação Consolidada (escala log)",
                             use_container_width=True)
        else:
            st.info("Injeção de falhas ainda não executada.")

    # ── Seção 3: Validação Formal ────────────────────────────
    with st.expander("✅ Validação Formal — AUC, F1, Recall", expanded=True):
        arq_csv = PASTA_AE / "validacao_tabela.csv"
        if arq_csv.exists():
            df_val = pd.read_csv(arq_csv)

            # Métricas em destaque
            melhores = df_val.loc[df_val.groupby("falha")["auc_roc"].idxmax()]
            cols_m   = st.columns(len(melhores))
            for col, (_, row) in zip(cols_m, melhores.iterrows()):
                with col:
                    st.metric(
                        label    = row["falha"].split(" ")[0] + "…",
                        value    = f"AUC {row['auc_roc']:.3f}",
                        delta    = f"F1={row['f1']:.3f} | sev={row['severidade']}"
                    )

            # Tabela completa formatada
            df_show = df_val[["falha","severidade","f1","auc_roc","recall","precision"]].copy()
            df_show.columns = ["Falha","Severidade","F1","AUC-ROC","Recall","Precision"]
            df_show = df_show.round(3)
            st.dataframe(
                df_show.style.background_gradient(
                    subset=["F1","AUC-ROC","Recall"], cmap="YlGn"
                ),
                use_container_width=True, hide_index=True
            )

            # Gráficos de validação
            arq = PASTA_AE / "validacao_roc.png"
            if arq.exists():
                st.image(str(arq), caption="Curvas ROC por Tipo de Falha",
                         use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                arq = PASTA_AE / "validacao_matriz.png"
                if arq.exists():
                    st.image(str(arq), caption="Matrizes de Confusão",
                             use_container_width=True)
            with col_b:
                arq = PASTA_AE / "validacao_metricas.png"
                if arq.exists():
                    st.image(str(arq), caption="Heatmap de Métricas",
                             use_container_width=True)
        else:
            st.info("Validação formal ainda não executada.")

    # ── Seção 4: RUL / Weibull ───────────────────────────────
    with st.expander("📈 RUL — Vida Útil Remanescente (Weibull)", expanded=True):
        arq_wb = PASTA_AE / "weibull_results.json"
        if arq_wb.exists():
            with open(arq_wb) as f:
                wb_data = json.load(f)

            # Tabela de parâmetros
            wb_rows = []
            for fid, falha in wb_data["falhas"].items():
                p = falha["weibull"]
                tipo_beta = ("Crescente ↑" if p["beta"] > 1.1
                             else "Constante →" if p["beta"] > 0.9
                             else "Decrescente ↓")
                wb_rows.append({
                    "Falha"  : falha["nome"],
                    "NPR"    : falha["npr"] or "D=10",
                    "β"      : f"{p['beta']:.3f}",
                    "η"      : f"{p['eta']:.1f}",
                    "MTTF"   : f"{p['mttf']:.1f}",
                    "B10"    : f"{p['b10']:.1f}",
                    "Taxa"   : tipo_beta,
                })

            st.caption("β > 1 confirma modelo de desgaste progressivo (esperado)")
            st.dataframe(pd.DataFrame(wb_rows), use_container_width=True,
                         hide_index=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                arq = PASTA_AE / "weibull_ttf.png"
                if arq.exists():
                    st.image(str(arq), caption="Distribuição TTF + Weibull",
                             use_container_width=True)
            with col_b:
                arq = PASTA_AE / "weibull_confiabilidade.png"
                if arq.exists():
                    st.image(str(arq), caption="Funções de Confiabilidade R(t)",
                             use_container_width=True)
            with col_c:
                arq = PASTA_AE / "weibull_rul.png"
                if arq.exists():
                    st.image(str(arq), caption="RUL Condicional E[T−t | T>t]",
                             use_container_width=True)
        else:
            st.info("Análise de Weibull ainda não executada.")


# ============================================================
# ABA — LITERATURA
# ============================================================

def renderizar_aba_literatura(colecao):
    """Lista os documentos indexados na base de conhecimento."""
    st.header("📚 Literatura Indexada")

    from src.conhecimento.agente import listar_documentos
    texto = listar_documentos(colecao)

    total = colecao.count()
    st.metric("Total de chunks", total)
    st.divider()
    st.text(texto)


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def main():

    # Inicializa session_state
    for chave, valor in [
        ("mensagens",         []),
        ("llm",               None),
        ("nome_provedor",     "Nenhum"),
        ("mostrar_artigos",   False),
        ("caminho_sessao",    None),
        ("pergunta_pendente", None),
    ]:
        if chave not in st.session_state:
            st.session_state[chave] = valor

    # Carrega componentes
    try:
        perfil, modelo, colecao, colecao_sessoes, relatorio = carregar_base()
    except Exception as e:
        st.error(f"❌ Erro ao carregar o agente: {e}")
        return

    # Sidebar
    renderizar_sidebar(perfil, modelo, colecao, colecao_sessoes)

    # Título
    st.title("⚡ Al IAdo PV")
    st.caption("Assistente especialista em inversores fotovoltaicos — UTFPR")

    # Feedback do orquestrador
    mostrar_novidades_orquestrador(relatorio)

    # ── Abas principais ──────────────────────────────────────
    aba_chat, aba_ml, aba_lit = st.tabs([
        "💬 Chat",
        "🔬 Resultados ML",
        "📚 Literatura",
    ])

    # ── Aba Chat ─────────────────────────────────────────────
    with aba_chat:

        if st.session_state.llm is None:
            st.info("👈 Selecione um provedor de LLM no painel lateral "
                    "e clique em **Conectar provedor** para começar.")
        else:
            # Histórico de mensagens
            for msg in st.session_state.mensagens:
                avatar = "🔬" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

            # Processa pergunta pendente (enviada pelo chat_input do rodapé)
            if st.session_state.pergunta_pendente:
                pergunta = st.session_state.pergunta_pendente
                st.session_state.pergunta_pendente = None

                with st.chat_message("user", avatar="🔬"):
                    st.markdown(pergunta)

                from src.conhecimento.agente import preparar_prompt
                historico = [{"role": m["role"], "content": m["content"]}
                             for m in st.session_state.mensagens]

                with st.spinner("🔍 Buscando na literatura..."):
                    prompt, citacoes = preparar_prompt(
                        pergunta          = pergunta,
                        perfil            = perfil,
                        modelo_embeddings = modelo,
                        colecao           = colecao,
                        historico         = historico,
                        colecao_sessoes   = colecao_sessoes
                    )

                with st.chat_message("assistant", avatar="🤖"):
                    try:
                        resposta_texto = st.write_stream(
                            stream_resposta(prompt, st.session_state.llm)
                        )
                        if citacoes:
                            st.divider()
                            st.caption("📚 **Fontes consultadas:**")
                            for citacao in citacoes.values():
                                st.caption(f"→ {citacao}")
                    except Exception as e:
                        erro = str(e)
                        if "429" in erro:
                            st.error("⏳ Limite da API atingido. Troque o provedor.")
                        else:
                            st.error(f"❌ Erro: {e}")
                        resposta_texto = f"[Erro: {e}]"

                resposta_completa = resposta_texto
                if citacoes:
                    resposta_completa += ("\n\n**Fontes:** "
                                          + ", ".join(citacoes.values()))

                st.session_state.mensagens.append(
                    {"role": "user", "content": pergunta}
                )
                st.session_state.mensagens.append(
                    {"role": "assistant", "content": resposta_completa}
                )

                n = len(st.session_state.mensagens) // 2
                salvar_sessao_streamlit(pergunta, resposta_completa, n, modelo)

    # ── Aba Resultados ML ─────────────────────────────────────
    with aba_ml:
        renderizar_aba_ml()

    # ── Aba Literatura ────────────────────────────────────────
    with aba_lit:
        renderizar_aba_literatura(colecao)

    # ── Input de chat — nível superior, fixo no rodapé ───────
    if st.session_state.llm is not None:
        if pergunta := st.chat_input(
            "Digite sua pergunta sobre inversores fotovoltaicos..."
        ):
            st.session_state.pergunta_pendente = pergunta
            st.rerun()

# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()