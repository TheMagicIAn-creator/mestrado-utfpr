"""Status, diagnostico e controles laterais da interface."""

from __future__ import annotations

import json

from src.core.config import RAIZ_PROJETO
from src.interface.apoio_streamlit import _estado, _falha_recuperavel

from src.interface.streamlit_proxy import st

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
    except Exception as exc:  # noqa: BLE001
        _falha_recuperavel("Não foi possível consultar o arquivo de log", exc)
        st.caption("Log operacional indisponível nesta execução.")


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


def _estado_persistencia() -> tuple[str, str, str]:
    """(nivel, rotulo, detalhe) da persistencia na nuvem.

    Saude vira UMA linha; so o problema abre espaco. A regra que originou
    este bloco continua valendo: cada alvo e avaliado em separado, para um
    alvo com sucesso nunca mascarar outro falhando em silencio.
    """
    try:
        from src.conhecimento.persistencia_nuvem import diagnostico

        diag = diagnostico()
    except Exception as exc:  # noqa: BLE001
        return "alerta", "Persistência: diagnóstico indisponível", type(exc).__name__

    if not diag["ativa"]:
        return (
            "alerta",
            f"Persistência desligada ({diag['resumo']})",
            f"{diag['detalhe']} Sem isto, sessões e memórias somem a cada reboot.",
        )

    falhas = [
        f"{info.get('rotulo', '')}: {info.get('detalhe', '')}"
        for info in diag.get("por_alvo", {}).values()
        if info.get("estado") == "erro"
    ]
    if falhas:
        return (
            "erro",
            "Persistência FALHANDO",
            " · ".join(falhas)
            + " Verifique a permissão Contents: Read and write do token.",
        )
    return "ok", "Persistência na nuvem ativa", ""


def renderizar_sidebar(modelo, colecao, colecao_sessoes, colecao_obsidian,
                       relatorio: list | None = None) -> None:
    from src.conhecimento.obsidian import contar_notas_indexadas
    from src.core.config import MARCADOR_BUILD

    with st.sidebar:
        st.markdown(
            '<div class="alp-marca">⚡ Al IAdo PV</div>'
            '<div class="alp-sub">Mestrado UTFPR · Engenharia Elétrica</div>',
            unsafe_allow_html=True,
        )

        # ── estado (uma linha quando tudo vai bem) ──────────────────────
        equipe = st.session_state.get("equipe")
        if equipe is None:
            st.markdown(_estado("Equipe de IA desconectada", "erro"),
                        unsafe_allow_html=True)
            erro = st.session_state.get("erro_equipe")
            if erro:
                st.caption(erro)
            if st.button("Ativar equipe", type="primary", width="stretch"):
                from src.interface.streamlit_app import conectar_equipe

                conectar_equipe(forcar=True)
                st.rerun()
        else:
            st.markdown(_estado("Equipe Gemini ativa"), unsafe_allow_html=True)

        nivel_p, rotulo_p, detalhe_p = _estado_persistencia()
        st.markdown(_estado(rotulo_p, nivel_p), unsafe_allow_html=True)
        if detalhe_p:
            st.caption(detalhe_p)

        # ── base de conhecimento em números ─────────────────────────────
        memorias = equipe.memoria.contar() if equipe is not None else 0
        st.markdown(
            '<div class="alp-stats">'
            f'<div><b>{colecao.count()}</b><span>literatura</span></div>'
            f'<div><b>{contar_notas_indexadas(colecao_obsidian)}</b><span>notas</span></div>'
            f'<div><b>{memorias}</b><span>memórias</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Fallback: o caminho normal da nuvem restaura o snapshot portátil no
        # carregamento. O botão só aparece se o snapshot estiver ausente/inválido.
        if colecao.count() == 0:
            st.caption("⚠️ Literatura não indexada (base vazia).")
            if st.button("Indexar literatura", icon=":material/sync:",
                         width="stretch",
                         help="Reconstrói a base a partir dos PDFs de literatura/."):
                try:
                    from src.conhecimento.indexador import indexar_literatura

                    with st.spinner("Indexando literatura… alguns minutos."):
                        resumo = indexar_literatura(modelo=modelo)
                    if resumo["erros"]:
                        st.warning(f"Indexação concluída com {resumo['erros']} erro(s).")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Falha ao indexar: {exc}")

        metadados_pendentes = _carregar_metadados_pendentes()
        if metadados_pendentes:
            st.markdown(
                _estado(f"{len(metadados_pendentes)} PDF(s) com metadados pendentes",
                        "alerta"),
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── tudo que não é estado recua para expanders ──────────────────
        with st.expander("Documentos"):
            arquivo_pdf = st.file_uploader("Adicionar PDF", type=["pdf"],
                                           label_visibility="collapsed")
            if arquivo_pdf is not None and st.button("Enviar para processamento"):
                # Sanitiza o nome vindo do navegador (anti path-traversal) e
                # confirma que o destino fica DENTRO de novos_pdfs/.
                from src.core.seguranca import (
                    caminho_dentro_do_projeto,
                    nome_arquivo_seguro,
                )

                pasta_novos = RAIZ_PROJETO / "novos_pdfs"
                pasta_novos.mkdir(exist_ok=True)
                nome = nome_arquivo_seguro(arquivo_pdf.name, padrao="upload.pdf")
                destino = caminho_dentro_do_projeto(nome, base=pasta_novos)
                destino.write_bytes(arquivo_pdf.getbuffer())
                st.success("PDF enviado. O watcher processará automaticamente.")

        with st.expander("Manutenção"):
            feedback = st.session_state.pop("feedback_manutencao", None)
            if feedback:
                st.success(feedback)

            if metadados_pendentes:
                st.caption("PDF(s) com autor/ano incompletos — confira na fonte "
                           "antes de citar:")
                for nome, info in metadados_pendentes.items():
                    st.caption(
                        f"• {nome} — {info.get('autor_atual') or 'autor ?'} "
                        f"({info.get('ano_atual') or '????'})"
                    )

            col_a, col_b = st.columns(2, gap="small")
            if col_a.button("Consolidar memória", width="stretch"):
                try:
                    from src.conhecimento.consolidar_memoria import consolidar

                    st.session_state.feedback_manutencao = (
                        "Memória consolidada." if consolidar(forcar=True)
                        else "Nada a consolidar."
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro: {exc}")
            if col_b.button("Corrigir metadados", width="stretch"):
                try:
                    from src.orquestrador import reprocessar_metadados_ruins

                    st.session_state.feedback_manutencao = reprocessar_metadados_ruins()
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro: {exc}")

        with st.expander("Diagnóstico"):
            st.caption(f"build: {MARCADOR_BUILD}")
            renderizar_diagnostico(colecao, colecao_sessoes, colecao_obsidian)
            novidades = [
                str(item) for item in (relatorio or [])
                if item and "nenhum pendente" not in str(item).lower()
            ]
            if novidades:
                st.caption("**Inicialização**")
                for item in novidades:
                    st.caption(f"• {item}")

        if st.session_state.get("mensagens"):
            st.markdown("")
            if st.button("Limpar conversa", icon=":material/close:"):
                st.session_state.mensagens = []
                st.session_state.caminho_sessao = None
                st.rerun()

        # Só a data e o número: o marcador completo (com a descrição da
        # entrega) fica no Diagnóstico, senão quebra em três linhas aqui.
        st.caption(" · ".join(MARCADOR_BUILD.split(" · ")[:2]))
