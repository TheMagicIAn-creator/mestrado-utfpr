"""Persistencia de sessao e execucao dos caminhos de resposta."""

from __future__ import annotations

from src.interface.streamlit_app import (
    RAIZ_PROJETO,
    _falha_recuperavel,
    _html_pensando,
    agora_local,
    renderizar_imagens,
    stream_resposta_limpa,
)

from src.interface.streamlit_proxy import st

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

    # Liga a sessão a memórias validadas relacionadas — sem isto, a sessão só
    # existia isolada no vault, sem nenhuma aresta no grafo do Obsidian.
    try:
        equipe = st.session_state.get("equipe")
        if equipe is not None:
            from src.conhecimento.vault_links import (
                bloco_notas_relacionadas,
                notas_relacionadas,
            )

            itens_memoria = equipe.memoria.listar()
            relacionadas = notas_relacionadas(f"{pergunta}\n{resposta}", itens_memoria)
            bloco += bloco_notas_relacionadas(relacionadas)
    except Exception as exc:
        _falha_recuperavel("Não foi possível relacionar a sessão às memórias", exc)

    with open(st.session_state.caminho_sessao, "a", encoding="utf-8") as f:
        f.write(bloco)

    try:
        indexar_sessao(st.session_state.caminho_sessao, modelo_embeddings, PASTA_CHROMADB)
    except Exception as exc:
        _falha_recuperavel(
            "Sessão salva, mas não foi indexada na memória", exc, notificar=True
        )


def _cadencia_atingida() -> int:
    """Nº de interações atual quando bate um múltiplo de AL_IADO_CONSOLIDAR_A_CADA
    (default 6), ou 0 caso contrário. Cadência única compartilhada pelo
    aprendizado automático e pela persistência de sessão na nuvem."""
    import os

    mensagens = st.session_state.get("mensagens") or []
    n = len(mensagens) // 2
    try:
        passo = max(1, int(os.getenv("AL_IADO_CONSOLIDAR_A_CADA", "6")))
    except ValueError:
        passo = 6
    return n if (n > 0 and n % passo == 0) else 0


def aprender_da_sessao_web() -> None:
    """Aprendizado automático entre sessões, no fluxo da conversa.

    A cada N interações, o auditor (Gemini Flash) varre o transcrito atual e
    extrai decisões/preferências duráveis para a memória validada — SEM depender
    do gatilho manual ("lembre") NEM do watcher (que não roda em modo_consulta,
    ou seja, nunca dispara na nuvem). É a peça que faz o agente "acumular" o que
    foi conversado. Best-effort; persiste no GitHub se AL_IADO_PERSISTIR_NUVEM
    estiver ligado (senão vale durante a instância). Deduplica via registrar().
    """
    auditor = st.session_state.get("auditor")
    if auditor is None or not hasattr(auditor, "consolidar_memoria_das_sessoes"):
        return
    if not _cadencia_atingida():
        return
    try:
        from src.core.conversa_export import montar_transcricao

        mensagens = st.session_state.get("mensagens") or []
        transcrito = montar_transcricao(mensagens)
        auditor.consolidar_memoria_das_sessoes(transcrito, origem="chat_web_auto")
    except Exception as exc:
        _falha_recuperavel(
            "Não foi possível atualizar a memória automática", exc, notificar=True
        )


def persistir_sessao_web() -> None:
    """Commita o arquivo de sessão atual (notas/sessoes/*.md) no GitHub, no
    MESMO ritmo do aprendizado automático (a cada N interações).

    Sem isto, o transcrito da conversa na nuvem só existe no disco EFÊMERO do
    container: qualquer reboot/redeploy o apaga por completo — foi exatamente
    o que aconteceu com as sessões do dia (só a memória validada, um resumo
    curado, sobrevivia). Best-effort; no-op se a persistência não estiver
    ligada (AL_IADO_PERSISTIR_NUVEM) ou se ainda não houver sessão salva.
    """
    caminho = st.session_state.get("caminho_sessao")
    if caminho is None:
        return
    n = _cadencia_atingida()
    if not n:
        return
    try:
        from src.conhecimento.persistencia_nuvem import (
            persistencia_ativa,
            persistir_arquivo,
        )

        if persistencia_ativa():
            persistir_arquivo(
                caminho,
                mensagem=f"chore(sessao): atualiza sessao web ({n} interacoes)",
                alvo="sessao",
            )
    except Exception as exc:
        _falha_recuperavel(
            "Sessão local salva, mas não persistida na nuvem", exc, notificar=True
        )


def _fechar_turno_simples(conteudo_usuario: str, resposta: str, modelo,
                          anexo_txt: dict | None = None) -> None:
    """Registra um par (usuário, assistente) resolvido localmente — sem LLM —
    na sessão, igual ao fluxo normal (para persistir e indexar).

    `anexo_txt` guarda o download junto da mensagem: o Streamlit re-renderiza
    o histórico a cada rerun, e sem isso o botão de baixar a conversa sumiria
    no turno seguinte.
    """
    st.session_state.mensagens.append(
        {"role": "user", "content": conteudo_usuario, "imagens": []}
    )
    mensagem = {"role": "assistant", "content": resposta, "imagens": []}
    if anexo_txt:
        mensagem["export_txt"] = anexo_txt
    st.session_state.mensagens.append(mensagem)
    salvar_sessao(
        conteudo_usuario, resposta, [], len(st.session_state.mensagens) // 2, modelo
    )


def _contexto_recente(n_trocas: int = 4) -> str:
    """Últimas trocas da conversa, para pedidos dêiticos ("guarde ESSE resultado")."""
    msgs = st.session_state.get("mensagens", [])[-(n_trocas * 2):]
    return "\n\n".join(
        f"{'Rodolfo' if m.get('role') == 'user' else 'Al IAdo PV'}: {m.get('content', '')}"
        for m in msgs if m.get("content")
    )


def responder_com_ferramenta(pergunta: str, perfil: str, llm) -> tuple[str, list[dict]]:
    from src.conhecimento.ferramentas import decidir_acao, processar_com_ferramentas

    # A máquina não aparece. Em vez da caixa "Executando solicitação..." com o
    # log rolando dentro, existe UMA linha pulsante que troca de texto conforme
    # o trabalho anda — o pesquisador lê "Treinando o classificador PV Farms",
    # não "Acionando ferramenta: treinar_classificador_pv".
    #
    # A informação não se perde: o mesmo callback de progresso que alimentava a
    # caixa agora alimenta a linha. O que sai de cena é o nome interno da
    # ferramenta (que foi para o log) e o contorno de painel de execução.
    espera = st.empty()

    def _pulsar(texto: str) -> None:
        rotulo = str(texto).strip().rstrip(".…") or "Trabalhando nisso"
        with espera.container():
            with st.chat_message("assistant", avatar="⚡"):
                st.markdown(_html_pensando(rotulo), unsafe_allow_html=True)

    _pulsar("Interpretando o pedido")
    try:
        decisao = decidir_acao(pergunta, llm)
    except Exception:
        espera.empty()
        raise

    if not decisao["usar_ferramenta"]:
        espera.empty()
        return "", []

    try:
        saida = processar_com_ferramentas(
            pergunta=pergunta,
            perfil=perfil,
            llm=llm,
            progresso=_pulsar,
            decisao=decisao,
            contexto=_contexto_recente(),
        )
    finally:
        # Sai de cena SEMPRE: se a ferramenta levantar, o balão pulsante não
        # pode sobreviver acima da mensagem de erro.
        espera.empty()

    with st.chat_message("assistant", avatar="⚡"):
        resposta = saida["resposta"] or "Sem resposta."
        imagens = saida["resultado"].get("imagens", []) if saida["resultado"] else []
        # Trava anti-invenção TAMBÉM no caminho de ferramenta/web: 'norma IEC/ISO'
        # cai em buscar_web, e o LLM pode inventar cláusula/página de uma norma
        # paga não indexada (ex.: IEC 60812). Aqui não há rodapé de literatura,
        # então checamos contra vazio: norma/página sem lastro vira aviso.
        if "Verificação de citações" not in resposta:
            from src.core.citacao_guarda import alerta_citacao_infundada

            aviso = alerta_citacao_infundada(resposta, {})
            if aviso:
                # Aviso no TOPO: o pesquisador vê antes de ler o palpite.
                resposta = aviso.strip() + "\n\n" + resposta
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
    )

    # Consultas simples de primeiro/último registro são resolvidas pela ordem
    # dos metadados. Isso evita que o LLM troque cronologia por similaridade.
    # Inventário do vault, cronologia e saudação já foram tratados por
    # resolver_atalho() em renderizar_chat, antes do roteador de ferramentas —
    # ver src/conhecimento/atalhos.py. Aqui chega só o que precisa de RAG.
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
        except Exception as exc:
            _falha_recuperavel(
                "Não foi possível sincronizar o Obsidian neste turno",
                exc,
                notificar=True,
            )
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

    # Coerencia prosa x rodape: o contexto do prompt nao e filtrado, mas o
    # rodape sim (o auditor roda depois). Sem isto, a prosa pode citar uma fonte
    # que o rodape nao mostra. Restringe as citacoes ao MESMO conjunto do rodape
    # (e, se vazio, proibe citar) — como ultima instrucao, a mais saliente.
    if consultar_literatura:
        from src.core.citacao_guarda import montar_restricao_fontes

        prompt = prompt + "\n\n" + montar_restricao_fontes(citacoes)

    # Quando ha imagem anexada E o provedor e multimodal, o conteudo vira uma
    # lista (texto + image_url); caso contrario, segue como string.
    conteudo_humano = montar_conteudo_humano(
        prompt, anexos, st.session_state.get("multimodal", False)
    )

    with st.chat_message("assistant", avatar="⚡"):
        placeholder = st.empty()
        try:
            refs_md = formatar_referencias_markdown(citacoes)
            resposta = stream_resposta_limpa(
                conteudo_humano,
                st.session_state.llm,
                placeholder,
                refs_md,
            )
            # Trava estrutural: sinaliza citações sem lastro (normas fora do
            # rodapé, páginas sem fonte recuperada) que o prompt sozinho não
            # segura. Roda SEMPRE — a fabricação mais grave (ex.: inventar
            # "IEC 60812, Clause 7.3.3, p.27") acontece justamente quando a
            # literatura NÃO foi consultada e o LLM responde de cabeça. Sem
            # citação real na resposta, o guard fica silencioso (alta precisão).
            from src.core.citacao_guarda import alerta_citacao_infundada

            aviso = alerta_citacao_infundada(resposta, citacoes if consultar_literatura else {})
            if aviso:
                # Aviso no TOPO: o pesquisador vê antes de ler o palpite.
                resposta = aviso.strip() + "\n\n" + resposta
                placeholder.markdown(resposta)
        except Exception as exc:
            # Se a falha veio antes do primeiro token, o placeholder ainda
            # exibe o brilho de espera — sem isto, ele pulsaria para sempre
            # logo acima da mensagem de erro.
            placeholder.empty()
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
