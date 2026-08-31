"""Busca de contexto, catalogo e preparacao final do prompt."""

from __future__ import annotations

import os

from src.conhecimento.agente import (
    N_RESULTADOS,
    PERFIL_COMPACTO,
    _logger,
    _llm_suporta_multimodal,
    montar_bloco_texto_anexos,
)
from src.conhecimento.agente_interacao import (
    _chave_citacao,
    _entrada_citacao,
    _formatar_historico,
    _limitar_texto,
    _orcamento_rag,
    _rotulo_paginas_meta,
    _trecho_relevante,
    deve_consultar_literatura,
)
from src.conhecimento.agente_recuperacao import (
    _busca_hibrida,
    _expandir_query,
    _montar_prompt,
    _rerankar,
    recuperar_hibrido_r4,
)


def perfil_retrieval_ativo() -> str:
    """Perfil canônico com rollback imediato para o ranking legado."""
    perfil = os.getenv("AL_IADO_RETRIEVAL_PROFILE", "r4_hybrid").strip().lower()
    return perfil if perfil in {"r4_hybrid", "baseline"} else "r4_hybrid"

def buscar_contexto(
    pergunta        : str,
    modelo_embeddings,
    colecao,
    colecao_sessoes = None,
    n_pool          : int | None = None,
    n_resultados    : int | None = None,
    contexto_chars  : int | None = None,
    sessao_chars    : int | None = None,
    consultar_literatura: bool = True,
    n_resultados_revisao: int | None = None,
    max_chunks_por_fonte: int = 2,
    indice_lexical = None,
    colecao_obsidian = None,
    obsidian_chars: int | None = None,
    retornar_pacote_evidencias: bool = False,
) -> tuple:
    """
    Recuperacao local em quatro camadas: literatura híbrida, vault Obsidian,
    memória de sessões e memória estruturada adicionada pelo coordenador.
    A auditoria e a síntese são papéis do Router no invocador web.
    Quando consultar_literatura=False, pula expansão/busca/reranking da base
    bibliográfica e usa apenas o cérebro do projeto e a memória de sessões.

    Quando a pergunta cheira a revisao bibliografica ("literatura completa",
    "estado da arte", "cite a literatura"), o orcamento sobe para
    `n_resultados_revisao` chunks e cap de `max_chunks_por_fonte=1` para
    maximizar diversidade.
    """
    contexto = ""
    citacoes = {}
    evidencias_incluidas = []

    if consultar_literatura:
        # ── CAMADA 1 — Expansão ───────────────────────────────
        expansao  = _expandir_query(pergunta)
        variacoes = expansao.get("variacoes", [pergunta])
        termos    = expansao.get("termos",    [])
        revisao   = expansao.get("revisao",   False)

        if pergunta not in variacoes:
            variacoes.insert(0, pergunta)

        if perfil_retrieval_ativo() == "r4_hybrid":
            melhores = recuperar_hibrido_r4(
                pergunta,
                modelo_embeddings,
                colecao,
                indice_lexical,
                n_pool=n_pool or 80,
                n_resultados=n_resultados or min(N_RESULTADOS, 8),
                n_resultados_revisao=(
                    n_resultados_revisao or (n_resultados or N_RESULTADOS) * 2
                ),
                max_chunks_por_fonte=max_chunks_por_fonte,
            )
        else:
            candidatos = _busca_hibrida(
                variacoes,
                termos,
                colecao,
                modelo_embeddings,
                n_pool=n_pool or 30,
                indice_lexical=indice_lexical,
            )
            if revisao:
                alvo = n_resultados_revisao or (n_resultados or N_RESULTADOS) * 2
                cap = 1
            else:
                alvo = n_resultados or min(N_RESULTADOS, 8)
                cap = max_chunks_por_fonte
            melhores = _rerankar(
                candidatos,
                pergunta,
                n_final=alvo,
                max_por_fonte=cap,
                termos_extra=termos,
            )
    else:
        melhores = []

    # Monta contexto da literatura em ROUND-ROBIN por fonte: a primeira
    # rodada inclui 1 chunk de cada fonte distinta (em ordem de score), so
    # entao as rodadas seguintes adicionam segundos/terceiros chunks. Assim,
    # mesmo com orcamento de caracteres apertado, o LLM recebe a MAIOR
    # diversidade possivel de fontes — em vez de ficar travado com 2-3 que
    # esgotaram o limite primeiro.
    if consultar_literatura and melhores:
        contexto += "\n📚 DA LITERATURA CIENTÍFICA:\n"
        usados = len(contexto)
        limite = contexto_chars or 10_000

        # Agrupa por fonte preservando ordem (e ordem dos chunks dentro
        # de cada fonte) — assim os chunks de maior score lideram cada rodada.
        por_fonte: dict[str, list] = {}
        ordem_fontes: list[str] = []
        for doc, meta in melhores:
            arquivo = meta.get("arquivo", "") or meta.get("citacao", "?")
            if arquivo not in por_fonte:
                por_fonte[arquivo] = []
                ordem_fontes.append(arquivo)
            por_fonte[arquivo].append((doc, meta))

        max_rondas = max((len(c) for c in por_fonte.values()), default=0)
        cheio = False
        for ronda in range(max_rondas):
            if cheio:
                break
            for arquivo in ordem_fontes:
                chunks_fonte = por_fonte[arquivo]
                if ronda >= len(chunks_fonte):
                    continue
                doc, meta = chunks_fonte[ronda]
                citacao = meta.get("citacao", arquivo)
                # Página do chunk (extração page-aware). Chunks antigos sem
                # essa metadado simplesmente não recebem página.
                pagina = _rotulo_paginas_meta(meta)
                trecho = _trecho_relevante(doc, pergunta, meta)
                trecho_linha = f'Trecho-chave: "{trecho}"\n' if trecho else ""
                rotulo = citacao + (f" - {pagina}" if pagina else "")
                evidence_id = f"E{len(evidencias_incluidas) + 1}"
                cabecalho = f"\n[{evidence_id}] [Fonte: {rotulo}]\n"
                bloco = f"{cabecalho}{trecho_linha}{doc}\n"
                if usados + len(bloco) > limite:
                    restante = limite - usados - len(cabecalho) - len(trecho_linha)
                    if restante <= 300:
                        cheio = True
                        break
                    bloco = f"{cabecalho}{trecho_linha}{_limitar_texto(doc, restante)}\n"
                citacoes[_chave_citacao(meta, doc)] = _entrada_citacao(meta, doc, pergunta)
                evidencias_incluidas.append((doc, meta))
                contexto += bloco
                usados += len(bloco)
                if usados >= limite:
                    cheio = True
                    break

    # ── Obsidian — todo o vault, classificado e sem valor bibliográfico ──
    contexto_obsidian = ""
    if colecao_obsidian is not None:
        try:
            from src.conhecimento.obsidian import buscar_notas_obsidian

            contexto_obsidian = buscar_notas_obsidian(
                pergunta,
                modelo_embeddings,
                colecao_obsidian,
                n_resultados=max(3, min(6, (n_resultados or 8) // 2)),
                max_chars=obsidian_chars or 3_200,
            )
            contexto += contexto_obsidian
        except Exception as exc:
            _logger.warning("busca no vault Obsidian indisponível: %s", exc)

    # ── Sessões legadas — fallback quando o vault não respondeu ──────────
    # obsidian_pv já inclui sessões atuais e arquivadas. sessoes_pv permanece
    # como compatibilidade e memória volátil durante a migração.
    if colecao_sessoes and not contexto_obsidian:
        try:
            vetor_pergunta = modelo_embeddings.encode([pergunta]).tolist()
            resultados_ses = colecao_sessoes.query(
                query_embeddings = vetor_pergunta,
                n_results        = max(2, min(4, (n_resultados or 8) // 2))
            )
            docs_ses  = resultados_ses.get("documents", [[]])[0]
            metas_ses = resultados_ses.get("metadatas",  [[]])[0]

            if docs_ses:
                contexto += "\n💭 DA MEMÓRIA DE SESSÕES ANTERIORES:\n"
                usados_ses = 0
                limite_ses = sessao_chars or 1_200
                for doc, meta in zip(docs_ses, metas_ses):
                    arquivo = meta.get("arquivo", "")
                    bloco = f"\n[Memória: {arquivo}]\n{doc}\n"
                    if usados_ses + len(bloco) > limite_ses:
                        restante = limite_ses - usados_ses - len(f"\n[Memória: {arquivo}]\n")
                        if restante <= 200:
                            break
                        bloco = f"\n[Memória: {arquivo}]\n{_limitar_texto(doc, restante)}\n"
                    contexto += bloco
                    usados_ses += len(bloco)
                    if usados_ses >= limite_ses:
                        break
        except Exception as exc:
            _logger.warning("fallback de sessões legadas indisponível: %s", exc)

    if retornar_pacote_evidencias:
        from src.conhecimento.evidence_guard import construir_evidence_package

        return (
            contexto,
            citacoes,
            construir_evidence_package(pergunta, evidencias_incluidas),
        )
    return contexto, citacoes

def listar_documentos(colecao) -> str:
    """
    Lista todos os documentos únicos indexados no ChromaDB.
    Usa paginação para evitar o limite de variáveis SQL.
    """

    arquivos_vistos = set()
    documentos      = []
    offset          = 0
    lote            = 200  # busca 200 por vez

    while True:
        try:
            resultados = colecao.get(
                limit   = lote,
                offset  = offset,
                include = ["metadatas"]
            )
        except Exception:
            break

        metadados = resultados.get("metadatas", [])
        if not metadados:
            break

        for meta in metadados:
            arquivo = meta.get("arquivo", "desconhecido")
            pasta   = meta.get("pasta",   "desconhecida")
            citacao = meta.get("citacao", arquivo)
            if arquivo not in arquivos_vistos:
                arquivos_vistos.add(arquivo)
                documentos.append((pasta, citacao))

        offset += lote
        if len(metadados) < lote:
            break

    # Ordena por pasta temática
    documentos.sort(key=lambda x: x[0])

    texto       = f"📚 Total de documentos: {len(documentos)}\n\n"
    pasta_atual = ""

    for pasta, citacao in documentos:
        if pasta != pasta_atual:
            texto      += f"\n📁 {pasta}/\n"
            pasta_atual = pasta
        texto += f"   → {citacao}\n"

    return texto


# Nomes amigaveis para os 5 temas (campo `pasta` no metadado do ChromaDB).
_NOMES_TEMAS = {
    "confiabilidade":   "Confiabilidade e FMEA",
    "inversores-pv":    "Inversores PV e modos de falha",
    "manutencao":       "Manutenção preditiva e RCM",
    "ml-preditivo":     "Machine Learning e predição de falhas",
    "sinais-eletricos": "Sinais elétricos e processamento",
}


def catalogo_literatura(colecao) -> str:
    """
    Catálogo COMPLETO e determinístico da literatura indexada.

    Diferente do RAG (que só traz os trechos mais relevantes e leva o LLM a
    truncar/inventar), aqui lemos TODOS os metadados do ChromaDB e devolvemos
    o inventário inteiro, agrupado por tema. A citação é reconstruída a partir
    de autor/ano/título (campos limpos) para não herdar o mojibake do campo
    `citacao`. Use isto sempre que o pesquisador pedir "liste tudo o que você
    tem", "a base bibliográfica completa", "quantos artigos", etc.
    """
    vistos: dict[str, dict] = {}
    offset, lote = 0, 200

    while True:
        try:
            res = colecao.get(limit=lote, offset=offset, include=["metadatas"])
        except Exception:
            break
        metas = res.get("metadatas", []) or []
        if not metas:
            break
        for m in metas:
            arq = m.get("arquivo", "desconhecido")
            if arq not in vistos:
                vistos[arq] = {
                    "pasta":  m.get("pasta", "outros"),
                    "autor":  (m.get("autor") or "Autor desconhecido").strip(),
                    "ano":    str(m.get("ano") or "s.d.").strip(),
                    "titulo": (m.get("titulo") or arq).strip(),
                    "chunks": int(m.get("total_chunks", 0) or 0),
                }
        offset += lote
        if len(metas) < lote:
            break

    if not vistos:
        return (
            "Não encontrei documentos indexados na base de conhecimento. "
            "Verifique a base com `python scripts/manter_base.py reconstruir-literatura`."
        )

    por_tema: dict[str, list[dict]] = {}
    for info in vistos.values():
        por_tema.setdefault(info["pasta"], []).append(info)

    total = len(vistos)
    linhas = [f"📚 **Base bibliográfica completa — {total} documentos indexados**"]

    # Temas conhecidos primeiro (na ordem do dicionário), depois quaisquer extras.
    ordem_temas = [t for t in _NOMES_TEMAS if t in por_tema]
    ordem_temas += [t for t in sorted(por_tema) if t not in _NOMES_TEMAS]

    for pasta in ordem_temas:
        docs = sorted(por_tema[pasta], key=lambda d: (d["autor"].lower(), d["ano"]))
        nome_tema = _NOMES_TEMAS.get(pasta, pasta)
        linhas.append(f"\n### {nome_tema} ({len(docs)})")
        for d in docs:
            linhas.append(f"- **{d['autor']} ({d['ano']})** — {d['titulo']}")

    linhas.append(
        f"\n_São {total} documentos no total. Posso detalhar qualquer um, "
        "comparar dois trabalhos ou buscar um tema específico — é só pedir._"
    )
    return "\n".join(linhas)


def preparar_prompt(
    pergunta: str,
    perfil: str,
    modelo_embeddings,
    colecao,
    historico: list    = None,
    colecao_sessoes    = None,
    nome_provedor: str | None = None,
    anexos: list | None = None,
    indice_lexical = None,
    colecao_obsidian = None,
) -> tuple:
    """
    Prepara o prompt completo sem invocar o LLM.
    Retorna (prompt_str, citacoes_dict, evidence_package).
    Usado pela camada HTTP para fazer streaming separado.

    `anexos` e a lista de dicts vinda de `leitor_anexos.ler_anexos(...)`: o texto
    extraido (PDF/CSV/Excel/Word/codigo/...) entra no prompt como bloco
    prioritario; imagens viram nota textual aqui (o pixel vai pela via
    multimodal em `montar_conteudo_humano`, chamada pelo invocador do LLM).
    """

    if historico is None:
        historico = []

    orcamento = _orcamento_rag(nome_provedor)
    consultar_literatura = deve_consultar_literatura(pergunta, colecao)
    contexto, citacoes, evidence_package = buscar_contexto(
        pergunta,
        modelo_embeddings,
        colecao,
        colecao_sessoes,
        n_pool=orcamento["n_pool"],
        n_resultados=orcamento["n_resultados"],
        n_resultados_revisao=orcamento.get("n_resultados_revisao"),
        max_chunks_por_fonte=orcamento.get("max_chunks_por_fonte", 2),
        contexto_chars=orcamento["contexto_chars"],
        sessao_chars=orcamento["sessao_chars"],
        obsidian_chars=orcamento.get("obsidian_chars", 3_200),
        consultar_literatura=consultar_literatura,
        indice_lexical=indice_lexical,
        colecao_obsidian=colecao_obsidian,
        retornar_pacote_evidencias=True,
    )

    suporta_imagem = _llm_suporta_multimodal(None, nome_provedor)
    anexos_texto = (
        montar_bloco_texto_anexos(anexos, suporta_imagem=suporta_imagem)
        if anexos else ""
    )

    historico_formatado = _formatar_historico(historico, orcamento)

    # Identidade estática que ENTRA no prompt. O chamador passa `perfil` (ex.:
    # CLAUDE.md). Para não inflar cada prompt com o documento inteiro, usamos o
    # perfil recebido apenas quando é compacto; caso contrário, o PERFIL_COMPACTO
    # curado. Assim o parâmetro deixa de ser ignorado, mas o custo fica contido.
    perfil_prompt = perfil if (perfil and perfil.strip() and len(perfil) <= 6000) else PERFIL_COMPACTO

    prompt = _montar_prompt(
        pergunta,
        contexto,
        historico_formatado,
        orcamento,
        consultar_literatura=consultar_literatura,
        anexos_texto=anexos_texto,
        perfil=perfil_prompt,
    )
    from src.conhecimento.evidence_graph import (
        construir_evidence_graph,
        consulta_relacional,
        resumir_grafo_para_prompt,
    )

    if consulta_relacional(pergunta) and evidence_package.get("evidences"):
        grafo = construir_evidence_graph(evidence_package)
        evidence_package["evidence_graph"] = grafo
        resumo_grafo = resumir_grafo_para_prompt(grafo)
        if resumo_grafo:
            prompt += (
                "\n\n## Evidence Graph R7 — apoio relacional rastreável\n"
                "Use somente as relações abaixo e preserve os marcadores [E#]. "
                "O grafo é auxiliar; o texto recuperado continua sendo a fonte.\n"
                + resumo_grafo
            )
    return prompt, citacoes, evidence_package
