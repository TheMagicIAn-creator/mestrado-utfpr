"""Consultas cronologicas, inventario e busca no vault Obsidian."""

from __future__ import annotations

from src.conhecimento.obsidian import (
    Path,
    _ALVOS_INVENTARIO,
    _DATA_ARQUIVO,
    _INTENCAO_HISTORICA,
    _ORDEM_ANTIGA,
    _PEDIDO_INVENTARIO,
    _PRIMEIRO_REGISTRO,
    _QUANTIDADE,
    _QUER_CONTEUDO,
    _STOPWORDS_BUSCA,
    _ULTIMO_REGISTRO,
    _normalizar,
    _tokens_nota,
    fnmatch,
    re,
)

def _termos_busca(texto: str) -> list[str]:
    return sorted(
        (token for token in _tokens_nota(texto) if token not in _STOPWORDS_BUSCA),
        key=lambda token: (-len(token), token),
    )


def _variacoes_lexicais(texto: str) -> list[str]:
    # Preserva caixa para o $contains case-sensitive do ChromaDB.
    candidatos = []
    vistos = set()
    for termo in re.findall(r"[^\W_][\w-]{2,}", texto, flags=re.UNICODE):
        normalizado = _normalizar(termo)
        if normalizado in _STOPWORDS_BUSCA or normalizado in vistos:
            continue
        vistos.add(normalizado)
        candidatos.append(termo)
    candidatos.sort(key=lambda termo: (-len(_normalizar(termo)), _normalizar(termo)))
    saida = []
    for termo in candidatos:
        for variante in (termo, termo.lower(), termo.capitalize(), termo.upper()):
            if variante not in saida:
                saida.append(variante)
    return saida


def _adicionar_candidato(
    candidatos: dict[tuple[str, int], dict],
    doc: str,
    meta: dict | None,
    *,
    semantico: float = 0.0,
    lexical: float = 0.0,
    temporal: float = 0.0,
) -> None:
    meta = meta or {}
    caminho = str(meta.get("caminho_obsidian", "?"))
    indice = int(meta.get("chunk_index", 0) or 0)
    chave = (caminho, indice)
    item = candidatos.setdefault(chave, {"doc": str(doc), "meta": meta, "score": 0.0})
    item["score"] = max(float(item["score"]), semantico + lexical + temporal)


def _registros_historicos(colecao, pergunta: str) -> list[tuple[str, dict, float]]:
    """Seleciona registros por data para consultas como 'primeira sessão'."""
    if not _INTENCAO_HISTORICA.search(pergunta) and not _DATA_ARQUIVO.search(pergunta):
        return []
    try:
        dados = colecao.get(include=["documents", "metadatas"])
    except Exception:
        return []
    ids = dados.get("ids") or []
    docs = dados.get("documents") or []
    metas = dados.get("metadatas") or []
    itens = []
    for ordem, (doc, meta) in enumerate(zip(docs, metas)):
        meta = meta or {}
        classe = str(meta.get("classe_fonte", ""))
        if classe not in {"sessao_atual", "sessao_arquivada", "memoria_consolidada"}:
            continue
        caminho = str(meta.get("caminho_obsidian", ids[ordem] if ordem < len(ids) else ""))
        itens.append((caminho, int(meta.get("chunk_index", 0) or 0), str(doc), meta))
    if not itens:
        return []

    data_pedida = _DATA_ARQUIVO.search(pergunta)
    if data_pedida:
        alvo = data_pedida.group(1)
        filtrados = [item for item in itens if str(item[3].get("data_registro", "")) == alvo]
    elif _PRIMEIRO_REGISTRO.search(pergunta):
        sessoes = [item for item in itens if "sessao" in str(item[3].get("classe_fonte", ""))]
        if not sessoes:
            return []
        primeiro = min(sessoes, key=lambda item: Path(item[0]).name)[0]
        filtrados = [item for item in itens if item[0] == primeiro]
    elif _ULTIMO_REGISTRO.search(pergunta):
        sessoes = [item for item in itens if "sessao" in str(item[3].get("classe_fonte", ""))]
        if not sessoes:
            return []
        ultimo = max(sessoes, key=lambda item: Path(item[0]).name)[0]
        filtrados = [item for item in itens if item[0] == ultimo]
    else:
        return []
    filtrados.sort(key=lambda item: (item[0], item[1]))
    return [(doc, meta, 1.2) for _, _, doc, meta in filtrados]


def identificar_registro_cronologico(colecao, pergunta: str) -> dict[str, str] | None:
    """Retorna o primeiro/último registro por metadados, sem inferência do LLM."""
    if not (_PRIMEIRO_REGISTRO.search(pergunta) or _ULTIMO_REGISTRO.search(pergunta)):
        return None
    registros = _registros_historicos(colecao, pergunta)
    if not registros:
        return None
    meta = registros[0][1] or {}
    caminho = str(meta.get("caminho_obsidian", ""))
    nome = Path(caminho).name
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})(?:_(\d{2})-(\d{2}))?", nome)
    data_legivel = str(meta.get("data_registro", ""))
    hora = ""
    if match:
        ano, mes, dia, hora_match, minuto = match.groups()
        data_legivel = f"{dia}/{mes}/{ano}"
        if hora_match and minuto:
            hora = f"{hora_match}:{minuto}"
    return {
        "ordem": "primeira" if _PRIMEIRO_REGISTRO.search(pergunta) else "última",
        "data": data_legivel,
        "hora": hora,
        "titulo": str(meta.get("titulo", Path(caminho).stem)),
        "arquivo": caminho,
        "classe_fonte": str(meta.get("classe_fonte", "")),
    }


def responder_consulta_cronologica(colecao, pergunta: str) -> str | None:
    """Responde consultas cronológicas simples diretamente a partir do índice."""
    normalizada = _normalizar(pergunta)
    if any(
        termo in normalizada
        for termo in ("resum", "conteudo", "assunto", "discut", "aconteceu", "falamos")
    ):
        return None
    registro = identificar_registro_cronologico(colecao, pergunta)
    if not registro:
        return None
    quando = registro["data"] or "data não informada"
    if registro["hora"]:
        quando += f", às {registro['hora']}"
    return (
        f"A {registro['ordem']} sessão registrada no vault é de **{quando}**. "
        f"O registro está em `{registro['arquivo']}` e tem o título "
        f"**{registro['titulo']}**. Essa identificação vem da ordenação dos "
        "metadados e nomes de arquivo do índice completo, não de similaridade semântica."
    )


# Onde cada classe VIVE no disco, e o que de fato conta como membro dela.
# O padrao de nome importa: `_classe_da_nota` classifica pela PASTA, entao
# qualquer .md em notas/memorias/ virava "memoria_consolidada" -- foi assim
# que `resultados-fase5-ml.md`, que nao e consolidacao, entrou na contagem.
_PASTAS_INVENTARIO: dict[str, tuple[str, str]] = {
    "memoria_consolidada": ("memorias", "*_consolidado.md"),
    "memoria_validada": ("Cerebro/Memorias validadas", "*.md"),
    "sessao_atual": ("sessoes", "*.md"),
    "sessao_arquivada": ("sessoes_arquivadas", "*.md"),
}


def _no_disco(classes) -> dict[str, str]:
    """Arquivos que EXISTEM no vault, por classe — independente do índice.

    O vault e versionado no Git, entao esses arquivos estao presentes tambem
    na nuvem, mesmo quando o snapshot portatil do indice esta defasado. Sem
    isto, a contagem responde "o que eu consigo buscar" quando o pesquisador
    perguntou "o que existe" — foi o que produziu "15" para um vault de 26.
    """
    from src.conhecimento import obsidian

    achados: dict[str, str] = {}
    for classe in classes:
        subpasta, padrao = _PASTAS_INVENTARIO.get(classe, (None, None))
        if not subpasta:
            continue
        base = obsidian.PASTA_VAULT_OBSIDIAN / subpasta
        if not base.is_dir():
            continue
        for caminho in base.glob(padrao):
            if caminho.is_file():
                achados[caminho.name] = classe
    return achados


def inventario_por_classe(colecao, classes) -> list[dict]:
    """Registros das classes pedidas, um por ARQUIVO, do mais recente.

    Une DUAS fontes e marca a diferença em `indexado`:
      - o disco, que responde "o que existe";
      - o índice, que responde "o que eu consigo buscar".

    Divergir é normal na nuvem, onde o índice vem de um snapshot portátil que
    só é regenerado sob demanda. O que não pode é a diferença ficar invisível.
    A ordenação usa o NOME do arquivo, que começa pelo carimbo de data —
    cronologia por nome, não por similaridade.
    """
    alvo = frozenset(classes)
    padroes = {c: _PASTAS_INVENTARIO.get(c, (None, "*.md"))[1] for c in alvo}

    por_nome: dict[str, dict] = {}

    # 1) disco — a verdade sobre o que existe
    for nome, classe in _no_disco(alvo).items():
        por_nome[nome] = {"arquivo": nome, "nome": nome, "titulo": Path(nome).stem,
                          "data": "", "classe": classe, "indexado": False}

    # 2) índice — o que é pesquisável
    try:
        dados = colecao.get(include=["metadatas"])
    except Exception:
        dados = {}
    ids = dados.get("ids") or []
    for ordem, meta in enumerate(dados.get("metadatas") or []):
        meta = meta or {}
        classe = str(meta.get("classe_fonte", ""))
        if classe not in alvo:
            continue
        caminho = str(meta.get("caminho_obsidian")
                      or (ids[ordem] if ordem < len(ids) else ""))
        if not caminho:
            continue
        nome = Path(caminho).name
        # Mesmo filtro de nome do disco: a classificação por pasta sozinha
        # deixa passar arquivo que não é da classe.
        if not fnmatch(nome, padroes.get(classe, "*.md")):
            continue
        item = por_nome.setdefault(nome, {"arquivo": caminho, "nome": nome,
                                          "classe": classe, "indexado": False})
        item["indexado"] = True
        item["arquivo"] = caminho
        item["titulo"] = str(meta.get("titulo", "") or Path(caminho).stem)
        item["data"] = str(meta.get("data_registro", "") or item.get("data", ""))

    return sorted(por_nome.values(), key=lambda item: item["nome"], reverse=True)


def _data_legivel(item: dict[str, str]) -> str:
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})(?:[_-](\d{2})-(\d{2}))?",
                      item["nome"])
    if not match:
        return item.get("data", "") or "—"
    ano, mes, dia, hora, minuto = match.groups()
    legivel = f"{dia}/{mes}/{ano}"
    return f"{legivel} {hora}:{minuto}" if hora and minuto else legivel


def responder_inventario_vault(colecao, pergunta: str) -> str | None:
    """Responde "quais/quantas <memórias|sessões>" contando de fato.

    Retorna None quando a pergunta não é de inventário — aí segue o fluxo
    normal de RAG.
    """
    # "as 10 últimas memórias consolidadas" é pedido de inventário sem verbo —
    # a quantidade explícita basta como gatilho.
    pediu = bool(_PEDIDO_INVENTARIO.search(pergunta) or _QUANTIDADE.search(pergunta))
    if not pediu or _QUER_CONTEUDO.search(pergunta):
        return None

    for rotulo, padrao, classes in _ALVOS_INVENTARIO:
        if padrao.search(pergunta):
            break
    else:
        return None

    itens = inventario_por_classe(colecao, classes)
    if not itens:
        return None

    total = len(itens)
    pedido = _QUANTIDADE.search(pergunta)
    limite = min(int(pedido.group(1)), total) if pedido else min(10, total)
    if _ORDEM_ANTIGA.search(pergunta):
        recorte, ordem = itens[-limite:][::-1], "mais antigas"
    else:
        recorte, ordem = itens[:limite], "mais recentes"

    linhas = [
        f"| {i} | {_data_legivel(item)} | `{item['nome']}` |"
        + ("" if item.get("indexado") else " ⚠️")
        for i, item in enumerate(recorte, start=1)
    ]
    cabecalho = (
        f"O vault tem **{total} {rotulo}**. "
        f"{'Todas' if limite >= total else f'As {limite} {ordem}'}:"
    )

    partes = []
    if limite < total:
        partes.append(f"As outras {total - limite} também estão no vault.")

    # A defasagem do índice PRECISA aparecer. Antes, a resposta dizia
    # "N indexadas" e o pesquisador lia "N existem" — na nuvem, onde o índice
    # vem de um snapshot congelado, isso escondia 12 de 26 consolidações.
    fora = [i for i in itens if not i.get("indexado")]
    if fora:
        partes.append(
            f"⚠️ **{len(fora)} de {total} ainda não estão no índice de busca** "
            f"(marcadas com ⚠️ acima): elas existem no vault, mas eu não consigo "
            "recuperá-las por conteúdo. O índice portátil da nuvem só é "
            "regenerado sob demanda — para incluí-las, rode no PC "
            "`python scripts/manter_base.py sincronizar-obsidian` e commite "
            "`artefatos/obsidian_indexado.jsonl.gz`."
        )
    partes.append(
        "Contagem obtida listando os arquivos do vault e cruzando com os "
        "metadados do índice — não é amostra de busca semântica."
    )
    return (
        f"{cabecalho}\n\n| # | Data | Arquivo |\n| ---: | :--- | :--- |\n"
        + "\n".join(linhas)
        + "\n\n"
        + "\n\n".join(partes)
    )


def buscar_notas_obsidian(
    pergunta: str,
    modelo_embeddings,
    colecao,
    *,
    n_resultados: int = 5,
    max_chars: int = 3200,
) -> str:
    """Busca híbrida em todo o vault, preservando classe e proveniência."""
    try:
        total = int(colecao.count())
    except Exception:
        return ""
    if total <= 0:
        return ""

    candidatos: dict[tuple[str, int], dict] = {}
    vetor = modelo_embeddings.encode([pergunta])
    if hasattr(vetor, "tolist"):
        vetor = vetor.tolist()
    resultado = colecao.query(
        query_embeddings=vetor,
        n_results=min(total, max(n_resultados * 6, n_resultados)),
        include=["documents", "metadatas", "distances"],
    )
    docs = (resultado.get("documents") or [[]])[0]
    metas = (resultado.get("metadatas") or [[]])[0]
    distancias = (resultado.get("distances") or [[]])[0]
    for ordem, (doc, meta) in enumerate(zip(docs, metas)):
        distancia = float(distancias[ordem]) if ordem < len(distancias) else 1.0
        _adicionar_candidato(
            candidatos,
            doc,
            meta,
            semantico=1.0 / (1.0 + max(0.0, distancia)),
        )

    # Complemento lexical: nomes, siglas e frases exatas não dependem apenas
    # da geometria dos embeddings. Chroma limita cada busca por termo.
    for termo in _variacoes_lexicais(pergunta)[:16]:
        try:
            exatos = colecao.get(
                where_document={"$contains": termo},
                limit=max(8, n_resultados * 4),
                include=["documents", "metadatas"],
            )
        except Exception:
            continue
        for doc, meta in zip(exatos.get("documents") or [], exatos.get("metadatas") or []):
            _adicionar_candidato(candidatos, doc, meta, lexical=0.72)

    for doc, meta, bonus in _registros_historicos(colecao, pergunta):
        _adicionar_candidato(candidatos, doc, meta, temporal=bonus)

    termos = set(_termos_busca(pergunta))
    historica = bool(
        _INTENCAO_HISTORICA.search(pergunta) or _DATA_ARQUIVO.search(pergunta)
    )
    pontuados = []
    for ordem, item in enumerate(candidatos.values()):
        doc = item["doc"]
        meta = item["meta"]
        campos = " ".join([
            str(meta.get("titulo", "")), str(meta.get("secao", "")),
            str(meta.get("tags", "")), str(meta.get("wikilinks", "")),
            str(meta.get("caminho_obsidian", "")), doc,
        ])
        sobreposicao = len(termos & _tokens_nota(campos))
        confianca = {"alta": 0.12, "media": 0.06, "baixa": 0.0}.get(
            str(meta.get("confianca", "baixa")), 0.0
        )
        classe = str(meta.get("classe_fonte", "nota_vault"))
        classe_bonus = {
            "curada": 0.10,
            "memoria_validada": 0.12,
            "memoria_consolidada": 0.08,
            "conceito_obsidian": 0.05,
            "experimento_obsidian": 0.05,
            "literatura_obsidian": -0.04,
            "sessao_atual": 0.24 if historica else -0.08,
            "sessao_arquivada": 0.24 if historica else -0.10,
        }.get(classe, 0.0)
        status_penalidade = -0.18 if str(meta.get("status", "ativo")) in {"rascunho", "superado"} else 0.0
        score = float(item["score"]) + 0.08 * sobreposicao + confianca + classe_bonus + status_penalidade
        pontuados.append((score, -ordem, doc, meta))
    pontuados.sort(reverse=True)

    registro_cronologico = identificar_registro_cronologico(colecao, pergunta)
    linhas = [
        "\n🧠 DO VAULT OBSIDIAN — MEMÓRIA PESQUISÁVEL DO PROJETO ",
        "(contexto interno; não é evidência bibliográfica. Sessões registram falas e respostas antigas, que podem conter hipóteses ou erros já superados):\n",
    ]
    if registro_cronologico:
        linhas.append(
            "\n[REGISTRO CRONOLÓGICO AUTORITATIVO — use este arquivo e esta "
            f"data na resposta: ordem={registro_cronologico['ordem']} | "
            f"data={registro_cronologico['data']} | hora={registro_cronologico['hora'] or '?'} | "
            f"arquivo={registro_cronologico['arquivo']} | "
            f"título={registro_cronologico['titulo']}]\n"
        )
    usados = sum(len(item) for item in linhas)
    por_nota: dict[str, int] = {}
    incluidos = 0
    limite_por_nota = 5 if historica else 2
    for _, _, doc, meta in pontuados:
        caminho = str(meta.get("caminho_obsidian", "?"))
        if por_nota.get(caminho, 0) >= limite_por_nota:
            continue
        classe = str(meta.get("classe_fonte", "nota_vault"))
        cabecalho = (
            f"\n[Registro Obsidian: {meta.get('titulo', '?')} > {meta.get('secao', '?')} | "
            f"origem={classe} | tipo={meta.get('tipo', '?')} | "
            f"confiança={meta.get('confianca', '?')} | "
            f"evidência={str(meta.get('nivel_evidencia', '?')).upper()} | "
            f"data={meta.get('data_registro', '') or '?'} | arquivo={caminho}]\n"
        )
        restante = max_chars - usados - len(cabecalho)
        if restante <= 180:
            break
        trecho = str(doc)
        if len(trecho) > restante:
            trecho = trecho[:restante].rsplit(" ", 1)[0].rstrip() + "…"
        linhas.extend([cabecalho, trecho, "\n"])
        usados += len(cabecalho) + len(trecho) + 1
        por_nota[caminho] = por_nota.get(caminho, 0) + 1
        incluidos += 1
        if incluidos >= n_resultados:
            break
    return "".join(linhas) if incluidos else ""
