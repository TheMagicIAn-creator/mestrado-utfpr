"""Integração governada entre o Al IAdo PV e todo o vault Obsidian.

Todo Markdown útil do vault entra na coleção ``obsidian_pv`` por padrão. Cada
nota recebe uma classe de origem para que sessões, memórias consolidadas,
rascunhos, notas de literatura e conhecimento curado sejam recuperáveis sem
serem confundidos com evidência científica ou artefatos recalculáveis.

Diretórios técnicos, templates, segredos aparentes e notas explicitamente
marcadas com ``al_iado: false`` ou ``privado: true`` não são indexados.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from src.core.config import (
    PASTA_CEREBRO_OBSIDIAN,
    PASTA_VAULT_OBSIDIAN,
    TAMANHO_LOTE,
)


TIPOS_NOTA = {
    "conceito",
    "contexto",
    "decisao",
    "decisao_metodologica",
    "experimento",
    "hipotese",
    "correcao",
    "preferencia",
    "memoria_validada",
    "memoria_consolidada",
    "sessao",
    "avaliacao",
    "nota_literatura",
    "nota_vault",
}
CONFIANCAS = {"alta", "media", "baixa"}
NIVEIS_EVIDENCIA = {"e0", "e1", "e2", "e3", "projeto", "usuario"}
DIRETORIOS_IGNORADOS = {".obsidian", ".smart-env", "templates"}
STATUS_PRIVADOS = {"privado", "excluido", "excluir", "ignorar"}

_SEGREDOS = (
    re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{25,}"),
    re.compile(r"(?<![A-Za-z0-9])gsk_[A-Za-z0-9_-]{15,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{15,}"),
    re.compile(r"(?i)(api[_ -]?key|token|senha|password)\s*[:=]\s*\S{8,}"),
)
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_CABECALHO = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]{2,}")
_DATA_ARQUIVO = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")
_INTENCAO_HISTORICA = re.compile(
    r"(?i)\b(sess(?:a|ã)o|sess(?:o|õ)es|conversa(?:s)?|conversamos|"
    r"lembr(?:a|e|ar|ança)|mem[oó]ria|hist[oó]rico|anterior(?:es)?)\b"
)
_PRIMEIRO_REGISTRO = re.compile(r"(?i)\b(primeir[ao]|mais antig[ao]|inicial)\b")
_ULTIMO_REGISTRO = re.compile(r"(?i)\b([uú]ltim[ao]|mais recent[ea]|atual)\b")

# ── Inventário: "quais as 10 últimas memórias consolidadas?" ────────────────
# Pergunta de CONTAGEM/LISTA não pode ser respondida por busca semântica: o
# top-K devolve uma AMOSTRA, e o LLM a apresenta como se fosse o total. Foi
# exatamente o que aconteceu — 26 memórias consolidadas no vault, e a resposta
# afirmou "apenas 4", confirmando o número quando contestada. Aqui a contagem
# vem da varredura dos metadados do índice, sem embeddings e sem LLM.
_PEDIDO_INVENTARIO = re.compile(
    r"(?i)\b(quais|quantas?|quantos|list[ae]|listar|liste|relacione|enumere|"
    r"mostre|me\s+d[êe])\b"
)
_ALVOS_INVENTARIO: tuple[tuple[str, re.Pattern, frozenset[str]], ...] = (
    (
        "memórias consolidadas",
        re.compile(r"(?i)(mem[oó]rias?\s+consolidad|consolida[çc][õo]es|"
                   r"consolidad[oa]s?\b)"),
        frozenset({"memoria_consolidada"}),
    ),
    (
        "memórias validadas",
        re.compile(r"(?i)mem[oó]rias?\s+validad"),
        frozenset({"memoria_validada"}),
    ),
    (
        "sessões",
        re.compile(r"(?i)\b(sess[oõ]es|sess[ãa]o)\b"),
        frozenset({"sessao_atual", "sessao_arquivada"}),
    ),
)
# "resuma a última sessão" quer CONTEÚDO, não inventário — segue para o RAG.
_QUER_CONTEUDO = re.compile(
    r"(?i)(resum|conte[uú]do|assunto|discut|aconteceu|falamos|sobre\s+o\s+que)"
)
_QUANTIDADE = re.compile(r"(?i)\b(\d{1,3})\s*(?:[uú]ltim|mais\s+recent|primeir)")
# Plural-cientes, e SÓ para a ordenação do inventário: _PRIMEIRO_REGISTRO e
# _ULTIMO_REGISTRO servem à consulta cronológica (um único registro) e não
# casam "primeiras"/"últimas"; alterá-los mudaria aquele comportamento.
_ORDEM_ANTIGA = re.compile(r"(?i)\b(primeir[ao]s?|mais\s+antig[ao]s?|iniciais?)\b")


@dataclass(frozen=True)
class NotaObsidian:
    caminho: Path
    relativo: str
    titulo: str
    corpo: str
    metadados: dict[str, Any]
    wikilinks: tuple[str, ...]
    hash_conteudo: str
    classe_fonte: str
    data_registro: str
    arquivada: bool


@dataclass
class RelatorioSincronizacao:
    notas_ativas: int = 0
    chunks_ativos: int = 0
    notas_atualizadas: int = 0
    notas_removidas: int = 0
    notas_ignoradas: int = 0
    fontes_por_classe: dict[str, int] = field(default_factory=dict)
    erros: list[str] = field(default_factory=list)

    def para_dict(self) -> dict:
        return {
            "notas_ativas": self.notas_ativas,
            "chunks_ativos": self.chunks_ativos,
            "notas_atualizadas": self.notas_atualizadas,
            "notas_removidas": self.notas_removidas,
            "notas_ignoradas": self.notas_ignoradas,
            "fontes_por_classe": dict(self.fontes_por_classe),
            "erros": list(self.erros),
        }


def _normalizar(texto: str) -> str:
    base = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in base if not unicodedata.combining(c))


def _tokens(texto: str) -> set[str]:
    return set(_TOKEN.findall(_normalizar(texto)))


def _carregar_yaml(texto: str) -> dict:
    try:
        import yaml

        dados = yaml.safe_load(texto) or {}
    except Exception as exc:
        raise ValueError(f"frontmatter YAML invalido: {exc}") from exc
    if not isinstance(dados, dict):
        raise ValueError("frontmatter deve ser um objeto YAML")
    return dados


def separar_frontmatter(texto: str) -> tuple[dict, str]:
    """Separa YAML inicial do corpo Markdown, sem aceitar bloco no meio."""
    linhas = texto.lstrip("\ufeff").splitlines()
    if not linhas or linhas[0].strip() != "---":
        return {}, texto
    primeira_util = next((linha.strip() for linha in linhas[1:] if linha.strip()), "")
    if primeira_util.startswith(("#", "**", "- ", "> ")):
        # Régua horizontal no início de um documento Markdown. Uma régua muito
        # posterior não deve ser confundida com fechamento de frontmatter.
        return {}, texto
    fim = next(
        (i for i, linha in enumerate(linhas[1:], 1) if linha.strip() == "---"),
        None,
    )
    if fim is None:
        # Sessões antigas às vezes começam com uma régua Markdown ``---``.
        # Sem delimitador final não há frontmatter; preserve o texto integral.
        cabecalho_incompleto = "\n".join(linhas[1:20])
        if re.search(r"(?im)^\s*al_iado\s*:\s*false\s*$", cabecalho_incompleto):
            return {"al_iado": False}, texto
        if re.search(r"(?im)^\s*privado\s*:\s*true\s*$", cabecalho_incompleto):
            return {"privado": True}, texto
        return {}, texto
    return _carregar_yaml("\n".join(linhas[1:fim])), "\n".join(linhas[fim + 1:]).strip()


def _booleano(valor: Any) -> bool:
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in {"1", "true", "sim", "yes"}


def _lista(valor: Any) -> list[str]:
    if valor is None:
        return []
    if isinstance(valor, (list, tuple, set)):
        return [str(item).strip() for item in valor if str(item).strip()]
    return [item.strip() for item in str(valor).split(",") if item.strip()]


def _classe_da_nota(relativo: str, meta: dict[str, Any]) -> tuple[str, str, bool]:
    partes = Path(relativo).parts
    topo = _normalizar(partes[0]) if len(partes) > 1 else ""
    nome = _normalizar(Path(relativo).name)
    tipo_declarado = _normalizar(str(meta.get("tipo", ""))).replace("-", "_")

    if topo == "cerebro":
        if tipo_declarado == "memoria_validada" or "memorias validadas" in _normalizar(relativo):
            return "memoria_validada", "memoria_validada", False
        tipo = tipo_declarado if tipo_declarado in TIPOS_NOTA else "contexto"
        return "curada", tipo, False
    if topo == "sessoes_arquivadas":
        tipo = "avaliacao" if "avaliacao" in nome else "sessao"
        return "sessao_arquivada", tipo, True
    if topo == "sessoes":
        tipo = "avaliacao" if "avaliacao" in nome else "sessao"
        return "sessao_atual", tipo, False
    if topo == "memorias":
        return "memoria_consolidada", "memoria_consolidada", False
    if topo == "literatura":
        return "literatura_obsidian", "nota_literatura", False
    if topo == "experimentos":
        return "experimento_obsidian", "experimento", False
    if topo == "conceitos":
        return "conceito_obsidian", "conceito", False
    tipo = tipo_declarado if tipo_declarado in TIPOS_NOTA else "nota_vault"
    return "nota_vault", tipo, False


def _confianca_padrao(classe_fonte: str) -> str:
    return {
        "curada": "media",
        "memoria_validada": "alta",
        "memoria_consolidada": "media",
        "experimento_obsidian": "media",
        "conceito_obsidian": "media",
    }.get(classe_fonte, "baixa")


def _nivel_padrao(classe_fonte: str) -> str:
    if classe_fonte in {"sessao_atual", "sessao_arquivada", "memoria_validada"}:
        return "usuario"
    return "projeto"


def _data_nota(caminho: Path, meta: dict[str, Any]) -> str:
    declarada = str(meta.get("data", meta.get("date", ""))).strip()
    match = _DATA_ARQUIVO.search(declarada) or _DATA_ARQUIVO.search(caminho.name)
    return match.group(1) if match else ""


def ler_nota(caminho: Path, raiz: Path = PASTA_VAULT_OBSIDIAN) -> NotaObsidian | None:
    """Lê uma nota do vault; ``None`` representa exclusão explícita ou técnica."""
    raiz = Path(raiz).resolve()
    caminho = Path(caminho)
    resolvido = caminho.resolve()
    try:
        relativo = resolvido.relative_to(raiz).as_posix()
    except ValueError as exc:
        raise ValueError("nota fora do vault Obsidian configurado") from exc
    if caminho.is_symlink() or caminho.suffix.lower() != ".md":
        return None
    if any(_normalizar(parte) in DIRETORIOS_IGNORADOS for parte in Path(relativo).parts):
        return None

    texto = caminho.read_text(encoding="utf-8", errors="strict")
    meta, corpo = separar_frontmatter(texto)
    if "al_iado" in meta and not _booleano(meta.get("al_iado")):
        return None
    if _booleano(meta.get("privado")):
        return None
    status = str(meta.get("status", "ativo")).strip().lower() or "ativo"
    if status in STATUS_PRIVADOS:
        return None

    classe_fonte, tipo, arquivada = _classe_da_nota(relativo, meta)
    confianca = str(meta.get("confianca", _confianca_padrao(classe_fonte))).strip().lower()
    if confianca not in CONFIANCAS:
        confianca = _confianca_padrao(classe_fonte)
    nivel = str(meta.get("nivel_evidencia", _nivel_padrao(classe_fonte))).strip().lower()
    if nivel not in NIVEIS_EVIDENCIA:
        nivel = _nivel_padrao(classe_fonte)
    if not corpo.strip():
        return None
    if any(p.search(texto) for p in _SEGREDOS):
        raise ValueError("nota contem segredo aparente e nao pode ser indexada")

    titulo_meta = str(meta.get("titulo", "")).strip()
    titulo_h1 = next(
        (m.group(2).strip() for linha in corpo.splitlines() if (m := _CABECALHO.match(linha)) and len(m.group(1)) == 1),
        "",
    )
    titulo = titulo_meta or titulo_h1 or caminho.stem
    links = tuple(dict.fromkeys(m.group(1).strip() for m in _WIKILINK.finditer(corpo)))
    hash_conteudo = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    normalizados = {
        **meta,
        "tipo": tipo,
        "status": status,
        "confianca": confianca,
        "nivel_evidencia": nivel,
        "tags": _lista(meta.get("tags")),
        "classe_fonte": classe_fonte,
        "data_registro": _data_nota(caminho, meta),
        "arquivada": arquivada,
    }
    return NotaObsidian(
        caminho=resolvido,
        relativo=relativo,
        titulo=titulo,
        corpo=corpo,
        metadados=normalizados,
        wikilinks=links,
        hash_conteudo=hash_conteudo,
        classe_fonte=classe_fonte,
        data_registro=normalizados["data_registro"],
        arquivada=arquivada,
    )


def _quebrar_texto(texto: str, limite: int = 1600, sobreposicao: int = 160) -> list[str]:
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    if len(texto) <= limite:
        return [texto] if texto else []
    partes = []
    inicio = 0
    while inicio < len(texto):
        fim = min(len(texto), inicio + limite)
        if fim < len(texto):
            candidatos = [texto.rfind(sep, inicio + limite // 2, fim) for sep in ("\n\n", ". ", "; ")]
            corte = max(candidatos)
            if corte > inicio:
                fim = corte + 1
        parte = texto[inicio:fim].strip()
        if parte:
            partes.append(parte)
        if fim >= len(texto):
            break
        inicio = max(inicio + 1, fim - sobreposicao)
    return partes


def dividir_nota(nota: NotaObsidian) -> list[tuple[str, str]]:
    """Divide por secoes Markdown e preserva o caminho de cabecalhos."""
    secoes: list[tuple[str, list[str]]] = []
    trilha: list[str] = []
    atual: list[str] = []
    rotulo = nota.titulo

    def fechar() -> None:
        nonlocal atual
        texto = "\n".join(atual).strip()
        if texto:
            secoes.append((" > ".join(trilha) or rotulo, atual))
        atual = []

    for linha in nota.corpo.splitlines():
        match = _CABECALHO.match(linha)
        if not match:
            atual.append(linha)
            continue
        fechar()
        nivel = len(match.group(1))
        titulo = match.group(2).strip()
        trilha = trilha[: max(0, nivel - 1)]
        while len(trilha) < nivel - 1:
            trilha.append(nota.titulo)
        trilha.append(titulo)
        atual.append(linha)
    fechar()

    chunks: list[tuple[str, str]] = []
    if nota.classe_fonte in {"sessao_atual", "sessao_arquivada"}:
        limite, sobreposicao = 900, 100
    elif nota.classe_fonte in {"memoria_consolidada", "memoria_validada"}:
        limite, sobreposicao = 1200, 120
    else:
        limite, sobreposicao = 1600, 160
    for cabecalho, linhas in secoes:
        texto = "\n".join(linhas).strip()
        for parte in _quebrar_texto(texto, limite=limite, sobreposicao=sobreposicao):
            chunks.append((cabecalho, parte))
    return chunks


def _estado_colecao(colecao) -> tuple[dict[str, list[tuple[str, dict]]], list[str]]:
    por_arquivo: dict[str, list[tuple[str, dict]]] = {}
    todos_ids: list[str] = []
    try:
        dados = colecao.get(include=["metadatas"])
    except Exception:
        return por_arquivo, todos_ids
    ids = dados.get("ids") or []
    metas = dados.get("metadatas") or []
    for item_id, meta in zip(ids, metas):
        meta = meta or {}
        relativo = str(meta.get("caminho_obsidian", ""))
        if relativo:
            por_arquivo.setdefault(relativo, []).append((item_id, meta))
            todos_ids.append(item_id)
    return por_arquivo, todos_ids


def sincronizar_obsidian(
    colecao,
    modelo_embeddings,
    *,
    raiz: Path = PASTA_VAULT_OBSIDIAN,
) -> dict:
    """Sincroniza incrementalmente todos os Markdown elegíveis do vault."""
    raiz = Path(raiz)
    raiz.mkdir(parents=True, exist_ok=True)
    relatorio = RelatorioSincronizacao()
    existentes, _ = _estado_colecao(colecao)
    ativos: dict[str, tuple[NotaObsidian, list[tuple[str, str]]]] = {}

    for caminho in sorted(raiz.rglob("*.md")):
        partes = caminho.relative_to(raiz).parts
        if any(
            parte.startswith(".") or _normalizar(parte) in DIRETORIOS_IGNORADOS
            for parte in partes
        ):
            relatorio.notas_ignoradas += 1
            continue
        relativo = caminho.resolve().relative_to(raiz.resolve()).as_posix()
        try:
            nota = ler_nota(caminho, raiz)
            if nota is None:
                relatorio.notas_ignoradas += 1
                continue
            chunks = dividir_nota(nota)
            if not chunks:
                raise ValueError("nota sem chunks indexaveis")
            ativos[relativo] = (nota, chunks)
            relatorio.fontes_por_classe[nota.classe_fonte] = (
                relatorio.fontes_por_classe.get(nota.classe_fonte, 0) + 1
            )
        except Exception as exc:
            relatorio.erros.append(f"{relativo}: {exc}")

    ids_remover: list[str] = []
    pendentes: list[tuple[NotaObsidian, int, str, str]] = []
    for relativo, (nota, chunks) in ativos.items():
        anteriores = existentes.get(relativo, [])
        hashes = {str(meta.get("conteudo_hash", "")) for _, meta in anteriores}
        if hashes == {nota.hash_conteudo} and len(anteriores) == len(chunks):
            continue
        ids_remover.extend(item_id for item_id, _ in anteriores)
        relatorio.notas_atualizadas += 1
        for indice, (secao, texto) in enumerate(chunks):
            pendentes.append((nota, indice, secao, texto))

    # Notas removidas, desativadas ou que perderam o opt-in saem da colecao.
    # Em erro de leitura, preservamos o ultimo indice valido.
    com_erro = {item.split(":", 1)[0] for item in relatorio.erros}
    for relativo, anteriores in existentes.items():
        if relativo in ativos or relativo in com_erro:
            continue
        ids_remover.extend(item_id for item_id, _ in anteriores)
        relatorio.notas_removidas += 1

    if ids_remover:
        colecao.delete(ids=list(dict.fromkeys(ids_remover)))

    if pendentes:
        textos = [texto for _, _, _, texto in pendentes]
        vetores = modelo_embeddings.encode(
            textos, batch_size=32, show_progress_bar=False
        )
        if hasattr(vetores, "tolist"):
            vetores = vetores.tolist()
        ids = []
        metadados = []
        for nota, indice, secao, _ in pendentes:
            prefixo = hashlib.sha256(nota.relativo.encode("utf-8")).hexdigest()[:16]
            ids.append(f"obs_{prefixo}_{indice:04d}")
            meta = nota.metadados
            metadados.append({
                "caminho_obsidian": nota.relativo,
                "titulo": nota.titulo[:240],
                "secao": secao[:300],
                "tipo": str(meta["tipo"]),
                "status": str(meta["status"]),
                "confianca": str(meta["confianca"]),
                "nivel_evidencia": str(meta["nivel_evidencia"]),
                "classe_fonte": nota.classe_fonte,
                "data_registro": nota.data_registro,
                "arquivada": nota.arquivada,
                "citavel": False,
                "tags": ";".join(meta.get("tags", []))[:800],
                "wikilinks": ";".join(nota.wikilinks)[:1200],
                "conteudo_hash": nota.hash_conteudo,
                "chunk_index": indice,
            })
        for inicio in range(0, len(ids), TAMANHO_LOTE):
            fim = inicio + TAMANHO_LOTE
            colecao.upsert(
                ids=ids[inicio:fim],
                embeddings=vetores[inicio:fim],
                documents=textos[inicio:fim],
                metadatas=metadados[inicio:fim],
            )

    relatorio.notas_ativas = len(ativos)
    try:
        relatorio.chunks_ativos = int(colecao.count())
    except Exception:
        relatorio.chunks_ativos = sum(len(chunks) for _, chunks in ativos.values())
    return relatorio.para_dict()


def contar_notas_indexadas(colecao) -> int:
    try:
        dados = colecao.get(include=["metadatas"])
        return len({
            str(meta.get("caminho_obsidian"))
            for meta in (dados.get("metadatas") or [])
            if meta and meta.get("caminho_obsidian")
        })
    except Exception:
        return 0


def hash_corpus_obsidian(raiz: Path = PASTA_VAULT_OBSIDIAN) -> str:
    """Hash determinístico dos Markdown elegíveis do vault para o snapshot."""
    raiz = Path(raiz)
    digest = hashlib.sha256()
    if not raiz.is_dir():
        return digest.hexdigest()
    for caminho in sorted(raiz.rglob("*.md")):
        partes = caminho.relative_to(raiz).parts
        if caminho.is_symlink() or any(
            parte.startswith(".") or _normalizar(parte) in DIRETORIOS_IGNORADOS
            for parte in partes
        ):
            continue
        relativo = caminho.resolve().relative_to(raiz.resolve()).as_posix()
        digest.update(relativo.encode("utf-8"))
        digest.update(b"\0")
        digest.update(caminho.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_STOPWORDS_BUSCA = {
    "ainda", "algo", "aquele", "aquela", "como", "com", "das", "dos",
    "ele", "ela", "entre", "essa", "esse", "esta", "este", "isso",
    "mais", "para", "pela", "pelo", "por", "qual", "que", "sobre",
    "uma", "uns", "nas", "nos", "the", "and", "from", "what", "when",
}


def _termos_busca(texto: str) -> list[str]:
    return sorted(
        (token for token in _tokens(texto) if token not in _STOPWORDS_BUSCA),
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
    achados: dict[str, str] = {}
    for classe in classes:
        subpasta, padrao = _PASTAS_INVENTARIO.get(classe, (None, None))
        if not subpasta:
            continue
        base = PASTA_VAULT_OBSIDIAN / subpasta
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
            "`python scripts/reconstruir_cerebro_obsidian.py` e commite "
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
        sobreposicao = len(termos & _tokens(campos))
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


def _yaml_string(valor: Any) -> str:
    return json.dumps(str(valor or ""), ensure_ascii=False)


def espelhar_memoria_validada(
    caminho_memoria: Path,
    *,
    raiz: Path = PASTA_CEREBRO_OBSIDIAN,
) -> dict:
    """Gera a visao Markdown das memorias JSON aprovadas pelo auditor."""
    caminho_memoria = Path(caminho_memoria)
    if not caminho_memoria.is_file():
        return {"escritas": 0, "inalteradas": 0}
    dados = json.loads(caminho_memoria.read_text(encoding="utf-8"))
    itens = dados.get("itens") or []
    if not isinstance(itens, list):
        raise ValueError("memoria_validada.json sem lista de itens")

    pasta = Path(raiz) / "Memorias validadas"
    pasta.mkdir(parents=True, exist_ok=True)
    escritas = inalteradas = 0
    for item in itens:
        item_id = str(item.get("id", "")).strip()
        if not item_id or any(p.search(json.dumps(item)) for p in _SEGREDOS):
            continue
        status = "ativo" if item.get("status") == "ativo" else "superado"
        confianca_num = float(item.get("confianca", 0.0) or 0.0)
        confianca = "alta" if confianca_num >= 0.9 else "media" if confianca_num >= 0.7 else "baixa"
        tipo_original = str(item.get("tipo", "contexto_projeto"))
        tipo_nota = {
            "decisao_metodologica": "decisao_metodologica",
            "contexto_projeto": "contexto",
        }.get(tipo_original, tipo_original)
        titulo = f"Memoria validada - {item_id}"
        from src.conhecimento.vault_links import notas_relacionadas

        relacionadas = notas_relacionadas(
            str(item.get("conteudo", "")), itens, excluir_id=item_id
        )
        links_relacionados = "\n".join(
            f"- [[Memoria validada - {r['id']}]] — "
            + str(r.get("conteudo", ""))[:90]
            for r in relacionadas
        )
        secao_relacionadas = (
            f"\n## Notas relacionadas\n{links_relacionados}\n" if links_relacionados else ""
        )
        conteudo = f"""---
al_iado: true
titulo: {_yaml_string(titulo)}
tipo: {tipo_nota}
status: {status}
confianca: {confianca}
nivel_evidencia: usuario
origem: memoria_validada
memoria_id: {_yaml_string(item_id)}
escopo: {_yaml_string(item.get('escopo', 'compartilhado'))}
criado_em: {_yaml_string(item.get('criado_em_utc', ''))}
tags: [al-iado, memoria-validada, {tipo_nota}]
---

# {titulo}

> [!info] Memória externa auditável
> Esta nota é uma projeção legível da memória validada pelo auditor. O arquivo JSON continua sendo a fonte de verdade.

## Conteúdo aprovado

{item.get('conteudo', '')}

## Evidência do pesquisador

> {item.get('evidencia_usuario', '')}

## Governança

- **Validado por:** {item.get('validado_por', '')}
- **Confiança numérica:** {confianca_num:.3f}
- **Origem:** {item.get('origem', '')}
- **Status:** {status}

## Conexões

- [[00 - Painel do cerebro]]
{secao_relacionadas}"""
        destino = pasta / f"{item_id}.md"
        anterior = destino.read_text(encoding="utf-8") if destino.is_file() else None
        if anterior == conteudo:
            inalteradas += 1
            continue
        temporario = destino.with_suffix(".md.tmp")
        temporario.write_text(conteudo, encoding="utf-8")
        temporario.replace(destino)
        escritas += 1
    return {"escritas": escritas, "inalteradas": inalteradas}
