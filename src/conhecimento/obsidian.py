"""Integracao governada entre o Al IAdo PV e o vault Obsidian.

O Obsidian e a camada navegavel do conhecimento do projeto, nao uma fonte
cientifica por si so. Apenas notas dentro de ``notas/Cerebro`` com
``al_iado: true`` e ``status: ativo`` entram nesta colecao. Literatura em PDF,
sessoes brutas e memorias consolidadas antigas permanecem fora dela.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import (
    PASTA_CEREBRO_OBSIDIAN,
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
}
STATUS_INDEXADO = "ativo"
CONFIANCAS = {"alta", "media", "baixa"}
NIVEIS_EVIDENCIA = {"e0", "e1", "e2", "e3", "projeto", "usuario"}

_SEGREDOS = (
    re.compile(r"AIza[A-Za-z0-9_-]{25,}"),
    re.compile(r"gsk_[A-Za-z0-9_-]{15,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{15,}"),
    re.compile(r"(?i)(api[_ -]?key|token|senha|password)\s*[:=]\s*\S{8,}"),
)
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_CABECALHO = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]{2,}")


@dataclass(frozen=True)
class NotaObsidian:
    caminho: Path
    relativo: str
    titulo: str
    corpo: str
    metadados: dict[str, Any]
    wikilinks: tuple[str, ...]
    hash_conteudo: str


@dataclass
class RelatorioSincronizacao:
    notas_ativas: int = 0
    chunks_ativos: int = 0
    notas_atualizadas: int = 0
    notas_removidas: int = 0
    notas_ignoradas: int = 0
    erros: list[str] = field(default_factory=list)

    def para_dict(self) -> dict:
        return {
            "notas_ativas": self.notas_ativas,
            "chunks_ativos": self.chunks_ativos,
            "notas_atualizadas": self.notas_atualizadas,
            "notas_removidas": self.notas_removidas,
            "notas_ignoradas": self.notas_ignoradas,
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
    fim = next(
        (i for i, linha in enumerate(linhas[1:], 1) if linha.strip() == "---"),
        None,
    )
    if fim is None:
        raise ValueError("frontmatter sem delimitador de fechamento")
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


def ler_nota(caminho: Path, raiz: Path = PASTA_CEREBRO_OBSIDIAN) -> NotaObsidian | None:
    """Le uma nota elegivel; retorna ``None`` quando ela nao e opt-in/ativa."""
    raiz = Path(raiz).resolve()
    caminho = Path(caminho)
    resolvido = caminho.resolve()
    try:
        relativo = resolvido.relative_to(raiz).as_posix()
    except ValueError as exc:
        raise ValueError("nota fora da area curada do Obsidian") from exc
    if caminho.is_symlink() or caminho.suffix.lower() != ".md":
        return None

    texto = caminho.read_text(encoding="utf-8", errors="strict")
    meta, corpo = separar_frontmatter(texto)
    if not _booleano(meta.get("al_iado")):
        return None
    status = str(meta.get("status", "")).strip().lower()
    if status != STATUS_INDEXADO:
        return None

    tipo = str(meta.get("tipo", "")).strip().lower()
    if tipo not in TIPOS_NOTA:
        raise ValueError(f"tipo de nota nao permitido: {tipo or '(vazio)'}")
    confianca = str(meta.get("confianca", "media")).strip().lower()
    if confianca not in CONFIANCAS:
        raise ValueError(f"confianca invalida: {confianca}")
    nivel = str(meta.get("nivel_evidencia", "projeto")).strip().lower()
    if nivel not in NIVEIS_EVIDENCIA:
        raise ValueError(f"nivel_evidencia invalido: {nivel}")
    if not corpo.strip():
        raise ValueError("nota ativa sem conteudo")
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
    }
    return NotaObsidian(
        caminho=resolvido,
        relativo=relativo,
        titulo=titulo,
        corpo=corpo,
        metadados=normalizados,
        wikilinks=links,
        hash_conteudo=hash_conteudo,
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
    for cabecalho, linhas in secoes:
        texto = "\n".join(linhas).strip()
        for parte in _quebrar_texto(texto):
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
    raiz: Path = PASTA_CEREBRO_OBSIDIAN,
) -> dict:
    """Sincroniza incrementalmente as notas curadas com a colecao vetorial."""
    raiz = Path(raiz)
    raiz.mkdir(parents=True, exist_ok=True)
    relatorio = RelatorioSincronizacao()
    existentes, _ = _estado_colecao(colecao)
    ativos: dict[str, tuple[NotaObsidian, list[tuple[str, str]]]] = {}

    for caminho in sorted(raiz.rglob("*.md")):
        if any(parte.startswith(".") for parte in caminho.relative_to(raiz).parts):
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


def hash_corpus_obsidian(raiz: Path = PASTA_CEREBRO_OBSIDIAN) -> str:
    """Hash deterministico dos Markdown da area curada para o snapshot."""
    raiz = Path(raiz)
    digest = hashlib.sha256()
    if not raiz.is_dir():
        return digest.hexdigest()
    for caminho in sorted(raiz.rglob("*.md")):
        if caminho.is_symlink():
            continue
        relativo = caminho.resolve().relative_to(raiz.resolve()).as_posix()
        digest.update(relativo.encode("utf-8"))
        digest.update(b"\0")
        digest.update(caminho.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def buscar_notas_obsidian(
    pergunta: str,
    modelo_embeddings,
    colecao,
    *,
    n_resultados: int = 5,
    max_chars: int = 3200,
) -> str:
    """Recupera notas curadas sem transforma-las em citacoes cientificas."""
    try:
        total = int(colecao.count())
    except Exception:
        return ""
    if total <= 0:
        return ""
    vetor = modelo_embeddings.encode([pergunta])
    if hasattr(vetor, "tolist"):
        vetor = vetor.tolist()
    resultado = colecao.query(
        query_embeddings=vetor,
        n_results=min(total, max(n_resultados * 3, n_resultados)),
        include=["documents", "metadatas", "distances"],
    )
    docs = (resultado.get("documents") or [[]])[0]
    metas = (resultado.get("metadatas") or [[]])[0]
    distancias = (resultado.get("distances") or [[]])[0]
    termos = _tokens(pergunta)
    pontuados = []
    for ordem, (doc, meta) in enumerate(zip(docs, metas)):
        meta = meta or {}
        distancia = float(distancias[ordem]) if ordem < len(distancias) else 1.0
        campos = " ".join([
            str(meta.get("titulo", "")),
            str(meta.get("secao", "")),
            str(meta.get("tags", "")),
            str(meta.get("wikilinks", "")),
        ])
        sobreposicao = len(termos & _tokens(campos))
        confianca = {"alta": 0.12, "media": 0.06, "baixa": 0.0}.get(
            str(meta.get("confianca", "media")), 0.0
        )
        score = (1.0 / (1.0 + max(0.0, distancia))) + 0.08 * sobreposicao + confianca
        pontuados.append((score, -ordem, doc, meta))
    pontuados.sort(reverse=True)

    linhas = [
        "\n🧠 DO CÉREBRO OBSIDIAN — NOTAS CURADAS DO PROJETO ",
        "(contexto interno; não é evidência bibliográfica nem substitui os artefatos):\n",
    ]
    usados = sum(len(item) for item in linhas)
    por_nota: dict[str, int] = {}
    incluidos = 0
    for _, _, doc, meta in pontuados:
        caminho = str(meta.get("caminho_obsidian", "?"))
        if por_nota.get(caminho, 0) >= 2:
            continue
        cabecalho = (
            f"\n[Nota curada: {meta.get('titulo', '?')} > {meta.get('secao', '?')} | "
            f"tipo={meta.get('tipo', '?')} | confiança={meta.get('confianca', '?')} | "
            f"evidência={str(meta.get('nivel_evidencia', '?')).upper()} | arquivo={caminho}]\n"
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
> Esta nota é uma projeção legível da memória validada pelo Groq. O arquivo JSON continua sendo a fonte de verdade.

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
"""
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
