"""Métricas determinísticas de recuperação, separadas da geração textual.

As funções simples operam sobre listas ordenadas de identificadores. O contrato
por grupos permite que mais de um chunk seja uma resposta aceitável para a mesma
evidência manual, sem premiar repetição de trechos vizinhos do mesmo documento.
"""

from __future__ import annotations

import math


def recall_at_k(recuperados: list, relevantes, k: int) -> float:
    rel = set(relevantes)
    if not rel:
        return 0.0
    topk = set(recuperados[:k])
    return len(rel & topk) / len(rel)


def precision_at_k(recuperados: list, relevantes, k: int) -> float:
    """Fração das ``k`` posições ocupadas por itens relevantes distintos."""
    if k <= 0:
        return 0.0
    rel = set(relevantes)
    topk = set(recuperados[:k])
    return len(rel & topk) / k


def hit_rate_at_k(recuperados: list, relevantes, k: int) -> float:
    """Indica se ao menos um item relevante aparece nas primeiras ``k`` posições."""
    rel = set(relevantes)
    return float(bool(rel & set(recuperados[:k])))


def mrr(recuperados: list, relevantes) -> float:
    """Reciprocal rank da PRIMEIRA fonte relevante (0 se nenhuma)."""
    rel = set(relevantes)
    for i, item in enumerate(recuperados, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(recuperados: list, relevantes, k: int) -> float:
    """nDCG@k com relevância binária."""
    rel = set(relevantes)
    dcg = 0.0
    for i, item in enumerate(recuperados[:k], start=1):
        if item in rel:
            dcg += 1.0 / math.log2(i + 1)
    n_ideal = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def cobertura_fontes(recuperados: list) -> int:
    """Número de fontes DISTINTAS recuperadas (diversidade)."""
    return len(set(recuperados))


def _normalizar_grupos(
    grupos_relevantes: list[dict],
) -> tuple[list[tuple[tuple[str, ...], int]], dict[str, list[int]]]:
    grupos = []
    por_chunk: dict[str, list[int]] = {}
    for indice, grupo in enumerate(grupos_relevantes):
        ids = tuple(dict.fromkeys(str(item) for item in grupo.get("chunk_ids", [])))
        relevancia = int(grupo.get("relevance", 1))
        if not ids:
            raise ValueError("Cada grupo relevante deve declarar ao menos um chunk_id.")
        if relevancia not in (1, 2, 3):
            raise ValueError("A relevância esperada deve estar entre 1 e 3.")
        grupos.append((ids, relevancia))
        for chunk_id in ids:
            por_chunk.setdefault(chunk_id, []).append(indice)
    return grupos, por_chunk


def _ganhos_por_rank(
    recuperados: list[str],
    grupos: list[tuple[tuple[str, ...], int]],
    por_chunk: dict[str, list[int]],
    k: int,
) -> tuple[set[int], list[int], int | None]:
    grupos_vistos: set[int] = set()
    ganhos: list[int] = []
    primeiro_rank: int | None = None
    for rank, chunk_id in enumerate(recuperados[:k], start=1):
        grupo_novo = next(
            (
                indice
                for indice in por_chunk.get(str(chunk_id), [])
                if indice not in grupos_vistos
            ),
            None,
        )
        if grupo_novo is None:
            ganhos.append(0)
            continue
        grupos_vistos.add(grupo_novo)
        ganhos.append(grupos[grupo_novo][1])
        if primeiro_rank is None:
            primeiro_rank = rank
    return grupos_vistos, ganhos, primeiro_rank


def metricas_por_grupos(
    recuperados: list[str],
    grupos_relevantes: list[dict],
    k: int,
) -> dict[str, float | int]:
    """Avalia evidências com alternativas de chunk e relevância graduada.

    Cada grupo representa uma unidade de informação esperada e contém
    ``chunk_ids`` aceitáveis e ``relevance`` entre 1 e 3. Apenas a primeira
    ocorrência de cada grupo pontua; assim, chunks vizinhos não inflam Recall,
    Precision ou nDCG.
    """
    grupos, por_chunk = _normalizar_grupos(grupos_relevantes)
    if not grupos:
        return {
            "k": k,
            f"recall@{k}": 0.0,
            f"precision@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            "mrr": 0.0,
            f"ndcg@{k}": 0.0,
            "grupos_relevantes": 0,
            "grupos_recuperados": 0,
        }

    grupos_vistos, ganhos, primeiro_rank = _ganhos_por_rank(
        recuperados,
        grupos,
        por_chunk,
        k,
    )

    dcg = sum(
        ((2**ganho) - 1) / math.log2(rank + 1)
        for rank, ganho in enumerate(ganhos, start=1)
        if ganho
    )
    ganhos_ideais = sorted((relevancia for _, relevancia in grupos), reverse=True)[:k]
    idcg = sum(
        ((2**ganho) - 1) / math.log2(rank + 1)
        for rank, ganho in enumerate(ganhos_ideais, start=1)
    )
    encontrados = len(grupos_vistos)
    return {
        "k": k,
        f"recall@{k}": encontrados / len(grupos),
        f"precision@{k}": encontrados / k if k > 0 else 0.0,
        f"hit_rate@{k}": float(bool(encontrados)),
        "mrr": (1.0 / primeiro_rank) if primeiro_rank else 0.0,
        f"ndcg@{k}": (dcg / idcg) if idcg else 0.0,
        "grupos_relevantes": len(grupos),
        "grupos_recuperados": encontrados,
    }


def _corresponde_ao_grupo(recuperado: dict, grupo: dict, nivel: str) -> bool:
    if nivel == "chunk":
        return str(recuperado.get("chunk_id", "")) in {
            str(item) for item in grupo.get("chunk_ids", [])
        }
    mesmo_documento = str(recuperado.get("document_id", "")) == str(
        grupo.get("document_id", "")
    )
    if nivel == "document":
        return mesmo_documento
    paginas_recuperadas = {int(item) for item in recuperado.get("pages", [])}
    paginas_esperadas = {int(item) for item in grupo.get("pages", [])}
    return mesmo_documento and bool(paginas_recuperadas & paginas_esperadas)


def _selecionar_grupo(
    recuperado: dict,
    grupos_relevantes: list[dict],
    nivel: str,
    grupos_atribuidos: set[int],
) -> int | None:
    candidatos = [
        (int(grupo.get("relevance", 1)), indice)
        for indice, grupo in enumerate(grupos_relevantes)
        if indice not in grupos_atribuidos
        and _corresponde_ao_grupo(recuperado, grupo, nivel)
    ]
    return max(candidatos)[1] if candidatos else None


def metricas_por_evidencias(
    recuperados: list[dict],
    grupos_relevantes: list[dict],
    k: int,
    *,
    nivel: str = "page",
) -> dict[str, float | int]:
    """Avalia chunks recuperados nos níveis ``chunk``, ``page`` ou ``document``.

    Página é o nível primário do benchmark porque corresponde à unidade
    verificável da citação. Chunk exato é mais estrito e sensível às fronteiras
    de segmentação; documento é um limite superior menos específico.
    """
    if nivel not in {"chunk", "page", "document"}:
        raise ValueError("Nível deve ser chunk, page ou document.")

    grupos_sinteticos = [
        {
            "chunk_ids": [f"grupo:{indice}"],
            "relevance": int(grupo.get("relevance", 1)),
        }
        for indice, grupo in enumerate(grupos_relevantes)
    ]
    recuperados_sinteticos = []
    grupos_atribuidos: set[int] = set()
    for rank, recuperado in enumerate(recuperados):
        indice = _selecionar_grupo(
            recuperado,
            grupos_relevantes,
            nivel,
            grupos_atribuidos,
        )
        if indice is not None:
            grupos_atribuidos.add(indice)
            recuperados_sinteticos.append(f"grupo:{indice}")
        else:
            recuperados_sinteticos.append(f"miss:{rank}")

    return metricas_por_grupos(
        recuperados_sinteticos,
        grupos_sinteticos,
        k,
    )


def agregimar(linhas: list[dict], k: int = 5) -> dict:
    """Agrega métricas por pergunta em médias (linhas: [{recuperados, relevantes}])."""
    if not linhas:
        return {"n": 0}
    n = len(linhas)
    rec = sum(recall_at_k(x["recuperados"], x["relevantes"], k) for x in linhas) / n
    rr = sum(mrr(x["recuperados"], x["relevantes"]) for x in linhas) / n
    nd = sum(ndcg_at_k(x["recuperados"], x["relevantes"], k) for x in linhas) / n
    precisao = sum(
        precision_at_k(x["recuperados"], x["relevantes"], k) for x in linhas
    ) / n
    acerto = sum(
        hit_rate_at_k(x["recuperados"], x["relevantes"], k) for x in linhas
    ) / n
    return {
        "n": n,
        "k": k,
        f"recall@{k}": round(rec, 4),
        f"precision@{k}": round(precisao, 4),
        f"hit_rate@{k}": round(acerto, 4),
        "mrr": round(rr, 4),
        f"ndcg@{k}": round(nd, 4),
        "taxa_fonte_correta": round(acerto, 4),
    }
