"""Evidence Graph leve, determinístico e sempre ancorado em chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.core.config import RAIZ_PROJETO
from src.core.texto import normalizar_busca

TAXONOMY_PATH = Path(RAIZ_PROJETO) / "literatura" / "evidence_graph_taxonomy_v1.json"
RELATIONAL_MARKERS = (
    "relacione",
    "conecte",
    "como se relaciona",
    "relação entre",
    "relacao entre",
    "multi-hop",
)


def consulta_relacional(pergunta: str) -> bool:
    texto = normalizar_busca(pergunta)
    return any(normalizar_busca(marcador) in texto for marcador in RELATIONAL_MARKERS)


def carregar_taxonomia(caminho: str | Path = TAXONOMY_PATH) -> dict[str, Any]:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if dados.get("schema_version") != 1 or not isinstance(dados.get("entities"), list):
        raise ValueError("Taxonomia do Evidence Graph inválida.")
    return dados


def _node_id(tipo: str, nome: str) -> str:
    digest = hashlib.sha256(f"{tipo}:{nome}".encode("utf-8")).hexdigest()[:16]
    return f"{tipo}:{digest}"


def _contem_alias(texto: str, alias: str) -> bool:
    termo = normalizar_busca(alias)
    if not termo:
        return False
    return bool(re.search(rf"(?:^|\s){re.escape(termo)}(?:$|\s)", texto))


def construir_evidence_graph(
    pacote: Mapping[str, Any],
    taxonomia: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Cria relações apenas quando a entidade aparece literalmente no raw_text."""
    taxonomia = dict(taxonomia or carregar_taxonomia())
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def node(tipo: str, nome: str, **extra: Any) -> str:
        node_id = _node_id(tipo, nome)
        nodes.setdefault(node_id, {"id": node_id, "type": tipo, "name": nome, **extra})
        return node_id

    def edge(origem: str, relacao: str, destino: str, evidence: Mapping[str, Any]) -> None:
        chave = (origem, relacao, destino)
        item = edges.setdefault(
            chave,
            {
                "source": origem,
                "relation": relacao,
                "target": destino,
                "evidence_ids": [],
                "chunk_ids": [],
            },
        )
        for campo, valor in (
            ("evidence_ids", evidence.get("evidence_id")),
            ("chunk_ids", evidence.get("chunk_id")),
        ):
            if valor and valor not in item[campo]:
                item[campo].append(valor)

    for evidence in pacote.get("evidences") or []:
        if evidence.get("source_type") != "scientific_pdf":
            continue
        evidence_id = str(evidence.get("evidence_id") or "")
        chunk_id = str(evidence.get("chunk_id") or "")
        document_id = str(evidence.get("document_id") or "")
        if not evidence_id or not chunk_id or not document_id:
            continue
        evidence_node = node("evidence", evidence_id, chunk_id=chunk_id)
        document_node = node("document", document_id, file=evidence.get("file"))
        edge(document_node, "SUPPORTED_BY", evidence_node, evidence)
        author_nodes = [
            node("author", str(author)) for author in evidence.get("authors") or []
        ]
        texto = normalizar_busca(evidence.get("raw_text"))
        for entity in taxonomia.get("entities") or []:
            aliases = entity.get("aliases") or []
            if not any(_contem_alias(texto, alias) for alias in aliases):
                continue
            entity_node = node(str(entity["type"]), str(entity["name"]))
            edge(document_node, "STUDIES", entity_node, evidence)
            edge(entity_node, "SUPPORTED_BY", evidence_node, evidence)
            for author_node in author_nodes:
                edge(author_node, "STUDIES", entity_node, evidence)

    return {
        "schema_version": 1,
        "graph_id": "aliado-evidence-graph-r7-pilot",
        "status": "pilot_not_primary_retrieval",
        "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
        "edges": sorted(
            edges.values(),
            key=lambda item: (item["source"], item["relation"], item["target"]),
        ),
    }


def caminhos_ancorados(
    grafo: Mapping[str, Any],
    origem: str,
    destino: str,
    *,
    max_hops: int = 3,
) -> list[list[str]]:
    """Busca caminhos curtos; cada aresta sem evidência é ignorada."""
    adjacency: dict[str, list[str]] = {}
    for edge in grafo.get("edges") or []:
        if not edge.get("evidence_ids") or not edge.get("chunk_ids"):
            continue
        adjacency.setdefault(str(edge["source"]), []).append(str(edge["target"]))
    encontrados = []
    fila = deque([(origem, [origem])])
    while fila:
        atual, caminho = fila.popleft()
        if len(caminho) - 1 >= max_hops:
            continue
        for vizinho in adjacency.get(atual, []):
            if vizinho in caminho:
                continue
            novo = [*caminho, vizinho]
            if vizinho == destino:
                encontrados.append(novo)
            else:
                fila.append((vizinho, novo))
    return encontrados


def resumir_grafo_para_prompt(grafo: Mapping[str, Any], *, limite: int = 12) -> str:
    nodes = {item["id"]: item["name"] for item in grafo.get("nodes") or []}
    linhas = []
    arestas_ancoradas = [
        edge
        for edge in grafo.get("edges") or []
        if edge.get("evidence_ids") and edge.get("chunk_ids")
    ]
    for edge in arestas_ancoradas[: max(0, limite)]:
        linhas.append(
            f"- {nodes.get(edge['source'], edge['source'])} "
            f"{edge['relation']} {nodes.get(edge['target'], edge['target'])} "
            f"[{', '.join(edge['evidence_ids'])}]"
        )
    return "\n".join(linhas)
