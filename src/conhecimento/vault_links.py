"""
vault_links.py — Al IAdo PV

Ligações reais entre notas do vault (memória validada <-> memória validada,
sessão <-> memória validada), além do link único e estático para o hub
("00 - Painel do cerebro") que as notas geradas automaticamente já tinham.

Motivação: sem isto, o grafo do Obsidian era uma estrela — todo nó auto-gerado
apontava só para o hub, sem nenhuma aresta entre si. Não havia "fluxo lógico"
navegável entre uma decisão e as sessões/decisões relacionadas.

Lógica pura (sem I/O, sem Streamlit): sobreposição lexical simples entre o
texto de origem e o conteúdo de cada item de memória validada ATIVO. Conservador
de propósito — exige um mínimo de termos em comum para não linkar por acaso.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import re

from src.core.texto import normalizar_busca as _normalizar

MIN_OVERLAP_PADRAO = 2
MAX_LINKS_PADRAO = 3


def _tokens_relacao(texto: str) -> set[str]:
    return {t for t in _normalizar(texto).split() if len(t) >= 4}


def notas_relacionadas(
    texto: str,
    itens: list[dict],
    *,
    excluir_id: str | None = None,
    max_links: int = MAX_LINKS_PADRAO,
    min_overlap: int = MIN_OVERLAP_PADRAO,
) -> list[dict]:
    """Retorna, em ordem de relevância, os itens de memória validada ATIVOS
    mais lexicalmente relacionados a `texto` (por sobreposição de termos).

    `itens` é a lista bruta de MemoriaPersistente.listar() (ou dados["itens"]).
    `excluir_id` evita que um item aponte para si mesmo (uso memória<->memória).
    """
    alvo = _tokens_relacao(texto)
    if not alvo:
        return []
    pontuados = []
    for item in itens or []:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ativo":
            continue
        item_id = str(item.get("id", ""))
        if excluir_id and item_id == str(excluir_id):
            continue
        candidato = _tokens_relacao(item.get("conteudo", ""))
        overlap = len(alvo & candidato)
        if overlap >= min_overlap:
            pontuados.append((overlap, item))
    pontuados.sort(key=lambda par: -par[0])
    return [item for _, item in pontuados[:max_links]]


def bloco_notas_relacionadas(itens: list[dict], *, titulo: str = "Notas relacionadas") -> str:
    """Bloco Markdown com wikilinks para os itens, ou '' se a lista for vazia."""
    if not itens:
        return ""
    linhas = []
    for item in itens:
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            continue
        trecho = re.sub(r"\s+", " ", str(item.get("conteudo", ""))).strip()[:90]
        linhas.append(f"- [[Memoria validada - {item_id}]] — {trecho}")
    if not linhas:
        return ""
    return f"\n## {titulo}\n" + "\n".join(linhas) + "\n"
