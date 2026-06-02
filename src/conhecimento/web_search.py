"""
web_search.py — Al IAdo PV
Busca leve na web (sem chave de API).

Usa Wikipedia (API REST) como fonte primária e DuckDuckGo Instant Answer
como complemento. Cobre lookups factuais (datas, definições, contexto
histórico) que ficam fora da literatura indexada do mestrado.

Não pretende substituir o RAG: é uma camada de "saber geral" para o agente
não responder algo errado por falta de contexto externo.
"""

from __future__ import annotations

import re
import urllib.parse

import requests

_TIMEOUT = 6
_HEADERS = {
    "User-Agent": "Al-IAdoPV/1.0 (Mestrado UTFPR; contato@al-iado-pv.local)",
    "Accept": "application/json",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.85,es;q=0.75,fr;q=0.7",
}


def _wikipedia_resumo(termo: str, idioma: str = "pt") -> dict | None:
    """Busca resumo na Wikipedia. Retorna {titulo, extrato, url} ou None."""
    termo_url = urllib.parse.quote(termo.replace(" ", "_"))
    url = f"https://{idioma}.wikipedia.org/api/rest_v1/page/summary/{termo_url}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        d = r.json()
        if d.get("type") == "disambiguation":
            return None
        extrato = (d.get("extract") or "").strip()
        if not extrato:
            return None
        return {
            "titulo": d.get("title", termo),
            "extrato": extrato,
            "url": d.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "fonte": f"Wikipedia ({idioma})",
        }
    except Exception:
        return None


def _ddg_instant(termo: str) -> dict | None:
    """DuckDuckGo Instant Answer (sem JS, sem chave)."""
    url = (
        "https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1"
        f"&q={urllib.parse.quote(termo)}"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        d = r.json()
        extrato = (d.get("AbstractText") or "").strip()
        if not extrato:
            return None
        return {
            "titulo": d.get("Heading") or termo,
            "extrato": extrato,
            "url": d.get("AbstractURL", ""),
            "fonte": d.get("AbstractSource") or "DuckDuckGo",
        }
    except Exception:
        return None


def _nivel_confianca(url: str, fonte: str) -> tuple[str, str]:
    """
    Classifica a confiança da fonte web (seção 11):
      A — norma oficial / DOI / documentação institucional
      B — universidade / fabricante / editora técnica
      C — Wikipedia / agregador
      D — conteúdo informal
    """
    u = (url or "").lower()
    f = (fonte or "").lower()
    nivel_a = ("doi.org", "iso.org", "iec.ch", "ieee.org", "abnt.org", ".gov",
               "crossref", "openalex", "semanticscholar")
    nivel_b = (".edu", "scholar.google", "researchgate", "springer", "elsevier",
               "sciencedirect", "mdpi", "nature.com")
    if any(t in u for t in nivel_a):
        return "A", "norma/DOI/institucional"
    if any(t in u for t in nivel_b):
        return "B", "universidade/editora técnica"
    if "wikipedia" in u or "wikipedia" in f or "duckduckgo" in f:
        return "C", "Wikipedia/agregador"
    return "D", "conteúdo informal"


def buscar_web(termo: str, max_chars: int = 1400) -> dict:
    """
    Tenta Wikipedia pt → en → DuckDuckGo.
    Retorna {ok, resultados: [{titulo, extrato, url, fonte, nivel_confianca}], mensagem}.
    """
    termo = (termo or "").strip()
    if not termo:
        return {
            "ok": False,
            "resultados": [],
            "mensagem": "Termo de busca vazio.",
        }

    resultados = []
    for tentativa in (
        lambda: _wikipedia_resumo(termo, "pt"),
        lambda: _wikipedia_resumo(termo, "en"),
        lambda: _wikipedia_resumo(termo, "es"),
        lambda: _wikipedia_resumo(termo, "fr"),
        lambda: _ddg_instant(termo),
    ):
        try:
            r = tentativa()
        except Exception:
            r = None
        if r and r.get("extrato"):
            extrato = r["extrato"]
            if len(extrato) > max_chars:
                extrato = extrato[:max_chars].rsplit(" ", 1)[0] + "…"
            r["extrato"] = extrato
            nivel, desc = _nivel_confianca(r.get("url", ""), r.get("fonte", ""))
            r["nivel_confianca"] = nivel
            r["confianca_desc"] = desc
            resultados.append(r)
            break  # primeiro hit válido já basta

    if not resultados:
        return {
            "ok": False,
            "resultados": [],
            "mensagem": (
                f"Não encontrei resultado factual público para '{termo}'. "
                "Posso responder com base na literatura indexada se você reformular."
            ),
        }

    linhas = [f"## Resultado da busca: {termo}\n"]
    for r in resultados:
        nivel = r.get("nivel_confianca", "?")
        linhas.append(
            f"**{r['titulo']}** — _{r['fonte']}_ · confiança **{nivel}** "
            f"({r.get('confianca_desc', '')})"
        )
        linhas.append(r["extrato"])
        if r.get("url"):
            linhas.append(f"🔗 {r['url']}")
        linhas.append("")

    if any(r.get("nivel_confianca") in ("C", "D") for r in resultados):
        linhas.append(
            "⚠️ Fonte nível C/D (Wikipedia/agregador) NÃO sustenta afirmação "
            "normativa. Para normas (IEC/ISO/IEEE/ABNT) ou artigos, prefira a "
            "fonte oficial / DOI."
        )

    return {
        "ok": True,
        "resultados": resultados,
        "mensagem": "\n".join(linhas).strip(),
    }
