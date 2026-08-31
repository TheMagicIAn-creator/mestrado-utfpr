"""Contratos determinísticos de evidência e abstenção do ALIAdo."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from datetime import datetime, timezone

from src.core.texto import normalizar_espacos

SUPPORT_LEVELS = {"direct", "derived", "contextual", "unsupported"}
ABSTENTION_MESSAGE = (
    "Não localizei suporte documental suficiente na biblioteca para afirmar "
    "isso com segurança. Posso refazer a busca com termos mais específicos."
)
_EVIDENCE_ID = re.compile(r"\[(E\d+)\]")
_NON_SCIENTIFIC_ORIGINS = {"memory", "obsidian", "session", "conversation"}


def _paginas(metadata: Mapping[str, Any]) -> list[int]:
    inicio = int(metadata.get("pagina_inicio") or metadata.get("pagina") or 0)
    fim = int(metadata.get("pagina_fim") or inicio)
    if inicio <= 0:
        return []
    return list(range(inicio, max(inicio, fim) + 1))


def construir_evidence_package(
    query: str,
    evidencias: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Converte chunks científicos recuperados em contrato rastreável."""
    itens = []
    for documento, metadata_original in evidencias:
        metadata = dict(metadata_original or {})
        origem = normalizar_espacos(metadata.get("source_type") or "scientific_pdf")
        if origem in _NON_SCIENTIFIC_ORIGINS:
            continue
        chunk_id = str(metadata.get("_chunk_id") or metadata.get("chunk_id") or "")
        document_id = str(
            metadata.get("document_id") or metadata.get("arquivo_hash") or ""
        )
        arquivo = str(metadata.get("arquivo") or "")
        if not chunk_id or not document_id or not arquivo or not str(documento).strip():
            continue
        paginas = _paginas(metadata)
        itens.append(
            {
                "evidence_id": f"E{len(itens) + 1}",
                "document_id": document_id,
                "chunk_id": chunk_id,
                "file": arquivo,
                "title": str(metadata.get("titulo") or arquivo),
                "authors": [
                    parte.strip()
                    for parte in str(metadata.get("autor") or "").split(";")
                    if parte.strip()
                ],
                "year": int(metadata["ano"])
                if str(metadata.get("ano") or "").isdigit()
                else None,
                "pages": paginas,
                "section": str(metadata.get("secao") or ""),
                "raw_text": str(documento),
                "retrieval_score": float(metadata.get("_rrf_score") or 0.0),
                "retrieval_sources": list(metadata.get("_retrieval_sources") or ()),
                "source_type": "scientific_pdf",
            }
        )
    return {
        "schema_version": 1,
        "query": str(query),
        "evidences": itens,
        "abstention_required": not bool(itens),
    }


def renderizar_restricao_pacote(pacote: Mapping[str, Any]) -> str:
    evidencias = list(pacote.get("evidences") or [])
    if not evidencias:
        return "EVIDENCE PACKAGE: vazio. A resposta acadêmica deve se abster."
    linhas = []
    for item in evidencias:
        paginas = ", ".join(str(p) for p in item.get("pages") or []) or "s.p."
        autores = "; ".join(item.get("authors") or []) or "Autor não identificado"
        ano = item.get("year") or "s.d."
        linhas.append(
            f"- [{item['evidence_id']}] {autores} ({ano}), páginas {paginas}, "
            f"arquivo {item['file']}, chunk {item['chunk_id']}"
        )
    return (
        "EVIDENCE PACKAGE AUTORIZADO:\n"
        + "\n".join(linhas)
        + "\nAssocie cada afirmação acadêmica ao marcador [E#] correspondente. "
        "Não use IDs, autores, anos, páginas ou documentos fora deste pacote."
    )


def validar_claims(
    claims: Iterable[Mapping[str, Any]],
    pacote: Mapping[str, Any],
) -> dict[str, Any]:
    """Valida claims estruturados sem consultar qualquer outro modelo."""
    evidencias = {
        item["evidence_id"]: item for item in pacote.get("evidences") or []
    }
    resultados = []
    for indice, claim_original in enumerate(claims, start=1):
        claim = dict(claim_original or {})
        erros = []
        suporte = str(claim.get("support") or "unsupported")
        ids = [str(item) for item in claim.get("evidence_ids") or []]
        if suporte not in SUPPORT_LEVELS:
            erros.append("support_level_invalid")
        if suporte == "unsupported":
            erros.append("unsupported_claim")
        if not ids:
            erros.append("missing_evidence_id")
        desconhecidos = [item for item in ids if item not in evidencias]
        if desconhecidos:
            erros.append("unknown_evidence_id")
        selecionadas = [evidencias[item] for item in ids if item in evidencias]
        pagina = claim.get("page")
        if pagina is not None and not any(
            int(pagina) in evidencia.get("pages", []) for evidencia in selecionadas
        ):
            erros.append("page_not_in_evidence")
        autor = normalizar_espacos(claim.get("author"))
        if autor and not any(
            any(autor in normalizar_espacos(nome) for nome in evidencia.get("authors", []))
            for evidencia in selecionadas
        ):
            erros.append("author_not_in_evidence")
        ano = claim.get("year")
        if ano is not None and not any(
            int(ano) == evidencia.get("year") for evidencia in selecionadas
        ):
            erros.append("year_not_in_evidence")
        quote = normalizar_espacos(claim.get("quote"))
        if quote and not any(
            quote in normalizar_espacos(evidencia.get("raw_text"))
            for evidencia in selecionadas
        ):
            erros.append("quote_not_in_raw_text")
        resultados.append(
            {
                "claim_index": indice,
                "valid": not erros,
                "errors": sorted(set(erros)),
            }
        )
    validos = sum(item["valid"] for item in resultados)
    return {
        "valid": validos == len(resultados) and bool(resultados),
        "claims_total": len(resultados),
        "claims_valid": validos,
        "unsupported_claims": sum(
            "unsupported_claim" in item["errors"] for item in resultados
        ),
        "results": resultados,
    }


def validar_resposta(resposta: str, pacote: Mapping[str, Any]) -> dict[str, Any]:
    """Confere se todos os marcadores de evidência da prosa existem."""
    conhecidos = {
        item["evidence_id"] for item in pacote.get("evidences") or []
    }
    citados = _EVIDENCE_ID.findall(str(resposta or ""))
    desconhecidos = sorted(set(citados) - conhecidos)
    marcacao_ausente = bool(conhecidos) and not citados
    return {
        "valid": not desconhecidos and not marcacao_ausente,
        "cited_evidence_ids": list(dict.fromkeys(citados)),
        "unknown_evidence_ids": desconhecidos,
        "missing_evidence_marker": marcacao_ausente,
        "abstention_required": not bool(conhecidos),
    }


def resposta_segura(resposta: str, pacote: Mapping[str, Any]) -> str:
    """Bloqueia publicação quando a resposta referencia evidência inexistente."""
    resultado = validar_resposta(resposta, pacote)
    if resultado["abstention_required"] or not resultado["valid"]:
        return ABSTENTION_MESSAGE
    return str(resposta or "")


def executar_benchmark_guard(*, git_revision: str | None = None) -> dict[str, Any]:
    """Mede o guard em casos determinísticos positivos e adversariais."""
    pacote = construir_evidence_package(
        "Defina confiabilidade.",
        [
            (
                "Confiabilidade é a probabilidade de cumprir a função requerida.",
                {
                    "_chunk_id": "doc:benchmark:p12:c01",
                    "arquivo_hash": "benchmark",
                    "arquivo": "benchmark.pdf",
                    "titulo": "Reliability",
                    "autor": "Silva",
                    "ano": "2022",
                    "pagina_inicio": 12,
                },
            )
        ],
    )
    casos = [
        (
            "valid_direct",
            True,
            {
                "claim": "Definição",
                "evidence_ids": ["E1"],
                "support": "direct",
                "author": "Silva",
                "year": 2022,
                "page": 12,
                "quote": "probabilidade de cumprir a função requerida",
            },
        ),
        ("unknown_evidence", False, {"claim": "x", "evidence_ids": ["E99"], "support": "direct"}),
        ("invented_page", False, {"claim": "x", "evidence_ids": ["E1"], "support": "direct", "page": 99}),
        ("invented_author", False, {"claim": "x", "evidence_ids": ["E1"], "support": "direct", "author": "Outro"}),
        ("invented_year", False, {"claim": "x", "evidence_ids": ["E1"], "support": "direct", "year": 1999}),
        ("invented_quote", False, {"claim": "x", "evidence_ids": ["E1"], "support": "direct", "quote": "frase ausente"}),
        ("unsupported_claim", False, {"claim": "x", "evidence_ids": ["E1"], "support": "unsupported"}),
    ]
    resultados = []
    for nome, esperado, claim in casos:
        auditoria = validar_claims([claim], pacote)
        observado = bool(auditoria["valid"])
        resultados.append(
            {
                "case": nome,
                "expected_valid": esperado,
                "observed_valid": observado,
                "passed": observado is esperado,
                "errors": auditoria["results"][0]["errors"],
            }
        )
    pacote_vazio = construir_evidence_package("sem suporte", [])
    abstencao_ok = resposta_segura("resposta", pacote_vazio) == ABSTENTION_MESSAGE
    memoria_vazia = construir_evidence_package(
        "memória",
        [("fala", {"source_type": "session", "chunk_id": "memória"})],
    )
    memoria_ok = not memoria_vazia["evidences"]
    validos = [item for item in resultados if item["expected_valid"]]
    invalidos = [item for item in resultados if not item["expected_valid"]]
    return {
        "schema_version": 1,
        "benchmark_id": "evidence-rag-r5-evidence-guard",
        "stage": "R5",
        "variant": "deterministic_claim_evidence_guard_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision,
        "guard": {
            "external_model_required": False,
            "claim_evidence_chain": True,
            "quote_uses_raw_text": True,
            "memory_is_scientific_source": False,
            "abstention_enabled": True,
            "support_levels": sorted(SUPPORT_LEVELS),
        },
        "summary": {
            "n_cases": len(resultados) + 2,
            "citation_validity": sum(item["passed"] for item in validos)
            / len(validos),
            "invalid_claim_rejection_rate": sum(item["passed"] for item in invalidos)
            / len(invalidos),
            "abstention_accuracy": float(abstencao_ok),
            "memory_rejection_accuracy": float(memoria_ok),
            "unsupported_claim_rate_after_guard": 0.0
            if all(item["passed"] for item in invalidos)
            else 1.0,
        },
        "cases": resultados,
        "abstention_case": {"passed": abstencao_ok},
        "memory_case": {"passed": memoria_ok},
    }


def relatorio_guard_markdown(relatorio: Mapping[str, Any]) -> str:
    resumo = relatorio["summary"]
    linhas = [
        "# Evidence RAG — Evidence Guard R5",
        "",
        "> Estado: guarda determinística integrada; promoção do retrieval permanece para R6.",
        "",
        "## Integridade de evidência",
        "",
        "| Métrica | Resultado |",
        "|---|---:|",
        f"| Citation validity | {resumo['citation_validity']:.1%} |",
        f"| Rejeição de claims inválidos | {resumo['invalid_claim_rejection_rate']:.1%} |",
        f"| Acerto de abstenção | {resumo['abstention_accuracy']:.1%} |",
        f"| Rejeição de memória como fonte | {resumo['memory_rejection_accuracy']:.1%} |",
        f"| Unsupported claim rate após guarda | {resumo['unsupported_claim_rate_after_guard']:.1%} |",
        "",
        "## Contrato",
        "",
        "- Cadeia obrigatória: claim → evidence_id → chunk_id → document_id → página → PDF.",
        "- Quotes são conferidas somente contra `raw_text` normalizado.",
        "- Memória, Obsidian e sessões não podem se tornar fonte científica.",
        "- Ausência de evidência produz abstenção explícita.",
        "- Nenhum segundo LLM é usado como verificador documental.",
    ]
    return "\n".join(linhas) + "\n"
