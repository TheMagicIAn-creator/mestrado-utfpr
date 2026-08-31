from src.conhecimento.evidence_guard import (
    ABSTENTION_MESSAGE,
    construir_evidence_package,
    executar_benchmark_guard,
    renderizar_restricao_pacote,
    resposta_segura,
    validar_claims,
    validar_resposta,
)


def _pacote():
    return construir_evidence_package(
        "Como funciona?",
        [
            (
                "A confiabilidade é a probabilidade de cumprir a função requerida.",
                {
                    "_chunk_id": "doc:abc:p12:c01",
                    "arquivo_hash": "abc",
                    "arquivo": "fonte.pdf",
                    "titulo": "Confiabilidade",
                    "autor": "Silva",
                    "ano": "2022",
                    "pagina_inicio": 12,
                    "pagina_fim": 12,
                    "_rrf_score": 0.4,
                },
            )
        ],
    )


def test_pacote_preserva_cadeia_documental_e_renderiza_id():
    pacote = _pacote()
    evidencia = pacote["evidences"][0]
    assert evidencia["evidence_id"] == "E1"
    assert evidencia["chunk_id"] == "doc:abc:p12:c01"
    assert evidencia["document_id"] == "abc"
    assert evidencia["pages"] == [12]
    assert "[E1]" in renderizar_restricao_pacote(pacote)


def test_pacote_rejeita_memoria_como_evidencia_cientifica():
    pacote = construir_evidence_package(
        "pergunta",
        [("fala antiga", {"source_type": "obsidian", "chunk_id": "x"})],
    )
    assert pacote["evidences"] == []
    assert pacote["abstention_required"] is True


def test_claim_direto_valida_autoria_pagina_ano_e_quote():
    auditoria = validar_claims(
        [
            {
                "claim": "Definição de confiabilidade",
                "evidence_ids": ["E1"],
                "support": "direct",
                "author": "Silva",
                "year": 2022,
                "page": 12,
                "quote": "probabilidade de cumprir a função requerida",
            }
        ],
        _pacote(),
    )
    assert auditoria["valid"] is True
    assert auditoria["claims_valid"] == 1


def test_claim_inventado_e_quote_parafraseado_sao_bloqueados():
    auditoria = validar_claims(
        [
            {
                "claim": "Texto sem lastro",
                "evidence_ids": ["E99"],
                "support": "direct",
                "page": 99,
                "quote": "frase que não existe",
            }
        ],
        _pacote(),
    )
    assert auditoria["valid"] is False
    assert "unknown_evidence_id" in auditoria["results"][0]["errors"]


def test_resposta_com_id_inexistente_abstem_em_vez_de_publicar():
    assert validar_resposta("Afirmação [E99]", _pacote())["valid"] is False
    assert resposta_segura("Afirmação [E99]", _pacote()) == ABSTENTION_MESSAGE


def test_resposta_academica_sem_marcador_de_evidencia_abstem():
    resultado = validar_resposta("Afirmação sem vínculo", _pacote())
    assert resultado["missing_evidence_marker"] is True
    assert resposta_segura("Afirmação sem vínculo", _pacote()) == ABSTENTION_MESSAGE


def test_pacote_vazio_exige_abstencao():
    vazio = construir_evidence_package("sem fonte", [])
    assert resposta_segura("Resposta confiante", vazio) == ABSTENTION_MESSAGE


def test_benchmark_guard_rejeita_todos_os_casos_adversariais():
    relatorio = executar_benchmark_guard(git_revision="teste")
    assert relatorio["summary"] == {
        "n_cases": 9,
        "citation_validity": 1.0,
        "invalid_claim_rejection_rate": 1.0,
        "abstention_accuracy": 1.0,
        "memory_rejection_accuracy": 1.0,
        "unsupported_claim_rate_after_guard": 0.0,
    }
