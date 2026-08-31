from src.conhecimento.evidence_graph import (
    caminhos_ancorados,
    construir_evidence_graph,
    consulta_relacional,
    resumir_grafo_para_prompt,
)


TAXONOMIA = {
    "schema_version": 1,
    "entities": [
        {"type": "component", "name": "IGBT", "aliases": ["igbt"]},
        {"type": "method", "name": "FMECA", "aliases": ["fmeca"]},
    ],
}


def _pacote():
    return {
        "evidences": [
            {
                "evidence_id": "E1",
                "document_id": "doc-1",
                "chunk_id": "doc-1:p2:c1",
                "file": "fonte.pdf",
                "authors": ["Silva"],
                "raw_text": "A FMECA estuda modos de falha do IGBT.",
                "source_type": "scientific_pdf",
            }
        ]
    }


def test_grafo_so_cria_relacoes_ancoradas_em_evidencia():
    grafo = construir_evidence_graph(_pacote(), TAXONOMIA)
    assert grafo["status"] == "pilot_not_primary_retrieval"
    assert any(node["name"] == "IGBT" for node in grafo["nodes"])
    assert any(node["name"] == "FMECA" for node in grafo["nodes"])
    assert all(edge["evidence_ids"] == ["E1"] for edge in grafo["edges"])
    assert all(edge["chunk_ids"] == ["doc-1:p2:c1"] for edge in grafo["edges"])


def test_entidade_ausente_nao_e_inferida():
    pacote = _pacote()
    pacote["evidences"][0]["raw_text"] = "O componente foi analisado."
    grafo = construir_evidence_graph(pacote, TAXONOMIA)
    assert not any(node["type"] in {"component", "method"} for node in grafo["nodes"])


def test_memoria_nao_entra_no_grafo_cientifico():
    pacote = _pacote()
    pacote["evidences"][0]["source_type"] = "session"
    grafo = construir_evidence_graph(pacote, TAXONOMIA)
    assert grafo["nodes"] == []
    assert grafo["edges"] == []


def test_roteamento_relacional_e_conservador():
    assert consulta_relacional("Relacione FMECA com IGBT") is True
    assert consulta_relacional("Onde o artigo define FMECA?") is False


def test_caminho_ignora_aresta_sem_lastro_e_resumo_preserva_id():
    grafo = construir_evidence_graph(_pacote(), TAXONOMIA)
    autor = next(node["id"] for node in grafo["nodes"] if node["name"] == "Silva")
    igbt = next(node["id"] for node in grafo["nodes"] if node["name"] == "IGBT")
    assert caminhos_ancorados(grafo, autor, igbt, max_hops=2)
    grafo["edges"].append({"source": autor, "relation": "USES", "target": "x"})
    assert caminhos_ancorados(grafo, autor, "x", max_hops=2) == []
    assert "[E1]" in resumir_grafo_para_prompt(grafo)
