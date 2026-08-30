"""Renderização textual dos benchmarks versionados de recuperação."""

from __future__ import annotations

from collections import Counter


def _linhas_comparacao(resultado: dict) -> list[str]:
    comparacao = resultado.get("comparison_to_baseline")
    if not comparacao:
        return []
    stage = resultado.get("stage")
    baseline_label = "R2" if stage == "R3" else "R0–R1"
    candidate_label = stage if stage in {"R2", "R3"} else "candidato"
    linhas = [
        "## Comparação baseline x candidato",
        "",
        f"| Métrica | {baseline_label} | {candidate_label} | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metrica, valores in comparacao["metrics"].items():
        linhas.append(
            f"| {metrica} | {float(valores['baseline']):.6f} | "
            f"{float(valores['candidate']):.6f} | {float(valores['delta']):+.6f} |"
        )
    aprovado_r2 = all(
        comparacao[campo]
        for campo in (
            "corpus_identity_preserved",
            "ranking_contract_preserved",
            "scientific_metrics_identical",
        )
    )
    linhas.extend(
        [
            "",
            f"- Identidade do corpus preservada: {str(comparacao['corpus_identity_preserved']).lower()}.",
            f"- Contrato de ranking preservado: {str(comparacao['ranking_contract_preserved']).lower()}.",
            f"- Métricas científicas idênticas: {str(comparacao['scientific_metrics_identical']).lower()}.",
        ]
    )
    if stage == "R3":
        linhas.extend(
            [
                f"- Ganho de qualidade observado: {str(comparacao['quality_gain_observed']).lower()}.",
                "- Consultas com regressão de Recall@5: "
                + (", ".join(comparacao["regressed_queries_at_5"]) or "nenhuma")
                + ".",
                "- Regressões críticas em perguntas simples: "
                + (", ".join(comparacao["critical_simple_regressions"]) or "nenhuma")
                + ".",
                "- Decisão: candidato não promovido; avaliação continua em R4–R6.",
                "- Latência é informativa e não participa do gate científico de qualidade.",
                "",
            ]
        )
        detalhes = comparacao["regression_details"]
        if detalhes:
            linhas.extend(["### Regressões auditadas", ""])
            linhas.extend(_linha_regressao(item) for item in detalhes)
            linhas.append("")
    else:
        linhas.extend(
            [
                "- Latência é informativa e não participa do gate científico de R2.",
                f"- Gate R2: {'APROVADO' if aprovado_r2 else 'REPROVADO'}.",
                "",
            ]
        )
    return linhas


def _linha_regressao(item: dict) -> str:
    return (
        f"- `{item['query_id']}` ({item['category']}): Recall de página@5 "
        f"{item['baseline_page_recall_at_5']:.1f}→"
        f"{item['candidate_page_recall_at_5']:.1f}; Recall documental R3="
        f"{item['candidate_document_recall_at_5']:.1f}; top-1 R2 "
        f"`{item['baseline_top_1']['file']}` p.{item['baseline_top_1']['pages']}; "
        f"top-1 R3 `{item['candidate_top_1']['file']}` p."
        f"{item['candidate_top_1']['pages']}."
    )


def _estado_relatorio(stage: str, schema_v2: bool) -> tuple[str, str, str, str]:
    if stage == "R3":
        return (
            "# Evidence RAG — Contextual Retrieval determinístico R3",
            "> Estado: candidato contextual paralelo medido e não promovido; "
            "gold set provisório e pendente de revisão do pesquisador em R6.",
            "`raw_text` preservado, `retrieval_text` contextualizado e embeddings "
            "recalculados com o mesmo encoder.",
            "- O contexto usa apenas título, autores, ano, coleção, página e idioma "
            "observados; nenhuma LLM inferiu ou resumiu conteúdo.",
        )
    if schema_v2:
        return (
            "# Evidence RAG — JSONL schema v2 R2",
            "> Estado: contrato JSONL v2 medido sem contextualização; gold set "
            "provisório e pendente de revisão do pesquisador em R6.",
            "campos `raw_text` e `retrieval_text` separados, ainda idênticos em R2, "
            "com embeddings preservados por chunk.",
            "- O schema v2 separa os contratos de texto sem acrescentar contexto; "
            "`retrieval_text` permanece idêntico a `raw_text` nesta etapa.",
        )
    return (
        "# Evidence RAG — baseline R0–R1",
        "> Estado: baseline vigente medido; gold set provisório e pendente de "
        "revisão do pesquisador em R6.",
        "texto bruto no campo `documento` e embeddings armazenados por chunk.",
        "- O schema v1 ainda não separa `raw_text` de `retrieval_text` e não "
        "registra contexto do chunk.",
    )


def relatorio_markdown(resultado: dict) -> str:
    resumo = resultado["summary"]
    retrieval = resultado["retrieval"]
    corpus = resultado["corpus"]
    falhas = [
        item
        for item in resultado["queries"]
        if item["expected_behavior"] == "retrieve"
        and not item["metrics"]["5"]["hit_rate@5"]
    ]
    categorias = Counter(item["category"] for item in resultado["queries"])
    abstencoes = resumo["n_future_abstention_queries"]
    rotulo_abstencao = "reservada" if abstencoes == 1 else "reservadas"
    stage = resultado.get("stage", "R0-R1")
    schema_v2 = int(corpus["snapshot_schema_version"]) >= 2
    titulo, estado, descricao_snapshot, estado_contrato = _estado_relatorio(
        stage, schema_v2
    )
    limitacao_snapshot = (
        "- O schema v2 registra estratégia e hash de conteúdo, mas tamanho e overlap "
        "do snapshot legado permanecem desconhecidos; nenhum valor foi inferido."
        if schema_v2
        else "- O snapshot v1 não registra tamanho/overlap de chunk nem proveniência do texto contextual."
    )
    hash_conteudo = corpus.get("snapshot_content_hash_sha256")
    linhas = [
        titulo,
        "",
        estado,
        "",
        f"## Auditoria {stage}",
        "",
        f"- Corpus: {corpus['n_documents']} PDFs e {corpus['n_chunks']} chunks.",
        f"- Snapshot portátil: schema v{corpus['snapshot_schema_version']}; "
        + descricao_snapshot,
        f"- Hash SHA-256 do corpus: `{corpus['hash_sha256']}`.",
        *(
            [f"- Hash de texto/embedding do snapshot: `{hash_conteudo}`."]
            if hash_conteudo
            else []
        ),
        f"- Índice semântico: {retrieval['semantic_index']} com `{retrieval['embedding_model']}` "
        f"na revisão `{retrieval['embedding_revision']}`.",
        f"- Backend de indexação: {retrieval['index_embedding_backend']}; "
        f"backend de consulta equivalente: {retrieval['query_embedding_backend']}.",
        f"- Índice lexical: {retrieval['lexical_index']} (disponível: "
        f"{str(retrieval['lexical_available']).lower()}).",
        "- Fusão: Reciprocal Rank Fusion com constante 60; reranking e diversificação locais vigentes.",
        "- IDs: SHA-256 documental mais índice ordinal do chunk; páginas preservadas nos metadados.",
        estado_contrato,
        "- O caminho atual não possui Evidence Package nem Evidence Guard determinístico.",
        "- Nenhum parâmetro de ranking foi modificado nesta etapa.",
        "",
        "## Gold set R1",
        "",
        f"- Perguntas: {resumo['n_queries']} ({resumo['n_retrieval_queries']} recuperáveis e "
        f"{abstencoes} {rotulo_abstencao} à futura avaliação de abstenção).",
        "- Categorias: "
        + ", ".join(
            f"{nome}={quantidade}"
            for nome, quantidade in sorted(categorias.items())
        )
        + ".",
        "- Todas as evidências provisórias foram validadas contra arquivo, hash, página e chunk do snapshot.",
        "- O conjunto não é verdade final: a promoção R6 permanece bloqueada até revisão humana do pesquisador.",
        "",
        "## Métricas de retrieval",
        "",
        "| Métrica | k=5 | k=8 |",
        "|---|---:|---:|",
        f"| Recall por página@k | {resumo['recall@5']:.4f} | {resumo['recall@8']:.4f} |",
        f"| Recall por chunk exato@k | {resumo['strict_chunk_recall@5']:.4f} | "
        f"{resumo['strict_chunk_recall@8']:.4f} |",
        f"| Recall por documento@k | {resumo['document_recall@5']:.4f} | "
        f"{resumo['document_recall@8']:.4f} |",
        f"| Precision@k | {resumo['precision@5']:.4f} | {resumo['precision@8']:.4f} |",
        f"| Hit Rate@k | {resumo['hit_rate@5']:.4f} | {resumo['hit_rate@8']:.4f} |",
        f"| MRR@k | {resumo['mrr@5']:.4f} | {resumo['mrr@8']:.4f} |",
        f"| nDCG@k | {resumo['ndcg@5']:.4f} | {resumo['ndcg@8']:.4f} |",
        "",
        f"- Latência aquecida média: {resumo['latency_ms_mean']:.1f} ms; "
        f"p50={resumo['latency_ms_p50']:.1f} ms; p95={resumo['latency_ms_p95']:.1f} ms.",
        f"- Contexto médio no maior k: {resumo['context_chars_mean_at_max_k']:.0f} caracteres.",
        f"- Consultas recuperáveis sem acerto no top-5: {len(falhas)}.",
        "",
        "A diferença entre Recall documental e Recall por página mostra que o baseline "
        "frequentemente localiza a fonte correta, mas não a passagem citável correta. "
        "O Recall por chunk exato permanece como controle estrito das fronteiras de segmentação.",
        "",
        *_linhas_comparacao(resultado),
        "## Diagnóstico por categoria",
        "",
        "| Categoria | Perguntas | Recall página@5 | Recall documento@5 | Latência média (ms) |",
        "|---|---:|---:|---:|---:|",
        *(
            f"| {categoria} | {dados['n_queries']} | {dados['recall@5']:.4f} | "
            f"{dados['document_recall@5']:.4f} | {dados['latency_ms_mean']:.1f} |"
            for categoria, dados in sorted(resultado["by_category"].items())
        ),
        "",
        "## Limitações e próximo gate",
        "",
        "- Métricas de retrieval não medem fidelidade da resposta gerada.",
        limitacao_snapshot,
        "- A expansão vigente possui regras temáticas manuais e ainda associa Paderborn a Stender; "
        "não existe regra ou fonte bibliográfica direta para os 16 ensaios GPVS-Faults.",
        "- A recuperação não aplica filtro de página e ainda não valida citações após a geração.",
        "- Perguntas de abstenção serão pontuadas apenas após o Evidence Guard (R5).",
        "- O baseline deve permanecer disponível para comparação e rollback durante R2–R6.",
        "- Contextual Retrieval só poderá ser promovido após ganho mensurável e ausência de regressão crítica.",
        "",
    ]
    if falhas:
        linhas.extend(
            [
                "### Consultas sem acerto no top-5",
                "",
                *(f"- `{item['query_id']}`: {item['question']}" for item in falhas),
                "",
            ]
        )
    return "\n".join(linhas)
