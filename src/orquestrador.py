"""
orquestrador.py - Al IAdo PV
Coordenacao leve do backend.

Na inicializacao do Streamlit, este modulo executa apenas tarefas rapidas:
reprocessamento completo se houver sinal explicito e indexacao de PDFs novos.
Pipeline de ML e resultados sao acionados por prompt via ferramentas.
"""

from __future__ import annotations

from src.core.config import (
    MODELO_EMBEDDINGS,
    NOME_COLECAO,
    PASTA_CHROMADB,
    PASTA_LITERATURA,
    PASTA_NOVOS_PDFS,
    PASTA_RESULTADOS,
    PASTA_SESSOES,
    RAIZ_PROJETO,
)
from src.ml.pipeline import (
    ARTEFATOS_ML,
    NOMES_ETAPAS,
    ORDEM_ETAPAS_ML,
)

# Reexports do pipeline de ML usados por codigo legado.
from src.ml.pipeline import (  # noqa: F401
    autoencoder_pendente,
    etapa_pendente,
    executar_etapa,
    executar_pipeline_ml,
    features_ca_pendente,
    injecao_falhas_pendente,
    limpar_artefatos,
    pipeline_status,
    regenerar_pipeline,
    rul_weibull_pendente,
    validacao_pendente,
)
from src.ml.resultados import indexar_resultados_ml  # noqa: F401


def ha_pdfs_novos() -> bool:
    return PASTA_NOVOS_PDFS.exists() and any(PASTA_NOVOS_PDFS.glob("*.pdf"))


def ha_sessoes_para_consolidar(minimo: int = 2) -> bool:
    return PASTA_SESSOES.exists() and len(list(PASTA_SESSOES.glob("*.md"))) >= minimo


def ha_arquivos_com_nome_ruim() -> bool:
    return any(PASTA_LITERATURA.rglob("autor-desconhecido_*.pdf"))


def ha_sinal_reprocessamento() -> bool:
    return (RAIZ_PROJETO / "REPROCESSAR").exists()


def eda_pendente() -> bool:
    pasta = PASTA_RESULTADOS / "eda"
    return not pasta.exists() or not any(pasta.iterdir())


def classificacao_pendente() -> bool:
    # Caminho legado: a classificacao PV oficial agora e o experimento Ghoneim
    # em src/ml/experimentos_artigos.py.
    return False


def etapa_indexar_pdfs(modelo_embeddings) -> str:
    if not ha_pdfs_novos():
        return "PDFs novos: nenhum pendente"
    try:
        from src.conhecimento.processador_pdf import processar_pasta

        processar_pasta(PASTA_NOVOS_PDFS, modelo_embeddings, PASTA_CHROMADB)
        return "PDFs novos: indexados com sucesso"
    except Exception as exc:
        return f"PDFs novos: erro - {exc}"


def reprocessar_metadados_ruins() -> str:
    """
    Corrige PDFs com 'autor-desconhecido' no nome.
    Operacao pesada, chamada sob demanda pela manutencao.
    """
    ruins = list(PASTA_LITERATURA.rglob("autor-desconhecido_*.pdf"))
    if not ruins:
        return "Metadados: todos os arquivos com nome correto"

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from src.conhecimento.indexador import indexar_pdf_unico
        from src.conhecimento.processador_pdf import (
            extrair_metadados_pdf,
            gerar_nome_padronizado,
        )

        modelo = SentenceTransformer(MODELO_EMBEDDINGS)
        client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = client.get_or_create_collection(NOME_COLECAO)
        corrigidos = 0

        for pdf in ruins:
            meta = extrair_metadados_pdf(pdf)
            autor = meta["autor"]
            if autor == "autor-desconhecido":
                continue

            nome_novo = gerar_nome_padronizado(
                autor, meta["titulo"], meta["ano"]
            )
            destino = pdf.parent / nome_novo
            if destino.exists() and destino != pdf:
                continue

            res = colecao.get(where={"arquivo": pdf.name}, include=["metadatas"])
            ids_antigos = res.get("ids", [])
            if ids_antigos:
                colecao.delete(ids=ids_antigos)

            pdf.rename(destino)
            indexar_pdf_unico(destino, modelo, PASTA_CHROMADB)
            corrigidos += 1

        return f"Metadados: {corrigidos} arquivo(s) corrigido(s)"
    except Exception as exc:
        return f"Metadados: erro no reprocessamento - {exc}"


def reprocessar_literatura() -> str:
    """
    Reprocessa toda a literatura quando o arquivo REPROCESSAR existe.
    """
    sinal = RAIZ_PROJETO / "REPROCESSAR"
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm
        from src.conhecimento.indexador import indexar_pdf_unico
        from src.conhecimento.processador_pdf import (
            classificar_tema,
            extrair_metadados_pdf,
            extrair_texto_pdf,
            gerar_nome_padronizado,
            gerar_nota_obsidian,
        )

        modelo = SentenceTransformer(MODELO_EMBEDDINGS)
        client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = client.get_or_create_collection(NOME_COLECAO)

        sucesso = renomeados = falhas = 0
        relatorio = ["REPROCESSAMENTO COMPLETO - Al IAdo PV", "=" * 60, ""]

        for pdf in tqdm(sorted(PASTA_LITERATURA.rglob("*.pdf")), desc="PDFs"):
            try:
                meta = extrair_metadados_pdf(pdf)
                nome_novo = gerar_nome_padronizado(
                    meta["autor"], meta["titulo"], meta["ano"]
                )
                destino = pdf.parent / nome_novo

                res = colecao.get(where={"arquivo": pdf.name}, include=["metadatas"])
                ids_antigos = res.get("ids", [])
                if ids_antigos:
                    colecao.delete(ids=ids_antigos)

                nome_ruim = (
                    "autor-desconhecido" in pdf.name
                    or pdf.name.startswith(("p_", "empresa_", "data-set_", "design_"))
                    or pdf.name.startswith(("for-facilities_", "with-ua_", "academic-editor_"))
                )
                if nome_ruim and destino.name != pdf.name:
                    if destino.exists():
                        destino = pdf.parent / nome_novo.replace(".pdf", "_v2.pdf")
                    pdf.rename(destino)
                    renomeados += 1
                else:
                    destino = pdf

                res = indexar_pdf_unico(destino, modelo, PASTA_CHROMADB)
                texto = extrair_texto_pdf(destino)
                tema = classificar_tema(destino.name, texto)
                gerar_nota_obsidian(
                    destino.name, meta["autor"], meta["titulo"], meta["ano"], tema, texto
                )
                relatorio.append(f"OK  {pdf.name[:48]} ({res.get('n_chunks', 0)} chunks)")
                sucesso += 1
            except Exception as exc:
                relatorio.append(f"ERRO {pdf.name[:48]} - {exc}")
                falhas += 1

        relatorio += [
            "",
            "=" * 60,
            f"Sucesso   : {sucesso}",
            f"Renomeados: {renomeados}",
            f"Falha     : {falhas}",
        ]
        arquivo_rel = PASTA_RESULTADOS / "reprocessamento.txt"
        arquivo_rel.parent.mkdir(parents=True, exist_ok=True)
        arquivo_rel.write_text("\n".join(relatorio), encoding="utf-8")

        sinal.unlink(missing_ok=True)
        return f"Reprocessamento: {sucesso} OK | {renomeados} renomeados | {falhas} falhas"
    except Exception as exc:
        return f"Reprocessamento: erro - {exc}"


def executar_pipeline(modelo_embeddings) -> list[str]:
    relatorio = []
    if ha_sinal_reprocessamento():
        relatorio.append(reprocessar_literatura())
    relatorio.append(etapa_indexar_pdfs(modelo_embeddings))
    return relatorio


# Compatibilidade com nomes antigos.
def etapa_eda() -> str:
    if not eda_pendente():
        return "EDA: ja realizada"
    try:
        from src.ml.eda import executar_eda

        return "EDA: gerada com sucesso" if executar_eda() else "EDA: falhou"
    except Exception as exc:
        return f"EDA: erro - {exc}"


def etapa_classificacao() -> str:
    return (
        "Classificacao PV legada depreciada. Use o experimento 'ghoneim' "
        "em src/ml/experimentos_artigos.py."
    )


def etapa_features_ca() -> str:
    return executar_etapa("features_ca")["mensagem"]


def etapa_autoencoder() -> str:
    return executar_etapa("autoencoder")["mensagem"]


def etapa_injecao_falhas() -> str:
    return executar_etapa("injecao_falhas")["mensagem"]


def etapa_validacao_ml() -> str:
    return executar_etapa("validacao")["mensagem"]


def etapa_rul_weibull() -> str:
    return executar_etapa("rul_weibull")["mensagem"]


if __name__ == "__main__":
    print("=" * 60)
    print("  AL IADO PV - ORQUESTRADOR")
    print("=" * 60)
    print(f"Sinal REPROCESSAR         : {ha_sinal_reprocessamento()}")
    print(f"PDFs novos pendentes      : {ha_pdfs_novos()}")
    print(f"Sessoes para consolidar   : {ha_sessoes_para_consolidar()}")
    print(f"Arquivos com nome ruim    : {ha_arquivos_com_nome_ruim()}")
    for etapa, pronto in pipeline_status().items():
        print(f"{NOMES_ETAPAS[etapa]:<26}: {'pronto' if pronto else 'pendente'}")
    print("=" * 60)
