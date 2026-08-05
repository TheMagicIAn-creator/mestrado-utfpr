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
from src.core.logs import get_logger
from src.ml.pipeline import NOMES_ETAPAS, pipeline_status

_logger = get_logger("orquestrador")


def ha_pdfs_novos() -> bool:
    return PASTA_NOVOS_PDFS.exists() and any(PASTA_NOVOS_PDFS.glob("*.pdf"))


def ha_sessoes_para_consolidar(minimo: int = 2) -> bool:
    return PASTA_SESSOES.exists() and len(list(PASTA_SESSOES.glob("*.md"))) >= minimo


def ha_arquivos_com_nome_ruim() -> bool:
    return any(PASTA_LITERATURA.rglob("autor-desconhecido_*.pdf"))


def ha_sinal_reprocessamento() -> bool:
    return (RAIZ_PROJETO / "REPROCESSAR").exists()


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
    Reconcilia pendências, nomes dos PDFs e metadados dos índices.

    A operação usa o cadastro de pendências e também revisita arquivos ``0000``;
    portanto não depende mais apenas do prefixo ``autor-desconhecido``.
    """
    try:
        from pathlib import Path

        from src.conhecimento.indice_portatil import (
            atualizar_metadados_snapshot,
            hash_corpus_pdfs,
        )
        from src.conhecimento.processador_pdf import (
            carregar_metadados_pendentes,
            extrair_metadados_pdf,
            gerar_nome_padronizado,
            metadados_resolvidos,
            salvar_metadados_pendentes,
        )
        from src.core.config import ARQUIVO_INDICE_LITERATURA
        from src.core.tempo import agora_local

        pendencias = carregar_metadados_pendentes()

        def caminho_local(nome: str, info: dict) -> Path | None:
            candidatos = []
            bruto = Path(str(info.get("arquivo", "")))
            partes_minusculas = [parte.lower() for parte in bruto.parts]
            if "literatura" in partes_minusculas:
                posicao = partes_minusculas.index("literatura")
                candidatos.append(PASTA_LITERATURA.joinpath(*bruto.parts[posicao + 1:]))
            if not bruto.is_absolute():
                candidatos.append(RAIZ_PROJETO / bruto)
            candidatos.extend(PASTA_LITERATURA.rglob(nome))
            raiz = PASTA_LITERATURA.resolve()
            for candidato in candidatos:
                try:
                    resolvido = candidato.resolve()
                    resolvido.relative_to(raiz)
                except (OSError, ValueError):
                    continue
                if resolvido.is_file():
                    return resolvido
            return None

        alvos: dict[Path, set[str]] = {}
        for nome, info in pendencias.items():
            if not isinstance(info, dict) or info.get("resolvido"):
                continue
            caminho = caminho_local(nome, info)
            if caminho is not None:
                alvos.setdefault(caminho, set()).add(nome)

        for padrao in ("autor-desconhecido_*.pdf", "*_0000.pdf"):
            for caminho in PASTA_LITERATURA.rglob(padrao):
                alvos.setdefault(caminho.resolve(), set()).add(caminho.name)

        atualizacoes_snapshot: dict[str, dict] = {}
        corrigidos = confirmados_sem_data = ainda_pendentes = 0

        colecao = None
        if PASTA_CHROMADB.is_dir():
            try:
                import chromadb

                cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
                colecao = cliente.get_collection(NOME_COLECAO)
            except Exception:
                colecao = None

        for pdf, chaves_pendencia in sorted(alvos.items(), key=lambda item: str(item[0])):
            nome_antigo = pdf.name
            meta = extrair_metadados_pdf(pdf, registrar_pendencia=False)
            if not metadados_resolvidos(meta):
                ainda_pendentes += 1
                for chave in chaves_pendencia:
                    if chave not in pendencias:
                        pendencias[chave] = {}
                    pendencias[chave].update({
                        "arquivo": pdf.relative_to(RAIZ_PROJETO).as_posix(),
                        "autor_atual": meta.get("autor", ""),
                        "titulo_atual": meta.get("titulo", ""),
                        "ano_atual": meta.get("ano", "0000"),
                        "ultima_verificacao": agora_local().isoformat(timespec="minutes"),
                        "resolvido": False,
                    })
                continue

            nome_novo = gerar_nome_padronizado(
                meta["autor"], meta["titulo"], meta["ano"]
            )
            destino = pdf.parent / nome_novo
            if destino != pdf and destino.exists():
                ainda_pendentes += 1
                continue

            if destino != pdf:
                pdf.rename(destino)
                corrigidos += 1
            if meta.get("ano_confirmado_ausente"):
                confirmados_sem_data += 1

            ano_citacao = meta["ano"] if meta["ano"] != "0000" else "s.d."
            novos_metadados = {
                "arquivo": destino.name,
                "autor": meta["autor"],
                "titulo": meta["titulo"],
                "ano": ano_citacao,
                "citacao": f"{meta['autor']} ({ano_citacao}) — {meta['titulo']}",
            }
            atualizacoes_snapshot[nome_antigo] = novos_metadados

            if colecao is not None:
                try:
                    registros = colecao.get(
                        where={"arquivo_hash": meta.get("arquivo_hash", "")},
                        include=["metadatas"],
                    ) if meta.get("arquivo_hash") else colecao.get(
                        where={"arquivo": nome_antigo}, include=["metadatas"]
                    )
                    ids = registros.get("ids") or []
                    metadados = registros.get("metadatas") or []
                    for inicio in range(0, len(ids), 250):
                        fim = inicio + 250
                        colecao.update(
                            ids=ids[inicio:fim],
                            metadatas=[
                                {**item, **novos_metadados}
                                for item in metadados[inicio:fim]
                            ],
                        )
                except Exception as exc:
                    _logger.warning(
                        "metadados revisados, mas atualização no ChromaDB falhou: %s",
                        exc,
                    )

            for chave in chaves_pendencia:
                if chave not in pendencias:
                    continue
                pendencias[chave].update({
                    "arquivo_final": destino.relative_to(RAIZ_PROJETO).as_posix(),
                    "autor_atual": meta["autor"],
                    "titulo_atual": meta["titulo"],
                    "ano_atual": meta["ano"],
                    "ano_confirmado_ausente": bool(
                        meta.get("ano_confirmado_ausente")
                    ),
                    "ultima_verificacao": agora_local().isoformat(timespec="minutes"),
                    "resolvido": True,
                    "motivo_resolucao": "metadados revisados e propagados",
                })

        orfaos = 0
        for nome, info in pendencias.items():
            if not isinstance(info, dict) or info.get("resolvido"):
                continue
            if caminho_local(nome, info) is None:
                info.update({
                    "resolvido": True,
                    "ultima_verificacao": agora_local().isoformat(timespec="minutes"),
                    "motivo_resolucao": "registro órfão de arquivo já removido ou renomeado",
                })
                orfaos += 1

        salvar_metadados_pendentes(pendencias)

        if atualizacoes_snapshot and ARQUIVO_INDICE_LITERATURA.is_file():
            hash_corpus, n_documentos = hash_corpus_pdfs(PASTA_LITERATURA)
            atualizar_metadados_snapshot(
                ARQUIVO_INDICE_LITERATURA,
                atualizacoes_snapshot,
                hash_corpus=hash_corpus,
                n_documentos=n_documentos,
            )

        return (
            f"Metadados: {corrigidos} arquivo(s) renomeado(s), "
            f"{confirmados_sem_data} fonte(s) confirmada(s) como s.d., "
            f"{orfaos} registro(s) órfão(s) encerrado(s) e "
            f"{ainda_pendentes} pendência(s) restante(s)."
        )
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
