"""
orquestrador.py — Al IAdo PV
Coordena a execução do fluxo completo no backend.

Verifica o estado do projeto e executa apenas as etapas
pendentes ou desatualizadas, na ordem correta de dependência.

É chamado pelo app.py na inicialização.

Autor: Rodolfo Torres (UTFPR)
"""

from pathlib import Path

from src.core.config import (
    PASTA_NOVOS_PDFS, PASTA_SESSOES, PASTA_RESULTADOS,
    PASTA_CHROMADB,RAIZ_PROJETO,
)


# ============================================================
# FUNÇÕES DE VERIFICAÇÃO DE ESTADO
# ============================================================

def ha_pdfs_novos() -> bool:
    """Verifica se há PDFs aguardando indexação em novos_pdfs/."""
    if not PASTA_NOVOS_PDFS.exists():
        return False
    return any(PASTA_NOVOS_PDFS.glob("*.pdf"))


def ha_sessoes_para_consolidar(minimo: int = 2) -> bool:
    """Verifica se há acúmulo de sessões que justifique consolidação."""
    if not PASTA_SESSOES.exists():
        return False
    sessoes = list(PASTA_SESSOES.glob("*.md"))
    return len(sessoes) >= minimo


def eda_pendente() -> bool:
    """Verifica se a análise exploratória ainda não foi gerada."""
    pasta_eda = PASTA_RESULTADOS / "eda"
    if not pasta_eda.exists():
        return True
    return not any(pasta_eda.iterdir())


def classificacao_pendente() -> bool:
    """Verifica se a classificação de ML ainda não foi gerada."""
    pasta_clf = PASTA_RESULTADOS / "classificacao_pv"
    if not pasta_clf.exists():
        return True
    return not any(pasta_clf.iterdir())


# ============================================================
# EXECUÇÃO DE CADA ETAPA
# ============================================================

def etapa_indexar_pdfs(modelo_embeddings) -> str:
    """Indexa PDFs novos, se houver."""
    if not ha_pdfs_novos():
        return "PDFs novos: nenhum pendente"

    try:
        from src.conhecimento.processador_pdf import processar_pasta
        processar_pasta(PASTA_NOVOS_PDFS, modelo_embeddings,PASTA_CHROMADB)
        return "PDFs novos: indexados com sucesso"
    except Exception as e:
        return f"PDFs novos: erro — {e}"


def etapa_consolidar_memoria() -> dict:
    """Consolida memória se algum gatilho estiver ativo."""
    from src.conhecimento.consolidar_memoria import consolidar, deve_consolidar

    deve, motivo = deve_consolidar()
    if not deve:
        return {"executou": False, "motivo": "nenhum gatilho ativo"}

    try:
        sucesso = consolidar()
        if sucesso:
            return {"executou": True, "motivo": motivo}
        else:
            return {"executou": False, "motivo": "sessões insuficientes"}
    except Exception as e:
        return {"executou": False, "erro": str(e)}


def etapa_eda() -> str:
    """Roda a análise exploratória, se ainda não foi feita."""
    if not eda_pendente():
        return "EDA: já realizada (resultados existentes)"

    try:
        from src.ml.eda import executar_eda
        executar_eda()
        return "EDA: gerada com sucesso"
    except Exception as e:
        return f"EDA: erro — {e}"


def etapa_classificacao() -> str:
    """Roda a classificação de ML, se ainda não foi feita."""
    if not classificacao_pendente():
        return "Classificação: já realizada (resultados existentes)"

    try:
        from src.ml.classificador_pv import executar_classificacao
        executar_classificacao()
        return "Classificação: gerada com sucesso"
    except Exception as e:
        return f"Classificação: erro — {e}"

def ha_arquivos_com_nome_ruim() -> bool:
    """Verifica se há PDFs com autor-desconhecido no nome em literatura/."""
    from src.core.config import PASTA_LITERATURA
    return any(PASTA_LITERATURA.rglob("autor-desconhecido_*.pdf"))

def ha_metadados_pendentes() -> bool:
    """Verifica se há PDFs com metadados não resolvidos."""
    import json as _json
    arquivo = RAIZ_PROJETO / "metadados_pendentes.json"
    if not arquivo.exists():
        return False
    try:
        pendencias = _json.loads(arquivo.read_text(encoding="utf-8"))
        return any(not p.get("resolvido", False) for p in pendencias.values())
    except Exception:
        return False

def reprocessar_metadados_ruins() -> str:
    """
    Detecta PDFs com 'autor-desconhecido' no nome dentro de literatura/,
    extrai metadados corretos via LLM, renomeia o arquivo e reindexa.
    Totalmente automático — sem curadoria manual.
    """
    from src.core.config import PASTA_LITERATURA, PASTA_CHROMADB, MODELO_EMBEDDINGS

    # Encontra arquivos com nome ruim
    ruins = list(PASTA_LITERATURA.rglob("autor-desconhecido_*.pdf"))

    if not ruins:
        return "Metadados: todos os arquivos estão com nome correto"

    print(f"\n🔧 Reprocessando {len(ruins)} arquivo(s) com metadados ruins...")

    try:
        from sentence_transformers import SentenceTransformer
        from src.conhecimento.processador_pdf import (
            extrair_metadados_pdf,
            gerar_nome_padronizado,
        )
        from src.conhecimento.indexador import indexar_pdf_unico
        import chromadb

        modelo  = SentenceTransformer(MODELO_EMBEDDINGS)
        client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = client.get_or_create_collection("literatura_pv")

        corrigidos = 0

        for pdf in ruins:
            print(f"   📄 {pdf.name}")

            # 1. Extrai metadados corretos via LLM
            meta   = extrair_metadados_pdf(pdf)
            autor  = meta["autor"]
            titulo = meta["titulo"]
            ano    = meta["ano"]

            # Se ainda ficou desconhecido, pula — LLM não conseguiu
            if autor == "autor-desconhecido":
                print(f"      ⚠️  LLM não resolveu — mantendo para revisão manual")
                continue

            # 2. Gera nome correto
            nome_novo   = gerar_nome_padronizado(autor, titulo, ano)
            pasta       = pdf.parent
            caminho_novo = pasta / nome_novo

            # Evita sobrescrever arquivo existente com nome diferente
            if caminho_novo.exists() and caminho_novo != pdf:
                print(f"      ⚠️  Arquivo destino já existe: {nome_novo} — pulando")
                continue

            # 3. Remove chunks antigos do ChromaDB
            resultado = colecao.get(
                where   = {"arquivo": pdf.name},
                include = ["metadatas"]
            )
            ids_antigos = resultado.get("ids", [])
            if ids_antigos:
                colecao.delete(ids=ids_antigos)

            # 4. Renomeia o arquivo
            pdf.rename(caminho_novo)
            print(f"      → {nome_novo}")

            # 5. Reindexa com nome e metadados corretos
            res = indexar_pdf_unico(caminho_novo, modelo, PASTA_CHROMADB)
            chunks = res.get("n_chunks", 0)
            print(f"      ✅ {chunks} chunks reindexados")

            corrigidos += 1

        return f"Metadados: {corrigidos} arquivo(s) corrigido(s) automaticamente"

    except Exception as e:
        return f"Metadados: erro no reprocessamento — {e}"

def ha_sinal_reprocessamento() -> bool:
    """Verifica se existe o arquivo REPROCESSAR na raiz do projeto."""
    return (RAIZ_PROJETO / "REPROCESSAR").exists()


def reprocessar_literatura() -> str:
    """
    Reprocessa todos os PDFs da literatura com extração de metadados
    via LLM, renomeia arquivos com nomes corretos e reindexa no ChromaDB.
    Disparado pela presença do arquivo REPROCESSAR na raiz do projeto.
    """
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer
    from src.core.config import MODELO_EMBEDDINGS, PASTA_LITERATURA
    from src.conhecimento.processador_pdf import (
        extrair_metadados_pdf,
        gerar_nome_padronizado,
        classificar_tema,
        gerar_nota_obsidian,
        extrair_texto_pdf,
    )
    from src.conhecimento.indexador import indexar_pdf_unico
    import chromadb

    sinal = RAIZ_PROJETO / "REPROCESSAR"

    print("\n🔄 Sinal REPROCESSAR detectado — iniciando reprocessamento completo...")

    try:
        modelo  = SentenceTransformer(MODELO_EMBEDDINGS)
        client  = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = client.get_or_create_collection("literatura_pv")

        pdfs      = sorted(PASTA_LITERATURA.rglob("*.pdf"))
        sucesso   = 0
        renomeados = 0
        falha     = 0

        relatorio = ["REPROCESSAMENTO COMPLETO — Al IAdo PV", "=" * 60, ""]

        for pdf in tqdm(pdfs, desc="Reprocessando", unit="PDF"):
            try:
                # 1. Extrai metadados via LLM
                meta   = extrair_metadados_pdf(pdf)
                autor  = meta["autor"]
                titulo = meta["titulo"]
                ano    = meta["ano"]

                # 2. Gera nome correto
                nome_novo    = gerar_nome_padronizado(autor, titulo, ano)
                pasta_atual  = pdf.parent
                caminho_novo = pasta_atual / nome_novo

                # 3. Remove chunks antigos
                res = colecao.get(
                    where   = {"arquivo": pdf.name},
                    include = ["metadatas"]
                )
                ids_antigos = res.get("ids", [])
                if ids_antigos:
                    colecao.delete(ids=ids_antigos)

                    # 4. Renomeia apenas se o nome atual indica problema
                    # Arquivos já bem nomeados são preservados — só reindexados
                    nome_tem_problema = (
                            "autor-desconhecido" in pdf.name or
                            pdf.name.startswith("p_") or
                            pdf.name.startswith("empresa_") or
                            pdf.name.startswith("data-set_") or
                            pdf.name.startswith("design_") or
                            pdf.name.startswith("for-facilities_") or
                            pdf.name.startswith("with-ua_") or
                            pdf.name.startswith("academic-editor_")
                    )

                    if nome_tem_problema and nome_novo != pdf.name:
                        if caminho_novo.exists():
                            caminho_novo = pasta_atual / nome_novo.replace(".pdf", "_v2.pdf")
                        pdf.rename(caminho_novo)
                        renomeados += 1
                    else:
                        caminho_novo = pdf
                        if not nome_tem_problema:
                            print(f"        → nome preservado (já correto)")

                # 5. Reindexa
                res    = indexar_pdf_unico(caminho_novo, modelo, PASTA_CHROMADB)
                chunks = res.get("n_chunks", 0)

                # 6. Atualiza nota Obsidian
                texto = extrair_texto_pdf(caminho_novo)
                tema  = classificar_tema(nome_novo, texto)
                gerar_nota_obsidian(nome_novo, autor, titulo, ano, tema, texto)

                relatorio.append(f"✅ {pdf.name[:50]} → {nome_novo[:50]} ({chunks} chunks)")
                sucesso += 1

            except Exception as e:
                relatorio.append(f"❌ {pdf.name[:50]} — {e}")
                falha += 1

        # Salva relatório
        relatorio += [
            "", "=" * 60,
            f"Sucesso   : {sucesso}",
            f"Renomeados: {renomeados}",
            f"Falha     : {falha}",
        ]
        arquivo_rel = PASTA_RESULTADOS / "reprocessamento.txt"
        arquivo_rel.parent.mkdir(parents=True, exist_ok=True)
        arquivo_rel.write_text("\n".join(relatorio), encoding="utf-8")

        # Remove o sinal
        sinal.unlink()
        print("✅ Arquivo REPROCESSAR removido — sinal consumido.")

        return f"Reprocessamento: {sucesso} OK | {renomeados} renomeados | {falha} falhas"

    except Exception as e:
        return f"Reprocessamento: erro — {e}"

# ============================================================
# ORQUESTRAÇÃO PRINCIPAL
# ============================================================

# ============================================================
# VERIFICAÇÕES DE ESTADO — FASE 5
# ============================================================

def features_ca_pendente() -> bool:
    return not (RAIZ_PROJETO / "dados" / "processados" /
                "features_paderborn.parquet").exists()

def autoencoder_pendente() -> bool:
    return not (RAIZ_PROJETO / "resultados" / "autoencoder" /
                "modelo_autoencoder.pt").exists()

def injecao_falhas_pendente() -> bool:
    return not (RAIZ_PROJETO / "resultados" / "autoencoder" /
                "injecao_falhas_report.json").exists()

def validacao_pendente() -> bool:
    return not (RAIZ_PROJETO / "resultados" / "autoencoder" /
                "validacao_report.json").exists()

def rul_weibull_pendente() -> bool:
    return not (RAIZ_PROJETO / "resultados" / "autoencoder" /
                "weibull_results.json").exists()


# ============================================================
# ETAPAS — FASE 5
# ============================================================

def etapa_features_ca() -> str:
    """Extrai 109 features CA do dataset de Paderborn."""
    if not features_ca_pendente():
        return "Features CA: já extraídas"
    try:
        from src.ml.features_ca import executar_features_ca
        sucesso = executar_features_ca()
        return "✅ Features CA extraídas" if sucesso else "⚠️  Features CA: falhou"
    except Exception as e:
        return f"⚠️  Features CA: {e}"


def etapa_autoencoder() -> str:
    """Treina o Autoencoder de detecção de anomalias."""
    if not autoencoder_pendente():
        return "Autoencoder: já treinado"
    if features_ca_pendente():
        return "Autoencoder: aguardando features CA"
    try:
        from src.ml.autoencoder import executar_autoencoder
        sucesso = executar_autoencoder()
        return "✅ Autoencoder treinado" if sucesso else "⚠️  Autoencoder: falhou"
    except Exception as e:
        return f"⚠️  Autoencoder: {e}"


def etapa_injecao_falhas() -> str:
    """Injeta falhas sintéticas fundamentadas no FMEA."""
    if not injecao_falhas_pendente():
        return "Injeção de falhas: já realizada"
    if autoencoder_pendente():
        return "Injeção de falhas: aguardando Autoencoder"
    try:
        from src.ml.injecao_falhas import executar_injecao_falhas
        sucesso = executar_injecao_falhas()
        return "✅ Injeção de falhas concluída" if sucesso else "⚠️  Injeção de falhas: falhou"
    except Exception as e:
        return f"⚠️  Injeção de falhas: {e}"


def etapa_validacao_ml() -> str:
    """Validação formal: AUC, F1, Recall por tipo de falha."""
    if not validacao_pendente():
        return "Validação ML: já realizada"
    if injecao_falhas_pendente():
        return "Validação ML: aguardando injeção de falhas"
    try:
        from src.ml.validacao import executar_validacao
        sucesso = executar_validacao()
        return "✅ Validação ML concluída" if sucesso else "⚠️  Validação ML: falhou"
    except Exception as e:
        return f"⚠️  Validação ML: {e}"


def etapa_rul_weibull() -> str:
    """Estimativa de RUL com Análise de Weibull."""
    if not rul_weibull_pendente():
        return "RUL/Weibull: já calculado"
    if validacao_pendente():
        return "RUL/Weibull: aguardando validação ML"
    try:
        from src.ml.rul_weibull import executar_rul_weibull
        sucesso = executar_rul_weibull()
        return "✅ RUL/Weibull calculado" if sucesso else "⚠️  RUL/Weibull: falhou"
    except Exception as e:
        return f"⚠️  RUL/Weibull: {e}"


# ============================================================
# ORQUESTRAÇÃO PRINCIPAL
# ============================================================

def executar_pipeline(modelo_embeddings) -> list:
    """
    Executa o fluxo completo na ordem de dependência.
    Cada etapa decide internamente se precisa rodar.
    Retorna lista de mensagens de status para o app exibir.
    """
    relatorio = []

    # ── Sinal de reprocessamento completo ────────────────────
    if ha_sinal_reprocessamento():
        relatorio.append(reprocessar_literatura())

    # ── Conhecimento e memória ────────────────────────────────
    relatorio.append(etapa_indexar_pdfs(modelo_embeddings))
    relatorio.append(etapa_consolidar_memoria())
    relatorio.append(reprocessar_metadados_ruins())

    # ── Fase 5 — ML básico ────────────────────────────────────
    relatorio.append(etapa_eda())
    relatorio.append(etapa_classificacao())

    # ── Fase 5 — ML avançado (detecção de anomalias + RUL) ────
    relatorio.append(etapa_features_ca())
    relatorio.append(etapa_autoencoder())
    relatorio.append(etapa_injecao_falhas())
    relatorio.append(etapa_validacao_ml())
    relatorio.append(etapa_rul_weibull())

    return relatorio

if __name__ == "__main__":
    print("=" * 60)
    print("  AL IADO PV — ORQUESTRADOR (teste de estado)")
    print("=" * 60)
    print(f"\nSinal REPROCESSAR         : {ha_sinal_reprocessamento()}")
    print(f"PDFs novos pendentes      : {ha_pdfs_novos()}")
    print(f"Sessões para consolidar   : {ha_sessoes_para_consolidar()}")
    print(f"Arquivos com nome ruim    : {ha_arquivos_com_nome_ruim()}")
    print(f"EDA pendente              : {eda_pendente()}")
    print(f"Classificação pendente    : {classificacao_pendente()}")
    print(f"Features CA pendente      : {features_ca_pendente()}")
    print(f"Autoencoder pendente      : {autoencoder_pendente()}")
    print(f"Injeção de falhas pendente: {injecao_falhas_pendente()}")
    print(f"Validação ML pendente     : {validacao_pendente()}")
    print(f"RUL/Weibull pendente      : {rul_weibull_pendente()}")
    print("=" * 60)