"""
indexador.py — Al IAdo PV

Indexador seguro para literatura e sessões.

Diretrizes atuais:
- manter chunks semanticamente legíveis, sem granularidade excessiva;
- preservar metadados suficientes para auditoria acadêmica;
- aceitar literatura em português, inglês, espanhol e francês;
- registrar idioma estimado do documento sem impedir a indexação;
- nunca duplicar documentos iguais: SHA256 é a identidade primária.

Correções principais desta versão:
- controle de duplicidade por SHA256 do PDF;
- IDs determinísticos por hash + chunk;
- uso de upsert em lotes;
- chunking de literatura menos granular;
- remoção da combinação problemática:
    chunking fixo + chunking por seções + tabelas;
- extração de tabelas opcional via variável de ambiente.

Execute pela raiz do projeto:
    python src/conhecimento/indexador.py
"""

import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader

from src.core.config import (
    MODELO_EMBEDDINGS,
    NOME_COLECAO,
    NOME_COLECAO_SESSOES,
    PASTA_CHROMADB,
    PASTA_LITERATURA,
    SOBREPOSICAO,
    TAMANHO_CHUNK,
    TAMANHO_LOTE,
)
from src.core.logs import get_logger
from src.core.utils import parsear_nome_arquivo

_logger = get_logger("conhecimento.indexador")

# ============================================================
# PARÂMETROS ESPECÍFICOS PARA LITERATURA
# ============================================================

# O valor antigo de 500 caracteres gerava uma coleção excessivamente granular.
# Para RAG acadêmico, 1600–2200 caracteres costuma ser mais equilibrado.
TAMANHO_CHUNK_LITERATURA = int(os.getenv("TAMANHO_CHUNK_LITERATURA", "1800"))
SOBREPOSICAO_LITERATURA = int(os.getenv("SOBREPOSICAO_LITERATURA", "200"))

# A extração de tabelas com pdfplumber é útil, mas cara e pode aumentar a base.
# Por padrão fica desligada. Para ativar:
# PowerShell:
#   $env:EXTRAIR_TABELAS_LITERATURA="1"
# CMD:
#   set EXTRAIR_TABELAS_LITERATURA=1
EXTRAIR_TABELAS_LITERATURA = os.getenv("EXTRAIR_TABELAS_LITERATURA", "0") == "1"


# ============================================================
# UTILITÁRIOS
# ============================================================

def calcular_hash_arquivo(caminho: Path) -> str:
    """Calcula SHA256 do arquivo, usado como identidade estável do PDF."""
    sha = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(bloco)
    return sha.hexdigest()


def documento_ja_indexado(colecao, arquivo_hash: str) -> bool:
    """Retorna True se o conteúdo exato do PDF já estiver indexado."""
    try:
        res = colecao.get(where={"arquivo_hash": arquivo_hash}, limit=1)
        return bool(res.get("ids"))
    except Exception:
        return False


def remover_documento_antigo(colecao, nome_arquivo: str | None = None, arquivo_hash: str | None = None) -> int:
    """
    Remove chunks antigos por nome e/ou hash.

    Observação:
    - por nome: remove resíduos de versões antigas sem hash;
    - por hash: remove resíduos do mesmo conteúdo.
    """
    removidos = 0
    filtros = []

    if nome_arquivo:
        filtros.append({"arquivo": nome_arquivo})

    if arquivo_hash:
        filtros.append({"arquivo_hash": arquivo_hash})

    for where in filtros:
        try:
            res = colecao.get(where=where)
            ids = res.get("ids", []) or []
            if ids:
                for inicio in range(0, len(ids), TAMANHO_LOTE):
                    fim = inicio + TAMANHO_LOTE
                    colecao.delete(ids=ids[inicio:fim])
                removidos += len(ids)
        except Exception as exc:
            # Não interrompe indexação por resíduo antigo problemático.
            _logger.warning("não foi possível remover chunks antigos (%s): %s", where, exc)

    return removidos


def upsert_em_lotes(colecao, ids, embeddings, documents, metadados, tamanho_lote: int = 500) -> None:
    """Executa upsert em lotes para evitar limites internos do ChromaDB/SQLite."""
    total = len(ids)

    for inicio in range(0, total, tamanho_lote):
        fim = min(inicio + tamanho_lote, total)
        colecao.upsert(
            ids=ids[inicio:fim],
            embeddings=embeddings[inicio:fim],
            documents=documents[inicio:fim],
            metadatas=metadados[inicio:fim],
        )


# ============================================================
# EXTRAÇÃO E CHUNKING DE PDF
# ============================================================

# Singleton do modelo de embeddings: ~500 MB e 2-3 s de carga; reutilizado
# entre chamadas de indexação na mesma sessão (watcher, chat, orquestrador).
_MODELO_EMBEDDINGS = None


def obter_modelo_embeddings():
    """Carrega o SentenceTransformer UMA vez por processo e reutiliza."""
    global _MODELO_EMBEDDINGS
    if _MODELO_EMBEDDINGS is None:
        from sentence_transformers import SentenceTransformer

        _MODELO_EMBEDDINGS = SentenceTransformer(MODELO_EMBEDDINGS)
    return _MODELO_EMBEDDINGS


def ler_pdf(caminho_pdf: Path) -> str:
    """Extrai texto do PDF usando pypdf."""
    try:
        reader = PdfReader(str(caminho_pdf))
        partes = []

        for pagina in reader.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                partes.append(texto_pagina)

        return "\n\n".join(partes).strip()

    except Exception as e:
        print(f"  ⚠️  Erro ao ler {caminho_pdf.name}: {e}")
        return ""


def _rotulos_paginas_pdf(caminho_pdf: Path) -> dict[int, str]:
    """
    Le os rotulos de pagina definidos no proprio PDF, quando existirem.

    O numero fisico (1, 2, 3...) continua sendo a referencia principal porque
    ele e auditavel em qualquer visualizador. O rotulo do PDF entra como
    complemento, util para documentos com capa, resumo em romano ou numeracao
    reiniciada.
    """
    try:
        reader = PdfReader(str(caminho_pdf))
        labels = getattr(reader, "page_labels", None)
        if callable(labels):
            labels = labels()
        if not labels:
            return {}
        rotulos: dict[int, str] = {}
        for i, label in enumerate(labels, 1):
            valor = str(label or "").strip()
            if valor:
                rotulos[i] = valor
        return rotulos
    except Exception:
        return {}


def ler_pdf_paginas(caminho_pdf: Path) -> list[tuple[str, int]]:
    """
    Extrai o texto do PDF preservando a fronteira de páginas.

    Retorna uma lista de (texto_da_pagina, numero_da_pagina), com numeração
    começando em 1 (como o leitor humano vê). Páginas sem texto extraível são
    omitidas. Diferente de ``ler_pdf`` (que concatena tudo e perde a página),
    esta função habilita a citação com página — ``Autor (ano, p. X)``.
    """
    try:
        reader = PdfReader(str(caminho_pdf))
        paginas: list[tuple[str, int]] = []

        for num_pag, pagina in enumerate(reader.pages, 1):
            texto_pagina = pagina.extract_text()
            if texto_pagina and texto_pagina.strip():
                paginas.append((texto_pagina, num_pag))

        return paginas

    except Exception as e:
        print(f"  ⚠️  Erro ao ler páginas de {caminho_pdf.name}: {e}")
        return []


def normalizar_texto_pdf(texto: str) -> str:
    """
    Normaliza texto extraído de PDF.

    A intenção é reduzir ruído sem destruir fórmulas, siglas ou símbolos técnicos.
    """
    if not texto:
        return ""

    texto = texto.replace("\x00", " ")

    # Une palavras quebradas por hifenização no fim da linha:
    # confiabili-\ndade -> confiabilidade
    texto = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", texto)

    # Normaliza quebras excessivas de linha.
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # Remove espaços antes de pontuação comum.
    texto = re.sub(r"[ \t]+([,.;:!?])", r"\1", texto)

    return texto.strip()


def trecho_auditavel(texto: str, limite: int = 360) -> str:
    """
    Gera um trecho curto, limpo e rastreavel para auditoria de citacoes.

    Esse texto vai para o metadado do chunk e aparece no bloco de fontes. Nao
    substitui o chunk completo; serve para o pesquisador conferir rapidamente
    se a pagina/trecho recuperado realmente sustenta a resposta.
    """
    texto = normalizar_texto_pdf(texto or "")
    texto = re.sub(r"\s+", " ", texto).strip()
    if not texto:
        return ""

    limite = max(120, int(limite))
    if len(texto) <= limite:
        return texto

    corte = max(
        texto.rfind(".", 0, limite),
        texto.rfind(";", 0, limite),
        texto.rfind(":", 0, limite),
        texto.rfind("?", 0, limite),
        texto.rfind("!", 0, limite),
    )
    if corte < int(limite * 0.55):
        corte = texto.rfind(" ", 0, limite)
    if corte < int(limite * 0.55):
        corte = limite

    return texto[:corte].strip().rstrip(",;:") + "..."


def detectar_idioma_texto(texto: str) -> str:
    """
    Heurística leve para metadado de idioma.

    O embedding local já é multilíngue; este campo existe para auditoria,
    filtros futuros e transparência do catálogo, não para bloquear indexação.
    """
    amostra = normalizar_texto_pdf(texto[:6000]).lower()
    sinais = {
        "pt": ("ção", "ções", "não", "falha", "manutenção", "confiabilidade"),
        "en": (" the ", " and ", "failure", "maintenance", "reliability", "inverter"),
        "es": ("ción", "ciones", "falla", "mantenimiento", "confiabilidad"),
        "fr": (" pour ", " avec ", "défaillance", "defaillance", "fiabilité", "fiabilite"),
    }
    pontuacao = {
        idioma: sum(amostra.count(sinal) for sinal in termos)
        for idioma, termos in sinais.items()
    }
    melhor = max(pontuacao, key=pontuacao.get)
    return melhor if pontuacao[melhor] > 0 else "desconhecido"


def _melhor_ponto_de_corte(texto: str, inicio: int, limite: int, tamanho_minimo: int) -> int:
    """
    Escolhe um corte próximo do limite, preferindo fim de frase/parágrafo.
    Evita gerar chunks cortados no meio de sentenças quando possível.
    """
    n = len(texto)

    if limite >= n:
        return n

    janela_inicio = max(inicio + tamanho_minimo, limite - 500)
    trecho = texto[janela_inicio:limite]

    # Preferência 1: fim de parágrafo.
    pos = trecho.rfind("\n\n")
    if pos != -1:
        return janela_inicio + pos

    # Preferência 2: fim de frase.
    candidatos = [trecho.rfind("."), trecho.rfind(";"), trecho.rfind(":"), trecho.rfind("?"), trecho.rfind("!")]
    pos = max(candidatos)
    if pos != -1:
        return janela_inicio + pos + 1

    # Preferência 3: último espaço.
    pos = trecho.rfind(" ")
    if pos != -1:
        return janela_inicio + pos

    return limite


def dividir_em_chunks(texto: str, tamanho: int, sobreposicao: int) -> list[str]:
    """
    Divide texto em chunks por caracteres, com corte em ponto semântico quando possível.

    Esta função continua existindo com a mesma assinatura para preservar compatibilidade
    com indexação de sessões e outros módulos.
    """
    texto = normalizar_texto_pdf(texto)

    if not texto:
        return []

    tamanho = max(int(tamanho), 300)
    sobreposicao = max(0, min(int(sobreposicao), tamanho // 3))
    tamanho_minimo = max(250, tamanho // 2)

    chunks = []
    inicio = 0
    n = len(texto)

    while inicio < n:
        limite = min(inicio + tamanho, n)
        corte = _melhor_ponto_de_corte(texto, inicio, limite, tamanho_minimo)

        if corte <= inicio:
            corte = limite

        chunk = texto[inicio:corte].strip()

        if len(chunk) >= 80:
            chunks.append(chunk)

        if corte >= n:
            break

        novo_inicio = max(corte - sobreposicao, inicio + 1)

        # Evita começar no meio de espaço/quebra.
        while novo_inicio < n and texto[novo_inicio].isspace():
            novo_inicio += 1

        inicio = novo_inicio

    return chunks


def extrair_tabelas_pdf(
    caminho_pdf: Path, metadados_doc: dict, com_pagina: bool = False
):
    """
    Extrai tabelas do PDF como chunks Markdown estruturados.

    Esta rotina é opcional porque aumenta tempo de indexação e tamanho da base.
    Ative apenas quando tabelas forem essenciais para a busca.

    Se ``com_pagina=True``, retorna lista de ``(markdown, num_pagina)`` para que
    a indexação registre a página da tabela; caso contrário (padrão), retorna
    lista de ``markdown`` (compatível com o comportamento anterior).
    """
    chunks_tabelas: list = []

    if not EXTRAIR_TABELAS_LITERATURA:
        return chunks_tabelas

    citacao = metadados_doc.get("citacao", caminho_pdf.name)

    try:
        import pdfplumber

        with pdfplumber.open(str(caminho_pdf)) as pdf:
            for num_pag, pagina in enumerate(pdf.pages, 1):
                tabelas = pagina.extract_tables()
                if not tabelas:
                    continue

                for num_tab, tabela in enumerate(tabelas, 1):
                    if not tabela or len(tabela) < 2:
                        continue

                    linhas_validas = [
                        linha for linha in tabela
                        if linha and any(cel and str(cel).strip() for cel in linha)
                    ]

                    if len(linhas_validas) < 2:
                        continue

                    def limpar_celula(cel):
                        if cel is None:
                            return ""
                        return str(cel).replace("\n", " ").strip()

                    cabecalho = linhas_validas[0]
                    corpo = linhas_validas[1:]
                    n_cols = len(cabecalho)

                    md = f"[TABELA — {citacao} — Página {num_pag}, Tabela {num_tab}]\n"
                    md += "| " + " | ".join(limpar_celula(c) for c in cabecalho) + " |\n"
                    md += "| " + " | ".join("---" for _ in cabecalho) + " |\n"

                    for linha in corpo:
                        linha_pad = list(linha) + [""] * max(0, n_cols - len(linha))
                        md += "| " + " | ".join(limpar_celula(c) for c in linha_pad[:n_cols]) + " |\n"

                    if len(md) >= 150:
                        if com_pagina:
                            chunks_tabelas.append((md.strip(), num_pag))
                        else:
                            chunks_tabelas.append(md.strip())

    except ImportError:
        print("  ⚠️  pdfplumber não instalado; tabelas ignoradas.")
    except Exception as e:
        print(f"  ⚠️  Erro ao extrair tabelas de {caminho_pdf.name}: {e}")

    return chunks_tabelas


def remover_chunks_duplicados(chunks: list[str]) -> list[str]:
    """
    Remove chunks textual ou quase textualmente idênticos.
    """
    vistos = set()
    unicos = []

    for chunk in chunks:
        normalizado = " ".join(chunk.split()).strip()
        if not normalizado:
            continue

        chave = hashlib.sha256(
            normalizado.lower().encode("utf-8", errors="ignore")
        ).hexdigest()

        if chave in vistos:
            continue

        vistos.add(chave)
        unicos.append(chunk.strip())

    return unicos


def remover_itens_duplicados(
    itens: list[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """
    Versão de ``remover_chunks_duplicados`` que preserva a página de cada chunk.

    Recebe e devolve tuplas ``(texto, pagina_inicio, pagina_fim)``; mantém a
    PRIMEIRA ocorrência (com a sua página) de cada texto normalizado.
    """
    vistos: set = set()
    unicos: list[tuple[str, int, int]] = []

    for texto, p_ini, p_fim in itens:
        normalizado = " ".join((texto or "").split()).strip()
        if not normalizado:
            continue

        chave = hashlib.sha256(
            normalizado.lower().encode("utf-8", errors="ignore")
        ).hexdigest()

        if chave in vistos:
            continue

        vistos.add(chave)
        unicos.append((texto.strip(), p_ini, p_fim))

    return unicos


# ============================================================
# INDEXAÇÃO DE LITERATURA
# ============================================================

def indexar_pdf_unico(
    caminho_pdf: Path,
    modelo_embeddings,
    pasta_chromadb: Path,
    *,
    forcar: bool = False,
    metadados_override: dict | None = None,
) -> dict:
    """Indexa um PDF sob lock compartilhado por todos os processos."""
    from src.conhecimento.index_lock import lock_indexacao

    with lock_indexacao():
        return _indexar_pdf_unico_sem_lock(
            caminho_pdf,
            modelo_embeddings,
            pasta_chromadb,
            forcar=forcar,
            metadados_override=metadados_override,
        )


def _indexar_pdf_unico_sem_lock(
    caminho_pdf: Path,
    modelo_embeddings,
    pasta_chromadb: Path,
    *,
    forcar: bool = False,
    metadados_override: dict | None = None,
) -> dict:
    """
    Indexa um único PDF no ChromaDB com proteção contra duplicidade.

    Estratégia:
    1. calcula SHA256 do arquivo;
    2. se o mesmo conteúdo já estiver indexado, pula;
    3. remove resíduos antigos por nome/hash;
    4. gera chunks apenas por uma estratégia principal;
    5. opcionalmente adiciona chunks de tabelas;
    6. usa IDs determinísticos e upsert.
    """
    resultado = {
        "sucesso": False,
        "nome_arquivo": caminho_pdf.name,
        "n_chunks": 0,
        "pulou": False,
        "motivo": "",
        "erro": None,
    }

    try:
        arquivo_hash = calcular_hash_arquivo(caminho_pdf)

        import chromadb

        client = chromadb.PersistentClient(path=str(pasta_chromadb))
        colecao = client.get_or_create_collection(
            name=NOME_COLECAO,
            metadata={"hnsw:space": "cosine"},
        )

        if not forcar and documento_ja_indexado(colecao, arquivo_hash):
            resultado["sucesso"] = True
            resultado["pulou"] = True
            resultado["motivo"] = "PDF já indexado pelo mesmo hash SHA256."
            try:
                resultado["n_chunks"] = len(colecao.get(where={"arquivo_hash": arquivo_hash}).get("ids", []))
            except Exception:
                resultado["n_chunks"] = 0
            return resultado

        # Extração PAGE-AWARE: preserva a página de origem de cada chunk para
        # permitir citação com página — "Autor (ano, p. X)".
        paginas = ler_pdf_paginas(caminho_pdf)
        texto = "\n\n".join(t for t, _ in paginas).strip()

        if not texto:
            resultado["erro"] = "Não foi possível extrair texto do PDF."
            return resultado

        info_arquivo = parsear_nome_arquivo(caminho_pdf.name)
        idioma = detectar_idioma_texto(texto)

        # Chunking por página: cada chunk herda o número da sua página.
        # itens: lista de (texto_chunk, pagina_inicio, pagina_fim).
        rotulos_paginas = _rotulos_paginas_pdf(caminho_pdf)
        itens: list[tuple[str, int, int]] = []
        for texto_pag, num_pag in paginas:
            for ch in dividir_em_chunks(
                texto_pag,
                TAMANHO_CHUNK_LITERATURA,
                SOBREPOSICAO_LITERATURA,
            ):
                itens.append((ch, num_pag, num_pag))

        # Tabelas são opcionais — também carregam a página de origem.
        for md, num_pag in extrair_tabelas_pdf(
            caminho_pdf, info_arquivo, com_pagina=True
        ):
            itens.append((md, num_pag, num_pag))

        itens = remover_itens_duplicados(itens)

        if not itens:
            resultado["erro"] = "Nenhum chunk gerado."
            return resultado

        chunks = [it[0] for it in itens]

        # A parte cara e falivel ocorre antes da troca. Assim, uma falha de
        # inferencia nao apaga os chunks ainda validos do documento.
        embeddings = modelo_embeddings.encode(chunks).tolist()

        override = dict(metadados_override or {})
        nome_pasta = str(override.get("pasta") or caminho_pdf.parent.name)
        ids = [f"{arquivo_hash}__chunk_{j:05d}" for j in range(len(chunks))]

        metadados = [
            {
                "arquivo": caminho_pdf.name,
                "arquivo_hash": arquivo_hash,
                "pasta": nome_pasta,
                "chunk_index": j,
                "total_chunks": len(itens),
                "pagina_inicio": int(itens[j][1]),
                "pagina_fim": int(itens[j][2]),
                "pagina_rotulo": rotulos_paginas.get(int(itens[j][1]), ""),
                "trecho": trecho_auditavel(chunks[j]),
                "chunk_sha256": hashlib.sha256(
                    chunks[j].encode("utf-8", errors="ignore")
                ).hexdigest(),
                "autor": str(override.get("autor") or info_arquivo.get("autor", "")),
                "titulo": str(override.get("titulo") or info_arquivo.get("titulo", "")),
                "ano": str(override.get("ano") or info_arquivo.get("ano", "")),
                "citacao": str(
                    override.get("citacao")
                    or info_arquivo.get("citacao", caminho_pdf.name)
                ),
                "idioma": str(override.get("idioma") or idioma),
            }
            for j in range(len(itens))
        ]

        removidos = remover_documento_antigo(
            colecao,
            nome_arquivo=caminho_pdf.name,
            arquivo_hash=arquivo_hash,
        )

        if removidos:
            print(f"  Removidos {removidos} chunks antigos de {caminho_pdf.name}")

        upsert_em_lotes(
            colecao=colecao,
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadados=metadados,
            tamanho_lote=TAMANHO_LOTE,
        )

        resultado["sucesso"] = True
        resultado["n_chunks"] = len(chunks)
        return resultado

    except Exception as e:
        resultado["erro"] = str(e)
        return resultado


def indexar_literatura(modelo=None, pasta_chromadb: Path = PASTA_CHROMADB) -> dict:
    """Indexa todos os PDFs e retorna um resumo estruturado da execução."""
    print("=" * 72)
    print("AL IADO PV — INDEXADOR SEGURO DE LITERATURA")
    print("=" * 72)

    if not PASTA_LITERATURA.exists():
        print(f"Pasta não encontrada: {PASTA_LITERATURA}")
        return {"pdfs": 0, "indexados": 0, "pulados": 0, "erros": 1, "chunks": 0}

    pdfs = sorted(PASTA_LITERATURA.rglob("*.pdf"))

    if not pdfs:
        print(f"Nenhum PDF encontrado em: {PASTA_LITERATURA}")
        return {"pdfs": 0, "indexados": 0, "pulados": 0, "erros": 1, "chunks": 0}

    print(f"PDFs encontrados: {len(pdfs)}")
    print(f"Modelo de embeddings: {MODELO_EMBEDDINGS}")
    print(f"Chunk literatura: {TAMANHO_CHUNK_LITERATURA}")
    print(f"Sobreposição literatura: {SOBREPOSICAO_LITERATURA}")
    print(f"Extração de tabelas: {'ativada' if EXTRAIR_TABELAS_LITERATURA else 'desativada'}")

    modelo = modelo or obter_modelo_embeddings()

    total_chunks = 0
    pdfs_indexados = 0
    pdfs_pulados = 0
    pdfs_com_erro = 0

    for i, caminho_pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {caminho_pdf.name}")
        resultado = indexar_pdf_unico(caminho_pdf, modelo, pasta_chromadb)

        if not resultado.get("sucesso"):
            pdfs_com_erro += 1
            print(f"  ERRO: {resultado.get('erro')}")
            continue

        if resultado.get("pulou"):
            pdfs_pulados += 1
            print(f"  SKIP: {resultado.get('motivo')}")
            continue

        n_chunks = int(resultado.get("n_chunks", 0))
        total_chunks += n_chunks
        pdfs_indexados += 1
        print(f"  OK: {n_chunks} chunks")

    print("=" * 72)
    print("INDEXAÇÃO CONCLUÍDA")
    print(f"PDFs indexados : {pdfs_indexados}")
    print(f"PDFs pulados   : {pdfs_pulados}")
    print(f"PDFs com erro  : {pdfs_com_erro}")
    print(f"Chunks novos   : {total_chunks}")
    print("=" * 72)
    return {
        "pdfs": len(pdfs),
        "indexados": pdfs_indexados,
        "pulados": pdfs_pulados,
        "erros": pdfs_com_erro,
        "chunks": total_chunks,
    }


# ============================================================
# INDEXAÇÃO DE SESSÕES
# ============================================================

def indexar_sessao(caminho_md: Path, modelo_embeddings, pasta_chromadb: Path) -> int:
    """Indexa uma sessão sob lock compartilhado por todos os processos."""
    from src.conhecimento.index_lock import lock_indexacao

    with lock_indexacao():
        return _indexar_sessao_sem_lock(caminho_md, modelo_embeddings, pasta_chromadb)


def _indexar_sessao_sem_lock(
    caminho_md: Path, modelo_embeddings, pasta_chromadb: Path
) -> int:
    """
    Indexa uma sessão salva (.md) na coleção de sessões do ChromaDB.

    Mantém a assinatura original para compatibilidade com o restante do sistema.
    """
    if not caminho_md.exists():
        print(f"  ⚠️  Arquivo não encontrado: {caminho_md}")
        return 0

    texto = caminho_md.read_text(encoding="utf-8", errors="ignore")

    if not texto.strip():
        return 0

    linhas_filtradas = []
    for linha in texto.split("\n"):
        if any(termo in linha.lower() for termo in [
            "rate limit", "429", "resource_exhausted",
            "quota", "error code", "❌ erro",
        ]):
            continue
        linhas_filtradas.append(linha)

    texto = "\n".join(linhas_filtradas)

    chunks = dividir_em_chunks(texto, TAMANHO_CHUNK, SOBREPOSICAO)

    if not chunks:
        return 0

    import chromadb

    client = chromadb.PersistentClient(path=str(pasta_chromadb))
    colecao_sessoes = client.get_or_create_collection(
        name=NOME_COLECAO_SESSOES,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = modelo_embeddings.encode(chunks).tolist()

    nome_arquivo = caminho_md.name
    ids = [f"{nome_arquivo}__chunk_{j:05d}" for j in range(len(chunks))]

    data_sessao = caminho_md.stem[:10]

    metadados = [
        {
            "arquivo": nome_arquivo,
            "tipo": "sessao",
            "data": data_sessao,
            "chunk_index": j,
            "total_chunks": len(chunks),
        }
        for j in range(len(chunks))
    ]

    upsert_em_lotes(
        colecao=colecao_sessoes,
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadados=metadados,
        tamanho_lote=TAMANHO_LOTE,
    )

    return len(chunks)


if __name__ == "__main__":
    indexar_literatura()
