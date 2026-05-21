"""
processador_pdf.py — Al IAdo PV
Pipeline completo para processamento de novos PDFs.

Uso via linha de comando:
  python src/processador_pdf.py
  → Processa todos os PDFs da pasta novos_pdfs/

Uso via import (Streamlit):
  from src.processador_pdf import processar_pdf_unico

Etapas:
  1. Extrai metadados (autor, título, ano)
  2. Gera nome padronizado
  3. Classifica tema automaticamente
  4. Copia para literatura/<tema>/
  5. Indexa no ChromaDB
  6. Gera nota .md no Obsidian

Autor: Rodolfo Torres (UTFPR)
"""

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from pypdf import PdfReader


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_LITERATURA = Path(__file__).parent.parent / "literatura"
PASTA_NOTAS      = Path(__file__).parent.parent / "notas" / "literatura"
PASTA_ENTRADA    = Path(__file__).parent.parent / "novos_pdfs"


# ============================================================
# TEMAS E PALAVRAS-CHAVE
# ============================================================

TEMAS = {
    "ml-preditivo": [
        "machine learning", "deep learning", "random forest",
        "neural network", "lstm", "gru", "autoencoder",
        "isolation forest", "xgboost", "lightgbm", "transformer",
        "anomaly detection", "predictive", "prognosis",
        "remaining useful life", "rul", "fault detection",
        "aprendizado de máquina", "rede neural", "detecção de falha",
        "classificação", "regressão", "clustering", "gaussian process",
        "support vector", "svm", "naive bayes", "k-nearest"
    ],
    "inversores-pv": [
        "inverter", "inversor", "photovoltaic", "fotovoltaic",
        "solar", "pv system", "grid-connected", "on-grid",
        "lcl filter", "filtro lcl", "igbt", "mosfet",
        "power electronics", "eletrônica de potência",
        "dc-ac", "pwm", "pulse width modulation", "switching",
        "contactor", "contactores", "transformador"
    ],
    "manutencao": [
        "maintenance", "manutenção", "fmea", "fmeca",
        "failure mode", "modo de falha", "rcm",
        "reliability centered maintenance",
        "preventive", "preventiva", "predictive maintenance",
        "manutenção preditiva", "cbm", "condition based",
        "work order", "corrective", "downtime"
    ],
    "confiabilidade": [
        "reliability", "confiabilidade", "mtbf", "mttf",
        "failure rate", "taxa de falha", "weibull",
        "availability", "disponibilidade", "risk assessment",
        "hazard", "survival analysis", "markov chain",
        "fault tree", "árvore de falhas", "rbd"
    ],
    "sinais-eletricos": [
        "signal processing", "processamento de sinais",
        "harmonic", "harmônico", "thd", "distortion",
        "fft", "fourier", "spectrum", "espectro",
        "power quality", "qualidade de energia",
        "current", "corrente", "voltage", "tensão",
        "rms", "waveform", "forma de onda", "sampling",
        "amostragem", "filter design", "bandwidth",
        "discrete time", "continuous time", "z-transform"
    ]
}

TEMA_PADRAO = "ml-preditivo"


# ============================================================
# EXTRAÇÃO DE TEXTO
# ============================================================

def extrair_texto_pdf(caminho_pdf: Path, n_paginas: int = 3) -> str:
    """Extrai texto das primeiras N páginas do PDF."""
    try:
        reader = PdfReader(str(caminho_pdf))
        texto  = ""
        for i in range(min(n_paginas, len(reader.pages))):
            texto += reader.pages[i].extract_text() or ""
        return texto
    except Exception:
        return ""


# ============================================================
# EXTRAÇÃO DE METADADOS
# ============================================================

def extrair_metadados_pdf(caminho_pdf: Path) -> dict:
    """
    Extrai autor, título e ano em cascata:
    1. Metadados internos do PDF
    2. Análise do texto
    3. Fallback para valores padrão
    """

    autor  = ""
    titulo = ""
    ano    = "0000"

    try:
        reader = PdfReader(str(caminho_pdf))
        meta   = reader.metadata or {}

        # Autor
        autor_raw = str(meta.get("/Author") or meta.get("Author") or "")
        if autor_raw.strip():
            autor = autor_raw.split(";")[0].split(",")[0].strip()

        # Título
        titulo_raw = str(meta.get("/Title") or meta.get("Title") or "")
        if titulo_raw.strip():
            titulo = titulo_raw.strip()

        # Ano a partir da data de criação
        data_raw  = str(meta.get("/CreationDate") or meta.get("/ModDate") or "")
        match_ano = re.search(r"(\d{4})", data_raw)
        if match_ano:
            ano_cand = int(match_ano.group(1))
            if 1990 <= ano_cand <= datetime.now().year:
                ano = str(ano_cand)

    except Exception:
        pass

    # Fallback via texto do PDF
    texto = extrair_texto_pdf(caminho_pdf, n_paginas=2)

    if ano == "0000" and texto:
        anos = re.findall(r"\b(199\d|20[0-3]\d)\b", texto)
        if anos:
            ano = anos[0]

    if not autor and texto:
        padroes = [
            r"(?:Authors?|Autores?)[:\s]+([A-Z][a-záàãâéêíóõôú]+(?:\s[A-Z]\.?\s?[A-Z][a-z]+)*)",
            r"^([A-Z][a-z]+(?:\s[A-Z]\.?\s?[A-Z][a-z]+){1,3})",
        ]
        for p in padroes:
            m = re.search(p, texto[:3000], re.MULTILINE)
            if m:
                autor = m.group(1).strip()
                break

    if not titulo and texto:
        linhas = [l.strip() for l in texto.split("\n") if len(l.strip()) > 20]
        if linhas:
            titulo = linhas[0][:100]

    return {
        "autor" : autor  or "autor-desconhecido",
        "titulo": titulo or caminho_pdf.stem,
        "ano"   : ano
    }


# ============================================================
# NOME PADRONIZADO
# ============================================================

def gerar_nome_padronizado(autor: str, titulo: str, ano: str) -> str:
    """Gera nome no padrão autor_titulo_ano.pdf"""

    def limpar(texto: str, limite: int = 50) -> str:
        subs = {
            "á":"a","à":"a","ã":"a","â":"a","ä":"a",
            "é":"e","è":"e","ê":"e","ë":"e",
            "í":"i","ì":"i","î":"i","ï":"i",
            "ó":"o","ò":"o","õ":"o","ô":"o","ö":"o",
            "ú":"u","ù":"u","û":"u","ü":"u",
            "ç":"c","ñ":"n"
        }
        for orig, rep in subs.items():
            texto = texto.replace(orig, rep).replace(orig.upper(), rep)
        texto = re.sub(r"[^a-zA-Z0-9\s]", " ", texto)
        texto = re.sub(r"\s+", "-", texto.strip())
        texto = re.sub(r"-+", "-", texto)
        return texto.lower()[:limite]

    sobrenome = autor.split(",")[0].split(" ")[-1]
    return f"{limpar(sobrenome, 30)}_{limpar(titulo, 60)}_{ano}.pdf"


# ============================================================
# CLASSIFICAÇÃO DE TEMA
# ============================================================

def classificar_tema(nome_arquivo: str, texto: str) -> str:
    """Classifica por pontuação de palavras-chave."""

    conteudo = (nome_arquivo + " " + texto).lower()
    pontos   = {tema: 0 for tema in TEMAS}

    for tema, palavras in TEMAS.items():
        for palavra in palavras:
            if palavra in conteudo:
                pontos[tema] += 1

    melhor = max(pontos, key=pontos.get)
    return melhor if pontos[melhor] > 0 else TEMA_PADRAO


# ============================================================
# NOTA OBSIDIAN
# ============================================================

def gerar_nota_obsidian(
    nome_final: str,
    autor: str,
    titulo: str,
    ano: str,
    tema: str,
    texto_pdf: str
) -> Path:
    """Gera nota .md no vault do Obsidian."""

    pasta = PASTA_NOTAS / tema
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome_final.replace(".pdf", ".md")

    abstract = " ".join(texto_pdf.split())[:800] if texto_pdf else "Não disponível."

    conteudo  = f"---\n"
    conteudo += f"titulo: \"{titulo[:100]}\"\n"
    conteudo += f"autor: \"{autor}\"\n"
    conteudo += f"ano: {ano}\n"
    conteudo += f"tema: {tema}\n"
    conteudo += f"arquivo: {nome_final}\n"
    conteudo += f"tags: [literatura, {tema}, mestrado-utfpr]\n"
    conteudo += f"data_insercao: {datetime.now().strftime('%Y-%m-%d')}\n"
    conteudo += f"---\n\n"
    conteudo += f"# {titulo[:100]}\n\n"
    conteudo += f"**Autor:** {autor}  \n"
    conteudo += f"**Ano:** {ano}  \n"
    conteudo += f"**Tema:** {tema}  \n"
    conteudo += f"**Arquivo:** `{nome_final}`\n\n"
    conteudo += f"## Abstract\n\n{abstract}\n\n"
    conteudo += f"## Anotações\n\n> _Adicione suas anotações aqui._\n\n"
    conteudo += f"## Conexões\n\n- [[indice-literatura]]\n"

    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# ============================================================
# PIPELINE — PDF ÚNICO
# ============================================================

def processar_pdf_unico(
    caminho_pdf     : Path,
    modelo_embeddings,
    pasta_chromadb  : Path,
    gerar_obsidian  : bool = True
) -> dict:
    """
    Pipeline completo para um único PDF.
    Retorna dicionário com resultado detalhado.
    """

    resultado = {
        "sucesso"      : False,
        "arquivo_orig" : caminho_pdf.name,
        "arquivo_final": "",
        "autor"        : "",
        "titulo"       : "",
        "ano"          : "",
        "tema"         : "",
        "n_chunks"     : 0,
        "nota_obsidian": "",
        "erro"         : None
    }

    try:
        # 1. Metadados
        meta   = extrair_metadados_pdf(caminho_pdf)
        autor  = meta["autor"]
        titulo = meta["titulo"]
        ano    = meta["ano"]

        resultado.update({"autor": autor, "titulo": titulo, "ano": ano})

        # 2. Nome padronizado
        nome_final             = gerar_nome_padronizado(autor, titulo, ano)
        resultado["arquivo_final"] = nome_final

        # 3. Tema
        texto = extrair_texto_pdf(caminho_pdf)
        tema  = classificar_tema(nome_final, texto)
        resultado["tema"] = tema

        # 4. Copia para literatura/<tema>/
        pasta_destino = PASTA_LITERATURA / tema
        pasta_destino.mkdir(parents=True, exist_ok=True)
        caminho_final = pasta_destino / nome_final
        shutil.copy2(str(caminho_pdf), str(caminho_final))

        # 5. Indexa no ChromaDB
        from src.indexador import indexar_pdf_unico
        res = indexar_pdf_unico(caminho_final, modelo_embeddings, pasta_chromadb)

        if not res["sucesso"]:
            resultado["erro"] = f"Erro na indexação: {res['erro']}"
            return resultado

        resultado["n_chunks"] = res["n_chunks"]

        # 6. Nota Obsidian
        if gerar_obsidian:
            nota = gerar_nota_obsidian(nome_final, autor, titulo, ano, tema, texto)
            resultado["nota_obsidian"] = str(nota)

        resultado["sucesso"] = True

    except Exception as e:
        resultado["erro"] = str(e)

    return resultado


# ============================================================
# PIPELINE — PASTA INTEIRA
# ============================================================

def processar_pasta(
    pasta_entrada   : Path,
    modelo_embeddings,
    pasta_chromadb  : Path,
    gerar_obsidian  : bool = True
) -> list:
    """
    Processa todos os PDFs de uma pasta.
    Use via linha de comando para inserção em lote.
    """

    pdfs = list(pasta_entrada.glob("*.pdf"))

    if not pdfs:
        print(f"\n⚠️  Nenhum PDF em: {pasta_entrada}")
        print(f"   Coloque os PDFs lá e rode novamente.")
        return []

    print("=" * 60)
    print(f"  PROCESSADOR DE PDFs — Al IAdo PV")
    print(f"  PDFs encontrados: {len(pdfs)}")
    print("=" * 60 + "\n")

    resultados = []
    sucesso    = 0
    falha      = 0

    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i}/{len(pdfs)}] {pdf.name}")

        r = processar_pdf_unico(pdf, modelo_embeddings, pasta_chromadb, gerar_obsidian)

        if r["sucesso"]:
            print(f"         Autor  : {r['autor']}")
            print(f"         Título : {r['titulo'][:50]}")
            print(f"         Ano    : {r['ano']}")
            print(f"         Tema   : {r['tema']}")
            print(f"         Arquivo: {r['arquivo_final']}")
            print(f"         Chunks : {r['n_chunks']}")
            print(f"         ✅ OK!\n")
            sucesso += 1
        else:
            print(f"         ❌ Erro: {r['erro']}\n")
            falha += 1

        resultados.append(r)

    print("=" * 60)
    print(f"  Sucesso: {sucesso} | Falha: {falha}")
    print("=" * 60)

    return resultados


# ============================================================
# LINHA DE COMANDO
# ============================================================

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from src.agente import MODELO_EMBEDDINGS, PASTA_CHROMADB

    PASTA_ENTRADA.mkdir(exist_ok=True)

    if not list(PASTA_ENTRADA.glob("*.pdf")):
        print(f"\n📂 Pasta de entrada criada: {PASTA_ENTRADA}")
        print(f"   Coloque os PDFs lá e rode novamente:")
        print(f"   python src/processador_pdf.py")
    else:
        print("\n🔄 Carregando modelo de embeddings...")
        modelo = SentenceTransformer(MODELO_EMBEDDINGS)
        processar_pasta(PASTA_ENTRADA, modelo, PASTA_CHROMADB)