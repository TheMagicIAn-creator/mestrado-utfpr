"""
config.py — Al IAdo PV / Núcleo
Configuração central do projeto.

Este módulo é o ponto único de verdade para caminhos, constantes
e parâmetros. Todos os demais módulos importam daqui, em vez de
recalcular caminhos ou redefinir constantes localmente.

Autor: Rodolfo Torres (UTFPR)
"""

import os

# ── Windows / OpenMP ────────────────────────────────────────────────────────
# Vários pacotes nativos (torch, numpy/MKL, onnxruntime do ChromaDB e as libs
# do Orange3) embarcam o PRÓPRIO runtime OpenMP. Quando dois deles inicializam
# no mesmo processo, o OpenMP ABORTA (access violation → segfault/EXIT 139),
# de forma INTERMITENTE conforme a ordem de carga — o que derrubava o app no
# startup às vezes. Permitir a coexistência dos runtimes evita o crash. Definido
# aqui porque config é importado antes de qualquer biblioteca pesada em todos
# os pontos de entrada (app, terminal, scripts, bateria).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# RAIZ DO PROJETO
# ============================================================
# Este arquivo está em: <raiz>/src/core/config.py
# Portanto, a raiz do projeto está três níveis acima.
# Calculado UMA vez aqui; todos os módulos importam os caminhos prontos.

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent


# ============================================================
# VARIÁVEIS DE AMBIENTE
# ============================================================
# Carrega o arquivo .env da raiz do projeto.

load_dotenv(RAIZ_PROJETO / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# CAMINHOS DE PASTAS
# ============================================================

PASTA_LITERATURA   = RAIZ_PROJETO / "literatura"
PASTA_DADOS        = RAIZ_PROJETO / "dados"
PASTA_DADOS_BRUTOS = PASTA_DADOS / "brutos"
PASTA_DADOS_PROC   = PASTA_DADOS / "processados"
PASTA_RESULTADOS   = RAIZ_PROJETO / "resultados"
PASTA_NOTAS        = RAIZ_PROJETO / "notas"
PASTA_SESSOES      = PASTA_NOTAS / "sessoes"
PASTA_MEMORIAS     = PASTA_NOTAS / "memorias"
PASTA_VAULT_OBSIDIAN = Path(
    os.getenv(
        "AL_IADO_OBSIDIAN_VAULT_DIR",
        str(PASTA_NOTAS),
    )
).expanduser().resolve()
PASTA_CEREBRO_OBSIDIAN = Path(
    os.getenv(
        "AL_IADO_OBSIDIAN_DIR",
        str(PASTA_VAULT_OBSIDIAN / "Cerebro"),
    )
).expanduser().resolve()
PASTA_MEMORIA_AGENTES = Path(
    os.getenv(
        "AL_IADO_MEMORIA_DIR",
        str(PASTA_MEMORIAS / "agentes"),
    )
).expanduser().resolve()
ARQUIVO_MEMORIA_VALIDADA = Path(
    os.getenv(
        "AL_IADO_MEMORIA_VALIDADA",
        str(PASTA_MEMORIA_AGENTES / "memoria_validada.json"),
    )
).expanduser().resolve()
PASTA_ARQUIVO      = PASTA_NOTAS / "sessoes_arquivadas"
PASTA_NOVOS_PDFS   = RAIZ_PROJETO / "novos_pdfs"
PASTA_CHROMADB     = Path(
    os.getenv("AL_IADO_CHROMADB_DIR", str(RAIZ_PROJETO / "base_conhecimento"))
).expanduser().resolve()
PASTA_ARTEFATOS    = RAIZ_PROJETO / "artefatos"
ARQUIVO_INDICE_LITERATURA = Path(
    os.getenv(
        "AL_IADO_INDICE_LITERATURA",
        str(PASTA_ARTEFATOS / "literatura_indexada.jsonl.gz"),
    )
).expanduser().resolve()
ARQUIVO_INDICE_OBSIDIAN = Path(
    os.getenv(
        "AL_IADO_INDICE_OBSIDIAN",
        str(PASTA_ARTEFATOS / "obsidian_indexado.jsonl.gz"),
    )
).expanduser().resolve()
ARQUIVO_INDICE_LEXICAL = Path(
    os.getenv(
        "AL_IADO_INDICE_LEXICAL",
        str(PASTA_CHROMADB / "literatura_fts.sqlite3"),
    )
).expanduser().resolve()
ARQUIVO_PERFIL     = RAIZ_PROJETO / "CLAUDE.md"


# ============================================================
# CONSTANTES DO AGENTE (RAG)
# ============================================================

# Marcador de build — atualizado a cada deploy relevante. Aparece na barra
# lateral do app para confirmar QUAL versão do código está no ar (resolve a
# ambiguidade de redeploy no Streamlit Cloud: se o marcador aqui não bate com
# o exibido, o app está rodando código antigo e precisa de Reboot).
MARCADOR_BUILD         = "2026-07-21 · #25 · rodapé = fontes citáveis (coerência prosa↔rodapé)"

MODELO_EMBEDDINGS      = "paraphrase-multilingual-MiniLM-L12-v2"
# Modelo principal de conversa (fonte de verdade: provedores.PROVEDORES / env
# AL_IADO_GEMINI_MODEL). O de fundo (metadados/memória) é gemini-2.5-flash.
MODELO_GEMINI          = "gemini-2.5-pro"
NOME_COLECAO           = "literatura_pv"
NOME_COLECAO_SESSOES   = "sessoes_pv"
NOME_COLECAO_OBSIDIAN  = "obsidian_pv"
# Memória de AVALIAÇÃO (baterias/harness) — SEPARADA da memória de produção,
# para que testes automatizados nunca contaminem as sessões reais (item 8.2/18).
NOME_COLECAO_AVALIACOES = "avaliacoes_agente"
N_RESULTADOS           = 25
TAMANHO_CHUNK          = 500
SOBREPOSICAO           = 50
TAMANHO_LOTE           = 500   # limite de upsert do ChromaDB


# ============================================================
# CONSTANTES DO MACHINE LEARNING
# ============================================================

TAXA_AMOSTRAGEM = 10_000   # Hz — dataset de inversor (Paderborn)
SEMENTE_ALEATORIA = 42     # random_state — reprodutibilidade


# ============================================================
# VERIFICAÇÃO DE INTEGRIDADE
# ============================================================

def verificar_estrutura() -> dict:
    """
    Verifica quais pastas essenciais existem.
    Útil para o orquestrador decidir o que precisa ser criado.
    """
    pastas = {
        "literatura"    : PASTA_LITERATURA,
        "dados_brutos"  : PASTA_DADOS_BRUTOS,
        "resultados"    : PASTA_RESULTADOS,
        "sessoes"       : PASTA_SESSOES,
        "novos_pdfs"    : PASTA_NOVOS_PDFS,
        "chromadb"      : PASTA_CHROMADB,
    }
    return {nome: caminho.exists() for nome, caminho in pastas.items()}


if __name__ == "__main__":
    print("=" * 60)
    print("  AL IADO PV — CONFIGURAÇÃO CENTRAL")
    print("=" * 60)
    print(f"\nRaiz do projeto: {RAIZ_PROJETO}")
    print(f"\nChave Google : {'✅ configurada' if GOOGLE_API_KEY else '❌ ausente'}")
    print(f"\nEstrutura de pastas:")
    for nome, existe in verificar_estrutura().items():
        status = "✅" if existe else "❌"
        print(f"   {status} {nome}")
    print("=" * 60)
