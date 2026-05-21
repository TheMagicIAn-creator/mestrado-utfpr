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
    PASTA_CHROMADB,
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


def etapa_consolidar_memoria() -> str:
    """Consolida sessões acumuladas, se houver."""
    if not ha_sessoes_para_consolidar():
        return "Memória: sem acúmulo para consolidar"

    try:
        from src.conhecimento.consolidar_memoria import consolidar
        consolidar()
        return "Memória: sessões consolidadas"
    except Exception as e:
        return f"Memória: erro — {e}"


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


# ============================================================
# ORQUESTRAÇÃO PRINCIPAL
# ============================================================

def executar_pipeline(modelo_embeddings) -> list:
    """
    Executa o fluxo completo, na ordem de dependência.
    Cada etapa decide internamente se precisa rodar.

    Retorna uma lista de mensagens de status para o app exibir.
    """
    relatorio = []

    # Etapas leves e incrementais
    relatorio.append(etapa_indexar_pdfs(modelo_embeddings))
    relatorio.append(etapa_consolidar_memoria())

    # Etapas de ML — rodam só na primeira vez (verificação de estado)
    relatorio.append(etapa_eda())
    relatorio.append(etapa_classificacao())

    return relatorio


if __name__ == "__main__":
    # Execução de teste pelo terminal — sem modelo de embeddings real
    print("=" * 60)
    print("  AL IADO PV — ORQUESTRADOR (teste de estado)")
    print("=" * 60)
    print(f"\nPDFs novos pendentes      : {ha_pdfs_novos()}")
    print(f"Sessões para consolidar   : {ha_sessoes_para_consolidar()}")
    print(f"EDA pendente              : {eda_pendente()}")
    print(f"Classificação pendente    : {classificacao_pendente()}")
    print("=" * 60)