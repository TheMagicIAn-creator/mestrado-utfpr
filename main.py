"""
main.py — Al IAdo PV
Chat no terminal com auto-salvamento de sessão.

Como usar:
  python main.py

Para encerrar: Ctrl+C

Autor: Rodolfo Torres (UTFPR)
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# ANTES de imports pesados: evitar crash de OpenMP duplicado (torch/numpy/
# onnxruntime/Orange) no Windows — access violation intermitente no startup.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent))

# Blinda stdout/stderr contra emoji no Windows (cp1252) antes de qualquer print.
from src.core.utils import configurar_saida_utf8

configurar_saida_utf8()

from src.conhecimento.agente import inicializar_agente, perguntar, listar_documentos
from src.conhecimento.provedores import selecionar_provedor
from src.conhecimento.indexador import indexar_sessao


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_SESSOES = Path(__file__).parent / "notas" / "sessoes"


# ============================================================
# AUTO-SALVAMENTO
# ============================================================

def iniciar_sessao() -> tuple:
    """
    Cria o arquivo .md da sessão no início da conversa.
    Retorna (caminho_arquivo, data_formatada).
    """
    PASTA_SESSOES.mkdir(parents=True, exist_ok=True)
    agora        = datetime.now()
    nome_arquivo = agora.strftime("%Y-%m-%d_%H-%M") + "_sessao.md"
    caminho      = PASTA_SESSOES / nome_arquivo
    data_fmt     = agora.strftime("%d/%m/%Y às %H:%M")

    # Cabeçalho inicial da nota
    cabecalho  = f"---\n"
    cabecalho += f"data: {agora.strftime('%Y-%m-%d')}\n"
    cabecalho += f"hora: {agora.strftime('%H:%M')}\n"
    cabecalho += f"tipo: sessao-terminal\n"
    cabecalho += f"tags: [al-iado-pv, sessao, mestrado]\n"
    cabecalho += f"---\n\n"
    cabecalho += f"# Sessão Al IAdo PV — {data_fmt}\n\n"

    caminho.write_text(cabecalho, encoding="utf-8")
    return caminho, data_fmt


def salvar_interacao(caminho: Path, pergunta: str, resposta: str, n: int):
    """
    Adiciona um par pergunta/resposta ao arquivo .md da sessão.
    Chamado automaticamente após cada interação.
    """
    bloco  = f"---\n\n"
    bloco += f"## Interação {n}\n\n"
    bloco += f"**🔬 Você:** {pergunta}\n\n"
    bloco += f"**🤖 Al IAdo PV:**\n\n{resposta}\n\n"

    with open(caminho, "a", encoding="utf-8") as f:
        f.write(bloco)


def finalizar_sessao(caminho: Path, historico: list, modelo_embeddings, n_interacoes: int):
    """
    Adiciona rodapé e indexa a sessão no ChromaDB ao encerrar.
    """
    from src.core.config import PASTA_CHROMADB

    rodape  = f"\n---\n"
    rodape += f"*Sessão encerrada — {n_interacoes} interações*\n"
    rodape += f"*Gerado automaticamente pelo Al IAdo PV*\n"

    with open(caminho, "a", encoding="utf-8") as f:
        f.write(rodape)

    # Indexa no ChromaDB para memória persistente
    if n_interacoes > 0:
        try:
            n_chunks = indexar_sessao(caminho, modelo_embeddings, PASTA_CHROMADB)
            print(f"\n🧠 Sessão indexada na memória: {n_chunks} chunks")
        except Exception as e:
            print(f"\n⚠️  Erro ao indexar sessão: {e}")

    print(f"📓 Sessão salva em: {caminho.name}")


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    # Seleção do provedor
    try:
        llm, nome_provedor, _ = selecionar_provedor()
    except KeyboardInterrupt:
        print("\n\nAté logo! 👋")
        return

    # Inicializa agente
    try:
        (
            perfil,
            modelo_embeddings,
            colecao,
            colecao_sessoes,
            colecao_obsidian,
            _,
        ) = inicializar_agente(llm_externo=llm)
    except Exception as e:
        print(f"\n❌ Erro ao inicializar: {e}")
        return

    # Inicia arquivo de sessão
    caminho_sessao, data_fmt = iniciar_sessao()

    # Boas-vindas
    print(f"\nOlá, Rodolfo! Sou o Al IAdo PV. 🤖")
    print(f"Provedor: {nome_provedor}")
    print(f"Sessão iniciada às {data_fmt}")
    print(f"Auto-salvamento ativo — cada resposta é salva automaticamente.")
    print(f"Para encerrar: Ctrl+C")
    print("-" * 60)

    historico    = []
    n_interacoes = 0

    while True:

        try:
            pergunta = input("\n🔬 Você: ").strip()
        except KeyboardInterrupt:
            print("\n\nEncerrando sessão...")
            finalizar_sessao(caminho_sessao, historico, modelo_embeddings, n_interacoes)
            print("Até logo, Rodolfo! 👋")
            break

        if not pergunta:
            continue

        # Processa a pergunta
        print("\n🤖 Al IAdo PV:\n")

        try:
            resposta = perguntar(
                pergunta          = pergunta,
                perfil            = perfil,
                modelo_embeddings = modelo_embeddings,
                colecao           = colecao,
                llm               = llm,
                historico         = historico,
                streaming         = True,
                colecao_sessoes   = colecao_sessoes,
                nome_provedor     = nome_provedor,
                colecao_obsidian  = colecao_obsidian,
            )

            print("\n" + "-" * 60)

            # Salva automaticamente
            n_interacoes += 1
            salvar_interacao(caminho_sessao, pergunta, resposta, n_interacoes)

            # Atualiza histórico
            historico.append({"role": "user",      "content": pergunta})
            historico.append({"role": "assistant",  "content": resposta})

        except Exception as e:
            erro = str(e)
            if "413" in erro or "Request too large" in erro:
                print("\nPedido grande demais para o limite do provedor.")
                print("Tente uma pergunta mais focada ou escolha Gemini.")
            elif "429" in erro:
                print(f"\n⏳ Limite da API atingido.")
                print(f"   Encerre com Ctrl+C e reinicie escolhendo outro provedor.")
            else:
                print(f"\n❌ Erro: {e}")


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()
