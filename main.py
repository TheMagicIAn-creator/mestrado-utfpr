"""
main.py — Al IAdo PV
Ponto de entrada do agente. Executa o chat no terminal
e salva a sessão automaticamente no Obsidian.

Como usar:
  python main.py

Comandos especiais durante o chat:
  'sair'    → salva sessão e encerra
  'limpar'  → limpa tela e memória
  'fontes'  → mostra total de chunks indexados
  'listar'  → lista todos os documentos indexados
  'memoria' → mostra resumo do histórico da sessão

Autor: Rodolfo Torres (UTFPR)
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Garante que Python encontra a pasta src/
sys.path.insert(0, str(Path(__file__).parent))

from src.agente import inicializar_agente, perguntar, listar_documentos
from src.provedores import selecionar_provedor, listar_provedores, inicializar_provedor


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_SESSOES = Path(__file__).parent / "notas" / "sessoes"


# ============================================================
# FUNÇÃO PARA SALVAR SESSÃO NO OBSIDIAN
# ============================================================

def salvar_sessao(historico: list):
    """
    Salva o histórico da conversa como nota .md no Obsidian.
    Cria a pasta notas/sessoes/ se não existir.
    """

    if not historico:
        print("\n  ⚠️  Nenhuma conversa para salvar.")
        return

    # Cria a pasta se não existir
    PASTA_SESSOES.mkdir(parents=True, exist_ok=True)

    # Nome do arquivo com data e hora
    agora      = datetime.now()
    nome_arquivo = agora.strftime("%Y-%m-%d_%H-%M") + "_sessao.md"
    caminho    = PASTA_SESSOES / nome_arquivo

    # Monta o conteúdo da nota
    data_formatada = agora.strftime("%d/%m/%Y às %H:%M")

    conteudo  = f"---\n"
    conteudo += f"data: {agora.strftime('%Y-%m-%d')}\n"
    conteudo += f"hora: {agora.strftime('%H:%M')}\n"
    conteudo += f"tipo: sessao-agente\n"
    conteudo += f"tags: [al-iado-pv, sessao, mestrado]\n"
    conteudo += f"---\n\n"
    conteudo += f"# Sessão Al IAdo PV — {data_formatada}\n\n"

    # Processa os pares de pergunta/resposta
    pares = []
    for i in range(0, len(historico) - 1, 2):
        if historico[i]["role"] == "user":
            pergunta  = historico[i]["content"]
            resposta  = historico[i + 1]["content"] if i + 1 < len(historico) else ""
            pares.append((pergunta, resposta))

    for n, (pergunta, resposta) in enumerate(pares, 1):
        conteudo += f"---\n\n"
        conteudo += f"## Pergunta {n}\n\n"
        conteudo += f"**🔬 Você:** {pergunta}\n\n"
        conteudo += f"**🤖 Al IAdo PV:**\n\n{resposta}\n\n"

    conteudo += f"\n---\n"
    conteudo += f"*Sessão gerada automaticamente pelo Al IAdo PV*\n"
    conteudo += f"*Total de interações: {len(pares)}*\n"

    # Salva o arquivo
    caminho.write_text(conteudo, encoding="utf-8")

    print(f"\n📓 Sessão salva no Obsidian:")
    print(f"   {caminho}")


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    # Seleção do provedor de LLM
    try:
        llm, nome_provedor, escolha_provedor = selecionar_provedor()
    except KeyboardInterrupt:
        print("\n\nAté logo! 👋")
        return

    # Inicializa agente com o LLM escolhido
    try:
        perfil, modelo_embeddings, colecao, _ = inicializar_agente(llm_externo=llm)
    except Exception as e:
        print(f"\n❌ Erro ao inicializar o agente:\n   {e}")
        return

    #Boas vindas
    print(f"\nOlá, Rodolfo! Sou o Al IAdo PV. 🤖")
    print(f"Provedor ativo: {nome_provedor}")
    print()
    print("Comandos disponíveis:")
    print("  'sair'    → salva sessão no Obsidian e encerra")
    print("  'limpar'  → limpa tela e memória")
    print("  'fontes'  → total de chunks indexados")
    print("  'listar'  → lista todos os documentos")
    print("  'memoria' → resumo do histórico")
    print("  'trocar'  → troca o provedor de LLM")
    print("-" * 60)

    # Loop de conversa com memória
    historico = []

    while True:

        try:
            pergunta = input("\n🔬 Você: ").strip()
        except KeyboardInterrupt:
            print("\n\nSalvando sessão antes de sair...")
            salvar_sessao(historico)
            print("Até logo, Rodolfo! 👋")
            break

        if not pergunta:
            continue

        # Comandos especiais
        if pergunta.lower() == "sair":
            salvar_sessao(historico)
            print("\nAté logo, Rodolfo! Bons estudos! 👋")
            break

        if pergunta.lower() == "limpar":
            os.system("cls" if os.name == "nt" else "clear")
            historico = []
            print("🧹 Tela e memória limpas!")
            continue

        if pergunta.lower() == "fontes":
            total = colecao.count()
            print(f"\n📚 Total de chunks indexados: {total}")
            continue

        if pergunta.lower() in ["listar", "listar artigos", "listar documentos"]:
            print("\n" + listar_documentos(colecao))
            continue

        if pergunta.lower() == "memoria":
            if not historico:
                print("\n🧠 Nenhuma conversa registrada ainda.")
            else:
                print(f"\n🧠 Turnos na memória: {len(historico)}")
                for i, turno in enumerate(historico, 1):
                    role = "Você" if turno["role"] == "user" else "Al IAdo PV"
                    print(f"  {i}. {role}: {turno['content'][:80]}...")
            continue

        if pergunta.lower() == "trocar":
            print(f"\n  Provedor atual: {nome_provedor}")
            try:
                llm, nome_provedor, escolha_provedor = selecionar_provedor()
                print(f"\n  ✅ Trocado para: {nome_provedor}")
            except Exception as e:
                print(f"  ❌ Erro ao trocar: {e}")
            continue

        # Processa pergunta
        print("\n🤖 Al IAdo PV: Buscando na literatura...\n")

        try:
            resposta = perguntar(
                pergunta          = pergunta,
                perfil            = perfil,
                modelo_embeddings = modelo_embeddings,
                colecao           = colecao,
                llm               = llm,
                historico         = historico,
                streaming = True
            )

            print("\n" + "-" * 60)

            # Salva na memória
            historico.append({"role": "user",      "content": pergunta})
            historico.append({"role": "assistant",  "content": resposta})

        except Exception as e:
            print(f"❌ Erro ao processar a pergunta: {e}")
            print("   Tente reformular ou verifique sua conexão.")


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()