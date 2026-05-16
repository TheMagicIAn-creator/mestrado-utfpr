"""
main.py — Al IAdo PV
Ponto de entrada do agente. Executa o chat no terminal.

Como usar:
  python main.py

Comandos especiais durante o chat:
  'sair'   → encerra o programa
  'limpar' → limpa o histórico da conversa
  'fontes' → mostra quantos chunks estão indexados

Autor: Rodolfo Torres (UTFPR)
"""

import sys
from pathlib import Path

# Garante que Python encontra a pasta src/
sys.path.insert(0, str(Path(__file__).parent))

from src.agente import inicializar_agente, perguntar, listar_documentos

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    # ----------------------------------------------------------
    # Inicializa todos os componentes do agente
    # ----------------------------------------------------------
    try:
        perfil, modelo_embeddings, colecao, llm = inicializar_agente()
    except Exception as e:
        print(f"\n❌ Erro ao inicializar o agente:\n   {e}")
        print("\nVerifique:")
        print("  1. O arquivo .env existe com GOOGLE_API_KEY")
        print("  2. O indexador já foi executado (src/indexador.py)")
        print("  3. O ambiente virtual está ativo (.venv)")
        return

    # ----------------------------------------------------------
    # Boas-vindas
    # ----------------------------------------------------------
    print("Olá, Rodolfo! Sou o Al IAdo PV. 🤖")
    print("Estou pronto para responder sobre sua literatura.")
    print()
    print("Comandos disponíveis:")
    print("  'sair'   → encerra o programa")
    print("  'limpar' → limpa a tela")
    print("  'fontes' → mostra total de chunks indexados")
    print("-" * 60)

    # ----------------------------------------------------------
    # Loop de conversa com memória
    # ----------------------------------------------------------
    historico = []  # guarda toda a conversa da sessão

    while True:

        try:
            pergunta = input("\n🔬 Você: ").strip()
        except KeyboardInterrupt:
            print("\n\nEncerrando... Até logo, Rodolfo! 👋")
            break

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            print("\nAté logo, Rodolfo! Bons estudos! 👋")
            break

        if pergunta.lower() == "limpar":
            import os
            os.system("cls" if os.name == "nt" else "clear")
            historico = []  # limpa memória também
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

        # Processa a pergunta
        print("\n🤖 Al IAdo PV: Buscando na literatura...\n")

        try:
            resposta = perguntar(
                pergunta=pergunta,
                perfil=perfil,
                modelo_embeddings=modelo_embeddings,
                colecao=colecao,
                llm=llm,
                historico=historico
            )

            print(resposta)
            print("\n" + "-" * 60)

            # Salva na memória da sessão
            historico.append({"role": "user", "content": pergunta})
            historico.append({"role": "assistant", "content": resposta})

        except Exception as e:
            print(f"❌ Erro ao processar a pergunta: {e}")
            print("   Tente reformular ou verifique sua conexão.")


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()