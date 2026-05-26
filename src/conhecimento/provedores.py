"""
provedores.py — Al IAdo PV
Gerencia os provedores de LLM disponíveis.
Permite escolher e trocar de provedor durante a sessão.

Provedores suportados:
  1. Google Gemini (gemini-2.5-flash) — 20 req/dia grátis
  2. Groq (llama-3.3-70b)            — 12k tokens/min no tier on-demand

Autor: Rodolfo Torres (UTFPR)
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# DEFINIÇÃO DOS PROVEDORES
# ============================================================

PROVEDORES = {
    "1": {
        "nome"   : "Google Gemini",
        "modelo" : "gemini-2.5-flash",
        "env_key": "GOOGLE_API_KEY",
        "limite" : "20 req/dia",
        "emoji"  : "🔵"
    },
    "2": {
        "nome"   : "Groq (LLaMA 3.3)",
        "modelo" : "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "limite" : "12k tokens/min no tier on-demand",
        "emoji"  : "🟢"
    }
}


# ============================================================
# FUNÇÕES
# ============================================================

def listar_provedores():
    """
    Exibe o menu de escolha de provedores.
    """
    print("\n" + "=" * 60)
    print("  ESCOLHA O PROVEDOR DE LLM")
    print("=" * 60)

    for chave, info in PROVEDORES.items():
        api_key      = os.getenv(info["env_key"])
        disponivel   = "✅ Chave configurada" if api_key else "❌ Chave não encontrada"
        print(f"\n  {info['emoji']} [{chave}] {info['nome']}")
        print(f"      Modelo : {info['modelo']}")
        print(f"      Limite : {info['limite']} (tier gratuito)")
        print(f"      Status : {disponivel}")

    print("\n" + "=" * 60)


def inicializar_provedor(escolha: str):
    """
    Inicializa e retorna o LLM do provedor escolhido.
    """

    if escolha not in PROVEDORES:
        raise ValueError(f"Provedor '{escolha}' não reconhecido.")

    info    = PROVEDORES[escolha]
    api_key = os.getenv(info["env_key"])

    if not api_key:
        raise ValueError(
            f"\n❌ Chave não encontrada para {info['nome']}!\n"
            f"   Adicione {info['env_key']}=sua_chave no arquivo .env"
        )

    print(f"\n  {info['emoji']} Inicializando {info['nome']}...")

    # Google Gemini
    if escolha == "1":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model         = info["modelo"],
            google_api_key= api_key,
            temperature   = 0.3
        )

    # Groq
    elif escolha == "2":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model    = info["modelo"],
            groq_api_key = api_key,
            temperature  = 0.3
        )

    print(f"  ✅ {info['nome']} pronto! (limite: {info['limite']})")
    return llm, info["nome"]


def selecionar_provedor() -> tuple:
    """
    Exibe o menu, lê a escolha do usuário e retorna o LLM.
    """

    listar_provedores()

    while True:
        opcoes = list(PROVEDORES.keys())
        escolha = input(f"\n  Digite sua escolha {opcoes}: ").strip()

        if escolha in opcoes:
            try:
                llm, nome = inicializar_provedor(escolha)
                return llm, nome, escolha
            except Exception as e:
                print(f"\n  ❌ Erro: {e}")
                print("  Tente outro provedor.")
        else:
            print(f"  ⚠️  Opção inválida. Digite {opcoes[0]} ou {opcoes[-1]}.")
