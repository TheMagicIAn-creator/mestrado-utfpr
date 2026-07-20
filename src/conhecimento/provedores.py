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
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# DEFINIÇÃO DOS PROVEDORES
# ============================================================

PROVEDORES = {
    "1": {
        "nome"      : "Google Gemini",
        "modelo"    : "gemini-2.5-flash",
        "env_key"   : "GOOGLE_API_KEY",
        "limite"    : "20 req/dia",
        "emoji"     : "🔵",
        "multimodal": True,   # entende imagens (visão)
    },
    "2": {
        "nome"      : "Groq (LLaMA 3.3)",
        "modelo"    : "llama-3.3-70b-versatile",
        "env_key"   : "GROQ_API_KEY",
        "limite"    : "12k tokens/min no tier on-demand",
        "emoji"     : "🟢",
        "multimodal": False,  # texto puro (não lê imagens)
    }
}


@dataclass(frozen=True)
class RespostaLLM:
    """Resposta mínima compatível com o contrato usado pelo agente."""

    content: str


def _conteudo_da_mensagem(mensagem):
    if isinstance(mensagem, dict):
        return mensagem.get("content", "")
    return getattr(mensagem, "content", mensagem)


def _texto_do_conteudo(conteudo) -> str:
    if isinstance(conteudo, str):
        return conteudo
    partes = []
    for item in conteudo or []:
        if isinstance(item, str):
            partes.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            partes.append(str(item.get("text", "")))
    return "\n".join(parte for parte in partes if parte)


class GeminiLeve:
    """Adaptador do SDK Google Gen AI sem carregar a integração LangChain."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.3,
                 client=None) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self._client = client

    def _obter_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _converter_conteudo(conteudo):
        if isinstance(conteudo, str):
            return conteudo

        import base64
        from google.genai import types

        partes = []
        for item in conteudo or []:
            if isinstance(item, str):
                partes.append(types.Part.from_text(text=item))
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                partes.append(types.Part.from_text(text=str(item.get("text", ""))))
                continue
            if item.get("type") != "image_url":
                continue
            imagem = item.get("image_url", {})
            url = imagem.get("url", "") if isinstance(imagem, dict) else str(imagem)
            if not url.startswith("data:") or ";base64," not in url:
                continue
            cabecalho, payload = url.split(",", 1)
            mime_type = cabecalho[5:].split(";", 1)[0] or "image/png"
            partes.append(types.Part.from_bytes(
                data=base64.b64decode(payload), mime_type=mime_type
            ))
        return partes

    def _conteudos(self, mensagens):
        convertidos = [
            self._converter_conteudo(_conteudo_da_mensagem(mensagem))
            for mensagem in mensagens
        ]
        return convertidos[0] if len(convertidos) == 1 else convertidos

    def invoke(self, mensagens) -> RespostaLLM:
        resposta = self._obter_client().models.generate_content(
            model=self.model,
            contents=self._conteudos(mensagens),
            config={"temperature": self.temperature},
        )
        return RespostaLLM(content=getattr(resposta, "text", "") or "")

    def stream(self, mensagens):
        fluxo = self._obter_client().models.generate_content_stream(
            model=self.model,
            contents=self._conteudos(mensagens),
            config={"temperature": self.temperature},
        )
        for item in fluxo:
            texto = getattr(item, "text", "") or ""
            if texto:
                yield RespostaLLM(content=texto)


class GroqLeve:
    """Adaptador do SDK Groq sem carregar a integração LangChain."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.3,
                 client=None) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self._client = client

    def _obter_client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    @staticmethod
    def _mensagens(mensagens) -> list[dict]:
        return [
            {"role": "user", "content": _texto_do_conteudo(
                _conteudo_da_mensagem(mensagem)
            )}
            for mensagem in mensagens
        ]

    def invoke(self, mensagens) -> RespostaLLM:
        resposta = self._obter_client().chat.completions.create(
            model=self.model,
            messages=self._mensagens(mensagens),
            temperature=self.temperature,
        )
        texto = resposta.choices[0].message.content or ""
        return RespostaLLM(content=texto)

    def stream(self, mensagens):
        fluxo = self._obter_client().chat.completions.create(
            model=self.model,
            messages=self._mensagens(mensagens),
            temperature=self.temperature,
            stream=True,
        )
        for item in fluxo:
            texto = item.choices[0].delta.content or ""
            if texto:
                yield RespostaLLM(content=texto)


def eh_multimodal(nome_ou_chave: str) -> bool:
    """
    Diz se um provedor entende imagens (visão), aceitando tanto a chave
    ("1"/"2") quanto o nome exibido ("Google Gemini", "Groq (LLaMA 3.3)").

    Default conservador: False (assume texto puro se não reconhecer).
    """
    if not nome_ou_chave:
        return False
    alvo = str(nome_ou_chave).strip()
    if alvo in PROVEDORES:
        return bool(PROVEDORES[alvo].get("multimodal", False))
    for info in PROVEDORES.values():
        if info["nome"] == alvo:
            return bool(info.get("multimodal", False))
    return False


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
        llm = GeminiLeve(
            model=info["modelo"],
            api_key=api_key,
            temperature=0.3,
        )

    # Groq
    elif escolha == "2":
        llm = GroqLeve(
            model=info["modelo"],
            api_key=api_key,
            temperature=0.3,
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
                # Erros de SDK podem citar URL/headers — mascarar antes de exibir.
                from src.core.seguranca import mascarar_segredos

                print(f"\n  ❌ Erro: {mascarar_segredos(str(e))}")
                print("  Tente outro provedor.")
        else:
            print(f"  ⚠️  Opção inválida. Digite {opcoes[0]} ou {opcoes[-1]}.")
