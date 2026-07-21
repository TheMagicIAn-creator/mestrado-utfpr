"""
provedores.py — Al IAdo PV
Gerencia os provedores de LLM disponíveis.

Equipe 100% Gemini, um modelo por nível de tarefa (a escolha segue os limites
de taxa por modelo do plano pago da API Gemini: quanto mais barato o modelo,
maior o RPM/TPM disponível — então o trabalho repetitivo desce de nível):

  • NÍVEL 1 — conversa, síntese final e interpretação de imagens
      gemini-2.5-pro (o mais capaz; limite de requisições menor, por isso é
      reservado à ÚNICA chamada que o Rodolfo lê por turno).
  • NÍVEL 2 — auditoria de evidências e porteiro da memória validada
      gemini-2.5-flash (rápido, JSON estruturado nativo, RPM alto; roda a cada
      turno com literatura sem competir com o orçamento do pro).
  • NÍVEL 3 — tarefas de fundo em lote: metadados de PDF e consolidação
      gemini-2.5-flash-lite (o mais barato/veloz, maior limite de taxa; ideal
      para varrer 39 PDFs ou resumir sessões sem custo relevante).
  • SEM LLM — expansão de query, BM25, RRF, reranking, cálculos e ferramentas
      continuam heurísticas locais determinísticas.

Autor: Rodolfo Torres (UTFPR)
"""

import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


# NÍVEL 3 — modelo de FUNDO (extração de metadados de PDF e consolidação de
# memória): o mais econômico da família, com o maior limite de taxa.
MODELO_GEMINI_FUNDO = os.getenv("AL_IADO_GEMINI_MODEL_FUNDO", "gemini-2.5-flash-lite")


# ============================================================
# DEFINIÇÃO DOS PROVEDORES
# ============================================================

PROVEDORES = {
    "1": {
        "nome"      : "Google Gemini",
        "modelo"    : os.getenv("AL_IADO_GEMINI_MODEL", "gemini-2.5-pro"),
        "env_key"   : "GOOGLE_API_KEY",
        "limite"    : "conforme o plano da API",
        "emoji"     : "🔵",
        "multimodal": True,   # entende imagens (visão)
    },
    "2": {
        "nome"      : "Google Gemini (auditor)",
        "modelo"    : os.getenv("AL_IADO_GEMINI_MODEL_AUDITOR", "gemini-2.5-flash"),
        "env_key"   : "GOOGLE_API_KEY",
        "limite"    : "conforme o plano da API",
        "emoji"     : "🟢",
        # O modelo até enxerga imagens, mas o PAPEL é textual por contrato:
        # o auditor só recebe pacotes compactos de texto, nunca anexos.
        "multimodal": False,
    }
}

# Os papeis sao fixos por arquitetura. A interface nao permite trocar os
# modelos entre si porque cada um tem contrato, orcamento e responsabilidade
# diferentes.
PAPEIS_AGENTES = {
    "conversa": {
        "provedor": "1",
        "rotulo": "Gemini - conversa e sintese",
    },
    "auditoria": {
        "provedor": "2",
        "rotulo": "Gemini Flash - auditoria e memoria",
    },
}


@dataclass(frozen=True)
class RespostaLLM:
    """Resposta mínima compatível com o contrato usado pelo agente."""

    content: str


def _conteudo_da_mensagem(mensagem):
    if isinstance(mensagem, dict):
        return mensagem.get("content", "")
    return getattr(mensagem, "content", mensagem)


class GeminiLeve:
    """Adaptador do SDK Google Gen AI sem carregar a integração LangChain."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.45,
                 max_output_tokens: int = 8192, client=None) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = max(256, int(max_output_tokens))
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
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )
        return RespostaLLM(content=getattr(resposta, "text", "") or "")

    def invoke_json(self, mensagens, max_tokens: int = 700) -> dict:
        """Executa uma chamada curta e exige um objeto JSON (papel de auditor)."""
        resposta = self._obter_client().models.generate_content(
            model=self.model,
            contents=self._conteudos(mensagens),
            config={
                "temperature": 0.0,
                "max_output_tokens": max(100, int(max_tokens)),
                "response_mime_type": "application/json",
            },
        )
        texto = getattr(resposta, "text", "") or "{}"
        payload = json.loads(texto)
        if not isinstance(payload, dict):
            raise ValueError("O auditor nao retornou um objeto JSON.")
        return payload

    def stream(self, mensagens):
        fluxo = self._obter_client().models.generate_content_stream(
            model=self.model,
            contents=self._conteudos(mensagens),
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )
        for item in fluxo:
            texto = getattr(item, "text", "") or ""
            if texto:
                yield RespostaLLM(content=texto)


def eh_multimodal(nome_ou_chave: str) -> bool:
    """
    Diz se um provedor entende imagens (visão), aceitando tanto a chave
    ("1"/"2") quanto o nome exibido ("Google Gemini", "Google Gemini (auditor)").
    O papel de auditor é textual por contrato, então responde False.

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
        print(f"      Limite : {info['limite']}")
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

    # NÍVEL 1 — Gemini Pro: conversa, síntese e imagens
    if escolha == "1":
        llm = GeminiLeve(
            model=info["modelo"],
            api_key=api_key,
            temperature=float(os.getenv("AL_IADO_GEMINI_TEMPERATURE", "0.45")),
            max_output_tokens=int(
                os.getenv("AL_IADO_GEMINI_MAX_OUTPUT_TOKENS", "8192")
            ),
        )

    # NÍVEL 2 — Gemini Flash: auditor de evidências e porteiro da memória.
    # Temperatura baixa; as chamadas JSON do papel forçam 0.0 no invoke_json.
    elif escolha == "2":
        llm = GeminiLeve(
            model=info["modelo"],
            api_key=api_key,
            temperature=0.2,
            max_output_tokens=2048,
        )

    print(f"  ✅ {info['nome']} pronto! (limite: {info['limite']})")
    return llm, info["nome"]


def inicializar_papel(papel: str):
    """Inicializa o provedor fixado para um papel da equipe."""
    if papel not in PAPEIS_AGENTES:
        raise ValueError(f"Papel de agente desconhecido: {papel}")
    escolha = PAPEIS_AGENTES[papel]["provedor"]
    llm, nome = inicializar_provedor(escolha)
    return llm, nome, PAPEIS_AGENTES[papel]["rotulo"]


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
