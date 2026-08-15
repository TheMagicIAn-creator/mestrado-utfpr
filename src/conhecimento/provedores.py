"""
provedores.py — Al IAdo PV
Gerencia os provedores de LLM disponíveis.

Equipe 100% Gemini, um modelo por nível de tarefa. Usa identificadores GA
explícitos e verificados na API para evitar a latência de aliases indisponíveis:

  • NÍVEL 1 — conversa, síntese final e interpretação de imagens
      gemini-3.6-flash por padrão (GA, rápido, estável). O Pro é opt-in via
      AL_IADO_GEMINI_MODEL=gemini-pro-latest para máximo raciocínio — mas é lento
      no trivial e sofre 503 de alta demanda, por isso não é o default.
  • NÍVEL 2 — auditoria de evidências e porteiro da memória validada
      gemini-3.5-flash-lite (rápido, JSON estruturado nativo, GA; roda a cada
      turno com literatura sem competir com o orçamento do pro).
  • NÍVEL 3 — tarefas de fundo em lote: metadados de PDF e consolidação
      gemini-3.5-flash-lite (o mais barato/veloz; ideal para varrer os PDFs
      ou resumir sessões sem custo relevante).
  • SEM LLM — expansão de query, BM25, RRF, reranking, cálculos e ferramentas
      continuam heurísticas locais determinísticas.

Resiliência: GeminiLeve tenta o modelo configurado e, se ele estiver
indisponível (404/aposentado), cai automaticamente para MODELO_GEMINI_FALLBACK
(gemini-3.5-flash, GA) — o app nao trava por rotacao de modelo.

Autor: Rodolfo Torres (UTFPR)
"""

import os
import json
import time
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


# Retentativas em erro TRANSITÓRIO (503/sobrecarga, 429/limite): o mesmo modelo
# é tentado de novo com backoff curto; se esgotar, cai para o próximo candidato.
_MAX_RETENTATIVAS = int(os.getenv("AL_IADO_GEMINI_RETENTATIVAS", "2"))
_BACKOFF_BASE_S = float(os.getenv("AL_IADO_GEMINI_BACKOFF_S", "1.2"))


def _dormir(segundos: float) -> None:
    if segundos > 0:
        time.sleep(segundos)


# NÍVEL 3 — modelo de FUNDO (extração de metadados de PDF e consolidação de
# memória): o mais econômico da família, com o maior limite de taxa. Usa o
# identificador GA verificado e mantém fallbacks para futura aposentadoria.
MODELO_GEMINI_FUNDO = os.getenv("AL_IADO_GEMINI_MODEL_FUNDO", "gemini-3.5-flash-lite")


# ============================================================
# DEFINIÇÃO DOS PROVEDORES
# ============================================================

PROVEDORES = {
    "1": {
        "nome"      : "Google Gemini",
        # Default = Flash GA: rápido e estável (o Pro/preview é lento no trivial
        # e sujeito a 503 de alta demanda). Para máximo raciocínio, o pesquisador
        # sobe para gemini-pro-latest via AL_IADO_GEMINI_MODEL (cai no Flash se
        # o Pro não estiver liberado/estiver sobrecarregado).
        "modelo"    : os.getenv("AL_IADO_GEMINI_MODEL", "gemini-3.6-flash"),
        "env_key"   : "GOOGLE_API_KEY",
        "limite"    : "conforme o plano da API",
        "emoji"     : "🔵",
        "multimodal": True,   # entende imagens (visão)
    },
    "2": {
        "nome"      : "Google Gemini (auditor)",
        "modelo"    : os.getenv("AL_IADO_GEMINI_MODEL_AUDITOR", "gemini-3.5-flash-lite"),
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


_SEM_ITEM = object()  # sentinela: stream vazio sem confundir com item None


@dataclass(frozen=True)
class RespostaLLM:
    """Resposta mínima compatível com o contrato usado pelo agente."""

    content: str


def _conteudo_da_mensagem(mensagem):
    if isinstance(mensagem, dict):
        return mensagem.get("content", "")
    return getattr(mensagem, "content", mensagem)


def texto_da_resposta(resposta) -> str:
    """Converte respostas dos SDKs/integrações em texto puro.

    Versões recentes dos provedores podem devolver ``content`` como uma lista
    de blocos em vez de ``str``. Centralizar a conversão evita que cada fluxo
    tente concatenar ou interpretar diretamente estruturas incompatíveis.
    Blocos não textuais são ignorados deliberadamente.
    """

    conteudo = resposta if isinstance(resposta, (str, list, tuple, dict)) else getattr(
        resposta, "content", resposta
    )

    def _partes(valor):
        if valor is None:
            return
        if isinstance(valor, str):
            if valor:
                yield valor
            return
        if isinstance(valor, (list, tuple)):
            for item in valor:
                yield from _partes(item)
            return
        if isinstance(valor, dict):
            tipo = str(valor.get("type", "")).lower()
            if tipo and tipo not in {"text", "output_text"}:
                return
            for chave in ("text", "content", "value"):
                if chave in valor:
                    yield from _partes(valor[chave])
                    return
            return

        texto = getattr(valor, "text", None)
        if texto is not None:
            yield from _partes(texto)
            return
        interno = getattr(valor, "content", None)
        if interno is not None and interno is not valor:
            yield from _partes(interno)

    return "".join(_partes(conteudo) or ())


# Fallback GA explicito: se o modelo configurado some, a chamada cai na versao
# estavel anterior do Flash sem pagar a tentativa em um alias inexistente.
MODELO_GEMINI_FALLBACK = os.getenv("AL_IADO_GEMINI_FALLBACK", "gemini-3.5-flash")

# Modelo ALTERNATIVO de último recurso quando o principal está sobrecarregado
# (503): pool de capacidade diferente do Flash, para não só re-bater no mesmo
# modelo lotado. Qualidade menor, mas responde em vez de estourar erro.
MODELO_GEMINI_ALTERNATIVO = os.getenv(
    "AL_IADO_GEMINI_ALTERNATIVO", "gemini-3.5-flash-lite"
)


def _erro_de_modelo_indisponivel(exc) -> bool:
    """True quando a exceção indica modelo inexistente/aposentado (404 etc.).

    Ex.: 'This model models/gemini-2.5-pro is no longer available...',
         'NOT_FOUND', 'is not supported', 'was not found'.
    """
    txt = str(exc).lower()
    return any(s in txt for s in (
        "not_found", "not found", "was not found", "404",
        "no longer available", "is not supported", "not supported for",
        "does not exist", "unknown model",
    ))


def _erro_transitorio(exc) -> bool:
    """True para erros TEMPORÁRIOS de disponibilidade/limite — vale retentar.

    Ex.: '503 UNAVAILABLE ... experiencing high demand', '429 RESOURCE_EXHAUSTED',
         'overloaded', 'try again later'. NÃO inclui 500/erro interno, que é
         tratado como falha dura (propaga).
    """
    txt = str(exc).lower()
    return any(s in txt for s in (
        "503", "unavailable", "high demand", "overloaded",
        "429", "resource_exhausted", "rate limit", "try again",
    ))


class GeminiLeve:
    """Adaptador do SDK Google Gen AI sem carregar a integração LangChain.

    Resiliência a rotação de modelos: cada chamada tenta o modelo configurado
    e, se ele estiver indisponível (404/aposentado), cai para os `fallbacks`
    (por fim MODELO_GEMINI_FALLBACK). O modelo que funcionar vira o novo
    `self.model`, então a troca custa no máximo a primeira chamada da sessão.
    """

    def __init__(self, api_key: str, model: str, temperature: float = 0.45,
                 max_output_tokens: int = 8192, client=None,
                 fallbacks: tuple[str, ...] = (),
                 thinking_level: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = max(256, int(max_output_tokens))
        self._client = client
        self.fallbacks = tuple(f for f in fallbacks if f)
        niveis = {"minimal", "low", "medium", "high"}
        if thinking_level is not None and thinking_level not in niveis:
            raise ValueError(f"thinking_level invalido: {thinking_level}")
        self.thinking_level = thinking_level

    def _obter_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _candidatos(self) -> list[str]:
        """Modelos a tentar, em ordem, sem repetição."""
        vistos: set[str] = set()
        ordem = []
        for m in (self.model, *self.fallbacks, MODELO_GEMINI_FALLBACK):
            if m and m not in vistos:
                vistos.add(m)
                ordem.append(m)
        return ordem

    def _config_para_modelo(self, modelo: str, config: dict) -> dict:
        """Adapta a configuracao ao contrato dos modelos Gemini 3.x."""
        saida = dict(config)
        if modelo.startswith("gemini-3."):
            # Sampling foi descontinuado na familia 3.x. O nivel de thinking
            # explicito evita que o chat pague o custo medio em toda pergunta.
            saida.pop("temperature", None)
            if self.thinking_level:
                saida["thinking_config"] = {
                    "thinking_level": self.thinking_level,
                }
        return saida

    def _gerar(self, contents, config, *, stream: bool = False):
        """Executa generate_content(_stream) com retry transitório + fallback.

        Para cada modelo candidato: retenta em erro TRANSITÓRIO (503/429) com
        backoff curto; se esgotar, ou se o modelo estiver INDISPONÍVEL (404),
        passa ao próximo candidato. Erro d/outro tipo (ex.: 500) propaga.
        Retorna a resposta (não-stream) ou um gerador de itens (stream).
        """
        erro_final = None
        for modelo in self._candidatos():
            config_modelo = self._config_para_modelo(modelo, config)
            for tentativa in range(1, _MAX_RETENTATIVAS + 2):
                try:
                    cliente = self._obter_client()
                    if stream:
                        fluxo = cliente.models.generate_content_stream(
                            model=modelo, contents=contents, config=config_modelo,
                        )
                        iterador = iter(fluxo)
                        primeiro = next(iterador, _SEM_ITEM)  # força a chamada
                        self._fixar_modelo(modelo)

                        def _fluxo():
                            if primeiro is not _SEM_ITEM:
                                yield primeiro
                            yield from iterador

                        return _fluxo()
                    resposta = cliente.models.generate_content(
                        model=modelo, contents=contents, config=config_modelo,
                    )
                    self._fixar_modelo(modelo)
                    return resposta
                except Exception as exc:  # noqa: BLE001
                    if _erro_de_modelo_indisponivel(exc):
                        erro_final = exc
                        break  # modelo aposentado → próximo candidato, sem retry
                    if _erro_transitorio(exc):
                        erro_final = exc
                        if tentativa <= _MAX_RETENTATIVAS:
                            _dormir(_BACKOFF_BASE_S * tentativa)
                            continue  # retenta o MESMO modelo
                        break  # esgotou retries → próximo candidato
                    raise  # erro não-transitório e não-de-modelo → propaga
        if erro_final is not None:
            raise erro_final
        raise RuntimeError("Nenhum modelo Gemini candidato disponível.")

    def _fixar_modelo(self, modelo: str) -> None:
        if modelo != self.model:
            print(
                f"   ⚠️  Gemini: modelo '{self.model}' indisponível — "
                f"usando '{modelo}'. Ajuste AL_IADO_GEMINI_MODEL se quiser fixar."
            )
            self.model = modelo

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
        resposta = self._gerar(
            self._conteudos(mensagens),
            {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )
        return RespostaLLM(content=getattr(resposta, "text", "") or "")

    def invoke_json(self, mensagens, max_tokens: int = 700) -> dict:
        """Executa uma chamada curta e exige um objeto JSON (papel de auditor)."""
        resposta = self._gerar(
            self._conteudos(mensagens),
            {
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
        fluxo = self._gerar(
            self._conteudos(mensagens),
            {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
            stream=True,
        )
        for item in fluxo:
            texto = getattr(item, "text", "") or ""
            if texto:
                yield RespostaLLM(content=texto)


def inicializar_llm_fundo(
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
) -> GeminiLeve:
    """Cria o Gemini econômico usado em tarefas administrativas de fundo."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY não configurada para a tarefa de fundo."
        )
    return GeminiLeve(
        model=MODELO_GEMINI_FUNDO,
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        fallbacks=(MODELO_GEMINI_ALTERNATIVO, MODELO_GEMINI_FALLBACK),
        thinking_level=os.getenv("AL_IADO_GEMINI_THINKING_LEVEL_FUNDO", "minimal"),
    )


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

    # NÍVEL 1 — conversa, síntese e imagens. Fallbacks dão um modelo ALTERNATIVO
    # de verdade (Flash → Flash-Lite): num 503 de alta demanda, escapar para um
    # pool de capacidade diferente responde em vez de só bater no modelo lotado.
    if escolha == "1":
        llm = GeminiLeve(
            model=info["modelo"],
            api_key=api_key,
            temperature=float(os.getenv("AL_IADO_GEMINI_TEMPERATURE", "0.45")),
            max_output_tokens=int(
                os.getenv("AL_IADO_GEMINI_MAX_OUTPUT_TOKENS", "8192")
            ),
            fallbacks=(MODELO_GEMINI_FALLBACK, MODELO_GEMINI_ALTERNATIVO),
            thinking_level=os.getenv("AL_IADO_GEMINI_THINKING_LEVEL", "low"),
        )

    # NÍVEL 2 — Gemini Flash: auditor de evidências e porteiro da memória.
    # Temperatura baixa; as chamadas JSON do papel forçam 0.0 no invoke_json.
    elif escolha == "2":
        llm = GeminiLeve(
            model=info["modelo"],
            api_key=api_key,
            temperature=0.2,
            max_output_tokens=2048,
            fallbacks=(MODELO_GEMINI_ALTERNATIVO,),
            thinking_level=os.getenv(
                "AL_IADO_GEMINI_THINKING_LEVEL_AUDITOR", "minimal"
            ),
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
