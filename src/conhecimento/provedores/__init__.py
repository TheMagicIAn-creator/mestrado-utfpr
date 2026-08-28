"""Gateway multi-provider e fachada compatível do ALIAdo."""

from src.conhecimento.provedores.gateway import ProviderGateway, build_default_gateway
from src.conhecimento.provedores.gemini import (
    MODELO_GEMINI_ALTERNATIVO,
    MODELO_GEMINI_FALLBACK,
    MODELO_GEMINI_FUNDO,
    PAPEIS_AGENTES,
    PROVEDORES,
    GeminiLeve,
    GeminiProvider,
    RespostaLLM,
    eh_multimodal,
    inicializar_llm_fundo,
    inicializar_papel,
    inicializar_provedor,
    listar_provedores,
    selecionar_provedor,
    texto_da_resposta,
)
from src.conhecimento.provedores.openai import OpenAIProvider
from src.conhecimento.provedores.registry import (
    ModelRegistration,
    ModelStatus,
    ProviderRegistry,
)

__all__ = [
    "GeminiLeve",
    "GeminiProvider",
    "MODELO_GEMINI_ALTERNATIVO",
    "MODELO_GEMINI_FALLBACK",
    "MODELO_GEMINI_FUNDO",
    "ModelRegistration",
    "ModelStatus",
    "OpenAIProvider",
    "PAPEIS_AGENTES",
    "PROVEDORES",
    "ProviderGateway",
    "ProviderRegistry",
    "RespostaLLM",
    "build_default_gateway",
    "eh_multimodal",
    "inicializar_llm_fundo",
    "inicializar_papel",
    "inicializar_provedor",
    "listar_provedores",
    "selecionar_provedor",
    "texto_da_resposta",
]
