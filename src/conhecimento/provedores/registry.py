"""Registry explícito de provedores e modelos lógicos do ALIAdo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.conhecimento.provedores.base import LLMProvider, ProviderError


class ModelStatus(StrEnum):
    OPERATIONAL = "operational"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ModelRegistration:
    provider: str
    alias: str
    model_id: str | None
    status: ModelStatus = ModelStatus.OPERATIONAL
    capabilities: frozenset[str] = frozenset({"text"})

    @property
    def configured(self) -> bool:
        return bool(self.model_id and self.status != ModelStatus.DISABLED)


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._models: dict[tuple[str, str], ModelRegistration] = {}

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Nome do provedor não pode ser vazio")
        self._providers[key] = provider

    def register_model(self, registration: ModelRegistration) -> None:
        key = (registration.provider.lower(), registration.alias.lower())
        if key[0] not in self._providers:
            raise ValueError(f"Provedor não registrado: {registration.provider}")
        self._models[key] = registration

    def provider(self, name: str) -> LLMProvider:
        key = str(name).lower()
        if key not in self._providers:
            raise ProviderError(f"Provedor desconhecido: {name}", unavailable=True)
        return self._providers[key]

    def model(
        self,
        provider: str,
        alias: str,
        *,
        allow_experimental: bool = False,
        capabilities: set[str] | frozenset[str] = frozenset(),
    ) -> ModelRegistration:
        key = (str(provider).lower(), str(alias).lower())
        if key not in self._models:
            raise ProviderError(
                f"Modelo lógico desconhecido: {provider}/{alias}", unavailable=True
            )
        model = self._models[key]
        if model.status == ModelStatus.DISABLED or not model.model_id:
            raise ProviderError(
                f"Modelo lógico indisponível: {provider}/{alias}", unavailable=True
            )
        if model.status == ModelStatus.EXPERIMENTAL and not allow_experimental:
            raise ProviderError(
                f"Modelo experimental bloqueado no caminho crítico: {provider}/{alias}",
                unavailable=True,
            )
        missing = set(capabilities) - set(model.capabilities)
        if missing:
            raise ProviderError(
                f"Modelo {provider}/{alias} não oferece: {sorted(missing)}",
                unavailable=True,
            )
        return model

    def status(self) -> dict:
        providers = {
            name: {"configured": bool(provider.configured)}
            for name, provider in sorted(self._providers.items())
        }
        models = [
            {
                "provider": item.provider,
                "alias": item.alias,
                "model_id": item.model_id,
                "status": item.status.value,
                "capabilities": sorted(item.capabilities),
                "configured": item.configured,
            }
            for item in sorted(
                self._models.values(), key=lambda value: (value.provider, value.alias)
            )
        ]
        return {"providers": providers, "models": models}


__all__ = ["ModelRegistration", "ModelStatus", "ProviderRegistry"]
