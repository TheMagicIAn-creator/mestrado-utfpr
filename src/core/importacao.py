"""Reexportacoes tardias para fachadas modulares sem ciclos de importacao."""

from __future__ import annotations

from importlib import import_module


def resolver_exportacao_tardia(
    nome: str,
    grupos: tuple[tuple[str, tuple[str, ...]], ...],
    namespace: dict,
):
    """Carrega um simbolo movido somente quando a fachada for consultada."""
    for modulo, nomes in grupos:
        if nome not in nomes:
            continue
        valor = getattr(import_module(modulo), nome)
        namespace[nome] = valor
        return valor
    raise AttributeError(f"modulo {namespace.get('__name__', '')!r} sem atributo {nome!r}")
