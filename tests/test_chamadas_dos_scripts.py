"""
Scripts de `scripts/` chamam funções de `src/` com nomes de argumento válidos.

POR QUE ESTE TESTE EXISTE
=========================
Um script pode morrer na última linha, depois de uma execução pesada, quando
usa um nome de argumento inexistente. Como `scripts/` não é exercitado por
completo na suíte, esta guarda valida estaticamente esse contrato.

Esta guarda fecha a classe inteira do problema em vez do caso: percorre os
scripts por AST, encontra as chamadas a funções importadas de `src/`, e confere
cada palavra-chave contra a assinatura real. Não executa nada — só compara nomes.

O que ela NÃO cobre, e é bom deixar explícito: argumentos posicionais, tipos,
valores, e chamadas por atributo (`modulo.funcao(...)`). É uma rede para o erro
que de fato aconteceu, não um type-checker.
"""

from __future__ import annotations

import ast
import inspect
import importlib

import pytest

from src.core.config import RAIZ_PROJETO

PASTA_SCRIPTS = RAIZ_PROJETO / "scripts"


def _scripts() -> list:
    return sorted(p for p in PASTA_SCRIPTS.glob("*.py") if p.name != "__init__.py")


def _importados_de_src(arvore: ast.Module) -> dict[str, str]:
    """`{nome_local: modulo_de_origem}` para tudo que vem de `src.` ou `scripts.`.

    Cobre imports no topo E dentro de funções — os scripts importam tarde de
    propósito, para não carregar torch quando só querem `--help`.
    """
    origem = {}
    for node in ast.walk(arvore):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not node.module.startswith(("src.", "scripts.")):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                origem[alias.asname or alias.name] = node.module
    return origem


def _chamadas_com_palavra_chave(arvore: ast.Module):
    """`(nome_chamado, [palavras-chave], linha)` para cada chamada por nome."""
    for node in ast.walk(arvore):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        chaves = [kw.arg for kw in node.keywords if kw.arg is not None]
        if chaves:
            yield node.func.id, chaves, node.lineno


def _resolver(modulo: str, nome: str):
    """Devolve a função, ou None se o módulo não puder ser importado aqui.

    Dependência pesada ausente no ambiente de CI não deve reprovar o teste —
    seria degradação desonesta ao contrário: falhar por motivo errado.
    """
    try:
        mod = importlib.import_module(modulo)
    except Exception:
        return None
    alvo = getattr(mod, nome, None)
    return alvo if inspect.isfunction(alvo) else None


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_palavras_chave_batem_com_a_assinatura(script):
    arvore = ast.parse(script.read_text(encoding="utf-8"))
    origem = _importados_de_src(arvore)
    if not origem:
        pytest.skip(f"{script.name} não importa nada de src/")

    problemas = []
    for nome, chaves, linha in _chamadas_com_palavra_chave(arvore):
        modulo = origem.get(nome)
        if modulo is None:
            continue
        alvo = _resolver(modulo, nome)
        if alvo is None:
            continue

        parametros = inspect.signature(alvo).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parametros.values()):
            continue          # aceita **kwargs: qualquer nome é válido
        for chave in chaves:
            if chave not in parametros:
                validos = ", ".join(
                    n for n, p in parametros.items()
                    if p.kind is not inspect.Parameter.VAR_POSITIONAL
                )
                problemas.append(
                    f"{script.name}:{linha} — {nome}(...) recebe '{chave}', "
                    f"que não existe em {modulo}.{nome}. Válidos: {validos}"
                )

    assert not problemas, "\n".join(problemas)


def test_a_guarda_detecta_keyword_inexistente_em_api_canonica(tmp_path):
    """Contraprova: sem isto, o teste acima poderia estar sempre passando.

    A contraprova usa uma API canônica e confirma que a análise não passa
    silenciosamente quando um script troca o nome de um parâmetro.
    """
    script = tmp_path / "regressao.py"
    script.write_text(
        "from src.ml.modelos_autoencoder import sequences_from_flow\n"
        "sequences_from_flow([], janela=8)\n",
        encoding="utf-8",
    )
    arvore = ast.parse(script.read_text(encoding="utf-8"))
    origem = _importados_de_src(arvore)
    alvo = _resolver(origem["sequences_from_flow"], "sequences_from_flow")
    assert alvo is not None, "o contrato dos modelos deve importar sem treino"

    parametros = inspect.signature(alvo).parameters
    assert "janela" not in parametros
    assert "length" in parametros
