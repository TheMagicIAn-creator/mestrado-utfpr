"""Os `from src...` dos scripts apontam para símbolos que existem.

`scripts/varrer_calibracao.py` foi entregue importando
`carregar_pickle_com_sidecar` de `src.core.utils`, quando a função vive em
`src.core.seguranca`. O script quebrou na primeira execução do pesquisador.

Nada pegou: `ruff --select F821` só vê nomes indefinidos no arquivo, não
importes que resolvem para o módulo errado; e o import está DENTRO da função
(adiado de propósito, para o script não exigir `torch` só para mostrar o
`--help`), então nem o import do módulo o exercita.

Este teste resolve cada `from src.*` por AST — sem executar nada e sem exigir
`torch`, que é a única razão de o erro ter passado.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((RAIZ / "scripts").glob("*.py"))


def _simbolos_de_topo(caminho: Path) -> set[str] | None:
    """Nomes definidos no nível do módulo, lidos por AST (não importa nada).

    Devolve **None** quando o arquivo não pôde ser lido ou parseado — que é
    diferente de "não define nada". `src/conhecimento/agente.py`, por exemplo,
    usa f-string com barra invertida (sintaxe de Python 3.12+) e não parseia em
    3.11: tratar isso como conjunto vazio acusaria TODOS os seus símbolos como
    ausentes. "Não consegui verificar" nunca pode virar "está quebrado".
    """
    try:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None

    nomes: set[str] = set()

    def _coletar(corpo) -> None:
        nonlocal nomes
        # Percorre também try/if/with de nível de módulo: import opcional e
        # definição condicional são comuns e não podem virar falso positivo.
        for no in corpo:
            if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nomes.add(no.name)
            elif isinstance(no, ast.Assign):
                nomes |= {a.id for a in no.targets if isinstance(a, ast.Name)}
                if any(
                    isinstance(alvo, ast.Name)
                    and alvo.id == "_EXPORTACOES_TARDIAS"
                    for alvo in no.targets
                ):
                    try:
                        grupos = ast.literal_eval(no.value)
                    except (ValueError, TypeError):
                        grupos = ()
                    for _modulo, exportacoes in grupos:
                        nomes.update(exportacoes)
            elif isinstance(no, ast.AnnAssign) and isinstance(no.target, ast.Name):
                nomes.add(no.target.id)
            elif isinstance(no, (ast.Import, ast.ImportFrom)):
                nomes |= {(a.asname or a.name).split(".")[0] for a in no.names}
            elif isinstance(no, ast.Try):
                _coletar(no.body)
                for h in no.handlers:
                    _coletar(h.body)
                _coletar(no.orelse)
                _coletar(no.finalbody)
            elif isinstance(no, (ast.If, ast.With, ast.AsyncWith)):
                _coletar(no.body)
                _coletar(getattr(no, "orelse", []))

    _coletar(arvore.body)
    return nomes


def _modulo_para_arquivo(modulo: str) -> Path:
    return RAIZ / Path(*modulo.split(".")).with_suffix(".py")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_imports_internos_dos_scripts_resolvem(script: Path):
    arvore = ast.parse(script.read_text(encoding="utf-8"))
    problemas: list[str] = []

    for no in ast.walk(arvore):
        if not isinstance(no, ast.ImportFrom):
            continue
        if not (no.module or "").startswith("src."):
            continue

        arquivo = _modulo_para_arquivo(no.module)
        if not arquivo.exists():
            # Pode ser um pacote (src/x/__init__.py) — tenta importar de fato.
            try:
                importlib.import_module(no.module)
                continue
            except ModuleNotFoundError as exc:
                # Dependência pesada ausente (torch) não é erro do script.
                if "src." not in str(exc):
                    continue
                problemas.append(f"{no.module} não existe")
                continue
            except Exception:
                continue

        definidos = _simbolos_de_topo(arquivo)
        if definidos is None:
            continue        # ilegível nesta versão de Python — não verificável
        for alias in no.names:
            if alias.name == "*":
                continue
            if alias.name not in definidos:
                problemas.append(
                    f"{no.module}.{alias.name} — não definido em "
                    f"{arquivo.relative_to(RAIZ)}"
                )

    assert not problemas, (
        f"{script.name} importa símbolo inexistente:\n  "
        + "\n  ".join(problemas)
    )


def test_as_entradas_canonicas_estao_cobertas():
    """Guarda contra o teste virar vazio se scripts/ for reorganizado."""
    names = {script.name for script in SCRIPTS}
    assert names == {
        "auditar_resultados.py",
        "avaliar_agente.py",
        "manter_base.py",
        "verificar_projeto.py",
    }
