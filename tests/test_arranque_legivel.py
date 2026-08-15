"""
Dependência ausente tem de dizer o que fazer, não produzir traceback duplo.

POR QUE ESTE TESTE EXISTE
=========================
Em 15/08/2026 o pesquisador rodou `python -m src.ml.macro_weibull` com o
ambiente virtual DESATIVADO. O que apareceu foi:

    ModuleNotFoundError: No module named 'dotenv'
    During handling of the above exception, another exception occurred:
    ...
    ModuleNotFoundError: No module named 'dotenv'

Duas vezes o mesmo erro, e nenhuma pista da causa. O bloco de arranque dos
módulos de `src/ml/` fazia::

    try:
        from src.core.logs import ...
    except ModuleNotFoundError:      # <- engolia QUALQUER módulo
        <insere a raiz no sys.path>
        from src.core.logs import ...   # <- reimportava, e falhava igual

O `except` existe para um caso só: rodar o arquivo direto, sem a raiz do
projeto no `sys.path`. Uma dependência genuinamente ausente não é isso — e
reimportar não a faz aparecer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _blocos_de_arranque():
    """Todo `except ModuleNotFoundError` que reinsere a raiz no sys.path."""
    for arquivo in sorted((RAIZ / "src").rglob("*.py")):
        texto = arquivo.read_text(encoding="utf-8")
        if "_sys.path.insert" not in texto:
            continue
        for no in ast.walk(ast.parse(texto)):
            if isinstance(no, ast.ExceptHandler) and "sys.path.insert" in ast.dump(no):
                yield arquivo.relative_to(RAIZ).as_posix(), no


def test_todo_arranque_repassa_dependencia_que_nao_e_do_pacote():
    """A guarda tem de existir em TODOS — um esquecido reproduz o sintoma."""
    sem_guarda = []
    for caminho, handler in _blocos_de_arranque():
        corpo = ast.dump(handler)
        # A guarda é um `raise` condicionado ao nome do módulo que faltou.
        if "Raise" not in corpo or "name" not in corpo:
            sem_guarda.append(f"{caminho}:{handler.lineno}")
    assert not sem_guarda, (
        "blocos de arranque que ainda engolem qualquer dependência ausente: "
        f"{sem_guarda}. Sem a guarda, venv desativado vira traceback duplo com "
        "a causa enterrada."
    )


def test_o_arranque_nomeia_a_excecao_para_poder_inspecionar():
    for caminho, handler in _blocos_de_arranque():
        assert handler.name, (
            f"{caminho}:{handler.lineno} captura sem nomear a exceção — sem o "
            "objeto não dá para saber QUAL módulo faltou"
        )


def test_dotenv_ausente_explica_o_ambiente_virtual():
    """A mensagem tem de dizer a causa provável e o comando de saída."""
    fonte = (RAIZ / "src/core/config.py").read_text(encoding="utf-8")
    assert "python-dotenv" in fonte
    assert "Activate.ps1" in fonte, "o pesquisador está no Windows"
    assert "source .venv/bin/activate" in fonte
    assert "requirements.txt" in fonte


@pytest.mark.parametrize("modulo", [
    "src.ml.macro_weibull", "src.ml.autoencoder", "src.ml.validacao",
    "src.ml.injecao_falhas", "src.ml.rul_weibull", "src.ml.macro_comparar",
])
def test_os_modulos_continuam_importaveis(modulo):
    """A guarda não pode ter quebrado o caminho normal."""
    import importlib

    assert importlib.import_module(modulo) is not None
