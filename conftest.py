"""
Configuração compartilhada do pytest — Al IAdo PV.

- Garante `KMP_DUPLICATE_LIB_OK` antes de qualquer import nativo pesado
  (ver src/core/config.py) para evitar crash de OpenMP no Windows.
- Coloca a raiz do projeto no sys.path para permitir `import src...` ao
  rodar `python -m pytest` a partir da raiz.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_RAIZ = Path(__file__).resolve().parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))


_MARCAS_POR_ARQUIVO = {
    # Dependem de bibliotecas opcionais ou validam mecânica de modelos pesados,
    # mas podem pular honestamente quando a dependência não está instalada.
    "test_ae_lstm_temporal.py": ("integracao",),
    # Usam ChromaDB real. Ficam fora do CI leve, que não instala chromadb.
    "test_indexacao_pagina.py": ("pesado",),
    "test_obsidian_cerebro.py": ("pesado",),
}


def pytest_collection_modifyitems(config, items):
    """Marca automaticamente a suíte para o CI rodar por expressão de marcador.

    O padrão é conservador: todo teste é `leve` até que um arquivo inteiro seja
    classificado como integração/pesado. Assim novos testes entram no CI leve por
    padrão e a lista do workflow não volta a ficar manual e incompleta.
    """
    del config
    for item in items:
        arquivo = Path(str(item.fspath)).name
        marcas = _MARCAS_POR_ARQUIVO.get(arquivo, ("leve",))
        for marca in marcas:
            item.add_marker(getattr(pytest.mark, marca))
