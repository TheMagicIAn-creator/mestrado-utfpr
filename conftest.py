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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_RAIZ = Path(__file__).resolve().parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))
