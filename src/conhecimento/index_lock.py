"""
index_lock.py — Al IAdo PV / Sprint 4 (robustez)

Lock de INDEXAÇÃO: serializa as escritas no ChromaDB que podem ocorrer em
paralelo (watcher de PDFs, chat, reprocessamento). No app, o watcher roda numa
thread de background DENTRO do processo do Streamlit — a concorrência real é
in-process, coberta por um threading.Lock compartilhado.

Uso:
    from src.conhecimento.index_lock import lock_indexacao
    with lock_indexacao():
        ... operação de indexação (escrita no ChromaDB) ...

Cross-processo (ex.: watcher.py standalone + app aberto) fica como backlog
(exigiria um file-lock dedicado).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

_LOCK = threading.Lock()


def indexacao_ocupada() -> bool:
    """True se o lock está tomado neste momento (sem bloquear)."""
    obtido = _LOCK.acquire(blocking=False)
    if obtido:
        _LOCK.release()
        return False
    return True


@contextmanager
def lock_indexacao(timeout: float = 180.0):
    """
    Garante indexação serializada. Bloqueia até obter o lock (ou estoura
    `timeout`, levantando TimeoutError) e o libera ao sair.
    """
    obtido = _LOCK.acquire(timeout=timeout)
    if not obtido:
        raise TimeoutError("Indexação ocupada — lock não obtido no tempo limite.")
    try:
        yield
    finally:
        _LOCK.release()
