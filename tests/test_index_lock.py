"""
Sprint 4 — lock de indexação (item 10.3 / critério #21).

O lock serializa operações concorrentes de indexação (watcher × chat).
"""

import threading
import time

from src.conhecimento.index_lock import indexacao_ocupada, lock_indexacao


def test_lock_serializa_threads():
    eventos = []

    def trabalho(nome):
        with lock_indexacao(timeout=5):
            eventos.append(f"{nome}-inicio")
            time.sleep(0.05)
            eventos.append(f"{nome}-fim")

    t1 = threading.Thread(target=trabalho, args=("A",))
    t2 = threading.Thread(target=trabalho, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # serializado → cada início é imediatamente seguido pelo seu próprio fim
    assert eventos[0].endswith("-inicio")
    assert eventos[1] == eventos[0].replace("-inicio", "-fim")
    assert eventos[2].endswith("-inicio")
    assert eventos[3] == eventos[2].replace("-inicio", "-fim")


def test_indexacao_ocupada():
    assert not indexacao_ocupada()
    with lock_indexacao():
        assert indexacao_ocupada()
    assert not indexacao_ocupada()
