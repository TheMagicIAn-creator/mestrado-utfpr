"""Serialização das escritas no ChromaDB entre threads e processos."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from src.core.config import RAIZ_PROJETO

_LOCK_THREAD = threading.RLock()
_ESTADO_THREAD = threading.local()
_INTERVALO_TENTATIVA_S = 0.05


def _caminho_padrao() -> Path:
    configurado = os.getenv("AL_IADO_INDEX_LOCK_PATH")
    if configurado:
        return Path(configurado).expanduser().resolve()
    return RAIZ_PROJETO / "logs" / "chromadb-index.lock"


def _travar_arquivo(arquivo: BinaryIO) -> None:
    arquivo.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(arquivo.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(arquivo.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _destravar_arquivo(arquivo: BinaryIO) -> None:
    arquivo.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(arquivo.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(arquivo.fileno(), fcntl.LOCK_UN)


class _LockArquivo:
    def __init__(self, caminho: Path):
        self.caminho = caminho
        self.arquivo: BinaryIO | None = None

    def adquirir(self, timeout: float) -> bool:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        arquivo = self.caminho.open("a+b")
        if arquivo.tell() == 0:
            arquivo.write(b"\0")
            arquivo.flush()

        limite = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                _travar_arquivo(arquivo)
                self.arquivo = arquivo
                return True
            except (BlockingIOError, OSError):
                if time.monotonic() >= limite:
                    arquivo.close()
                    return False
                time.sleep(min(_INTERVALO_TENTATIVA_S, max(0.0, limite - time.monotonic())))

    def liberar(self) -> None:
        if self.arquivo is None:
            return
        try:
            _destravar_arquivo(self.arquivo)
        finally:
            self.arquivo.close()
            self.arquivo = None


def indexacao_ocupada(caminho_lock: Path | str | None = None) -> bool:
    """Informa sem bloquear se outra thread ou processo está escrevendo."""
    if getattr(_ESTADO_THREAD, "profundidade", 0):
        return True
    if not _LOCK_THREAD.acquire(blocking=False):
        return True
    try:
        lock_arquivo = _LockArquivo(Path(caminho_lock) if caminho_lock else _caminho_padrao())
        if not lock_arquivo.adquirir(0.0):
            return True
        lock_arquivo.liberar()
        return False
    finally:
        _LOCK_THREAD.release()


@contextmanager
def lock_indexacao(
    timeout: float = 180.0,
    *,
    caminho_lock: Path | str | None = None,
):
    """Serializa uma escrita; suporta chamadas aninhadas na mesma thread."""
    timeout = float(timeout)
    if timeout < 0:
        raise ValueError("timeout deve ser maior ou igual a zero")

    inicio = time.monotonic()
    if not _LOCK_THREAD.acquire(timeout=timeout):
        raise TimeoutError("Indexação ocupada: lock de thread não obtido no prazo.")

    profundidade = getattr(_ESTADO_THREAD, "profundidade", 0)
    lock_arquivo = None
    try:
        if profundidade == 0:
            restante = max(0.0, timeout - (time.monotonic() - inicio))
            caminho = Path(caminho_lock) if caminho_lock else _caminho_padrao()
            lock_arquivo = _LockArquivo(caminho)
            if not lock_arquivo.adquirir(restante):
                raise TimeoutError(
                    "Indexação ocupada por outro processo: file lock não obtido no prazo."
                )
            _ESTADO_THREAD.lock_arquivo = lock_arquivo
        _ESTADO_THREAD.profundidade = profundidade + 1
        yield
    finally:
        if getattr(_ESTADO_THREAD, "profundidade", 0):
            _ESTADO_THREAD.profundidade -= 1
        if getattr(_ESTADO_THREAD, "profundidade", 0) == 0:
            ativo = getattr(_ESTADO_THREAD, "lock_arquivo", lock_arquivo)
            if ativo is not None:
                ativo.liberar()
            _ESTADO_THREAD.lock_arquivo = None
        _LOCK_THREAD.release()
