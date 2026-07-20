"""Indice lexical BM25 em SQLite FTS5 para complementar o ChromaDB.

O indice e derivado da colecao de literatura e pode ser reconstruido. Ele nao
substitui embeddings: fornece a segunda lista de candidatos para fusao RRF.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from src.core.config import ARQUIVO_INDICE_LEXICAL


_LOCK = threading.RLock()
_STOPWORDS = {
    "para", "como", "com", "dos", "das", "uma", "que", "the", "and",
    "from", "por", "les", "des", "une", "los", "las", "del",
}


def _normalizar(texto: str) -> str:
    base = unicodedata.normalize("NFKD", str(texto).lower())
    sem_acentos = "".join(c for c in base if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", sem_acentos)).strip()


def _tokens(textos) -> list[str]:
    vistos = set()
    saida = []
    for texto in textos:
        for token in _normalizar(texto).split():
            if len(token) < 3 or token in _STOPWORDS or token in vistos:
                continue
            vistos.add(token)
            saida.append(token)
    return saida


@dataclass(frozen=True)
class ResultadoLexical:
    chunk_id: str
    documento: str
    metadata: dict
    rank: int
    score_bm25: float


class IndiceLexicalSQLite:
    """Indice FTS5 persistente e de baixo consumo de memoria."""

    def __init__(self, caminho: str | Path | None = None) -> None:
        self.caminho = Path(caminho or ARQUIVO_INDICE_LEXICAL)
        self._disponivel = True
        try:
            self._inicializar()
        except sqlite3.Error:
            self._disponivel = False

    @property
    def disponivel(self) -> bool:
        return self._disponivel

    def _conectar(self):
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        conexao = sqlite3.connect(str(self.caminho), timeout=30)
        conexao.execute("PRAGMA journal_mode=WAL")
        conexao.execute("PRAGMA synchronous=NORMAL")
        return conexao

    def _inicializar(self) -> None:
        with _LOCK, self._conectar() as conexao:
            conexao.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    documento,
                    metadata_json UNINDEXED,
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
            conexao.execute(
                "CREATE TABLE IF NOT EXISTS estado (chave TEXT PRIMARY KEY, valor TEXT)"
            )

    @staticmethod
    def _estado(conexao, chave: str) -> str | None:
        linha = conexao.execute(
            "SELECT valor FROM estado WHERE chave = ?", (chave,)
        ).fetchone()
        return str(linha[0]) if linha else None

    def sincronizar(
        self,
        colecao,
        *,
        versao: str,
        tamanho_lote: int = 300,
    ) -> dict:
        """Reconstrui o FTS somente quando versao ou contagem mudaram."""
        if not self.disponivel:
            return {"disponivel": False, "reconstruido": False, "n_chunks": 0}
        esperado = int(colecao.count())
        with _LOCK, self._conectar() as conexao:
            atual = self._estado(conexao, "versao")
            quantidade = int(
                conexao.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
            )
            if atual == str(versao) and quantidade == esperado:
                return {
                    "disponivel": True,
                    "reconstruido": False,
                    "n_chunks": quantidade,
                }

            conexao.execute("DELETE FROM chunks_fts")
            inseridos = 0
            for offset in range(0, esperado, tamanho_lote):
                lote = colecao.get(
                    limit=tamanho_lote,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                ids = lote.get("ids") or []
                documentos = lote.get("documents") or []
                metadados = lote.get("metadatas") or []
                registros = [
                    (
                        str(chunk_id),
                        str(documento or ""),
                        json.dumps(metadata or {}, ensure_ascii=False, default=str),
                    )
                    for chunk_id, documento, metadata in zip(
                        ids, documentos, metadados
                    )
                ]
                conexao.executemany(
                    "INSERT INTO chunks_fts(chunk_id, documento, metadata_json) "
                    "VALUES (?, ?, ?)",
                    registros,
                )
                inseridos += len(registros)
            if inseridos != esperado:
                raise ValueError(
                    f"Indice lexical incompleto: {inseridos}/{esperado} chunks."
                )
            conexao.execute(
                "INSERT OR REPLACE INTO estado(chave, valor) VALUES ('versao', ?)",
                (str(versao),),
            )
            conexao.execute(
                "INSERT OR REPLACE INTO estado(chave, valor) VALUES ('n_chunks', ?)",
                (str(inseridos),),
            )
            return {
                "disponivel": True,
                "reconstruido": True,
                "n_chunks": inseridos,
            }

    def buscar(
        self,
        consultas: list[str] | tuple[str, ...] | str,
        *,
        termos: list[str] | None = None,
        limite: int = 80,
    ) -> list[ResultadoLexical]:
        if not self.disponivel:
            return []
        textos = [consultas] if isinstance(consultas, str) else list(consultas)
        tokens = _tokens(textos + list(termos or []))[:40]
        if not tokens:
            return []
        expressao = " OR ".join(f'"{token}"' for token in tokens)
        try:
            with _LOCK, self._conectar() as conexao:
                linhas = conexao.execute(
                    """
                    SELECT chunk_id, documento, metadata_json, bm25(chunks_fts)
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25(chunks_fts)
                    LIMIT ?
                    """,
                    (expressao, max(1, int(limite))),
                ).fetchall()
        except sqlite3.Error:
            return []

        resultados = []
        for rank, (chunk_id, documento, metadata_json, score) in enumerate(linhas, 1):
            try:
                metadata = json.loads(metadata_json or "{}")
            except json.JSONDecodeError:
                metadata = {}
            resultados.append(
                ResultadoLexical(
                    chunk_id=str(chunk_id),
                    documento=str(documento or ""),
                    metadata=metadata,
                    rank=rank,
                    score_bm25=float(score),
                )
            )
        return resultados
