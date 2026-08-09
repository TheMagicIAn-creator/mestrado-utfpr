"""Normalizacao textual compartilhada entre os modulos do projeto."""

from __future__ import annotations

import re
import unicodedata


def normalizar_sem_acentos(texto: object) -> str:
    """Converte para minusculas e remove marcas diacriticas."""
    base = unicodedata.normalize(
        "NFKD", str("" if texto is None else texto).lower()
    )
    return "".join(caractere for caractere in base if not unicodedata.combining(caractere))


def normalizar_espacos(texto: object) -> str:
    """Remove acentos e compacta espacos, preservando pontuacao."""
    return re.sub(r"\s+", " ", normalizar_sem_acentos(texto)).strip()


def normalizar_busca(texto: object) -> str:
    """Produz texto ASCII alfanumerico adequado a busca lexical."""
    sem_pontuacao = re.sub(r"[^a-z0-9\s]", " ", normalizar_sem_acentos(texto))
    return re.sub(r"\s+", " ", sem_pontuacao).strip()
