"""Catalogo bibliografico rastreavel da literatura do ALIAdo.

O PDF e a identidade primaria: seu SHA-256 permanece estavel mesmo quando os
metadados curatoriais mudam. O catalogo e pequeno e versionavel; o snapshot
vetorial continua sendo um artefato reconstruivel separado.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit

from pypdf import PdfReader

from src.core.utils import parsear_nome_arquivo


SCHEMA_VERSION = 1
CATEGORIAS = (
    "confiabilidade",
    "inversores-pv",
    "manutencao",
    "ml-preditivo",
    "sinais-eletricos",
)
IDIOMAS = ("pt", "en", "es", "fr", "desconhecido")
_SOURCE_ID = re.compile(r"^[0-9a-f]{64}$")
_LOCK = threading.RLock()
_UNKNOWN_AUTHOR = "Autor desconhecido"
_EDITABLE_FIELDS = {"title", "authors", "year", "category", "language"}
_INTERNAL_FIELDS = {
    "relative_path",
    "file_name",
    "chunk_count",
    "size_bytes",
    "index_status",
    "extraction_warnings",
    "metadata_origin",
}
_METADADO_PDF_INVALIDO = (
    "acrobat",
    "microsoft word",
    "openoffice",
    "template for",
    "untitled",
)


class CatalogoBibliograficoInvalido(ValueError):
    """Indica catalogo ausente, inconsistente ou entrada invalida."""


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _agora_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def limpar_metadado(valor, *, limite: int = 500) -> str:
    """Normaliza metadados sem alterar o conteudo original dos PDFs."""
    texto = unicodedata.normalize("NFC", str(valor or ""))
    texto = texto.replace("\ufffd", " ").replace("\x00", " ")
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:limite].strip()


def _autores(valor) -> list[str]:
    if isinstance(valor, list):
        candidatos = valor
    else:
        texto = limpar_metadado(valor, limite=1000)
        candidatos = texto.split(";") if texto else []
    autores = []
    for item in candidatos:
        autor = limpar_metadado(item, limite=200)
        if autor and autor not in autores:
            autores.append(autor)
    return autores or [_UNKNOWN_AUTHOR]


def _autores_pdf(valor) -> list[str]:
    texto = limpar_metadado(valor, limite=1000)
    if not texto:
        return [_UNKNOWN_AUTHOR]
    normalizado = re.sub(r"\band\b", ";", texto, flags=re.IGNORECASE)
    return _autores(normalizado.replace(",", ";").replace("&", ";"))


def _metadados_pdf(caminho: Path) -> tuple[str, str, list[str]]:
    """Le metadados declarativos, sem tratar o PDF como fonte infalivel."""
    try:
        metadata = PdfReader(str(caminho)).metadata or {}
    except Exception:
        return "", "", ["metadado_pdf_interno_ilegivel"]
    titulo = limpar_metadado(metadata.get("/Title"), limite=800)
    autor = limpar_metadado(metadata.get("/Author"), limite=800)
    return titulo, autor, []


def _titulo_pdf_plausivel(titulo: str) -> bool:
    texto = titulo.casefold()
    esquema = urlsplit(titulo).scheme.casefold()
    sufixo = texto.rsplit(maxsplit=1)[-1] if texto else ""
    return bool(
        len(titulo) >= 12
        and esquema not in {"http", "https"}
        and not any(marcador in texto for marcador in _METADADO_PDF_INVALIDO)
        and not re.search(r"_[a-z]$", texto)
        and not re.fullmatch(r"n\d+", sufixo)
        and re.search(r"[a-zA-ZÀ-ÿ]{4}", titulo)
    )


def _titulos_relacionados(referencia: str, candidato: str) -> bool:
    def normalizar(texto: str) -> str:
        ascii_texto = unicodedata.normalize("NFKD", texto).encode(
            "ascii", "ignore"
        ).decode("ascii")
        return " ".join(re.findall(r"[a-z0-9]+", ascii_texto.casefold()))

    base = normalizar(referencia)
    interno = normalizar(candidato)
    if not base or not interno:
        return False
    menor, maior = sorted((base, interno), key=len)
    return menor in maior or SequenceMatcher(None, base, interno).ratio() >= 0.72


def _ano(valor) -> tuple[int | None, str]:
    texto = limpar_metadado(valor, limite=20)
    if texto in {"", "0", "0000", "s.d.", "s.d", "none", "null"}:
        return None, "desconhecido"
    if re.fullmatch(r"(?:18|19|20)\d{2}", texto):
        return int(texto), "informado"
    return None, "desconhecido"


def _citacao(autores: list[str], ano: int | None, titulo: str) -> str:
    autor = autores[0] if len(autores) == 1 else f"{autores[0]} et al."
    return f"{autor} ({ano if ano is not None else 's.d.'}) — {titulo}"


def _ler_curadoria(caminho: Path | None) -> dict[str, dict]:
    if caminho is None or not Path(caminho).is_file():
        return {}
    try:
        payload = json.loads(Path(caminho).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    documentos = payload.get("documentos", {}) if isinstance(payload, dict) else {}
    return documentos if isinstance(documentos, dict) else {}


def _registro_indice(linha: str, numero: int) -> dict:
    try:
        return json.loads(linha)
    except json.JSONDecodeError as exc:
        raise CatalogoBibliograficoInvalido(
            f"Indice portatil invalido na linha {numero}: {exc}"
        ) from exc


def _acumular_documento_indice(documentos: dict[str, dict], registro: dict) -> None:
    if registro.get("tipo") not in {"chunk_indice_portatil", "chunk_literatura"}:
        return
    metadata = registro.get("metadata") or {}
    source_id = str(metadata.get("arquivo_hash") or "").lower()
    if not _SOURCE_ID.fullmatch(source_id):
        return
    item = documentos.setdefault(
        source_id,
        {
            "metadata": metadata,
            "chunk_count": 0,
            "replacement_character": False,
        },
    )
    item["chunk_count"] += 1
    if "\ufffd" in json.dumps(metadata, ensure_ascii=False):
        item["replacement_character"] = True


def _ler_indice(caminho: Path | None) -> tuple[dict, dict[str, dict]]:
    manifesto: dict = {}
    documentos: dict[str, dict] = {}
    if caminho is None or not Path(caminho).is_file():
        return manifesto, documentos

    with gzip.open(caminho, "rt", encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, 1):
            registro = _registro_indice(linha, numero)
            if numero == 1 and registro.get("tipo", "").startswith("manifesto_"):
                manifesto = registro
                continue
            _acumular_documento_indice(documentos, registro)
    return manifesto, documentos


def _resumo(documentos: list[dict], *, index_records: int) -> dict:
    categorias = Counter(item["category"] for item in documentos)
    idiomas = Counter(item["language"] for item in documentos)
    return {
        "documents": len(documentos),
        "indexed_chunks": sum(int(item.get("chunk_count", 0)) for item in documentos),
        "portable_index_records": int(index_records),
        "categories": dict(sorted(categorias.items())),
        "languages": dict(sorted(idiomas.items())),
        "metadata_warnings": sum(bool(item.get("extraction_warnings")) for item in documentos),
    }


def _origem_metadado(revisao: dict, metadata: dict) -> str:
    if revisao:
        return "curadoria"
    if metadata:
        return "indice_portatil"
    return "nome_arquivo"


def _titulo_catalogo(
    pdf: Path,
    metadata: dict,
    fallback: dict,
    revisao: dict,
    titulo_pdf: str,
) -> tuple[str, bool, str]:
    titulo_base = limpar_metadado(
        metadata.get("titulo") or fallback.get("titulo") or pdf.stem
    )
    titulo = limpar_metadado(revisao.get("titulo") or titulo_base)
    origem = _origem_metadado(revisao, metadata)
    relacionado = _titulos_relacionados(titulo_base, titulo_pdf)
    usar_interno = (
        not revisao
        and _titulo_pdf_plausivel(titulo_pdf)
        and relacionado
        and len(titulo_pdf) >= len(titulo_base) + 3
    )
    if usar_interno:
        return titulo_pdf, relacionado, "pdf_interno"
    return titulo, relacionado, origem


def _autores_catalogo(
    metadata: dict,
    fallback: dict,
    revisao: dict,
    autor_pdf: str,
    relacionado: bool,
    origem: str,
) -> tuple[list[str], str]:
    autores_base = revisao.get("autor") or metadata.get("autor") or fallback.get("autor")
    autores = _autores(autores_base)
    if revisao or not autor_pdf or not relacionado:
        return autores, origem
    candidatos = _autores_pdf(autor_pdf)
    if candidatos == [_UNKNOWN_AUTHOR]:
        return autores, origem
    return candidatos, "pdf_interno"


def _ano_catalogo(metadata: dict, fallback: dict, revisao: dict) -> tuple[int | None, str]:
    ano, status = _ano(
        revisao.get("ano") or metadata.get("ano") or fallback.get("ano")
    )
    if revisao.get("ano_confirmado_ausente"):
        return None, "nao_declarado_na_fonte"
    return ano, status


def _categoria_catalogo(pdf: Path, metadata: dict) -> str:
    categoria = limpar_metadado(metadata.get("pasta") or pdf.parent.name, limite=80)
    if categoria in CATEGORIAS:
        return categoria
    if pdf.parent.name in CATEGORIAS:
        return pdf.parent.name
    return "ml-preditivo"


def _idioma_catalogo(metadata: dict) -> str:
    idioma = limpar_metadado(metadata.get("idioma") or "desconhecido", limite=20).lower()
    return idioma if idioma in IDIOMAS else "desconhecido"


def _alertas_catalogo(indice: dict, alertas_pdf: list[str]) -> list[str]:
    alertas = list(alertas_pdf)
    if indice.get("replacement_character"):
        alertas.append("caractere_substituto_na_extracao_pdf")
    if int(indice.get("chunk_count", 0)) == 0:
        alertas.append("ausente_do_indice_portatil")
    return alertas


def _restaurar_metadados_editados(item: dict, anterior: dict | None) -> None:
    if not anterior or not anterior.get("metadata_edited"):
        return
    campos = (
        "title", "authors", "year", "year_status", "category", "language",
        "citation", "relative_path",
    )
    for campo in campos:
        if campo in anterior:
            item[campo] = anterior[campo]
    item["metadata_edited"] = True
    item["updated_at"] = anterior.get("updated_at")
    item["index_status"] = anterior.get("index_status", "metadata_stale")


def _documento_catalogo(
    pdf: Path,
    raiz: Path,
    source_id: str,
    indice: dict,
    revisao: dict,
    anterior: dict | None,
) -> dict:
    metadata = indice.get("metadata", {})
    fallback = parsear_nome_arquivo(pdf.name)
    titulo_pdf, autor_pdf, alertas_pdf = _metadados_pdf(pdf)
    titulo, relacionado, origem = _titulo_catalogo(
        pdf, metadata, fallback, revisao, titulo_pdf
    )
    autores, origem = _autores_catalogo(
        metadata, fallback, revisao, autor_pdf, relacionado, origem
    )
    ano, ano_status = _ano_catalogo(metadata, fallback, revisao)
    chunks = int(indice.get("chunk_count", 0))
    item = {
        "source_id": source_id,
        "sha256": source_id,
        "file_name": pdf.name,
        "relative_path": pdf.relative_to(raiz).as_posix(),
        "title": titulo,
        "authors": autores,
        "year": ano,
        "year_status": ano_status,
        "category": _categoria_catalogo(pdf, metadata),
        "language": _idioma_catalogo(metadata),
        "chunk_count": chunks,
        "size_bytes": pdf.stat().st_size,
        "citation": _citacao(autores, ano, titulo),
        "metadata_origin": origem,
        "metadata_edited": False,
        "index_status": "indexed" if chunks else "not_indexed",
        "extraction_warnings": _alertas_catalogo(indice, alertas_pdf),
    }
    _restaurar_metadados_editados(item, anterior)
    return item


def construir_catalogo(
    raiz_literatura: Path,
    indice_portatil: Path | None = None,
    *,
    curadoria: Path | None = None,
    catalogo_anterior: dict | None = None,
) -> dict:
    """Constroi o catalogo a partir dos PDFs e do snapshot, sem editar ambos."""
    raiz = Path(raiz_literatura).resolve()
    manifesto, indexados = _ler_indice(indice_portatil)
    revisados = _ler_curadoria(curadoria)
    anteriores = {
        item.get("source_id"): item
        for item in (catalogo_anterior or {}).get("documents", [])
        if isinstance(item, dict)
    }
    documentos = []
    hashes_vistos: dict[str, Path] = {}

    for pdf in sorted(raiz.rglob("*.pdf")):
        source_id = sha256_arquivo(pdf)
        if source_id in hashes_vistos:
            raise CatalogoBibliograficoInvalido(
                f"PDF duplicado por hash: {hashes_vistos[source_id]} e {pdf}"
            )
        hashes_vistos[source_id] = pdf
        indice = indexados.get(source_id, {})
        revisao = revisados.get(source_id, {})
        documentos.append(
            _documento_catalogo(
                pdf,
                raiz,
                source_id,
                indice,
                revisao,
                anteriores.get(source_id),
            )
        )

    documentos.sort(key=lambda item: (item["title"].casefold(), item["source_id"]))
    index_chunks = int(manifesto.get("n_chunks", 0) or 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_id": "aliado-literatura",
        "source_index": {
            "schema_version": manifesto.get("schema_version"),
            "hash_corpus_sha256": manifesto.get("hash_corpus_sha256"),
            "generated_at_utc": manifesto.get("gerado_em_utc"),
            "declared_chunks": index_chunks,
        },
        "summary": _resumo(documentos, index_records=index_chunks + (1 if manifesto else 0)),
        "documents": documentos,
    }


def validar_catalogo(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise CatalogoBibliograficoInvalido("Schema do catalogo incompativel")
    documentos = payload.get("documents")
    if not isinstance(documentos, list):
        raise CatalogoBibliograficoInvalido("Catalogo sem lista de documentos")
    ids = [str(item.get("source_id", "")) for item in documentos if isinstance(item, dict)]
    if len(ids) != len(documentos) or any(not _SOURCE_ID.fullmatch(item) for item in ids):
        raise CatalogoBibliograficoInvalido("source_id invalido no catalogo")
    if len(set(ids)) != len(ids):
        raise CatalogoBibliograficoInvalido("source_id duplicado no catalogo")
    return payload


def carregar_catalogo(caminho: Path) -> dict:
    try:
        payload = json.loads(Path(caminho).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogoBibliograficoInvalido(f"Catalogo ilegivel: {exc}") from exc
    return validar_catalogo(payload)


def _caminho_catalogo(caminho: Path) -> Path:
    resolvido = Path(caminho).resolve()
    if resolvido.name != "catalogo.json":
        raise CatalogoBibliograficoInvalido(
            "O catalogo deve usar o nome fixo catalogo.json"
        )
    return resolvido


def salvar_catalogo(caminho: Path, payload: dict) -> None:
    validar_catalogo(payload)
    caminho = _caminho_catalogo(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.parent / ".catalogo.json.tmp"
    with _LOCK:
        try:
            temporario.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporario, caminho)
        finally:
            temporario.unlink(missing_ok=True)


def _aplicar_metadados_editaveis(item: dict, patch: dict) -> None:
    if "title" in patch:
        titulo = limpar_metadado(patch["title"])
        if not titulo:
            raise CatalogoBibliograficoInvalido("Titulo nao pode ser vazio")
        item["title"] = titulo
    if "authors" in patch:
        item["authors"] = _autores(patch["authors"])
    if "year" in patch:
        ano, status = _ano(patch["year"])
        item["year"] = ano
        item["year_status"] = status
    if "category" in patch:
        categoria = limpar_metadado(patch["category"], limite=80)
        if categoria not in CATEGORIAS:
            raise CatalogoBibliograficoInvalido("Categoria invalida")
        item["category"] = categoria
    if "language" in patch:
        idioma = limpar_metadado(patch["language"], limite=20).lower()
        if idioma not in IDIOMAS:
            raise CatalogoBibliograficoInvalido("Idioma invalido")
        item["language"] = idioma


class CatalogoStore:
    """Le e atualiza o catalogo com escrita atomica e validacao de campos."""

    def __init__(self, caminho: Path):
        self.caminho = Path(caminho)
        self._lock = threading.RLock()

    def load(self) -> dict:
        with self._lock:
            return carregar_catalogo(self.caminho)

    def save(self, payload: dict) -> None:
        with self._lock:
            salvar_catalogo(self.caminho, payload)

    @staticmethod
    def _find(payload: dict, source_id: str) -> dict:
        if not _SOURCE_ID.fullmatch(str(source_id)):
            raise KeyError(source_id)
        for item in payload["documents"]:
            if item["source_id"] == source_id:
                return item
        raise KeyError(source_id)

    def get(self, source_id: str) -> dict:
        return dict(self._find(self.load(), source_id))

    def update(self, source_id: str, patch: dict, *, internal: bool = False) -> dict:
        permitidos = _EDITABLE_FIELDS | (_INTERNAL_FIELDS if internal else set())
        desconhecidos = set(patch) - permitidos
        if desconhecidos:
            raise CatalogoBibliograficoInvalido(
                "Campos nao editaveis: " + ", ".join(sorted(desconhecidos))
            )
        with self._lock:
            payload = self.load()
            item = self._find(payload, source_id)
            _aplicar_metadados_editaveis(item, patch)
            for campo in permitidos - _EDITABLE_FIELDS:
                if campo in patch:
                    item[campo] = patch[campo]
            item["citation"] = _citacao(item["authors"], item.get("year"), item["title"])
            if not internal:
                item["metadata_edited"] = True
                item["index_status"] = "metadata_stale"
            item["updated_at"] = _agora_utc()
            payload["summary"] = _resumo(
                payload["documents"],
                index_records=int(payload.get("summary", {}).get("portable_index_records", 0)),
            )
            self.save(payload)
            return dict(item)

    def add(self, item: dict) -> dict:
        source_id = str(item.get("source_id", ""))
        if not _SOURCE_ID.fullmatch(source_id):
            raise CatalogoBibliograficoInvalido("source_id invalido")
        with self._lock:
            payload = self.load()
            if any(doc["source_id"] == source_id for doc in payload["documents"]):
                raise CatalogoBibliograficoInvalido("PDF ja catalogado")
            payload["documents"].append(item)
            payload["documents"].sort(
                key=lambda doc: (doc["title"].casefold(), doc["source_id"])
            )
            payload["summary"] = _resumo(
                payload["documents"],
                index_records=int(payload.get("summary", {}).get("portable_index_records", 0)),
            )
            self.save(payload)
            return dict(item)

    def update_index(
        self,
        source_id: str,
        *,
        chunk_count: int,
        status: str,
        snapshot: dict | None = None,
        warning: str | None = None,
    ) -> dict:
        """Atualiza o estado reconstruivel do indice e seu snapshot."""
        if int(chunk_count) < 0:
            raise CatalogoBibliograficoInvalido("Quantidade de chunks invalida")
        with self._lock:
            payload = self.load()
            item = self._find(payload, source_id)
            item["chunk_count"] = int(chunk_count)
            item["index_status"] = limpar_metadado(status, limite=80)
            alertas = [
                value
                for value in item.get("extraction_warnings", [])
                if value != "snapshot_portatil_desatualizado"
            ]
            if warning:
                alertas.append(limpar_metadado(warning, limite=160))
            item["extraction_warnings"] = list(dict.fromkeys(alertas))
            item["updated_at"] = _agora_utc()

            index_records = int(
                payload.get("summary", {}).get("portable_index_records", 0)
            )
            if snapshot:
                declarados = int(snapshot.get("n_chunks", 0))
                payload["source_index"] = {
                    "schema_version": snapshot.get("schema_version"),
                    "hash_corpus_sha256": snapshot.get("hash_corpus_sha256"),
                    "generated_at_utc": snapshot.get("gerado_em_utc"),
                    "declared_chunks": declarados,
                }
                index_records = declarados + 1

            payload["summary"] = _resumo(
                payload["documents"], index_records=index_records
            )
            self.save(payload)
            return dict(item)


def gerar_catalogo_padrao() -> dict:
    from src.core.config import ARQUIVO_INDICE_LITERATURA, PASTA_LITERATURA, RAIZ_PROJETO

    caminho = PASTA_LITERATURA / "catalogo.json"
    anterior = carregar_catalogo(caminho) if caminho.is_file() else None
    payload = construir_catalogo(
        PASTA_LITERATURA,
        ARQUIVO_INDICE_LITERATURA,
        curadoria=RAIZ_PROJETO / "metadados_revisados.json",
        catalogo_anterior=anterior,
    )
    salvar_catalogo(caminho, payload)
    return payload


if __name__ == "__main__":
    catalogo = gerar_catalogo_padrao()
    print(
        f"Catalogo: {catalogo['summary']['documents']} PDFs, "
        f"{catalogo['summary']['indexed_chunks']} chunks."
    )


__all__ = [
    "CATEGORIAS",
    "IDIOMAS",
    "CatalogoBibliograficoInvalido",
    "CatalogoStore",
    "carregar_catalogo",
    "construir_catalogo",
    "gerar_catalogo_padrao",
    "limpar_metadado",
    "salvar_catalogo",
    "sha256_arquivo",
]
