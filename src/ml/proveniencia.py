"""
proveniencia.py — Al IAdo PV / Sprint 1 (rastreabilidade)

Manifesto de proveniência por etapa do pipeline de ML + detecção de artefato
STALE (desatualizado). Uma etapa NÃO é válida só porque os arquivos existem:
ela é válida quando os artefatos são compatíveis com o código, os parâmetros e
os artefatos upstream que os geraram.

Três estados:
    pending — não há artefatos OU não há manifesto
    ready   — artefatos presentes E manifesto compatível
    stale   — artefatos presentes MAS algo mudou (código da etapa, parâmetros
              ou um artefato upstream regenerado com hash diferente)

Princípio: NUNCA apaga artefatos automaticamente — apenas sinaliza. Recalcular
é sempre sob comando explícito do pesquisador.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from collections.abc import Iterable

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.core.utils import to_project_relative_path

PASTA_MANIFESTOS = Path(RAIZ_PROJETO) / "resultados" / "manifestos"

READY = "ready"
STALE = "stale"
PENDING = "pending"

SUFIXOS_TEXTO_PORTAVEL = {
    ".csv", ".json", ".md", ".toml", ".txt", ".yaml", ".yml",
}


def sha256_arquivo(caminho) -> str | None:
    """SHA-256 de um arquivo (None se não existe)."""
    p = Path(caminho)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_arquivo_texto_normalizado(caminho) -> str | None:
    """SHA-256 textual estável entre Windows/Linux (CRLF, CR e LF -> LF)."""
    p = Path(caminho)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("r", encoding="utf-8", newline=None) as arquivo:
        for bloco in iter(lambda: arquivo.read(65536), ""):
            h.update(bloco.encode("utf-8"))
    return h.hexdigest()


# Campos que registram QUANDO o artefato foi escrito, não O QUE ele afirma.
# Entram no artefato de propósito — proveniência legítima — mas não podem entrar
# no hash: o pipeline é determinístico por semente, então rodar de novo produz o
# MESMO resultado com relógio diferente.
CAMPOS_VOLATEIS = frozenset({
    "data_treino", "data_geracao", "data_execucao", "created_at",
    "generated_at", "timestamp", "gerado_em", "executado_em",
})


def _sem_campos_volateis(valor):
    if isinstance(valor, dict):
        return {
            chave: _sem_campos_volateis(sub)
            for chave, sub in valor.items()
            if chave not in CAMPOS_VOLATEIS
        }
    if isinstance(valor, list):
        return [_sem_campos_volateis(item) for item in valor]
    return valor


def sha256_json_estavel(caminho) -> str | None:
    """SHA-256 do CONTEÚDO CIENTÍFICO de um JSON, sem os carimbos de tempo.

    POR QUE EXISTE
    ==============
    Em 15/08/2026 o pesquisador re-rodou a etapa `autoencoder`. O treino é
    determinístico: o limiar saiu bit a bit idêntico, `0.8577015399932861`. A
    ÚNICA diferença no arquivo foi ``data_treino``, de 12/08 para 15/08.

    Mesmo assim o SHA-256 mudou, e com ele quebraram o manifesto de proveniência
    e o `threshold_sha256` do registro E3 — dois testes reprovando no `main` por
    causa de um relógio. Pior que o incômodo: treina o leitor a ignorar
    divergência de hash, que é exatamente o alarme que deveria ser levado a
    sério quando um artefato muda de verdade.

    Hashear os bytes de um arquivo que contém a hora de escrita torna a cadeia
    de proveniência instável por construção. Aqui o JSON é lido, os campos
    voláteis são removidos em profundidade, e o resto é serializado
    canonicamente (chaves ordenadas) antes de hashear.

    Se o arquivo não for JSON válido, cai no hash textual — nunca devolve nada
    silenciosamente errado.
    """
    p = Path(caminho)
    if not p.exists() or not p.is_file():
        return None
    try:
        dados = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return sha256_arquivo_texto_normalizado(p)
    canonico = json.dumps(
        _sem_campos_volateis(dados),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _mapear_arquivos(arquivos) -> list[tuple[str, object]]:
    if not arquivos:
        return []
    if isinstance(arquivos, dict):
        return [(str(nome), caminho) for nome, caminho in arquivos.items()]
    if isinstance(arquivos, (str, bytes, Path)):
        arquivos = [arquivos]
    elif not isinstance(arquivos, Iterable):
        arquivos = [arquivos]
    return [(to_project_relative_path(caminho), caminho) for caminho in arquivos]


def _hashes_arquivos(arquivos, *, texto_normalizado: bool = False) -> dict[str, str | None]:
    func = sha256_arquivo_texto_normalizado if texto_normalizado else sha256_arquivo
    return {
        nome: func(caminho)
        for nome, caminho in _mapear_arquivos(arquivos)
    }


def funcao_de_hash_para(caminho):
    """Escolhe o hash pela natureza do arquivo, não pelo acaso da extensão.

    JSON leva o hash ESTÁVEL (sem os campos de data), porque é onde o pipeline
    grava carimbo de tempo junto com o resultado. Demais textos levam o hash com
    EOL normalizada; binário científico é hasheado byte a byte.
    """
    sufixo = Path(caminho).suffix.lower()
    if sufixo == ".json":
        return sha256_json_estavel
    if sufixo in SUFIXOS_TEXTO_PORTAVEL:
        return sha256_arquivo_texto_normalizado
    return sha256_arquivo


def _hashes_artefatos_portaveis(arquivos) -> dict[str, str | None]:
    """Normaliza EOL de textos; preserva bytes de binários científicos."""
    return {
        nome: funcao_de_hash_para(caminho)(caminho)
        for nome, caminho in _mapear_arquivos(arquivos)
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(RAIZ_PROJETO), capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def gerar_manifesto(
    stage: str,
    code_path,
    parameters: dict | None,
    input_artifacts: dict | None,
    outputs,
    *,
    code_dependencies: dict | Iterable | None = None,
    created_at: str | None = None,
    evidence_level: str | None = None,
) -> dict:
    """
    Monta o manifesto de uma etapa. `input_artifacts` é {nome: caminho}; o
    manifesto guarda o HASH de cada entrada (não o caminho), para detectar
    regeneração upstream. `code_path` é o arquivo-fonte da etapa.

    Manifesto v2:
    - normaliza CRLF/LF antes de hashear código Python;
    - registra dependências científicas compartilhadas;
    - registra hash de cada saída, normalizando EOL de formatos textuais.
    """
    outputs_lista = [to_project_relative_path(o) for o in (outputs or [])]
    manifesto = {
        "manifest_version": 2,
        "stage": stage,
        "created_at": created_at or agora_local().isoformat(),
        "git_commit": _git_commit(),
        "code_hash_mode": "text_lf_utf8",
        "input_hash_mode": "text_lf_utf8_by_suffix_else_binary",
        "output_hash_mode": "text_lf_utf8_by_suffix_else_binary",
        "code_sha256": sha256_arquivo_texto_normalizado(code_path) or "",
        "code_dependencies": _hashes_arquivos(
            code_dependencies or {}, texto_normalizado=True
        ),
        "parameters": parameters or {},
        "input_artifacts": _hashes_artefatos_portaveis(input_artifacts or {}),
        "outputs": outputs_lista,
        "output_artifacts": _hashes_artefatos_portaveis(outputs or {}),
    }
    if evidence_level:
        manifesto["evidence_level"] = evidence_level
    return manifesto


def caminho_manifesto(stage: str) -> Path:
    return PASTA_MANIFESTOS / f"{stage}.json"


def salvar_manifesto(manifesto: dict) -> Path:
    PASTA_MANIFESTOS.mkdir(parents=True, exist_ok=True)
    p = caminho_manifesto(manifesto["stage"])
    p.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def carregar_manifesto(stage: str) -> dict | None:
    p = caminho_manifesto(stage)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _inputs_compativeis(
    salvo: dict, atual: dict, *, permitir_ausentes: bool
) -> bool:
    if not permitir_ausentes:
        return salvo == atual
    if set(salvo) != set(atual):
        return False
    return all(
        hash_atual is None or salvo.get(nome) == hash_atual
        for nome, hash_atual in atual.items()
    )


def comparar(
    manifesto_salvo: dict | None,
    manifesto_atual: dict,
    *,
    permitir_inputs_ausentes: bool = False,
) -> list[str]:
    """Motivos de incompatibilidade; entradas ausentes podem ficar não verificadas."""
    if not manifesto_salvo:
        return ["sem manifesto"]
    motivos = []
    versao_salva = int(manifesto_salvo.get("manifest_version") or 1)
    versao_atual = int(manifesto_atual.get("manifest_version") or 1)
    if versao_salva < versao_atual:
        motivos.append("manifesto v2 ausente")
    if versao_salva >= 2 and (
        manifesto_salvo.get("input_hash_mode")
        != manifesto_atual.get("input_hash_mode")
    ):
        motivos.append("modo de hash das entradas ausente ou alterado")
    if versao_salva >= 2 and (
        manifesto_salvo.get("output_hash_mode")
        != manifesto_atual.get("output_hash_mode")
    ):
        motivos.append("modo de hash das saídas ausente ou alterado")
    if manifesto_salvo.get("code_sha256") != manifesto_atual.get("code_sha256"):
        motivos.append("código da etapa alterado")
    if versao_salva >= 2 and (
        manifesto_salvo.get("code_dependencies")
        != manifesto_atual.get("code_dependencies")
    ):
        motivos.append("dependência científica alterada")
    if manifesto_salvo.get("parameters") != manifesto_atual.get("parameters"):
        motivos.append("parâmetros alterados")
    if not _inputs_compativeis(
        manifesto_salvo.get("input_artifacts") or {},
        manifesto_atual.get("input_artifacts") or {},
        permitir_ausentes=permitir_inputs_ausentes,
    ):
        motivos.append("artefato upstream regenerado")
    if versao_salva >= 2 and (
        manifesto_salvo.get("output_artifacts")
        != manifesto_atual.get("output_artifacts")
    ):
        motivos.append("artefato de saída alterado")
    return motivos


def estado_etapa(
    stage: str,
    artefatos,
    code_path,
    parameters: dict | None = None,
    input_artifacts: dict | None = None,
    code_dependencies: dict | Iterable | None = None,
) -> dict:
    """
    Retorna {"estado": ready|stale|pending, "motivos": [...]}.

    - pending: algum artefato ausente OU sem manifesto;
    - stale  : artefatos presentes, manifesto presente, mas algo mudou;
    - ready  : tudo presente e compatível.
    """
    artefatos = list(artefatos)
    artefatos_ok = bool(artefatos) and all(Path(a).exists() for a in artefatos)
    if not artefatos_ok:
        return {"estado": PENDING, "motivos": ["artefato(s) ausente(s)"]}

    salvo = carregar_manifesto(stage)
    if not salvo:
        return {"estado": PENDING, "motivos": ["sem manifesto de proveniência"]}

    atual = gerar_manifesto(
        stage, code_path, parameters, input_artifacts, artefatos,
        code_dependencies=code_dependencies,
    )
    motivos = comparar(salvo, atual)
    return {"estado": (STALE if motivos else READY), "motivos": motivos}
