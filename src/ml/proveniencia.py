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

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local
from src.core.utils import to_project_relative_path

PASTA_MANIFESTOS = Path(RAIZ_PROJETO) / "resultados" / "manifestos"

READY = "ready"
STALE = "stale"
PENDING = "pending"


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


def sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


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
    created_at: str | None = None,
    evidence_level: str | None = None,
) -> dict:
    """
    Monta o manifesto de uma etapa. `input_artifacts` é {nome: caminho}; o
    manifesto guarda o HASH de cada entrada (não o caminho), para detectar
    regeneração upstream. `code_path` é o arquivo-fonte da etapa.
    """
    manifesto = {
        "stage": stage,
        "created_at": created_at or agora_local().isoformat(),
        "git_commit": _git_commit(),
        "code_sha256": sha256_arquivo(code_path) or "",
        "parameters": parameters or {},
        "input_artifacts": {
            nome: sha256_arquivo(caminho)
            for nome, caminho in (input_artifacts or {}).items()
        },
        "outputs": [to_project_relative_path(o) for o in (outputs or [])],
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


def comparar(manifesto_salvo: dict | None, manifesto_atual: dict) -> list[str]:
    """Motivos de incompatibilidade (lista vazia = compatível)."""
    if not manifesto_salvo:
        return ["sem manifesto"]
    motivos = []
    if manifesto_salvo.get("code_sha256") != manifesto_atual.get("code_sha256"):
        motivos.append("código da etapa alterado")
    if manifesto_salvo.get("parameters") != manifesto_atual.get("parameters"):
        motivos.append("parâmetros alterados")
    if manifesto_salvo.get("input_artifacts") != manifesto_atual.get("input_artifacts"):
        motivos.append("artefato upstream regenerado")
    return motivos


def estado_etapa(
    stage: str,
    artefatos,
    code_path,
    parameters: dict | None = None,
    input_artifacts: dict | None = None,
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

    atual = gerar_manifesto(stage, code_path, parameters, input_artifacts, artefatos)
    motivos = comparar(salvo, atual)
    return {"estado": (STALE if motivos else READY), "motivos": motivos}
