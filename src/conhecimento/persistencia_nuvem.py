"""
persistencia_nuvem.py — Al IAdo PV

Persistência transacional na nuvem via GitHub Contents API.

Hospedagens com sistema de arquivos EFÊMERO perdem gravações de runtime
somem no próximo redeploy/reinício. Como o repositório Git já é a fonte de
verdade da memória do agente (o deploy restaura o JSON versionado), o backend
durável natural é o próprio GitHub — sem provisionar banco novo.

Quando ativa, esta camada faz commit do arquivo alterado (a memória validada) de
volta ao repositório via API. No próximo deploy, o app já traz o estado mais
recente. É a peça que faz o aprendizado sobreviver a redeploys sem depender de
`git commit` manual do PC.

Ativação (master switch + credencial), configurados como segredos da plataforma:
  AL_IADO_PERSISTIR_NUVEM = 1
  GITHUB_TOKEN            = <PAT com permissão de escrita em conteúdo>
Opcionais:
  AL_IADO_GITHUB_REPO    = "owner/repo"   (default: detectado de .git/config)
  AL_IADO_GITHUB_BRANCH  = "main"          (branch de deploy)

Segurança e robustez:
  - O token nunca é logado; mensagens de erro passam por mascarar_segredos.
  - Best-effort: QUALQUER falha (sem token, rede, conflito) é engolida — a
    gravação LOCAL, que já ocorreu, nunca é invalidada por isto.
  - No PC não se ativa (o pesquisador versiona manualmente): sem o flag e o
    token, `persistencia_ativa()` é False e nada acontece.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from src.core.config import RAIZ_PROJETO

_API = "https://api.github.com"
_TIMEOUT = float(os.getenv("AL_IADO_GITHUB_TIMEOUT", "12"))


# Último resultado de tentativa de commit, POR ALVO ("sessao"/"memoria"/...).
# A persistência é best-effort e engolia erros em silêncio — isto os torna
# VISÍVEIS (ex.: token sem permissão de escrita → 403/404). Rastrear por alvo
# (não um único status global) importa porque sessão e memória commitam
# separadamente no mesmo turno: um status compartilhado deixava o resultado
# mais recente (ex.: sessão OK) mascarar uma falha silenciosa no outro alvo
# (ex.: memória FALHOU) — exatamente o cenário que perdeu uma memória do
# pesquisador sem nenhum aviso visível antes deste fix.
_ALVOS_CONHECIDOS = ("sessao", "memoria", "consolidado", "snippet")
_STATUS_POR_ALVO: dict[str, dict] = {
    alvo: {"estado": "sem_tentativa", "detalhe": ""} for alvo in _ALVOS_CONHECIDOS
}


def _registrar_status(alvo: str, estado: str, detalhe: str = "") -> None:
    registro = _STATUS_POR_ALVO.setdefault(alvo, {"estado": "sem_tentativa", "detalhe": ""})
    registro["estado"] = estado
    registro["detalhe"] = _mascarar(detalhe)[:200]


def _token() -> str | None:
    return os.getenv("GITHUB_TOKEN") or os.getenv("AL_IADO_GITHUB_TOKEN")


def _mascarar(texto: str) -> str:
    try:
        from src.core.seguranca import mascarar_segredos

        return mascarar_segredos(str(texto))
    except Exception:
        return "<erro mascarado>"


def persistencia_ativa() -> bool:
    """True somente com o master switch LIGADO e um token presente."""
    flag = os.getenv("AL_IADO_PERSISTIR_NUVEM", "").strip().lower()
    ligado = flag in {"1", "true", "sim", "yes", "on"}
    return ligado and bool(_token()) and bool(_repo_alvo())


def _repo_alvo() -> str | None:
    """Retorna "owner/repo": do env, ou detectado de .git/config (origin)."""
    env = os.getenv("AL_IADO_GITHUB_REPO", "").strip()
    if env and "/" in env:
        return env.removesuffix(".git")
    return _repo_do_git_config()


def _repo_do_git_config() -> str | None:
    config = RAIZ_PROJETO / ".git" / "config"
    if not config.is_file():
        return None
    try:
        texto = config.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    # Ex.: https://github.com/owner/repo.git  |  git@github.com:owner/repo.git
    m = re.search(r"github\.com[/:]([^/\s]+/[^/\s]+?)(?:\.git)?\s*$", texto, re.M)
    return m.group(1) if m else None


def _caminho_no_repo(caminho: Path) -> str | None:
    """Caminho do arquivo relativo à raiz do repo, com barras de URL."""
    try:
        rel = Path(caminho).resolve().relative_to(RAIZ_PROJETO.resolve())
    except (ValueError, OSError):
        return None
    return rel.as_posix()


def _requisitar(metodo: str, url: str, token: str, payload: dict | None = None):
    """Chamada única à API do GitHub. Retorna (status, corpo_json|None)."""
    import urllib.error
    import urllib.request

    dados = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=dados, method=metodo)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "al-iado-pv")
    if dados is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            corpo = resp.read().decode("utf-8") or "{}"
            return resp.status, json.loads(corpo)
    except urllib.error.HTTPError as exc:
        corpo = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        try:
            corpo_json = json.loads(corpo) if corpo else None
        except json.JSONDecodeError:
            corpo_json = None
        return exc.code, corpo_json


def _sha_atual(repo: str, caminho_repo: str, branch: str, token: str) -> str | None:
    url = f"{_API}/repos/{repo}/contents/{caminho_repo}?ref={branch}"
    status, corpo = _requisitar("GET", url, token)
    if status == 200 and isinstance(corpo, dict):
        return corpo.get("sha")
    return None  # 404 = arquivo novo; qualquer outro status = trata como novo


def persistir_arquivo(caminho: str | Path, *, mensagem: str, alvo: str = "geral") -> bool:
    """Faz commit do arquivo para o GitHub (create/update). Best-effort.

    `alvo` identifica QUEM está commitando ("sessao"/"memoria"/...) para o
    diagnóstico rastrear cada um separadamente — ver nota em _STATUS_POR_ALVO.
    Retorna True se o commit foi aceito (200/201), False caso contrário —
    nunca levanta exceção para o chamador.
    """
    if not persistencia_ativa():
        return False
    token = _token()
    repo = _repo_alvo()
    branch = os.getenv("AL_IADO_GITHUB_BRANCH", "main").strip() or "main"
    caminho = Path(caminho)
    caminho_repo = _caminho_no_repo(caminho)
    if not (token and repo and caminho_repo and caminho.is_file()):
        return False

    try:
        conteudo_b64 = base64.b64encode(caminho.read_bytes()).decode("ascii")
    except OSError as exc:
        print(f"   ⚠️  Persistência nuvem: leitura falhou ({_mascarar(exc)})")
        return False

    url = f"{_API}/repos/{repo}/contents/{caminho_repo}"

    # Uma tentativa + 1 retry para o caso de o sha ter mudado (conflito 409/422).
    for tentativa in (1, 2):
        payload = {
            "message": mensagem[:200],
            "content": conteudo_b64,
            "branch": branch,
        }
        sha = _sha_atual(repo, caminho_repo, branch, token)
        if sha:
            payload["sha"] = sha
        try:
            status, corpo = _requisitar("PUT", url, token, payload)
        except Exception as exc:  # rede/urllib — best-effort
            print(f"   ⚠️  Persistência nuvem: rede falhou ({_mascarar(exc)})")
            return False
        if status in (200, 201):
            print(f"   ☁️  Persistido no GitHub ({repo}@{branch}).")
            _registrar_status(alvo, "ok", f"{caminho_repo} @ {repo}")
            return True
        if status in (409, 422) and tentativa == 1:
            continue  # sha desatualizado: recarrega e tenta de novo
        motivo = ""
        if isinstance(corpo, dict):
            motivo = str(corpo.get("message", ""))
        print(f"   ⚠️  Persistência nuvem: HTTP {status} {_mascarar(motivo)}")
        _registrar_status(alvo, "erro", f"HTTP {status}: {motivo}")
        return False
    return False


_ROTULOS_ALVO = {"sessao": "Sessão", "memoria": "Memória", "consolidado": "Consolidação", "snippet": "Trechos"}


def diagnostico() -> dict:
    """Estado legível da persistência na nuvem, para exibir na barra lateral.

    Torna VISÍVEL o que era silencioso: se está desligada e por quê (flag/token/
    repo), ou se está ligada e qual foi o resultado do último commit de CADA
    alvo (sessão / memória) separadamente — um alvo com sucesso não pode mais
    mascarar o outro falhando silenciosamente no mesmo turno.
    """
    flag = os.getenv("AL_IADO_PERSISTIR_NUVEM", "").strip().lower()
    ligado = flag in {"1", "true", "sim", "yes", "on"}
    tem_token = bool(_token())
    repo = _repo_alvo()

    if not ligado:
        return {"ativa": False, "resumo": "desligada",
                "detalhe": "AL_IADO_PERSISTIR_NUVEM não está em 1 nos Secrets.",
                "por_alvo": {}}
    if not tem_token:
        return {"ativa": False, "resumo": "sem token",
                "detalhe": "GITHUB_TOKEN ausente nos Secrets.", "por_alvo": {}}
    if not repo:
        return {"ativa": False, "resumo": "sem repositório",
                "detalhe": "Não detectei owner/repo (defina AL_IADO_GITHUB_REPO).",
                "por_alvo": {}}

    por_alvo = {}
    for alvo in _ALVOS_CONHECIDOS:
        info = _STATUS_POR_ALVO.get(alvo, {"estado": "sem_tentativa", "detalhe": ""})
        por_alvo[alvo] = {"rotulo": _ROTULOS_ALVO.get(alvo, alvo), **info}

    estados = {v["estado"] for v in por_alvo.values()}
    if "erro" in estados:
        resumo = "ativa, com FALHA em pelo menos um alvo"
        detalhe = "Verifique se o token tem permissão Contents: Read and write."
    elif "ok" in estados:
        resumo = "ativa ✓"
        detalhe = f"Pronta ({repo})."
    else:
        resumo = "ativa (aguardando)"
        detalhe = f"Pronta ({repo}). O 1º commit ocorre ao criar memória ou a cada 6 interações."

    return {"ativa": True, "resumo": resumo, "detalhe": detalhe, "por_alvo": por_alvo}


def persistir_memoria_validada(caminho: str | Path) -> bool:
    """Atalho semântico usado pela camada de memória após aprovar um item."""
    return persistir_arquivo(
        caminho,
        mensagem="chore(memoria): atualiza memoria validada (persistencia nuvem)",
        alvo="memoria",
    )
