"""Persistencia transacional na nuvem (GitHub Contents API) — sem rede.

Os testes trocam `_requisitar` por um fake, entao nenhuma chamada HTTP real
acontece. Cobrem: ativacao (master switch + token), deteccao de repo, o fluxo
create/update e o retry em conflito de sha.
"""

from __future__ import annotations

import base64

from src.conhecimento import persistencia_nuvem as pn


def _limpar_env(monkeypatch):
    for var in ("AL_IADO_PERSISTIR_NUVEM", "GITHUB_TOKEN", "AL_IADO_GITHUB_TOKEN",
                "AL_IADO_GITHUB_REPO", "AL_IADO_GITHUB_BRANCH"):
        monkeypatch.delenv(var, raising=False)


def test_desligado_por_padrao(monkeypatch):
    _limpar_env(monkeypatch)
    assert pn.persistencia_ativa() is False


def test_token_sem_flag_nao_ativa(monkeypatch):
    _limpar_env(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x" * 8)
    monkeypatch.setenv("AL_IADO_GITHUB_REPO", "dono/repo")
    assert pn.persistencia_ativa() is False  # falta o master switch


def test_flag_mais_token_mais_repo_ativa(monkeypatch):
    _limpar_env(monkeypatch)
    monkeypatch.setenv("AL_IADO_PERSISTIR_NUVEM", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x" * 8)
    monkeypatch.setenv("AL_IADO_GITHUB_REPO", "dono/repo")
    assert pn.persistencia_ativa() is True


def test_repo_do_env_tira_sufixo_git(monkeypatch):
    _limpar_env(monkeypatch)
    monkeypatch.setenv("AL_IADO_GITHUB_REPO", "dono/repo.git")
    assert pn._repo_alvo() == "dono/repo"


def test_caminho_no_repo_relativo(tmp_path, monkeypatch):
    alvo = pn.RAIZ_PROJETO / "notas" / "memorias" / "agentes" / "memoria_validada.json"
    assert pn._caminho_no_repo(alvo) == "notas/memorias/agentes/memoria_validada.json"
    # Fora da raiz do repo → None (nunca commita arquivo externo).
    assert pn._caminho_no_repo(tmp_path / "fora.json") is None


def test_persistir_inativo_nao_faz_rede(monkeypatch, tmp_path):
    _limpar_env(monkeypatch)
    chamou = {"n": 0}
    monkeypatch.setattr(pn, "_requisitar", lambda *a, **k: chamou.__setitem__("n", chamou["n"] + 1))
    arq = tmp_path / "m.json"
    arq.write_text("{}", encoding="utf-8")
    assert pn.persistir_arquivo(arq, mensagem="x") is False
    assert chamou["n"] == 0


def test_update_com_sha_existente(monkeypatch):
    _limpar_env(monkeypatch)
    monkeypatch.setenv("AL_IADO_PERSISTIR_NUVEM", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x" * 8)
    monkeypatch.setenv("AL_IADO_GITHUB_REPO", "dono/repo")

    alvo = pn.RAIZ_PROJETO / "notas" / "memorias" / "agentes" / "memoria_validada.json"
    chamadas = []

    def fake(metodo, url, token, payload=None):
        chamadas.append((metodo, payload))
        if metodo == "GET":
            return 200, {"sha": "abc123"}
        return 201, {"content": {"path": "x"}}

    monkeypatch.setattr(pn, "_requisitar", fake)
    ok = pn.persistir_arquivo(alvo, mensagem="msg")
    assert ok is True
    put = [c for c in chamadas if c[0] == "PUT"][0]
    assert put[1]["sha"] == "abc123"
    assert put[1]["branch"] == "main"
    # conteudo e base64 valido
    base64.b64decode(put[1]["content"])


def test_conflito_de_sha_faz_um_retry(monkeypatch):
    _limpar_env(monkeypatch)
    monkeypatch.setenv("AL_IADO_PERSISTIR_NUVEM", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x" * 8)
    monkeypatch.setenv("AL_IADO_GITHUB_REPO", "dono/repo")

    alvo = pn.RAIZ_PROJETO / "notas" / "memorias" / "agentes" / "memoria_validada.json"
    puts = {"n": 0}

    def fake(metodo, url, token, payload=None):
        if metodo == "GET":
            return 200, {"sha": "s"}
        puts["n"] += 1
        return (409, {"message": "conflict"}) if puts["n"] == 1 else (200, {})

    monkeypatch.setattr(pn, "_requisitar", fake)
    assert pn.persistir_arquivo(alvo, mensagem="msg") is True
    assert puts["n"] == 2  # 1 conflito + 1 sucesso
