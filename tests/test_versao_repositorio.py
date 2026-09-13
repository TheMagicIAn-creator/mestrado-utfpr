"""Guarda de clone defasado, exercitada contra repositórios git reais.

Os repositórios são criados em `tmp_path`: um bare local faz o papel de
`origin`, então nada aqui depende de rede externa. Testar isso com mock de
`subprocess` provaria apenas que o mock responde o que eu mandei — o que
importa é o veredito diante de um git de verdade nos cinco estados possíveis.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.core.versao_repositorio import (
    ADIANTADO,
    ATRASADO,
    EM_DIA,
    INDETERMINADO,
    CloneDefasado,
    estado_do_clone,
    exigir_clone_atualizado,
)

pytestmark = pytest.mark.leve


def _git(*argumentos: str, cwd: Path) -> str:
    processo = subprocess.run(
        ["git", *argumentos],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return processo.stdout.strip()


def _commitar(raiz: Path, texto: str) -> None:
    (raiz / "arquivo.txt").write_text(texto, encoding="utf-8")
    _git("add", "-A", cwd=raiz)
    _git("commit", "-q", "-m", texto, cwd=raiz)


@pytest.fixture
def repositorios(tmp_path):
    """Devolve (autor, atrasado): dois clones do mesmo `origin` bare local."""
    remoto = tmp_path / "remoto.git"
    remoto.mkdir()
    _git("init", "-q", "--bare", cwd=remoto)
    # `git init -b` só existe a partir do 2.28; symbolic-ref funciona sempre.
    _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remoto)

    autor = tmp_path / "autor"
    _git("clone", "-q", str(remoto), str(autor), cwd=tmp_path)
    _git("config", "user.email", "pesquisador@exemplo.test", cwd=autor)
    _git("config", "user.name", "Pesquisador", cwd=autor)
    _git("checkout", "-q", "-B", "main", cwd=autor)
    _commitar(autor, "primeiro")
    _git("push", "-q", "-u", "origin", "main", cwd=autor)

    atrasado = tmp_path / "atrasado"
    _git("clone", "-q", str(remoto), str(atrasado), cwd=tmp_path)

    _commitar(autor, "segundo")
    _git("push", "-q", "origin", "main", cwd=autor)
    return autor, atrasado


def test_clone_em_dia_nao_bloqueia(repositorios):
    autor, _ = repositorios
    estado = estado_do_clone(autor)
    assert estado.estado == EM_DIA
    assert estado.precisa_de_pull is False
    assert exigir_clone_atualizado(raiz=autor).estado == EM_DIA


def test_clone_atrasado_bloqueia_antes_de_carregar_o_gpvs(repositorios):
    """O caso real: o remoto avançou e o clone publicaria código velho."""
    _, atrasado = repositorios
    estado = estado_do_clone(atrasado)
    assert estado.estado == ATRASADO
    assert estado.precisa_de_pull is True
    assert estado.branch == "main"
    assert estado.local != estado.remoto

    with pytest.raises(CloneDefasado) as erro:
        exigir_clone_atualizado(raiz=atrasado)
    # A mensagem precisa trazer o comando; quem lê está no meio da execução.
    assert "git pull origin main" in str(erro.value)


def test_commit_local_ainda_nao_enviado_nao_bloqueia(repositorios):
    """Estar à frente do remoto é trabalho em curso, não defasagem."""
    autor, _ = repositorios
    _git("commit", "-q", "--allow-empty", "-m", "local", cwd=autor)
    estado = estado_do_clone(autor)
    assert estado.estado == ADIANTADO
    assert estado.precisa_de_pull is False
    exigir_clone_atualizado(raiz=autor)


def test_head_destacado_e_fora_de_repositorio_ficam_indeterminados(
    repositorios, tmp_path
):
    """Ambiente sem referência para comparar nunca impede de trabalhar."""
    autor, _ = repositorios
    anterior = _git("rev-parse", "HEAD~1", cwd=autor)
    _git("checkout", "-q", anterior, cwd=autor)
    assert estado_do_clone(autor).estado == INDETERMINADO

    fora = tmp_path / "sem_git"
    fora.mkdir()
    estado = estado_do_clone(fora)
    assert estado.estado == INDETERMINADO
    assert estado.precisa_de_pull is False
    exigir_clone_atualizado(raiz=fora)


def test_progresso_avisa_sem_precisar_do_bloqueio(repositorios):
    """O aviso sai para qualquer estado fora do normal, inclusive adiantado."""
    autor, atrasado = repositorios
    avisos: list[str] = []
    exigir_clone_atualizado(raiz=autor, progresso=avisos.append)
    assert avisos == []

    with pytest.raises(CloneDefasado):
        exigir_clone_atualizado(raiz=atrasado, progresso=avisos.append)
    assert avisos and "git pull" in avisos[0]


def test_executar_etapa_recusa_rodar_com_clone_defasado(monkeypatch, repositorios):
    """A guarda cobre o caminho do pipeline, não só o da CLI."""
    import src.ml.pipeline as pipeline

    _, atrasado = repositorios
    chamou_runner = []
    monkeypatch.setattr(
        pipeline,
        "capacidade_recalculo_pipeline",
        lambda: {"disponivel": True},
    )
    monkeypatch.setattr(
        pipeline.PipelineStage,
        "load_runner",
        lambda self: lambda **_kwargs: chamou_runner.append(True),
    )
    monkeypatch.setattr(
        pipeline,
        "exigir_clone_atualizado",
        lambda **kwargs: exigir_clone_atualizado(raiz=atrasado, **kwargs),
    )

    resultado = pipeline.executar_etapa("detectabilidade")

    assert resultado["ok"] is False
    assert "git pull" in resultado["mensagem"]
    assert chamou_runner == [], "a etapa não pode rodar com o clone atrasado"


def test_executar_etapa_permite_ignorar_a_verificacao(monkeypatch, repositorios):
    """`ignorar_versao` existe para reproduzir um commit anterior de propósito."""
    import src.ml.pipeline as pipeline

    _, atrasado = repositorios
    chamou_runner = []
    monkeypatch.setattr(
        pipeline,
        "capacidade_recalculo_pipeline",
        lambda: {"disponivel": True},
    )
    monkeypatch.setattr(
        pipeline.PipelineStage,
        "load_runner",
        lambda self: lambda **_kwargs: chamou_runner.append(True) or {},
    )
    monkeypatch.setattr(
        pipeline,
        "exigir_clone_atualizado",
        lambda **kwargs: exigir_clone_atualizado(raiz=atrasado, **kwargs),
    )

    pipeline.executar_etapa("detectabilidade", ignorar_versao=True)

    assert chamou_runner == [True]
