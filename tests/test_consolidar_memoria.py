from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.conhecimento.consolidar_memoria as cm
import src.conhecimento.provedores as provedores


def _sessao(tmp_path: Path) -> dict:
    return {
        "arquivo": tmp_path / "2026-07-21_10-00_sessao_web.md",
        "conteudo": "## Interação\nPergunta e resposta com conteúdo suficiente.",
        "data": "2026-07-21",
        "interacoes": 1,
    }


def test_consolidacao_aceita_resposta_em_blocos(tmp_path, monkeypatch):
    class LLM:
        def invoke(self, _mensagens):
            return SimpleNamespace(content=[
                {"type": "text", "text": "## Síntese\n"},
                {"type": "output_text", "text": "Decisão preservada."},
            ])

    monkeypatch.setattr(
        provedores, "inicializar_llm_fundo", lambda **_kwargs: LLM()
    )

    resumo = cm.consolidar_com_llm([_sessao(tmp_path)], "")

    assert resumo == "## Síntese\nDecisão preservada."


def test_consolidacao_falha_sem_salvar_texto_de_erro(tmp_path, monkeypatch):
    monkeypatch.setattr(
        provedores,
        "inicializar_llm_fundo",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("API indisponível")),
    )

    with pytest.raises(RuntimeError, match="API indisponível"):
        cm.consolidar_com_llm([_sessao(tmp_path)], "")


def test_salvar_consolidado_normaliza_blocos_e_escreve_atomicamente(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cm, "PASTA_MEMORIAS", tmp_path)
    resumo = SimpleNamespace(content=[{"type": "text", "text": "Resumo válido"}])

    caminho = cm.salvar_consolidado(resumo, [_sessao(tmp_path)])

    assert caminho.is_file()
    assert caminho.name.endswith("_consolidado.md")
    assert "Resumo válido" in caminho.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_forcar_consolida_uma_unica_sessao(tmp_path, monkeypatch):
    sessao = _sessao(tmp_path)
    chamadas = []
    destino = tmp_path / "memoria.md"
    destino.write_text("memória", encoding="utf-8")

    monkeypatch.setattr(cm, "ler_sessoes", lambda: [sessao])
    monkeypatch.setattr(cm, "ler_memoria_anterior", lambda: "")
    monkeypatch.setattr(cm, "consolidar_com_llm", lambda *_args: "Resumo")
    monkeypatch.setattr(cm, "salvar_consolidado", lambda *_args: destino)
    monkeypatch.setattr(cm, "consolidar_memoria_validada", lambda _s: None)
    monkeypatch.setattr(
        cm, "atualizar_chromadb", lambda _c, _s: chamadas.append("indice")
    )
    monkeypatch.setattr(cm, "arquivar_sessoes", lambda _s: chamadas.append("arquivo"))

    assert cm.consolidar(forcar=True) is True
    assert chamadas == ["indice", "arquivo"]


def test_persiste_consolidacao_na_nuvem_quando_ativa(tmp_path, monkeypatch):
    """Bug real: consolidar() MOVIA sessoes para fora do caminho persistido e
    gerava um resumo novo, ambos so em disco local ate este fix. Verifica que,
    com a persistencia ligada, o resumo consolidado E a sessao arquivada sao
    ambos enviados para a nuvem."""
    import src.core.config as config

    sessao = _sessao(tmp_path)
    arquivo_pasta = tmp_path / "arquivadas"
    arquivo_pasta.mkdir()
    destino = tmp_path / "memoria.md"
    destino.write_text("memória", encoding="utf-8")
    arquivo_sessao_arquivada = arquivo_pasta / sessao["arquivo"].name
    arquivo_sessao_arquivada.write_text("sessao arquivada", encoding="utf-8")

    persistidos = []

    monkeypatch.setattr(cm, "ler_sessoes", lambda: [sessao])
    monkeypatch.setattr(cm, "ler_memoria_anterior", lambda: "")
    monkeypatch.setattr(cm, "consolidar_com_llm", lambda *_args: "Resumo")
    monkeypatch.setattr(cm, "salvar_consolidado", lambda *_args: destino)
    monkeypatch.setattr(cm, "consolidar_memoria_validada", lambda _s: None)
    monkeypatch.setattr(cm, "atualizar_chromadb", lambda _c, _s: None)
    monkeypatch.setattr(cm, "arquivar_sessoes", lambda _s: None)
    monkeypatch.setattr(config, "PASTA_ARQUIVO", arquivo_pasta)
    monkeypatch.setattr(cm, "PASTA_ARQUIVO", arquivo_pasta)

    class _FakePersistencia:
        @staticmethod
        def persistencia_ativa():
            return True

        @staticmethod
        def persistir_arquivo(caminho, *, mensagem, alvo):
            persistidos.append((Path(caminho).name, alvo))
            return True

    monkeypatch.setitem(
        __import__("sys").modules,
        "src.conhecimento.persistencia_nuvem",
        _FakePersistencia,
    )

    assert cm.consolidar(forcar=True) is True
    nomes_e_alvos = set(persistidos)
    assert (destino.name, "consolidado") in nomes_e_alvos
    assert (sessao["arquivo"].name, "sessao") in nomes_e_alvos


def test_nao_persiste_na_nuvem_quando_desligada(tmp_path, monkeypatch):
    sessao = _sessao(tmp_path)
    destino = tmp_path / "memoria.md"
    destino.write_text("memória", encoding="utf-8")

    monkeypatch.setattr(cm, "ler_sessoes", lambda: [sessao])
    monkeypatch.setattr(cm, "ler_memoria_anterior", lambda: "")
    monkeypatch.setattr(cm, "consolidar_com_llm", lambda *_args: "Resumo")
    monkeypatch.setattr(cm, "salvar_consolidado", lambda *_args: destino)
    monkeypatch.setattr(cm, "consolidar_memoria_validada", lambda _s: None)
    monkeypatch.setattr(cm, "atualizar_chromadb", lambda _c, _s: None)
    monkeypatch.setattr(cm, "arquivar_sessoes", lambda _s: None)
    monkeypatch.delenv("AL_IADO_PERSISTIR_NUVEM", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # Nao deve levantar nem exigir rede: persistencia_ativa() e False.
    assert cm.consolidar(forcar=True) is True
