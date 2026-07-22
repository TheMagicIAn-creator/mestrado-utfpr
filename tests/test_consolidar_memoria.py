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
