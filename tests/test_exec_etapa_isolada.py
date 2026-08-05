"""Contrato do processo-filho usado pelo pipeline de ML."""

from src.ml import exec_etapa_isolada as executor
from src.ml import pipeline


class _Etapa:
    def __init__(self, retorno=None, erro=None):
        self.retorno = retorno
        self.erro = erro

    def load_runner(self):
        if self.erro:
            raise self.erro
        return lambda: self.retorno


def test_main_exige_nome_da_etapa(capsys):
    assert executor.main(["executor"]) == 2
    assert "uso:" in capsys.readouterr().out


def test_main_reflete_sucesso_e_falha_do_runner(monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "get_stage", lambda nome: _Etapa(nome == "ok"))

    assert executor.main(["executor", "ok"]) == 0
    assert "etapa=ok ok=true" in capsys.readouterr().out
    assert executor.main(["executor", "falha"]) == 1
    assert "etapa=falha ok=false" in capsys.readouterr().out


def test_main_converte_excecao_em_codigo_controlado(monkeypatch):
    monkeypatch.setattr(pipeline, "get_stage", lambda nome: _Etapa(erro=RuntimeError("boom")))
    assert executor.main(["executor", "x"]) == 2
