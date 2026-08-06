"""
Regressão: executar_etapa deve RE-EXECUTAR uma etapa STALE, não pular.

Bug corrigido: o SKIP ("ja esta pronto") disparava em is_complete() (artefatos
existem no disco), ignorando o estado stale. Após um `git pull` que muda o
código de uma etapa (ex.: injecao_falhas com nova taxonomia FMECA), os
artefatos/gráficos antigos ficavam no disco e o pipeline os dava como prontos —
deixando os gráficos defasados. Agora só pula quando READY.
"""

from __future__ import annotations

import src.ml.pipeline as pipe


class _StageFake:
    label = "Etapa X"

    def is_complete(self):
        return True  # artefatos existem no disco (não deve bastar p/ pular)


def _prepara(monkeypatch, estado):
    monkeypatch.setattr(pipe, "get_stage", lambda k: _StageFake())
    monkeypatch.setattr(pipe, "estado_etapa_completo", lambda k: {"estado": estado, "motivos": []})
    monkeypatch.setattr(pipe, "dependencias_pendentes", lambda k: [])
    limpezas = []
    monkeypatch.setattr(pipe, "limpar_artefatos", lambda k: limpezas.append(k) or [])
    rodou = []

    class _StageComRunner(_StageFake):
        def load_runner(self):
            return lambda: rodou.append(True) or True

    monkeypatch.setattr(pipe, "get_stage", lambda k: _StageComRunner())
    monkeypatch.setattr(pipe, "registrar_manifesto", lambda *a, **k: None)
    return rodou, limpezas


def test_ready_pula_sem_rodar(monkeypatch):
    """READY pula o runner — e a mensagem tem de DIZER que pulou.

    A mensagem antiga era "já está pronto", e o chamador concatenava a tabela
    de resultados logo abaixo: lia-se como execução fresca. Com o treino
    determinístico (semente fixa), não havia como distinguir SKIP de recálculo
    olhando os arquivos. Ver docs/auditoria_total_src.md §2.
    """
    rodou, _ = _prepara(monkeypatch, "ready")
    res = pipe.executar_etapa("injecao_falhas", auto_deps=False)
    assert res["executou"] is False
    assert res["recalculou"] is False
    assert "NAO recalculei" in res["mensagem"]
    assert res["artefatos_de"]                # carimbo de origem, sempre
    assert rodou == []                       # NÃO rodou o runner


def test_stale_reexecuta(monkeypatch):
    rodou, limpezas = _prepara(monkeypatch, "stale")
    res = pipe.executar_etapa("injecao_falhas", auto_deps=False)
    assert res["executou"] is True           # RE-EXECUTOU
    assert rodou == [True]
    assert "injecao_falhas" in limpezas      # limpou antes de regenerar


def test_pending_roda(monkeypatch):
    rodou, _ = _prepara(monkeypatch, "pending")
    res = pipe.executar_etapa("injecao_falhas", auto_deps=False)
    assert res["executou"] is True
    assert rodou == [True]
