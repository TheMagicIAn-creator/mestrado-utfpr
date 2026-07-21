from datetime import timedelta

from src.core.tempo import agora_local, fuso_projeto


def test_relogio_padrao_usa_fuso_de_sao_paulo(monkeypatch):
    monkeypatch.delenv("AL_IADO_TIMEZONE", raising=False)

    agora = agora_local()

    assert fuso_projeto().key == "America/Sao_Paulo"
    assert agora.tzinfo is not None
    assert agora.utcoffset() == timedelta(hours=-3)


def test_relogio_aceita_fuso_configuravel(monkeypatch):
    monkeypatch.setenv("AL_IADO_TIMEZONE", "Europe/Lisbon")

    assert fuso_projeto().key == "Europe/Lisbon"
