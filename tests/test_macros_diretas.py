"""Contratos diretos dos três macro-códigos acadêmicos."""

import numpy as np

from src.ml import escore_anomalia, features_ca, macro_comparar, macro_ibrahim, macro_proposto
from src.ml import macro_comum, modelos_anomalia


class _Scaler:
    def transform(self, valores):
        return np.asarray(valores) * 2


def test_scorer_proposto_encadeia_features_residuo_e_escore(monkeypatch):
    capturado = {}
    monkeypatch.setattr(features_ca, "extrair_janela", lambda janela: janela)

    def residuo(modelo, valores, device):
        capturado["valores"] = valores
        return valores + 1

    monkeypatch.setattr(escore_anomalia, "residuo_por_feature", residuo)
    monkeypatch.setattr(
        escore_anomalia,
        "pontuar",
        lambda residuos, estat, metodo, k: residuos.sum(axis=1),
    )
    detector = {
        "colunas": ["a", "b"],
        "scaler": _Scaler(),
        "modelo": object(),
        "device": "cpu",
        "estat": object(),
        "metodo": "localizado",
        "k": 2,
    }

    escores = macro_proposto.construir_scorer(detector)([{"a": 1, "b": 2}])

    assert np.allclose(capturado["valores"], [[2, 4]])
    assert np.allclose(escores, [8])


def test_scorer_ibrahim_preserva_contexto_temporal(monkeypatch):
    contexto = np.array([[1.0, 2.0]])
    sequencias = np.ones((2, 3, 2))
    capturado = {}
    monkeypatch.setattr(
        macro_ibrahim,
        "features_das_janelas",
        lambda janelas, colunas, scaler: np.array([[3.0, 4.0], [5.0, 6.0]]),
    )

    def com_contexto(base, atual, tamanho):
        capturado.update(base=base, atual=atual, tamanho=tamanho)
        return sequencias

    monkeypatch.setattr(modelos_anomalia, "sequencias_com_contexto", com_contexto)
    monkeypatch.setattr(modelos_anomalia, "pontuar_ae_lstm", lambda model, seq: np.array([.1, .2]))

    escores = macro_ibrahim.construir_scorer("modelo", contexto, ["a", "b"], _Scaler())([1, 2])

    assert np.allclose(escores, [.1, .2])
    assert capturado["base"] is contexto
    assert capturado["tamanho"] == macro_ibrahim.SEQ_LEN
    assert capturado["atual"].shape == (2, 2)


def test_macro_comparar_executa_metodos_e_publica_uma_saida(monkeypatch, tmp_path):
    proposto = {"metodo": "proposto"}
    ibrahim = {"metodo": "ibrahim"}
    monkeypatch.setattr(macro_proposto, "executar", lambda n: proposto)
    monkeypatch.setattr(macro_ibrahim, "executar", lambda n: ibrahim)
    monkeypatch.setattr(macro_comparar, "PASTA_SAIDA", tmp_path)
    monkeypatch.setattr(macro_comum, "tabela_enxuta", lambda itens: "tabela")
    monkeypatch.setattr(
        macro_comum,
        "salvar_saidas",
        lambda itens, pasta, prefixo: {"json": pasta / f"{prefixo}.json"},
    )

    assert macro_comparar.executar(12) == [proposto, ibrahim]
