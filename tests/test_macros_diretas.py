"""Contratos diretos dos três macro-códigos acadêmicos."""

import sys
from types import SimpleNamespace

import numpy as np

from src.ml import escore_anomalia, gpvs_principal, macro_comparar, macro_ibrahim, macro_proposto
from src.ml import macro_comum


class _Scaler:
    def transform(self, valores):
        return np.asarray(valores) * 2


def test_scorer_proposto_encadeia_features_residuo_e_escore(monkeypatch):
    capturado = {}
    # O extrator canonico e o do GPVS desde 15/08/2026. Antes este monkeypatch
    # apontava para features_ca (Stender) -- o mesmo modulo que, em producao,
    # devolvia 0,0 para as 24 features do GPVS sem levantar erro.
    monkeypatch.setattr(
        gpvs_principal, "vetores_de_janelas",
        lambda janelas, colunas, normalizacao=None: np.asarray(
            [[j["a"], j["b"]] for j in janelas], dtype=np.float32
        ),
    )

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
        # Sem esta chave o scorer estoura KeyError -- e isso e a guarda: o
        # detector que nao carregar a normalizacao de comissionamento nao
        # pontua, em vez de pontuar errado.
        "normalizacao": None,
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
        lambda janelas, colunas, scaler, normalizacao=None: (
            np.array([[3.0, 4.0], [5.0, 6.0]])
        ),
    )

    def com_contexto(base, atual, tamanho):
        capturado.update(base=base, atual=atual, tamanho=tamanho)
        return sequencias

    modulo_falso = SimpleNamespace(
        sequencias_com_contexto=com_contexto,
        pontuar_ae_lstm=lambda model, seq: np.array([.1, .2]),
    )
    monkeypatch.setitem(sys.modules, "src.ml.modelos_anomalia", modulo_falso)

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
    monkeypatch.setattr(
        macro_comparar,
        "registrar_manifesto",
        lambda n: tmp_path / "macro_comparacao.json",
    )

    assert macro_comparar.executar(12) == [proposto, ibrahim]


def test_macro_comparar_declara_todas_as_saidas_versionaveis(monkeypatch, tmp_path):
    monkeypatch.setattr(macro_comparar, "PASTA_SAIDA", tmp_path)

    saidas = macro_comparar._saidas_macro()

    assert len(saidas) == 12
    assert len(set(saidas)) == 12
    assert tmp_path / "comparacao_resultado.json" in saidas
    assert tmp_path / "proposto_deteccao_severidade.png" in saidas
    assert tmp_path / "ibrahim_tabela.md" in saidas
