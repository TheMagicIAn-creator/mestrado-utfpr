from __future__ import annotations

import pandas as pd

from scripts import verificar_datasets as vd


def _gravar_classes(caminho, distribuicao: dict[int, int]) -> None:
    linhas = []
    indice = 0
    for classe, quantidade in distribuicao.items():
        for _ in range(quantidade):
            linhas.append({"sinal": float(indice), "class": classe})
            indice += 1
    pd.DataFrame(linhas).to_csv(caminho, sep=";", index=False)


def _somente_pv(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(vd, "BASE", tmp_path)
    monkeypatch.setattr(vd, "MANIFESTO", tmp_path / "dataset_manifest.json")
    monkeypatch.setattr(vd, "DATASETS", vd.DATASETS[:2])


def test_arquivos_pv_farms_completos_sao_utilizaveis(monkeypatch, tmp_path):
    _somente_pv(monkeypatch, tmp_path)
    _gravar_classes(
        tmp_path / "train_data.csv",
        {0: 100, 1: 153, 2: 149, 3: 198},
    )
    _gravar_classes(tmp_path / "test_data.csv", {0: 25, 1: 25, 2: 25, 3: 25})

    resultado = vd.verificar(silencioso=True)

    assert resultado["PV Farms simulado (treino)"]["utilizavel"] is True
    assert resultado["PV Farms simulado (teste)"]["utilizavel"] is True
    assert resultado["PV Farms simulado (treino)"]["linhas"] == 600
    assert resultado["PV Farms simulado (teste)"]["linhas"] == 100


def test_previa_de_treino_com_cem_linhas_e_rejeitada(monkeypatch, tmp_path):
    _somente_pv(monkeypatch, tmp_path)
    _gravar_classes(tmp_path / "train_data.csv", {0: 25, 1: 25, 2: 25, 3: 25})

    resultado = vd.verificar(silencioso=True)["PV Farms simulado (treino)"]

    assert resultado["utilizavel"] is False
    assert resultado["truncado"] is True
    assert resultado["linhas"] == 100
    assert "mínimo 600" in resultado["aviso"]


def test_duplicatas_sao_registradas_sem_invalidar_estrutura(monkeypatch, tmp_path):
    _somente_pv(monkeypatch, tmp_path)
    linhas = []
    for classe in range(4):
        linhas.extend({"sinal": float(classe), "class": classe} for _ in range(150))
    pd.DataFrame(linhas).to_csv(tmp_path / "train_data.csv", sep=";", index=False)

    resultado = vd.verificar(silencioso=True)["PV Farms simulado (treino)"]

    assert resultado["utilizavel"] is True
    assert resultado["linhas_duplicadas"] == 596
    assert "grupos na validação cruzada" in resultado["aviso"]
