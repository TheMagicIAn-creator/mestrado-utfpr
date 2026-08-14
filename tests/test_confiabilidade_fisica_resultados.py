from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageStat

from src.ml.proveniencia import (
    SUFIXOS_TEXTO_PORTAVEL,
    sha256_arquivo,
    sha256_arquivo_texto_normalizado,
)

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "resultados" / "v2" / "confiabilidade"


def _json(nome: str) -> dict:
    return json.loads((PASTA / nome).read_text(encoding="utf-8"))


def test_resultado_publica_cenarios_fontes_e_limite_do_dataset():
    resultado = _json("resultado.json")
    assert resultado["schema_version"] == 2
    assert resultado["status"] == "bibliographic_sensitivity_not_dataset_estimate"
    assert resultado["dataset_role"] == "detector_evaluation_only_not_physical_reliability"
    assert len(resultado["scenarios"]) == 5
    for cenario in resultado["scenarios"]:
        assert len(cenario["source_sha256"]) == 64
        assert cenario["source_artifact"].startswith("literatura/inversores-pv/")


def test_weibull_fisico_nao_e_fabricado():
    resultado = _json("resultado.json")
    weibull = resultado["physical_weibull"]
    assert weibull["status"] == "not_estimable_from_current_dataset"
    assert weibull["beta"] is None
    assert weibull["eta"] is None


def test_csvs_reconciliam_taxas_curvas_e_marcos():
    cenarios = pd.read_csv(PASTA / "cenarios.csv").set_index("scenario_id")
    curvas = pd.read_csv(PASTA / "curvas.csv")
    marcos = pd.read_csv(PASTA / "marcos.csv").set_index("scenario_id")
    assert len(cenarios) == 5
    assert len(curvas) == 5 * 401
    for scenario_id, bloco in curvas.groupby("scenario_id"):
        taxa = cenarios.loc[scenario_id, "lambda_per_year"]
        assert bloco["hazard_per_year"].nunique() == 1
        assert bloco["hazard_per_year"].iloc[0] == pytest.approx(taxa)
        assert bloco["time_years"].min() == 0
        assert bloco["time_years"].max() == 20
        assert marcos.loc[scenario_id, "reciprocal_time_years"] == pytest.approx(
            1 / taxa
        )


def test_figuras_png_sao_grandes_e_nao_vazias():
    nomes = (
        "confiabilidade_cenarios.png",
        "probabilidade_falha_cenarios.png",
        "densidade_taxa_falha.png",
        "marcos_confiabilidade.png",
    )
    for nome in nomes:
        path = PASTA / nome
        assert path.exists(), nome
        assert path.stat().st_size > 80_000, nome
        with Image.open(path) as imagem:
            assert imagem.width >= 3_000
            assert imagem.height >= 1_400
            canais = ImageStat.Stat(imagem.convert("RGB"))
            assert min(canais.var) > 50, nome


def test_figuras_pdf_vetoriais_foram_publicadas():
    for stem in (
        "confiabilidade_cenarios",
        "probabilidade_falha_cenarios",
        "densidade_taxa_falha",
        "marcos_confiabilidade",
    ):
        path = PASTA / f"{stem}.pdf"
        assert path.exists(), path.name
        assert path.stat().st_size > 10_000, path.name


def test_manifesto_v2_hasheia_fontes_codigo_e_saidas():
    manifesto = _json("manifesto_v2.json")
    assert manifesto["manifest_version"] == 2
    assert manifesto["evidence_level"] == "bibliographic_sensitivity"
    assert len(manifesto["input_artifacts"]) == 4
    assert len(manifesto["code_dependencies"]) == 3
    assert manifesto["output_artifacts"]
    assert all(len(value) == 64 for value in manifesto["input_artifacts"].values())
    assert all(len(value) == 64 for value in manifesto["output_artifacts"].values())


def test_manifesto_v2_esta_reconciliado_com_os_arquivos_publicados():
    manifesto = _json("manifesto_v2.json")

    def hash_portavel(path: Path) -> str:
        if path.suffix.lower() in SUFIXOS_TEXTO_PORTAVEL:
            return sha256_arquivo_texto_normalizado(path)
        return sha256_arquivo(path)

    for relativo, esperado in manifesto["input_artifacts"].items():
        assert hash_portavel(RAIZ / relativo) == esperado, relativo
    for relativo, esperado in manifesto["output_artifacts"].items():
        assert hash_portavel(RAIZ / relativo) == esperado, relativo
    for relativo, esperado in manifesto["code_dependencies"].items():
        assert sha256_arquivo_texto_normalizado(RAIZ / relativo) == esperado, relativo
    assert sha256_arquivo_texto_normalizado(
        RAIZ / "scripts" / "gerar_confiabilidade_fisica_v2.py"
    ) == manifesto["code_sha256"]
