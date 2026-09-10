"""A publicação E2 é dado-fonte auditável: resumo, POD, âncora e manifesto.

Sem torch nem dataset: alimenta o publicador com resultados sintéticos e fixa a
forma dos artefatos e do manifesto. O manifesto é redirecionado para um diretório
temporário para não escrever em `resultados/manifestos/` durante o teste.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import src.ml.proveniencia as proveniencia
from src.ml.campanha_detectabilidade import ResultadoInjecao
from src.ml.detectabilidade import Detectabilidade
from src.ml.injecao_e2 import ESPECIFICACOES, GRADE_SEVERIDADE
from src.ml.publicacao_detectabilidade import RESSALVA_DE_ESCALA, salvar_resultados
from src.ml.treino_comparacao import MODEL_IDS

pytestmark = pytest.mark.leve


def _detectabilidade(injecao: str, model_id: str, *, n: int = 14) -> Detectabilidade:
    grade = np.asarray(GRADE_SEVERIDADE, dtype=float)
    rng = np.random.default_rng(sum(map(ord, injecao + model_id)))
    detectados = rng.choice(grade[grade < 0.8], size=n - 3, replace=True)
    a_dets = np.concatenate([detectados, np.ones(3)])
    eventos = np.concatenate([np.ones(n - 3, dtype=bool), np.zeros(3, dtype=bool)])
    return Detectabilidade(
        modelo=model_id,
        injecao=injecao,
        a_dets=a_dets,
        eventos=eventos,
        n_trajetorias=n,
        grade=tuple(float(a) for a in grade),
    )


def _ancora(distancia: float) -> dict:
    return {
        "distancia_euclidiana_iqr": distancia,
        "distancia_mediana_iqr": distancia / 4.0,
        "distancia_maxima_iqr": distancia,
        "feature_mais_distante": "i_a_rms",
        "n_injetadas": 14,
        "n_reais": 3,
        "unidade": "IQR do bloco saudável de treino",
    }


def _resultados() -> list[ResultadoInjecao]:
    saida = []
    for indice, especificacao in enumerate(ESPECIFICACOES):
        por_modelo = {
            model_id: _detectabilidade(especificacao.id, model_id)
            for model_id in MODEL_IDS
        }
        distancia = 0.0 if especificacao.id == "controle" else float(indice + 1)
        saida.append(
            ResultadoInjecao(
                especificacao=especificacao,
                por_modelo=por_modelo,
                ancora=_ancora(distancia),
                registros_falha=[{"experiment": especificacao.ensaios_reais[0], "window_index": 0}],
            )
        )
    return saida


def _parametros() -> dict:
    return {
        "grade_severidade": [float(a) for a in GRADE_SEVERIDADE],
        "confirmacoes": 3,
        "n_trajetorias": 14,
        "holdout_experiments": ["F0L", "F0M"],
        "sequence_policy": "sustained_severity_repeated_frame",
        "normalization_policy": "f0_baseline_of_source_experiment_then_frozen_scaler",
        "threshold_source": "frozen_from_comparacao",
        "a_axis": "fraction_of_nominal_signature_not_time",
    }


@pytest.fixture
def _publicado(tmp_path, monkeypatch):
    monkeypatch.setattr(proveniencia, "PASTA_MANIFESTOS", tmp_path / "manifestos")
    saved = salvar_resultados(
        _resultados(),
        dataset_manifest={"windows": 320},
        holdout_meta={
            "dataset": "GPVS-Faults",
            "doi": "10.17632/n76t439f65.1",
            "protocol": "purged_temporal_holdout_F0L_F0M",
            "purge_windows": 2,
            "n_windows": 14,
            "records": [{"experiment": "F0L"}],
        },
        parametros=_parametros(),
        results_dir=tmp_path / "detectabilidade",
    )
    return tmp_path, saved


def test_escreve_todos_os_dados_fonte(_publicado):
    tmp_path, saved = _publicado
    destino = tmp_path / "detectabilidade"
    for nome in (
        "detectabilidade_resumo.csv",
        "pod_curvas.csv",
        "ancoras.csv",
        "fidelidade_injecao.csv",
        "detectabilidade.json",
        "relatorio.md",
        "e2_pod_curvas.png",
        "e2_pod_curvas.pdf",
        "e2_fidelidade.png",
        "e2_fidelidade.pdf",
    ):
        assert (destino / nome).is_file(), nome
    assert len(saved["outputs"]) == 10


def test_contrato_json_declara_e2_e_os_tres_itens(_publicado):
    tmp_path, _ = _publicado
    payload = json.loads(
        (tmp_path / "detectabilidade" / "detectabilidade.json").read_text(encoding="utf-8")
    )
    assert payload["evidence_level"] == "E2"
    assert payload["stage"] == "detectabilidade"
    assert len(payload["injections"]) == len(ESPECIFICACOES)
    assert payload["methodology"]["a_axis"] == "fraction_of_nominal_signature_not_time"
    assert "weibull_adoption_gate" in payload["methodology"]
    metodos = {item["injection_method"] for item in payload["injections"]}
    assert metodos == {"electrical_signature", "measured_state_interpolation"}


def test_pod_e_resumo_tem_a_forma_esperada(_publicado):
    import pandas as pd

    tmp_path, _ = _publicado
    destino = tmp_path / "detectabilidade"
    resumo = pd.read_csv(destino / "detectabilidade_resumo.csv")
    pod = pd.read_csv(destino / "pod_curvas.csv")
    ancoras = pd.read_csv(destino / "ancoras.csv")

    assert len(resumo) == len(ESPECIFICACOES) * len(MODEL_IDS)
    assert {"injection_method", "model", "a10_empirico", "weibull_2p_adotada"} <= set(
        resumo.columns
    )
    assert resumo["evidence_level"].eq("E2").all()
    assert len(pod) == len(ESPECIFICACOES) * len(MODEL_IDS) * len(GRADE_SEVERIDADE)
    assert pod["pod"].between(0.0, 1.0).all()
    assert len(ancoras) == len(ESPECIFICACOES)


def test_fidelidade_confronta_pod_a1_com_o_recall_da_e3(_publicado):
    """O confronto existe e carrega a ressalva de escala junto do número."""
    import pandas as pd

    tmp_path, _ = _publicado
    destino = tmp_path / "detectabilidade"
    fidelidade = pd.read_csv(destino / "fidelidade_injecao.csv")

    assert len(fidelidade) == len(ESPECIFICACOES) * len(MODEL_IDS)
    assert {"pod_a1", "e3_recall_min", "e3_recall_max", "status"} <= set(
        fidelidade.columns
    )
    assert fidelidade["pod_a1"].between(0.0, 1.0).all()
    assert fidelidade["escalas_de_normalizacao_distintas"].all()

    payload = json.loads(
        (destino / "detectabilidade.json").read_text(encoding="utf-8")
    )
    assert payload["injection_fidelity"]["scale_caveat"] == RESSALVA_DE_ESCALA
    assert len(payload["injection_fidelity"]["comparisons"]) == len(fidelidade)
    assert RESSALVA_DE_ESCALA in (destino / "relatorio.md").read_text(encoding="utf-8")


def test_sem_artefato_da_e3_a_fidelidade_degrada_sem_derrubar_a_etapa(
    tmp_path, monkeypatch
):
    """A E2 tem de continuar publicável sozinha: status no lugar de exceção."""
    import pandas as pd

    monkeypatch.setattr(proveniencia, "PASTA_MANIFESTOS", tmp_path / "manifestos")
    saved = salvar_resultados(
        _resultados(),
        dataset_manifest={"windows": 320},
        holdout_meta={"dataset": "GPVS-Faults", "n_windows": 14},
        parametros=_parametros(),
        results_dir=tmp_path / "detectabilidade",
        e3_metricas_path=tmp_path / "nao_existe" / "e3_metricas_por_ensaio.csv",
    )

    assert len(saved["outputs"]) == 10
    fidelidade = pd.read_csv(tmp_path / "detectabilidade" / "fidelidade_injecao.csv")
    assert fidelidade["status"].eq("sem_artefato_e3").all()
    assert fidelidade["e3_recall_min"].isna().all()
    assert fidelidade["divergencia_ate_faixa_e3"].isna().all()
    # A POD continua publicada: o que falta é o confronto, não o resultado.
    assert fidelidade["pod_a1"].notna().all()


def test_manifesto_registra_a_etapa_e_as_saidas(_publicado):
    tmp_path, saved = _publicado
    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    assert manifest["stage"] == "detectabilidade"
    assert manifest["evidence_level"] == "E2"
    assert len(manifest["outputs"]) == 10
    # As figuras entram no manifesto com o módulo que as desenha: mudar o estilo
    # sem regenerar passa a ser divergência de hash, não silêncio.
    assert "plots" in manifest["code_dependencies"]
    assert "plot_style" in manifest["code_dependencies"]
    assert manifest["parameters"]["sequence_policy"] == "sustained_severity_repeated_frame"
