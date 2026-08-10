import json
from pathlib import Path


def _png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(path)


def test_imagens_experimento_distingue_graficos_e_matrizes(tmp_path, monkeypatch):
    from src.ml import resultados

    pasta = tmp_path / "ibrahim"
    dados = {
        "referencia": "Ibrahim et al. (2022)",
        "modelos": {
            "AE-LSTM": {
                "disponivel": True,
                "grafico_metricas": _png(pasta / "ae_metricas.png"),
                "grafico_matriz_confusao": _png(pasta / "ae_matriz.png"),
            },
        },
    }
    (pasta / "resultado.json").write_text(
        json.dumps(dados, ensure_ascii=False), encoding="utf-8"
    )
    _png(pasta / "comparacao_metricas.png")
    _png(pasta / "comparacao_metricas_barras.png")
    _png(pasta / "comparacao_metricas_pontos.png")
    _png(pasta / "anomalias_detectadas.png")

    monkeypatch.setattr(resultados, "PASTA_EXPERIMENTOS", tmp_path)

    somente_matriz = resultados.imagens_relevantes(
        "Mostre a matriz de confusao do Ibrahim."
    )
    ambos = resultados.imagens_relevantes(
        "Mostre os graficos e matrizes do Ibrahim."
    )

    assert len(somente_matriz) == 1
    assert all("matriz de confusao" in img["caption"] for img in somente_matriz)
    assert [img["caption"] for img in ambos] == [
        "Ibrahim et al. (2022) - comparacao de metricas",
        "Ibrahim et al. (2022) - anomalias detectadas",
        "Ibrahim et al. (2022) - resultado individual (AE-LSTM)",
        "Ibrahim et al. (2022) - matriz de confusao (AE-LSTM)",
    ]

    barras = resultados.imagens_relevantes(
        "Mostre um grafico de barras do Ibrahim."
    )
    pontos = resultados.imagens_relevantes(
        "Mostre a comparacao por pontos do Ibrahim."
    )

    assert Path(barras[0]["path"]).name == "comparacao_metricas_barras.png"
    assert "barras horizontais" in barras[0]["caption"]
    assert Path(pontos[0]["path"]).name == "comparacao_metricas_pontos.png"
    assert "por pontos" in pontos[0]["caption"]

    comparativo = resultados.imagens_relevantes(
        "Compare Ibrahim por AUC, diga quantas anomalias o modelo detectou "
        "e mostre a comparacao por pontos."
    )
    assert [Path(img["path"]).name for img in comparativo] == [
        "comparacao_metricas_pontos.png",
        "anomalias_detectadas.png",
    ]


def test_tabela_de_anomalias_e_compacta_e_especifica(tmp_path, monkeypatch):
    from src.ml import resultados

    pasta = tmp_path / "ibrahim"
    pasta.mkdir(parents=True)
    (pasta / "resultado.json").write_text(
        json.dumps({
            "referencia": "Ibrahim et al. (2022)",
            "modelos": {
                "AE-LSTM": {
                    "disponivel": True,
                    "accuracy": 0.5,
                    "auc": 0.5,
                    "recall": 0.2,
                    "anomalias_detectadas": 12,
                    "anomalias_reais": 40,
                    "taxa_anomalias_detectadas": 0.3,
                }
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(resultados, "PASTA_EXPERIMENTOS", tmp_path)

    tabela = resultados._resumo_experimentos(
        "Quais modelos detectaram mais anomalias?"
    )

    assert "| Detectadas | Reais | Taxa marcada | Recall |" in tabela
    assert "| Accuracy | Precision |" not in tabela

    tabela_composta = resultados._resumo_experimentos(
        "Faca um ranking por AUC e diga quantas anomalias cada modelo detectou."
    )
    assert "| AUC | Detectadas | Reais | Taxa marcada | Recall |" in tabela_composta
    assert "| 1 | Ibrahim et al. (2022) | AE-LSTM | 0.500 | 12 |" in tabela_composta


def test_gpvs_entra_no_resumo_e_nas_figuras_do_agente(tmp_path, monkeypatch):
    from src.ml import resultados

    metrica = {
        "mean": 0.8,
        "ci95_low": 0.7,
        "ci95_high": 0.9,
        "n_experiments": 14,
    }
    bloco = {
        "auc": dict(metrica),
        "sensitivity": dict(metrica, mean=0.45),
        "specificity": dict(metrica, mean=0.97),
        "balanced_accuracy": dict(metrica, mean=0.71),
    }
    (tmp_path / "validacao_gpvs_e3.json").write_text(
        json.dumps({
            "schema_version": 2,
            "macro_summary": {
                "canonical_ae": {"all": bloco},
            }
        }),
        encoding="utf-8",
    )
    for nome in (
        "gpvs_macro_comparacao.png",
        "gpvs_transferencia_estrita.png",
        "gpvs_metricas_por_cenario.png",
    ):
        _png(tmp_path / nome)
    monkeypatch.setattr(resultados, "PASTA_GPVS", tmp_path)

    resumo = resultados.resumir_resultados(
        "Mostre a validação externa GPVS.", incluir_imagens=True
    )

    assert "E3 de bancada" in resumo["mensagem"]
    assert "detector ajustado somente" in resumo["mensagem"]
    assert "não é campo" in resumo["mensagem"]
    assert [Path(img["path"]).name for img in resumo["imagens"]] == [
        "gpvs_macro_comparacao.png",
        "gpvs_transferencia_estrita.png",
        "gpvs_metricas_por_cenario.png",
    ]
