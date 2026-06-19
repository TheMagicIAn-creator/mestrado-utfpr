import json
from pathlib import Path


def _png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(path)


def test_imagens_experimento_distingue_graficos_e_matrizes(tmp_path, monkeypatch):
    # Usa um experimento do NÚCLEO curado (Ibrahim) como veículo do teste de
    # agrupamento de imagens — Sharma/Ghoneim foram removidos do registry.
    from src.ml import resultados

    pasta = tmp_path / "ibrahim"
    dados = {
        "referencia": "Ibrahim et al. (2022)",
        "modelos": {
            "Isolation Forest": {
                "disponivel": True,
                "grafico_metricas": _png(pasta / "if_metricas.png"),
                "grafico_matriz_confusao": _png(pasta / "if_matriz.png"),
            },
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
    _png(pasta / "anomalias_detectadas.png")

    monkeypatch.setattr(resultados, "PASTA_EXPERIMENTOS", tmp_path)

    somente_matriz = resultados.imagens_relevantes(
        "Mostre a matriz de confusao do Ibrahim."
    )
    ambos = resultados.imagens_relevantes(
        "Mostre os graficos e matrizes do Ibrahim."
    )

    assert len(somente_matriz) == 2
    assert all("matriz de confusao" in img["caption"] for img in somente_matriz)
    assert [img["caption"] for img in ambos] == [
        "Ibrahim et al. (2022) - comparacao de metricas",
        "Ibrahim et al. (2022) - anomalias detectadas",
        "Ibrahim et al. (2022) - resultado individual (Isolation Forest)",
        "Ibrahim et al. (2022) - matriz de confusao (Isolation Forest)",
        "Ibrahim et al. (2022) - resultado individual (AE-LSTM)",
        "Ibrahim et al. (2022) - matriz de confusao (AE-LSTM)",
    ]
