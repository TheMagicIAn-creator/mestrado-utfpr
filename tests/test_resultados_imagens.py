import json
from pathlib import Path


def _png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(path)


def test_imagens_experimento_distingue_graficos_e_matrizes(tmp_path, monkeypatch):
    from src.ml import resultados

    pasta = tmp_path / "sharma"
    dados = {
        "referencia": "Sharma et al. (2026)",
        "modelos": {
            "SVM": {
                "disponivel": True,
                "grafico_metricas": _png(pasta / "svm_metricas.png"),
                "grafico_matriz_confusao": _png(pasta / "svm_matriz.png"),
            },
            "CNN": {
                "disponivel": True,
                "grafico_metricas": _png(pasta / "cnn_metricas.png"),
                "grafico_matriz_confusao": _png(pasta / "cnn_matriz.png"),
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
        "Mostre a matriz de confusao do Sharma."
    )
    ambos = resultados.imagens_relevantes(
        "Mostre os graficos e matrizes do Sharma."
    )

    assert len(somente_matriz) == 2
    assert all("matriz de confusao" in img["caption"] for img in somente_matriz)
    assert [img["caption"] for img in ambos] == [
        "Sharma et al. (2026) - comparacao de metricas",
        "Sharma et al. (2026) - anomalias detectadas",
        "Sharma et al. (2026) - resultado individual (SVM)",
        "Sharma et al. (2026) - matriz de confusao (SVM)",
        "Sharma et al. (2026) - resultado individual (CNN)",
        "Sharma et al. (2026) - matriz de confusao (CNN)",
    ]
