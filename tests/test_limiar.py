"""
Sprint 1 — integridade acadêmica.

Garante a definição OFICIAL do limiar de anomalia do Autoencoder:

    Referência MSE       = percentil 99 do erro de reconstrução saudável
    Limiar operacional   = score_threshold do score_method vigente
    Referência comparativa = μ + 3σ  (NUNCA usado como limiar em uso)
    Referência adicional   = percentil 95

E que nenhuma mensagem pública trate μ + 3σ como o limiar operacional atual.
"""

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
LIMIAR_JSON = RAIZ / "resultados" / "autoencoder" / "limiar.json"


def test_calcular_limiar_operacional_e_p99():
    from src.ml.autoencoder import calcular_limiar

    rng = np.random.default_rng(0)
    erros = rng.gamma(2.0, 1.0, size=1000)  # erro assimétrico (não-normal)
    info = calcular_limiar(erros)

    assert info["threshold_method"] == "p99"
    assert info["limiar"] == info["limiar_p99"]
    assert info["limiar_operacional"] == info["limiar_p99"]
    assert info["score_method"] == "mse"
    assert info["score_threshold"] == info["limiar_p99"]
    assert info["mse_p99"] == info["limiar_p99"]
    assert info["sigma_multiplier"] == info["k"]
    assert info["threshold_effective_percentile"] == 99.0
    assert "limiar_mu3sigma" in info
    assert "limiar_p95" in info
    # o p99 é, de fato, o percentil 99
    assert abs(info["limiar_p99"] - float(np.percentile(erros, 99))) < 1e-9
    # o limiar operacional NÃO é o μ + 3σ
    assert info["limiar"] != info["limiar_mu3sigma"]


def test_limiar_json_declara_threshold_method():
    if not LIMIAR_JSON.exists():
        import pytest

        pytest.skip("limiar.json indisponível (rode: python src/ml/autoencoder.py)")
    d = json.loads(LIMIAR_JSON.read_text(encoding="utf-8"))
    assert d.get("threshold_method") == "p99"
    assert d.get("limiar") == d.get("score_threshold")
    assert d.get("limiar_operacional") == d.get("score_threshold")
    assert d.get("mse_p99") == d.get("limiar_p99")
    assert d.get("score_method") in {"mse", "localizado"}
    if d.get("score_method") == "localizado":
        assert d.get("top_k") == d.get("k_localizado")
        assert d.get("threshold_effective_percentile") == d.get("percentil_limiar")
    else:
        assert d.get("score_threshold") == d.get("limiar_p99")
    assert "limiar_mu3sigma" in d


def test_resumo_publico_nomeia_ponto_operacional_sem_confundir_com_mu3sigma():
    """O resumo exibido no chat nomeia o ponto operacional, não μ+3σ."""
    if not LIMIAR_JSON.exists():
        import pytest

        pytest.skip("limiar.json indisponível")
    from src.ml.resultados import _resumo_autoencoder

    msg = _resumo_autoencoder() or ""
    low = msg.lower()
    if "limiar" in low:
        assert "limiar operacional" in low
        assert "escore operacional" in low
        assert "mse p99" in low
    # se μ+3σ aparecer, tem de ser como referência — nunca como limiar em uso
    if "3σ" in msg or "3sigma" in low or "mu3" in low:
        assert "refer" in low


def test_diagnostico_le_artefato_historico_com_localizado_operacional():
    from src.ml.diagnostico_escore import _limiares_comparacao

    scores = np.linspace(0.0, 10.0, 1001)
    info = {
        "limiar": 8.5,
        "score_method": "localizado",
        "score_threshold": 8.5,
        "mse_p99": 2.5,
        "limiar_p99": 2.5,
        "threshold_effective_percentile": 99.9,
        "top_k": 5,
    }

    mse, localizado, percentil, operacional = _limiares_comparacao(
        info, scores, k=5
    )

    assert mse == 2.5
    assert localizado == 8.5
    assert percentil == 99.9
    assert operacional is True


def test_diagnostico_recalibra_localizado_quando_k_muda():
    from src.ml.diagnostico_escore import _limiares_comparacao

    scores = np.linspace(0.0, 10.0, 1001)
    info = {
        "score_method": "localizado",
        "score_threshold": 8.5,
        "mse_p99": 2.5,
        "threshold_effective_percentile": 99.9,
        "top_k": 5,
    }

    _, localizado, _, operacional = _limiares_comparacao(info, scores, k=3)

    assert localizado == float(np.percentile(scores, 99.9))
    assert operacional is False
