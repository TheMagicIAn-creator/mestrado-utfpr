"""
Sprint 1 — integridade acadêmica.

Garante a definição OFICIAL do limiar de anomalia do Autoencoder:

    Limiar operacional  = percentil 99 do erro de reconstrução saudável
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
    assert d.get("limiar") == d.get("limiar_p99")
    assert d.get("limiar_operacional") == d.get("limiar_p99")
    assert "limiar_mu3sigma" in d


def test_resumo_publico_rotula_p99_nao_mu3sigma():
    """O resumo exibido no chat nomeia o limiar como p99, não como μ+3σ."""
    if not LIMIAR_JSON.exists():
        import pytest

        pytest.skip("limiar.json indisponível")
    from src.ml.resultados import _resumo_autoencoder

    msg = _resumo_autoencoder() or ""
    low = msg.lower()
    if "limiar" in low:
        assert "p99" in low, "o resumo deve nomear o limiar operacional como p99"
    # se μ+3σ aparecer, tem de ser como referência — nunca como limiar em uso
    if "3σ" in msg or "3sigma" in low or "mu3" in low:
        assert "refer" in low
