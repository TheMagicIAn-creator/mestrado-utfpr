"""As figuras do Autoencoder plotam MSE — o limiar desenhado tem de ser o de MSE.

O contrato canônico atual também opera com MSE p99. Artefatos históricos e
execuções experimentais, contudo, podem guardar em `limiar` o limiar do escore
localizado. Os três gráficos deste módulo devem continuar na escala MSE.

`regenerar_graficos_autoencoder` passava o dicionário cru para os plots. A
figura resultante teria a linha de limiar bem acima da nuvem de pontos e
reportaria quase nenhum alarme — figura ERRADA, não figura desatualizada. O
caminho do pipeline escapava porque monta seu próprio `info_mse`; a
regeneração a partir do disco, não. A função não tinha nenhum chamador nem
teste, e foi assim que o defeito passou.

Estes testes rodam sem `torch`: por isso os gráficos vivem em
`src/ml/graficos_autoencoder.py`, separados de `src/ml/autoencoder.py`.
"""

from __future__ import annotations

import json

import numpy as np

from src.ml.graficos_autoencoder import (
    _info_em_escala_mse,
    regenerar_graficos_autoencoder,
    salvar_resumo_calibracao,
)


# Reproduz o formato real de resultados/autoencoder/limiar.json.
_LIMIAR_JSON = {
    "limiar": 7.826175715408156,          # artefato localizado histórico
    "limiar_mse": 2.5454330444335938,     # o que os gráficos precisam
    "limiar_mu3sigma": 2.2560129165649414,
    "k": 3.0,
    "percentil_limiar": 99.9,
    "fp_test_pct": 10.227272727272728,    # medido contra o operacional
}


def test_usa_o_limiar_de_mse_em_artefato_localizado_historico():
    info = _info_em_escala_mse(dict(_LIMIAR_JSON))
    assert info["limiar"] == _LIMIAR_JSON["limiar_mse"]
    assert info["limiar"] != _LIMIAR_JSON["limiar"]


def test_recalcula_o_falso_positivo_na_escala_certa():
    """O fp salvo é contra o operacional; na escala de MSE ele é outro."""
    erros = np.array([0.1, 0.5, 1.0, 3.0, 4.0, 9.0])   # 3 acima de 2,545
    info = _info_em_escala_mse(dict(_LIMIAR_JSON), erros)
    assert info["fp_test_pct"] == 50.0
    assert info["fp_test_pct"] != _LIMIAR_JSON["fp_test_pct"]


def test_nao_muda_artefato_anterior_ao_escore_localizado():
    """Sem `limiar_mse`, o campo `limiar` já era o de MSE — deixar quieto."""
    antigo = {"limiar": 2.5, "k": 3.0, "limiar_mu3sigma": 2.2}
    assert _info_em_escala_mse(dict(antigo))["limiar"] == 2.5


def test_nao_mexe_no_dicionario_recebido():
    original = dict(_LIMIAR_JSON)
    _info_em_escala_mse(original, np.array([1.0, 5.0]))
    assert original == _LIMIAR_JSON


def test_erros_vazios_preservam_o_fp_salvo():
    info = _info_em_escala_mse(dict(_LIMIAR_JSON), np.array([]))
    assert info["fp_test_pct"] == _LIMIAR_JSON["fp_test_pct"]


# ── regeneração ponta a ponta, sem torch e sem dataset ───────────────────────

def _artefatos(pasta, n=40):
    """diagnostico_autoencoder.npz + limiar.json mínimos, como no disco."""
    rng = np.random.default_rng(42)
    erros = rng.lognormal(mean=-1.5, sigma=0.8, size=n)
    np.savez(
        pasta / "diagnostico_autoencoder.npz",
        historico_treino=np.linspace(1.0, 0.1, 20),
        historico_calibracao=np.linspace(1.1, 0.15, 20),
        epoca_melhor=np.asarray([17], dtype=np.int32),
        erros_treino=erros, erros_calibracao=erros[:10], erros_teste=erros[:8],
        erros_todos=erros, tempos=np.arange(n, dtype=float),
        scores_operacionais_treino=erros * 3,
        scores_operacionais_calibracao=erros[:10] * 3,
        scores_operacionais_teste=erros[:8] * 3,
        scores_operacionais_todos=erros * 3,
        indices_teste=np.arange(n - 8, n),
    )
    (pasta / "limiar.json").write_text(json.dumps(_LIMIAR_JSON), encoding="utf-8")


def test_regenera_as_tres_figuras_sem_torch(tmp_path):
    _artefatos(tmp_path)
    assert regenerar_graficos_autoencoder(tmp_path) is True
    for nome in (
        "curva_treino.png",
        "distribuicao_erro.png",
        "erro_temporal.png",
        "calibracao_autoencoder.csv",
        "calibracao_autoencoder.md",
    ):
        arquivo = tmp_path / nome
        assert arquivo.is_file(), nome
        minimo = 5000 if arquivo.suffix == ".png" else 100
        assert arquivo.stat().st_size > minimo, f"{nome} saiu vazio"


def test_regenerar_recusa_sem_artefatos(tmp_path):
    assert regenerar_graficos_autoencoder(tmp_path) is False
    (tmp_path / "limiar.json").write_text("{}", encoding="utf-8")
    assert regenerar_graficos_autoencoder(tmp_path) is False


def test_figura_muda_quando_o_limiar_de_mse_muda(tmp_path):
    """Prova que o limiar plotado é mesmo o de MSE, e não o operacional.

    Alterar SÓ `limiar_mse` tem de mudar o pixel; alterar SÓ `limiar`
    (operacional) não pode mudar nada, porque essa escala não é plotada.
    """
    import hashlib

    def _hash_erro_temporal(info):
        _artefatos(tmp_path)
        (tmp_path / "limiar.json").write_text(json.dumps(info), encoding="utf-8")
        regenerar_graficos_autoencoder(tmp_path)
        return hashlib.sha256((tmp_path / "erro_temporal.png").read_bytes()).hexdigest()

    base = _hash_erro_temporal(dict(_LIMIAR_JSON))

    so_operacional = dict(_LIMIAR_JSON, limiar=99.0)
    assert _hash_erro_temporal(so_operacional) == base, (
        "o limiar operacional não deve influenciar um gráfico de MSE"
    )

    so_mse = dict(_LIMIAR_JSON, limiar_mse=0.05)
    assert _hash_erro_temporal(so_mse) != base, (
        "mudar o limiar de MSE tem de mudar a figura"
    )


def test_resumo_calibracao_documenta_ic95_e_duas_escalas(tmp_path):
    erros_treino = np.array([0.10, 0.20, 0.30, 3.00])
    erros_calib = np.array([0.10, 0.20, 0.30, 2.60])
    erros_teste = np.array([0.10, 0.20, 0.30, 4.00])
    scores_teste = np.array([1.0, 2.0, 3.0, 9.0])
    info = dict(
        _LIMIAR_JSON,
        score_method="localizado",
        score_threshold=7.826175715408156,
        threshold_effective_percentile=99.9,
        mse_p99=_LIMIAR_JSON["limiar_mse"],
    )

    csv_path, md_path = salvar_resumo_calibracao(
        erros_treino, erros_calib, erros_teste, info, tmp_path,
        scores_treino=np.array([1.0, 2.0, 3.0, 4.0]),
        scores_calibracao=np.array([1.0, 2.0, 3.0, 8.0]),
        scores_teste=scores_teste,
    )

    md = md_path.read_text(encoding="utf-8")
    csv = csv_path.read_text(encoding="utf-8")
    assert "IC95% de Wilson" in md
    assert "Referência MSE p99" in md
    assert "limiar operacional" in md
    assert "localizado / percentil efetivo 99.9" in md
    assert "score_operacional_rate_pct" in csv
    assert "teste_isolado" in csv
