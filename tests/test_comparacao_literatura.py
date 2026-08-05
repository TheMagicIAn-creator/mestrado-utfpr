"""
Comparacao do metodo proposto com a literatura ativa.

O comparativo vigente usa apenas Ibrahim/AE-LSTM como concorrente executavel.
"""

from __future__ import annotations

import pytest

from src.ml import comparacao_literatura as C


LINHAS_FAKE = [
    {"metodo": "Autoencoder (metodo proposto)", "papel": "proposto",
     "fonte": "pipeline principal (este trabalho)", "auc": 0.81, "evidencia": "E1"},
    {"metodo": "AE-LSTM", "papel": "concorrente",
     "fonte": "Ibrahim et al. (2022)", "auc": 0.63, "evidencia": "E1"},
]


def test_sem_modelo_avisa_rodar_pipeline_sem_treinar(tmp_path, monkeypatch):
    """Sem artefatos do Autoencoder: fail-fast com instrucao, nunca treina."""
    monkeypatch.setattr(C, "PASTA_AE", tmp_path / "vazio")
    out = C.comparar_com_literatura()
    assert out["ok"] is False
    assert "rode o pipeline" in out["mensagem"].lower()
    assert "nunca treina" in out["mensagem"].lower()


def test_tabela_ranqueada_com_destaque_do_metodo():
    md = C._tabela_md(LINHAS_FAKE)
    linhas = [l for l in md.strip().splitlines() if l.startswith("|")]
    assert "Autoencoder" in linhas[2]
    assert "**" in linhas[2]
    assert "0.810" in linhas[2]
    assert "AE-LSTM" in linhas[-1]


def test_grafico_gera_png_com_tamanho_canonico(tmp_path):
    pytest.importorskip("matplotlib")
    destino = tmp_path / "comparacao.png"
    C._grafico(LINHAS_FAKE, "Banco comum E1 (teste)", destino)
    assert destino.exists()
    from PIL import Image

    with Image.open(destino) as im:
        assert 1400 <= im.width <= 1900


def test_experimentos_ausentes_geram_aviso(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PASTA_EXPERIMENTOS", tmp_path / "sem_nada")
    linhas, avisos = C._linhas_experimentos(n_te_banco=100)
    assert linhas == []
    assert len(avisos) == 1
    assert "ibrahim" in avisos[0]
    assert "rode" in avisos[0].lower()
