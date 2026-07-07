"""
Comparação do método proposto com a literatura (src/ml/comparacao_literatura).

CI-leve: nada aqui carrega torch/parquet — testa o fail-fast sem modelo,
os construtores puros de tabela e o gráfico (matplotlib está no CI).
"""

from __future__ import annotations

import pytest

from src.ml import comparacao_literatura as C

LINHAS_FAKE = [
    {"metodo": "Autoencoder (método proposto)", "papel": "proposto",
     "fonte": "pipeline principal (este trabalho)", "auc": 0.81, "evidencia": "E1"},
    {"metodo": "Isolation Forest", "papel": "concorrente",
     "fonte": "Ibrahim et al. (2022)", "auc": 0.63, "evidencia": "E1"},
    {"metodo": "Z-score (estatístico)", "papel": "baseline",
     "fonte": "Francisti et al. (2025)", "auc": 0.55, "evidencia": "E1"},
]


def test_sem_modelo_avisa_rodar_pipeline_sem_treinar(tmp_path, monkeypatch):
    """Sem artefatos do Autoencoder: fail-fast com instrução, nunca treina."""
    monkeypatch.setattr(C, "PASTA_AE", tmp_path / "vazio")
    out = C.comparar_com_literatura()
    assert out["ok"] is False
    assert "rode o pipeline" in out["mensagem"].lower()
    assert "nunca treina" in out["mensagem"].lower()


def test_tabela_ranqueada_com_destaque_do_metodo():
    md = C._tabela_md(LINHAS_FAKE)
    linhas = [l for l in md.strip().splitlines() if l.startswith("|")]
    # ranqueada por AUC desc: proposto (0.81) vem primeiro após o cabeçalho
    assert "Autoencoder" in linhas[2]
    assert "**" in linhas[2]                  # método proposto em negrito
    assert "0.810" in linhas[2]               # política de 3 casas (fmt_metrica)
    assert linhas[2].index("0.810") > 0
    assert "Z-score" in linhas[-1]            # menor AUC por último


def test_grafico_gera_png_com_tamanho_canonico(tmp_path):
    pytest.importorskip("matplotlib")
    destino = tmp_path / "comparacao.png"
    C._grafico(LINHAS_FAKE, "Banco comum E1 (teste)", destino)
    assert destino.exists()
    from PIL import Image

    with Image.open(destino) as im:
        # 12 pol de largura a 150 dpi, com corte tight → entre 1400 e 1900 px
        assert 1400 <= im.width <= 1900


def test_experimentos_ausentes_geram_aviso(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PASTA_EXPERIMENTOS", tmp_path / "sem_nada")
    linhas, avisos = C._linhas_experimentos(n_te_banco=100)
    assert linhas == []
    assert len(avisos) == 2                   # francisti e ibrahim
    assert all("rode" in a.lower() for a in avisos)
