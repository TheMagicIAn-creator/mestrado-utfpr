"""Saída uniforme dos macro-códigos (tabela enxuta + gráfico comparável)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.ml.macro_comum import plotar_deteccao_severidade, salvar_saidas, tabela_enxuta


def _resultado(nome: str, auc_ct: float, det_ct: float, auc_ig: float,
               det_ig: float, fp: float = 1.1) -> dict:
    sevs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    def por_sev(det_final):
        # rampa simples até a taxa final (só para o gráfico ter forma)
        return {s: {"taxa": det_final * (s ** 2), "ci_low": 0.0, "ci_high": 1.0,
                    "erro_mediano": 1.0, "atinge_smd": det_final * (s ** 2) >= 0.95}
                for s in sevs}

    return {
        "nome": nome, "cor": "#2a78d6", "limiar": 2.5, "percentil": 99.0,
        "fp_pct": fp, "severidades": sevs,
        "falhas": {
            "contator_ac": {"nome": "Contator AC", "npr": 315, "cor": "#2a78d6",
                            "auc": auc_ct, "tpr_fpr10": det_ct, "smd_fpr10": 0.5, "por_sev": por_sev(det_ct)},
            "igbt": {"nome": "IGBT", "npr": 90, "cor": "#1baf7a",
                     "auc": auc_ig, "tpr_fpr10": det_ig, "smd_fpr10": 0.7, "por_sev": por_sev(det_ig)},
        },
    }


def test_tabela_enxuta_tem_5_colunas_e_uma_linha_por_metodo_falha():
    tab = tabela_enxuta([_resultado("Proposto", 0.99, 1.0, 0.94, 0.86),
                         _resultado("Ibrahim", 0.91, 0.8, 0.72, 0.44)])
    linhas = [l for l in tab.splitlines() if l.startswith("|")]
    assert linhas[0].count("|") == 6          # 5 colunas → 6 pipes
    assert len(linhas) == 2 + 4               # cabeçalho + separador + 2×2
    assert "Proposto" in tab and "Ibrahim" in tab
    assert "TPR @FPR=10%, sev=1.0" in tab


def test_salvar_saidas_gera_md_csv_json_e_png(tmp_path):
    res = [_resultado("Proposto", 0.99, 1.0, 0.94, 0.86),
           _resultado("Ibrahim", 0.91, 0.8, 0.72, 0.44)]
    saidas = salvar_saidas(res, tmp_path, prefixo="cmp")
    for chave in ("tabela_md", "tabela_csv", "resultado_json", "grafico"):
        assert saidas[chave].exists(), f"{chave} não foi gerado"
    assert (tmp_path / "cmp_resultado.json").exists()
    # CSV enxuto: 8 colunas de dados, 4 linhas (2 métodos × 2 falhas)
    with (tmp_path / "cmp_tabela.csv").open(encoding="utf-8") as fh:
        linhas = list(csv.reader(fh))
    assert linhas[0] == [
        "metodo", "falha", "npr", "auc", "smd_fpr10",
        "tpr_fpr10_sev1", "deteccao_limiar_sev1", "fp_pct",
    ]
    assert len(linhas) == 5
    # JSON é recarregável (auditoria)
    dados = json.loads((tmp_path / "cmp_resultado.json").read_text(encoding="utf-8"))
    assert len(dados) == 2 and dados[0]["nome"] == "Proposto"

    # Roundtrip real: no JSON, chaves float de `por_sev` tornam-se strings.
    # O artefato recarregado precisa continuar apto a regerar tabela e figura.
    saidas_roundtrip = salvar_saidas(dados, tmp_path / "roundtrip", prefixo="cmp")
    assert all(caminho.exists() for caminho in saidas_roundtrip.values())


def test_grafico_um_painel_por_falha(tmp_path):
    res = [_resultado("Proposto", 0.99, 1.0, 0.94, 0.86)]
    png = plotar_deteccao_severidade(res, tmp_path, prefixo="g")
    assert png.exists() and png.stat().st_size > 5000   # figura real, não vazia


def test_tabela_macro_publicada_deriva_do_json_versionado():
    pasta = Path(__file__).resolve().parents[1] / "resultados" / "macro"
    dados = json.loads((pasta / "comparacao_resultado.json").read_text(encoding="utf-8"))
    tabela = (pasta / "comparacao_tabela.md").read_text(encoding="utf-8")

    assert tabela == tabela_enxuta(dados)
    assert "TPR @FPR=10%, sev=1.0" in tabela
