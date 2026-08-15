"""
As quatro curvas saindo POR MODELO — e não por componente.

POR QUE ESTE TESTE EXISTE
=========================
`weibull_por_modelo` deu a CAPACIDADE (varredura com scorer plugável);
`macro_weibull` é a orquestração que a torna um artefato. O que pode quebrar
aqui é integração, não matemática:

  - o bloco por modelo tem de encaixar nas funções de `graficos_rul` sem
    adaptador — elas esperam ``(a_dets, eventos, params, pasta)``;
  - cada modelo tem de escrever na SUA pasta, senão o segundo sobrescreve as
    figuras do primeiro em silêncio (os nomes de arquivo são iguais de
    propósito);
  - a tabela tem de dizer quando a Weibull 2P NÃO foi adotada, porque é o
    estado corrente das três falhas.

Rodam sem torch, sem dataset e sem checkpoint: o scorer é uma função de mentira
e a saída vai para `tmp_path`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ml import macro_weibull
from src.ml.gpvs_principal import JANELA
from src.ml.gpvs import COLUNAS_PRIMARIAS
from src.ml.injecao_falhas import FALHAS
from src.ml.weibull_por_modelo import (
    comparar_detectabilidade, detectabilidade_do_modelo,
)


def _janela(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(JANELA) / JANELA
    dados = {
        col: np.sin(2 * np.pi * t + k) + 0.01 * rng.normal(size=JANELA) + 10.0
        for k, col in enumerate(COLUNAS_PRIMARIAS)
    }
    df = pd.DataFrame(dados)
    df.attrs["ensaio"] = "F0L" if seed % 2 == 0 else "F0M"
    return df


def _scorer(ganho: float):
    def scorer(janelas):
        return np.asarray(
            [ganho * float(np.std(j[COLUNAS_PRIMARIAS[0]].to_numpy()))
             for j in janelas],
            dtype=float,
        )
    return scorer


@pytest.fixture
def blocos():
    janelas = [_janela(i) for i in range(6)]
    denso = detectabilidade_do_modelo(
        "Proposto (AE denso + MSE p99)", _scorer(8.0), 3.0, janelas, n_steps=21)
    denso["cor"] = "#2a78d6"
    lstm = detectabilidade_do_modelo(
        "Ibrahim 2022 (AE-LSTM temporal)", _scorer(2.0), 3.0, janelas, n_steps=21)
    lstm["cor"] = "#1baf7a"
    return [denso, lstm]


# ── pastas: um modelo não pode apagar as figuras do outro ──────────────────

def test_cada_modelo_tem_pasta_propria():
    p = macro_weibull._pasta_do_modelo("Proposto (AE denso + MSE p99)")
    i = macro_weibull._pasta_do_modelo("Ibrahim 2022 (AE-LSTM temporal)")
    assert p != i
    assert p.name == "proposto" and i.name == "ibrahim"


def test_modelo_desconhecido_ainda_ganha_pasta_estavel():
    """Sem isto, um terceiro modelo cairia na raiz e colidiria com os dois."""
    a = macro_weibull._pasta_do_modelo("PCA linear (ablação)")
    b = macro_weibull._pasta_do_modelo("PCA linear (ablação)")
    assert a == b
    assert a.parent == macro_weibull.PASTA_SAIDA
    assert a != macro_weibull.PASTA_SAIDA


# ── o encaixe com graficos_rul ─────────────────────────────────────────────

def test_conversao_para_o_formato_dos_plots(blocos):
    a_dets, eventos, params = macro_weibull._dicts_para_plot(blocos[0])
    ids = {f["id"] for f in FALHAS}
    assert set(a_dets) == set(eventos) == set(params) == ids
    for fid in ids:
        assert a_dets[fid].dtype == float
        assert eventos[fid].dtype == bool
        assert len(a_dets[fid]) == len(eventos[fid])
        assert "fit_converged" in params[fid]


def test_desenhar_modelo_emite_as_quatro_figuras(blocos, tmp_path, monkeypatch):
    monkeypatch.setattr(macro_weibull, "PASTA_SAIDA", tmp_path)
    figuras = macro_weibull.desenhar_modelo(blocos[0])

    assert set(figuras) == {
        "papel_weibull", "confiabilidade", "funcoes_distribuicao", "intensidade",
    }
    for chave, caminho in figuras.items():
        assert caminho.exists(), f"{chave} não foi escrita"
        assert caminho.stat().st_size > 0
        assert caminho.parent == tmp_path / "proposto"


def test_dois_modelos_nao_sobrescrevem_um_ao_outro(blocos, tmp_path, monkeypatch):
    """Os nomes de arquivo são iguais nos dois — a pasta é que separa."""
    monkeypatch.setattr(macro_weibull, "PASTA_SAIDA", tmp_path)
    a = macro_weibull.desenhar_modelo(blocos[0])
    b = macro_weibull.desenhar_modelo(blocos[1])

    assert a["confiabilidade"].name == b["confiabilidade"].name
    assert a["confiabilidade"] != b["confiabilidade"]
    assert all(caminho.exists() for caminho in (*a.values(), *b.values()))


def test_sobreposicao_compara_os_modelos(blocos, tmp_path, monkeypatch):
    monkeypatch.setattr(macro_weibull, "PASTA_SAIDA", tmp_path)
    caminho = macro_weibull.plotar_comparacao_confiabilidade(blocos)
    assert caminho.exists() and caminho.stat().st_size > 0
    assert caminho.name == "comparacao_confiabilidade.png"


# ── a tabela tem de carregar a ressalva ────────────────────────────────────

def test_tabela_declara_quando_a_2p_nao_foi_adotada(blocos):
    comp = comparar_detectabilidade(blocos)
    for linha in comp["linhas"]:
        assert "resumo_parametrico_recomendado" in linha
    md = macro_weibull.tabela_markdown(comp)
    assert "2P adotada" in md
    assert "POD_mon@a=1" in md
    # 6 linhas de dado (2 modelos × 3 falhas) + cabeçalho + separador
    assert len(md.splitlines()) == 8


def test_tabela_nao_quebra_com_ajuste_nao_convergido(blocos):
    comp = comparar_detectabilidade(blocos)
    for linha in comp["linhas"]:
        linha["a10"] = None
        linha["a_det_mediana"] = float("nan")
    md = macro_weibull.tabela_markdown(comp)
    assert "—" in md, "marcador de ausência tem de aparecer, não 'None'"
    assert "None" not in md
    assert "nan" not in md


def test_salvar_saidas_grava_json_md_e_csv(blocos, tmp_path, monkeypatch):
    monkeypatch.setattr(macro_weibull, "PASTA_SAIDA", tmp_path)
    comp = comparar_detectabilidade(blocos)
    saidas = macro_weibull.salvar_saidas(blocos, comp)

    assert set(saidas) == {"json", "tabela_md", "tabela_csv"}
    for caminho in saidas.values():
        assert caminho.exists()

    dados = json.loads(saidas["json"].read_text(encoding="utf-8"))
    assert len(dados["modelos"]) == 2
    assert len(dados["comparacao"]["linhas"]) == 6
    for bloco in dados["modelos"]:
        assert bloco["evidence_level"] == "E2"
        assert bloco["eixo_nao_e_tempo"] is True

    csv_txt = saidas["tabela_csv"].read_text(encoding="utf-8")
    assert "resumo_parametrico_recomendado" in csv_txt.splitlines()[0]


# ── o limiar é POR MODELO, e isso é obrigatório ────────────────────────────

def test_limiar_e_calibrado_por_modelo():
    """Escores de detectores diferentes vivem em escalas diferentes.

    Um limiar único faria a comparação medir a razão entre as escalas, não a
    sensibilidade dos modelos.
    """
    from src.ml.macro_comum import calibrar_limiar

    janelas = [_janela(i) for i in range(40)]
    lim_a, _ = calibrar_limiar(_scorer(8.0), janelas)
    lim_b, _ = calibrar_limiar(_scorer(2.0), janelas)

    assert lim_a != lim_b
    assert lim_a == pytest.approx(lim_b * 4.0, rel=1e-6)


def test_avaliar_deteccao_usa_a_mesma_calibracao(monkeypatch):
    """Uma única definição de limiar entre a rota AUC e a rota Weibull."""
    from src.ml import macro_comum

    chamadas = []
    original = macro_comum.calibrar_limiar

    def espiao(scorer, janelas):
        chamadas.append(len(janelas))
        return original(scorer, janelas)

    monkeypatch.setattr(macro_comum, "calibrar_limiar", espiao)
    macro_comum.avaliar_deteccao(
        "x", "#000", _scorer(1.0), [_janela(i) for i in range(12)],
        [_janela(i) for i in range(12, 18)],
    )
    assert chamadas == [12]
