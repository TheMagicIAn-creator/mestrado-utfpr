"""
Testes da retroalimentação da FMECA — conversão POD_mon → D_mon → NPR projetado.

Estes testes rodam SEM torch e SEM o dataset: a conversão é aritmética pura
sobre um artefato JSON já existente. É de propósito — a ponte entre o detector
e a FMECA é o resultado que a banca mais vai cobrar, e ela precisa ser
verificável na CI, não só na máquina do pesquisador.
"""

from __future__ import annotations

import json

import pytest

from src.ml.retroalimentacao_fmeca import (
    BORDAS_D,
    SEVERIDADE_REFERENCIA,
    carregar_pod_mon,
    d_projetado,
    formatar_markdown,
    indice_d,
    tabela_retroalimentacao,
)

# ------------------------------------------------------------------
# A escala da Tab. 4.8
# ------------------------------------------------------------------

def test_escala_cobre_todo_o_intervalo_sem_buraco():
    """Qualquer fração em [0, 1] tem índice. Sem faixa, sem exceção."""
    for i in range(1001):
        d = indice_d(i / 1000.0)
        assert 1 <= d <= 10


def test_extremos_batem_com_o_que_fmeca_md_registra():
    """Os dois extremos são os ÚNICOS valores documentados do TCC."""
    assert indice_d(0.0) == 1        # 0% de não detectar
    assert indice_d(0.05) == 1       # borda superior da faixa 1
    assert indice_d(0.86) == 10      # início da faixa 10
    assert indice_d(1.0) == 10       # 100% de não detectar


def test_escala_e_monotona():
    """Detectar pior nunca pode devolver índice melhor."""
    anterior = 0
    for i in range(101):
        d = indice_d(i / 100.0)
        assert d >= anterior, f"quebra de monotonia em {i}%"
        anterior = d


def test_ponto_flutuante_nao_empurra_para_a_faixa_seguinte():
    """`1 - 0.85` vale 0.15000000000000002 e cairia em D=3 sem arredondamento.

    Este é o caso REAL do IGBT em severidade máxima (recall 0,85).
    """
    assert (1.0 - 0.85) > 0.15          # a armadilha existe
    assert indice_d(1.0 - 0.85) == 2    # e está tratada


def test_valores_fracionarios_entre_faixas_inteiras():
    """A Tab. 4.8 é escrita em % inteiros; 5,5% não caía em faixa nenhuma."""
    assert indice_d(0.055) == 2
    assert indice_d(0.155) == 3


def test_fracao_fora_do_intervalo_e_recusada():
    for invalido in (-0.01, 1.01, 2.0):
        with pytest.raises(ValueError, match="fora de"):
            indice_d(invalido)


def test_bordas_sao_crescentes_e_terminam_em_100():
    indices = [d for d, _ in BORDAS_D]
    bordas = [b for _, b in BORDAS_D]
    assert indices == list(range(1, 11))
    assert bordas == sorted(bordas)
    assert bordas[-1] == 100.0


# ------------------------------------------------------------------
# A emenda min
# ------------------------------------------------------------------

def test_min_nunca_piora_o_indice():
    """O monitoramento é ADICIONAL: não pode tornar falha mais difícil de ver."""
    for d_campo in range(1, 11):
        for d_mon in range(1, 11):
            assert d_projetado(d_campo, d_mon) <= d_campo


def test_min_absorve_o_caso_falha_nunca_detectada():
    """POD_mon = 0 → D_mon = 10 → mantém o D original, sem exceção no código."""
    assert indice_d(1.0) == 10
    assert d_projetado(3, 10) == 3


def test_npr_projetado_nunca_sobe():
    """Consequência direta da emenda — é a razão de ela existir."""
    for d_campo in range(1, 11):
        for d_mon in range(1, 11):
            s, o = 5, 7
            assert s * o * d_projetado(d_campo, d_mon) <= s * o * d_campo


# ------------------------------------------------------------------
# Leitura do artefato de validação
# ------------------------------------------------------------------

def _relatorio_sintetico(tmp_path, recalls):
    """Monta um validacao_report.json mínimo com os recalls pedidos."""
    dados = {"__meta__": {"limiar_operacional": 7.83, "threshold_method": "p99"}}
    for fid, por_sev in recalls.items():
        for sev, rec in por_sev.items():
            dados[f"{fid}_sev{sev}"] = {
                "recall": rec, "recall_ci_low": max(0.0, rec - 0.1),
                "recall_ci_high": min(1.0, rec + 0.1),
            }
    arq = tmp_path / "validacao_report.json"
    arq.write_text(json.dumps(dados), encoding="utf-8")
    return arq


def test_carregar_separa_falha_de_severidade(tmp_path):
    arq = _relatorio_sintetico(tmp_path, {
        "contator_ac": {"0.3": 0.6, "1.0": 1.0},
        "fusivel_ac": {"1.0": 0.5},
    })
    lido = carregar_pod_mon(arq)
    assert set(lido["curvas"]) == {"contator_ac", "fusivel_ac"}
    assert lido["curvas"]["contator_ac"][1.0]["pod"] == 1.0
    assert lido["meta"]["threshold_method"] == "p99"


def test_carregar_ignora_meta_e_chaves_sem_severidade(tmp_path):
    arq = _relatorio_sintetico(tmp_path, {"igbt": {"1.0": 0.85}})
    dados = json.loads(arq.read_text(encoding="utf-8"))
    dados["resumo_geral"] = {"recall": 0.9}      # chave sem "_sev"
    arq.write_text(json.dumps(dados), encoding="utf-8")
    lido = carregar_pod_mon(arq)
    assert set(lido["curvas"]) == {"igbt"}


def test_severidade_ausente_falha_com_mensagem_util(tmp_path):
    arq = _relatorio_sintetico(tmp_path, {
        "contator_ac": {"0.3": 0.6}, "igbt": {"0.3": 0.2},
        "fusivel_ac": {"0.3": 0.1},
    })
    with pytest.raises(KeyError, match="não consta"):
        tabela_retroalimentacao(arq, severidade=1.0)


# ------------------------------------------------------------------
# Tabela completa
# ------------------------------------------------------------------

def _tabela_com_recall_maximo(tmp_path, recalls_sev_max):
    por_falha = {fid: {"1.0": r} for fid, r in recalls_sev_max.items()}
    return tabela_retroalimentacao(_relatorio_sintetico(tmp_path, por_falha))


def test_tabela_carrega_indices_da_fonte_unica(tmp_path):
    """S/O/D vêm de injecao_falhas.FALHAS, que espelha docs/fmeca.md."""
    res = _tabela_com_recall_maximo(tmp_path, {
        "contator_ac": 1.0, "igbt": 0.85, "fusivel_ac": 1.0,
    })
    por_id = {r["id"]: r for r in res["linhas"]}
    assert por_id["contator_ac"]["npr_oficial"] == 315
    assert por_id["igbt"]["npr_oficial"] == 90
    assert por_id["fusivel_ac"]["npr_oficial"] == 30


def test_deteccao_perfeita_leva_d_ao_minimo(tmp_path):
    res = _tabela_com_recall_maximo(tmp_path, {
        "contator_ac": 1.0, "igbt": 1.0, "fusivel_ac": 1.0,
    })
    for linha in res["linhas"]:
        assert linha["d_mon"] == 1
        assert linha["d_projetado"] == 1


def test_deteccao_nula_preserva_a_fmeca_original(tmp_path):
    """Detector cego não pode mudar nada — nem para melhor, nem para pior."""
    res = _tabela_com_recall_maximo(tmp_path, {
        "contator_ac": 0.0, "igbt": 0.0, "fusivel_ac": 0.0,
    })
    for linha in res["linhas"]:
        assert linha["d_projetado"] == linha["d_campo"]
        assert linha["npr_projetado"] == linha["npr_oficial"]
    assert res["ordem_inverte"] is False


def test_resultado_carrega_o_nivel_de_evidencia(tmp_path):
    res = _tabela_com_recall_maximo(tmp_path, {
        "contator_ac": 1.0, "igbt": 0.85, "fusivel_ac": 1.0,
    })
    assert res["evidence_level"] == "E2"
    assert "E3" in res["evidence_note"]          # diz o que NÃO é
    assert res["severidade_referencia"] == SEVERIDADE_REFERENCIA
    assert "min(" in res["regra"]


def test_deteccao_uniformemente_perfeita_PRESERVA_a_ordem(tmp_path):
    """Retroalimentar não inverte a ordem por si só.

    Com POD_mon = 1 nos três, todo D_proj vai a 1 e o NPR vira S×O puro:
    35 > 30 > 15 — a mesma ordem. Este teste existe para impedir a leitura
    errada de que "recalcular o NPR inverte a prioridade".
    """
    res = _tabela_com_recall_maximo(tmp_path, {
        "contator_ac": 1.0, "igbt": 1.0, "fusivel_ac": 1.0,
    })
    por_id = {r["id"]: r["npr_projetado"] for r in res["linhas"]}
    assert (por_id["contator_ac"], por_id["igbt"], por_id["fusivel_ac"]) == (35, 30, 15)
    assert res["ordem_inverte"] is False


def test_inversao_vem_da_falha_QUE_O_DETECTOR_TRATA_PIOR(tmp_path):
    """O caso REAL, com os recalls vigentes em severidade máxima.

    Contator e Fusível em 1,000; IGBT em 0,850 → D_mon = 2, e `min(3, 2) = 2`
    segura o NPR do IGBT em 60 enquanto o do Contator cai de 315 para 35.

    A inversão não é mecânica: ela é causada justamente pelo componente que o
    detector enxerga PIOR. Trocar 0,850 por 1,0 preserva a ordem (teste acima).
    """
    res = _tabela_com_recall_maximo(tmp_path, {
        "contator_ac": 1.0, "igbt": 0.85, "fusivel_ac": 1.0,
    })
    por_id = {r["id"]: r for r in res["linhas"]}
    assert por_id["igbt"]["d_mon"] == 2
    assert por_id["igbt"]["npr_projetado"] == 60
    assert por_id["contator_ac"]["npr_projetado"] == 35
    assert res["ordem_oficial"] == ["contator_ac", "igbt", "fusivel_ac"]
    assert res["ordem_projetada"][0] == "igbt"
    assert res["ordem_inverte"] is True
    assert "inverte" in formatar_markdown(res)


def test_markdown_traz_as_ressalvas_junto_dos_numeros(tmp_path):
    """Tabela sem ressalva vira número solto colado na dissertação."""
    md = formatar_markdown(_tabela_com_recall_maximo(tmp_path, {
        "contator_ac": 1.0, "igbt": 0.85, "fusivel_ac": 1.0,
    }))
    assert "E2" in md
    assert "Tab. 4.8" in md
    assert "NPR de campo" in md


# ------------------------------------------------------------------
# Ponto de operação: percentil EFETIVO, não o rótulo de método
# ------------------------------------------------------------------

def test_percentil_efetivo_vem_do_limiar_json(tmp_path):
    """`threshold_method` diz "p99"; a auto-calibração escolheu 99,9.

    Imprimir "(p99)" numa tabela destinada à dissertação seria enganoso — são
    pontos de operação diferentes. A divergência está documentada, e a saída
    tem de mostrar o valor que de fato vigora.
    """
    arq = _relatorio_sintetico(tmp_path, {
        "contator_ac": {"1.0": 1.0}, "igbt": {"1.0": 0.85},
        "fusivel_ac": {"1.0": 1.0},
    })
    (tmp_path / "limiar.json").write_text(
        json.dumps({"percentil_limiar": 99.9, "percentil_auto": True}),
        encoding="utf-8")
    res = tabela_retroalimentacao(arq)
    assert res["percentil_efetivo"] == 99.9
    md = formatar_markdown(res)
    assert "percentil 99.9" in md
    assert "(p99)" not in md


def test_sem_limiar_json_cai_no_rotulo_sem_inventar_precisao(tmp_path):
    arq = _relatorio_sintetico(tmp_path, {
        "contator_ac": {"1.0": 1.0}, "igbt": {"1.0": 0.85},
        "fusivel_ac": {"1.0": 1.0},
    })
    res = tabela_retroalimentacao(arq)
    assert res["percentil_efetivo"] is None
    assert "método p99" in formatar_markdown(res)


def test_limiar_json_corrompido_nao_derruba_a_tabela(tmp_path):
    arq = _relatorio_sintetico(tmp_path, {
        "contator_ac": {"1.0": 1.0}, "igbt": {"1.0": 0.85},
        "fusivel_ac": {"1.0": 1.0},
    })
    (tmp_path / "limiar.json").write_text("{ nao e json", encoding="utf-8")
    assert tabela_retroalimentacao(arq)["percentil_efetivo"] is None
