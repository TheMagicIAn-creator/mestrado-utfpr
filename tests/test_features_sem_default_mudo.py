"""
Feature que falta tem de ESTOURAR, nunca virar 0,0.

POR QUE ESTE TESTE EXISTE
=========================
O comparativo denso × Ibrahim publicou AUC de um autoencoder alimentado com um
vetor de 24 zeros. A causa foi uma linha:

    [extrair_janela(j).get(c, 0.0) for c in colunas_feat]

Depois da migração para o GPVS, `colunas_feat` vinha do checkpoint (24 nomes do
GPVS) e o extrator importado era o do Stender (108 nomes OUTROS). Nenhum nome
batia, todos caíram no default, e o modelo reconstruiu o nada — sem erro de
shape, sem aviso, com número plausível na saída.

Aquilo foi corrigido nos macro-códigos. A auditoria seguinte encontrou o MESMO
padrão em três pontos da **cadeia canônica**:

    injecao_falhas.py    etapa 3 — alimenta a SMD
    validacao.py         etapa 4 — alimenta as taxas de detecção E2
    varredura_a_det.py   etapa 5 — decide quais trajetórias entram no Weibull

Pior: `varredura_a_det` contradizia a si mesmo — `.get(c, 0.0)` no filtro de
elegibilidade e `[c]` estrito na varredura, no mesmo módulo e com o mesmo
`colunas_feat`. O filtro aceitaria em silêncio a janela zerada que a varredura
logo adiante rejeitaria.

Estes são os números que vão para a dissertação. O default mudo não pode voltar.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.gpvs import COLUNAS_PRIMARIAS
from src.ml.gpvs_principal import JANELA, extrair_janela, vetor_de_features

RAIZ = Path(__file__).resolve().parents[1]

# Todo módulo que converte janela em vetor na ordem do modelo.
MODULOS_DA_CADEIA = (
    "src/ml/injecao_falhas.py",
    "src/ml/validacao.py",
    "src/ml/varredura_a_det.py",
    "src/ml/macro_proposto.py",
    "src/ml/macro_ibrahim.py",
    "src/ml/gpvs_principal.py",
)


def _janela() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = np.arange(JANELA) / JANELA
    return pd.DataFrame({
        col: np.sin(2 * np.pi * t + k) + 0.01 * rng.normal(size=JANELA) + 10.0
        for k, col in enumerate(COLUNAS_PRIMARIAS)
    })


# ── o comportamento ────────────────────────────────────────────────────────

def test_feature_desconhecida_estoura_em_vez_de_virar_zero():
    """O caso exato que produziu o vetor de zeros publicado."""
    with pytest.raises(KeyError) as erro:
        vetor_de_features(_janela(), ["i_a_rms", "i_a_harm_5", "f0_estimado"])

    mensagem = str(erro.value)
    assert "i_a_rms" in mensagem, "a mensagem tem de dizer QUAL feature falta"
    assert "features_gpvs" in mensagem, (
        "a mensagem tem de dizer como sair do erro, não só que ele ocorreu"
    )


def test_o_vetor_sai_na_ordem_pedida_e_nao_na_do_extrator():
    """Ordem trocada é o mesmo desastre silencioso, com números não nulos."""
    feats = extrair_janela(_janela())
    colunas = list(feats)[:6][::-1]          # invertida de propósito
    vetor = vetor_de_features(_janela(), colunas)

    assert vetor.shape == (6,)
    assert vetor.dtype == np.float32
    np.testing.assert_allclose(
        vetor, np.array([feats[c] for c in colunas], dtype=np.float32)
    )


def test_subconjunto_das_features_continua_valido():
    """Nem toda coluna do extrator precisa entrar — só não pode FALTAR."""
    feats = extrair_janela(_janela())
    vetor = vetor_de_features(_janela(), list(feats)[:3])
    assert vetor.shape == (3,)


# ── a guarda estrutural ────────────────────────────────────────────────────

class _CacaDefaultMudo(ast.NodeVisitor):
    """Acha `<algo>.get(<nome>, <default>)` dentro de uma compreensão de lista.

    É a assinatura exata do defeito: iterar sobre nomes de feature e engolir os
    que não existem. Um `.get` de duas casas fora de compreensão (leitura de
    JSON de metadados, por exemplo) não é o alvo.
    """

    def __init__(self) -> None:
        self.achados: list[int] = []
        self._em_comprehension = 0

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._em_comprehension += 1
        self.generic_visit(node)
        self._em_comprehension -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if (
            self._em_comprehension
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
        ):
            self.achados.append(node.lineno)
        self.generic_visit(node)


@pytest.mark.parametrize("caminho", MODULOS_DA_CADEIA)
def test_nenhum_modulo_da_cadeia_engole_feature_ausente(caminho):
    arquivo = RAIZ / caminho
    caca = _CacaDefaultMudo()
    caca.visit(ast.parse(arquivo.read_text(encoding="utf-8")))
    assert not caca.achados, (
        f"{caminho} voltou a usar `.get(nome, default)` dentro de compreensão "
        f"nas linhas {caca.achados}. Feature ausente tem de estourar: use "
        f"gpvs_principal.vetor_de_features."
    )


@pytest.mark.parametrize("caminho", MODULOS_DA_CADEIA)
def test_a_conversao_janela_para_vetor_tem_fonte_unica(caminho):
    """Cada cópia da conversão é uma chance de a próxima divergir.

    `varredura_a_det` chegou a ter as duas versões — permissiva e estrita — no
    mesmo arquivo.
    """
    texto = (RAIZ / caminho).read_text(encoding="utf-8")
    if "extrair_janela(" not in texto:
        pytest.skip("módulo não converte janela em vetor")
    if caminho.endswith("gpvs_principal.py"):
        return   # é a fonte
    assert "vetor_de_features" in texto or "det[\"colunas\"]" in texto, (
        f"{caminho} extrai features sem passar pela fonte única"
    )
