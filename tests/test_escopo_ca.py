"""
O vetor de features é do LADO CA. Este teste impede que o CC volte por descuido.

POR QUE ESTE TESTE EXISTE
=========================
`tensao_dc_media` — a média do barramento CC — ficou no vetor de features por
toda a história do projeto, num pipeline cujo escopo declarado é o lado CA do
inversor (docs/fmeca.md: Contator AC, IGBT, Fusível AC). A auditoria
(docs/auditoria_total_src.md §4) mostrou que não era só uma impropriedade de
escopo:

- **inerte**: nenhuma injeção, E1 ou E2, jamais perturbou `u_dc_k`;
- **com alavanca desproporcional**: σ = 0,0883 contra mediana de 0,1998 nas 109
  features. Como o escore operacional é a média dos 5 maiores z = (|r| − μ)/σ,
  a alavanca era 2,26× a do canal mediano. Parte do limiar que define SMD,
  POD_mon, D_mon, NPR projetado e o eixo do Weibull vinha daí.

Removida em 08/08/2026. O teste roda sem torch e sem o dataset bruto: lê o
código-fonte por AST e monta uma janela sintética.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from src.core.config import RAIZ_PROJETO
from src.ml.features_ca import (
    COLUNAS_CORRENTE,
    COLUNAS_TENSAO,
    FEATURES_EXCLUIR,
    FS,
    JANELA,
    extrair_janela,
    features_interfase,
)

# Qualquer grandeza do barramento CC. Não inclui "dc" solto de propósito:
# "dc" aparece dentro de palavras legítimas do domínio CA.
PADROES_CC = ("u_dc", "tensao_dc", "_dc_k", "barramento")


def _janela_sintetica(n: int = JANELA, f0: float = 60.0) -> pd.DataFrame:
    """Três fases equilibradas, defasadas de 120°, sem coluna de CC."""
    t = np.arange(n) / FS
    dados = {}
    for k, col in enumerate(COLUNAS_CORRENTE):
        dados[col] = 10.0 * np.sin(2 * np.pi * f0 * t + k * 2 * np.pi / 3)
    for k, col in enumerate(COLUNAS_TENSAO):
        dados[col] = 220.0 * np.sin(2 * np.pi * f0 * t + k * 2 * np.pi / 3)
    return pd.DataFrame(dados)


# ── o vetor de features ────────────────────────────────────────────────────

def test_extrair_janela_funciona_sem_a_coluna_de_cc():
    """A prova mais direta de que o CC saiu: a janela nem precisa trazê-lo.

    Antes, `extrair_janela` levantava KeyError sem `u_dc_k`.
    """
    feats = extrair_janela(_janela_sintetica())
    assert feats, "extrair_janela devolveu vetor vazio"


def test_nenhuma_feature_e_do_barramento_cc():
    nomes = set(extrair_janela(_janela_sintetica()))
    culpadas = [n for n in nomes if any(p in n for p in PADROES_CC)]
    assert not culpadas, (
        f"features de barramento CC no vetor do lado CA: {culpadas}. "
        "Ver 'ESCOPO CA' em src/ml/features_ca.py"
    )


def test_features_interfase_nao_aceita_mais_o_argumento_dc():
    """Assinatura antiga: `features_interfase(rms_i, rms_u, dc)`.

    Se alguém restaurar o parâmetro, este teste avisa antes de o valor voltar a
    entrar no vetor.
    """
    with pytest.raises(TypeError):
        features_interfase({}, {}, 700.0)


def test_interfase_devolve_so_desbalanceamento_e_potencia():
    rms_i = {f"i_{f}_rms": v for f, v in zip("abc", (7.0, 7.1, 6.9))}
    rms_u = {f"u_{f}_rms": v for f, v in zip("abc", (155.0, 156.0, 154.0))}
    assert set(features_interfase(rms_i, rms_u)) == {
        "desbalanceamento_corrente", "desbalanceamento_tensao",
        "potencia_a", "potencia_b", "potencia_c",
    }


# ── leitura do CSV: o CC não deve nem ser carregado ────────────────────────

def _fonte(relativo: str) -> ast.Module:
    return ast.parse((RAIZ_PROJETO / relativo).read_text(encoding="utf-8"))


def _nomes_citados(arvore: ast.Module) -> set[str]:
    return {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)}


@pytest.mark.parametrize("modulo", [
    "src/ml/dados_avaliacao.py",     # carrega o bruto para injeção/validação
    "src/ml/injecao_falhas.py",      # importava COLUNA_DC sem usar
])
def test_modulos_do_pipeline_nao_referenciam_a_coluna_de_cc(modulo):
    assert "COLUNA_DC" not in _nomes_citados(_fonte(modulo)), (
        f"{modulo} ainda referencia COLUNA_DC — o barramento voltaria a ser "
        "lido do CSV sem alimentar feature nenhuma"
    )


def test_a_constante_sobrevive_para_a_eda():
    """`COLUNA_DC` continua existindo: a EDA descreve o dataset inteiro.

    Descrever o dado bruto é legítimo; o que saiu de escopo foi alimentar o
    VETOR DE FEATURES com ele.
    """
    from src.ml.features_ca import COLUNA_DC
    assert COLUNA_DC == "u_dc_k"


# ── a decisão em aberto, registrada em vez de silenciada ───────────────────

def test_offset_cc_do_lado_ca_continua_excluido_e_documentado():
    """As seis médias de sinal CA seguem fora — mas por decisão, não por acaso.

    São o componente CC dos sinais CA (injeção de CC na rede, limitada por
    IEC 61727 e IEEE 1547). Reintroduzi-las exigiria antes estender a assinatura
    do IGBT em docs/fmeca.md; sem isso, seriam seis dimensões inertes sob
    injeção — o mesmo defeito pelo qual `tensao_dc_media` saiu.
    """
    assert set(FEATURES_EXCLUIR) == {
        "i_a_media", "i_b_media", "i_c_media",
        "u_a_media", "u_b_media", "u_c_media",
    }
    fonte = (RAIZ_PROJETO / "src/ml/features_ca.py").read_text(encoding="utf-8")
    for marca in ("DECISÃO EM ABERTO", "IEC 61727", "IEEE 1547"):
        assert marca in fonte, f"a justificativa perdeu a marca {marca!r}"
