"""
Artefato de resultado não pode ser apagado por automação de backup do vault.

POR QUE ESTE TESTE EXISTE
=========================
Em 06/08/2026 o commit `e5518db`, um **`vault backup` automático** do plugin
Obsidian Git, deletou 24 arquivos de `resultados/autoencoder/` — entre eles
`limiar.json`, `validacao_report.json`, `weibull_results.json` e
`weibull_confiabilidade.png`, que é justamente a figura com a sobrevivência
empírica do detector que o pesquisador relatou "não ter visto".

Não foi caso isolado: pelo menos cinco commits `vault backup` tocaram
`resultados/`. A causa é o plugin estar commitando o repositório inteiro em vez
de só `notas/`.

A correção de raiz é escopar o plugin na máquina do pesquisador. Este teste é a
rede de segurança enquanto isso: se os artefatos essenciais sumirem de novo, a
CI reprova em vez de o problema aparecer semanas depois, quando alguém for
procurar a curva para a dissertação.

Recuperação, se acontecer de novo:

    git log --oneline --diff-filter=D -- resultados/autoencoder/
    git checkout <commit_anterior> -- resultados/autoencoder/
"""

from __future__ import annotations

import pytest

from src.core.config import RAIZ_PROJETO

# Artefatos sem os quais a dissertação perde evidência. Não é a lista completa
# de saídas do pipeline — é o conjunto mínimo que sustenta afirmação no texto.
ARTEFATOS_ESSENCIAIS = [
    # ponto de operação e proveniência do detector
    "resultados/autoencoder/limiar.json",
    "resultados/autoencoder/calibracao_autoencoder.csv",
    # detecção por magnitude de injeção (POD_mon, SMD)
    "resultados/autoencoder/injecao_falhas_report.json",
    "resultados/autoencoder/injecao_smd_tabela.csv",
    # validação E2
    "resultados/autoencoder/validacao_report.json",
    "resultados/autoencoder/validacao_tabela.csv",
    # confiabilidade
    "resultados/autoencoder/weibull_results.json",
    "resultados/autoencoder/weibull_tabela.csv",
    # a figura que motivou esta guarda
    "resultados/autoencoder/weibull_confiabilidade.png",
    "resultados/autoencoder/weibull_intensidade_deteccao.png",
    "resultados/autoencoder/weibull_funcoes_distribuicao.png",
    "resultados/autoencoder/weibull_distribuicao.png",
]


@pytest.mark.parametrize("relativo", ARTEFATOS_ESSENCIAIS)
def test_artefato_essencial_existe(relativo):
    caminho = RAIZ_PROJETO / relativo
    assert caminho.exists(), (
        f"{relativo} sumiu do repositório. Antes de regerar, verifique se foi "
        "apagado por commit automático:\n"
        f"    git log --oneline --diff-filter=D -- {relativo}\n"
        "Se foi, RESTAURE em vez de recalcular — recalcular muda os números."
    )


@pytest.mark.parametrize("relativo", ARTEFATOS_ESSENCIAIS)
def test_artefato_essencial_nao_esta_vazio(relativo):
    """Arquivo de 0 byte é pior que ausente: passa por existente."""
    assert (RAIZ_PROJETO / relativo).stat().st_size > 0, f"{relativo} está vazio"


def test_curva_de_confiabilidade_esta_publicada():
    """As funções do detector ficam publicadas sem alegação física indevida.

    Foi a que o pesquisador reportou não ter visto, e a que o backup apagou.
    """
    nomes = (
        "weibull_confiabilidade.png",
        "weibull_intensidade_deteccao.png",
        "weibull_funcoes_distribuicao.png",
        "weibull_distribuicao.png",
    )
    for nome in nomes:
        png = RAIZ_PROJETO / "resultados/autoencoder" / nome
        assert png.exists() and png.stat().st_size > 10_000, (
            f"{nome} ausente ou truncada — a evidência Weibull ficou incompleta"
        )
