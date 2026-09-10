"""Guardas da publicação científica canônica."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.auditar_resultados import auditar_publicacao
from src.core.config import RAIZ_PROJETO
from src.ml.detectabilidade import MAXIMO_A_PUBLICAVEL
from src.ml.proveniencia import funcao_de_hash_para

RAIZ = Path(RAIZ_PROJETO)
RESULTADOS = RAIZ / "resultados"
MANIFESTOS = RESULTADOS / "manifestos"
DETECTABILIDADE = RESULTADOS / "detectabilidade"

MANIFESTOS_CANONICOS = (
    MANIFESTOS / "comparacao_autoencoders.json",
    MANIFESTOS / "confiabilidade_componentes.json",
    MANIFESTOS / "detectabilidade.json",
)


@pytest.mark.parametrize("manifesto_path", MANIFESTOS_CANONICOS)
def test_manifesto_canonico_protege_todos_os_outputs(manifesto_path: Path):
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    assert manifesto["manifest_version"] == 2
    assert set(manifesto["outputs"]) == set(manifesto["output_artifacts"])

    for relativo, esperado in manifesto["output_artifacts"].items():
        caminho = RAIZ / relativo
        assert caminho.is_file(), f"artefato canônico ausente: {relativo}"
        assert caminho.stat().st_size > 0, f"artefato canônico vazio: {relativo}"
        assert funcao_de_hash_para(caminho)(caminho) == esperado


def test_resultados_contem_somente_as_quatro_pastas_canonicas():
    entradas = {item.name for item in RESULTADOS.iterdir()}
    assert entradas == {
        "comparacao",
        "confiabilidade",
        "detectabilidade",
        "manifestos",
    }


def test_manifestos_contem_execucoes_cientificas_e_evidence_rag():
    assert {item.name for item in MANIFESTOS.iterdir()} == {
        "comparacao_autoencoders.json",
        "confiabilidade_componentes.json",
        "detectabilidade.json",
        "evidence_rag_baseline_v1.json",
        "evidence_rag_contextual_r3.json",
        "evidence_rag_hybrid_r4.json",
        "evidence_rag_guard_r5.json",
        "evidence_rag_promotion_r6.json",
        "evidence_graph_pilot_r7.json",
        "evidence_rag_schema_v2_r2.json",
    }


def test_auditoria_canonica_aprova_publicacao():
    relatorio = auditar_publicacao(RAIZ)
    assert relatorio["ok"], "\n".join(relatorio["errors"])
    assert relatorio["manifests"] == 10
    assert relatorio["artifacts"] == 47


def _artefato_e2_predata_o_codigo() -> bool:
    """O contrato E2 publicado ainda desconhece a guarda de domínio?

    Em 2026-09-08 o portão de adoção da Weibull ganhou um quarto critério, que
    recusa ajuste cujo percentil publicável escape de `[0,1]`. O código entrou
    no `main` sem a regeneração, então `resultados/detectabilidade/` seguiu
    publicando `a50_parametrico = 10,28` — um número que o próprio código já
    declara impublicável. Só

        python -m src.ml.campanha_detectabilidade

    fecha essa divergência.

    A condição é lida do próprio artefato, some sozinha quando ele for
    regenerado, e `strict=True` faz o CI reprovar se a guarda passar enquanto a
    condição ainda vale — ninguém "conserta" isso mexendo no teste.
    """
    contrato = DETECTABILIDADE / "detectabilidade.json"
    if not contrato.is_file():
        return False
    portao = (
        json.loads(contrato.read_text(encoding="utf-8"))
        .get("methodology", {})
        .get("weibull_adoption_gate", {})
    )
    return portao.get("max_publishable_a") != MAXIMO_A_PUBLICAVEL


FIGURAS_E2 = (
    "e2_pod_curvas.png",
    "e2_pod_curvas.pdf",
    "e2_fidelidade.png",
    "e2_fidelidade.pdf",
)


def _manifesto_e2_ainda_nao_conhece_as_figuras() -> bool:
    """O manifesto da E2 publicado ainda ignora o módulo que desenha as figuras?

    Em 2026-09-09 a etapa passou a publicar quatro figuras. Enquanto a campanha
    não for reexecutada, os arquivos existem em `resultados/detectabilidade/`
    mas não estão no manifesto — presentes e NÃO hasheados. A auditoria conta o
    que o manifesto lista, então essa defasagem passaria em silêncio, que é
    exatamente o modo de falha fechado pela guarda acima.

    Só

        python -m src.ml.campanha_detectabilidade

    fecha a divergência. A condição vem do próprio manifesto e some sozinha.
    """
    manifesto = MANIFESTOS / "detectabilidade.json"
    if not manifesto.is_file():
        return False
    dependencias = json.loads(manifesto.read_text(encoding="utf-8")).get(
        "code_dependencies", {}
    )
    return "plots" not in dependencias


PENDENTE_DE_FIGURAS_E2 = pytest.mark.xfail(
    _manifesto_e2_ainda_nao_conhece_as_figuras(),
    strict=True,
    reason=(
        "resultados/detectabilidade/ foi gerado antes das figuras da E2; rode "
        "`python -m src.ml.campanha_detectabilidade` e commite resultados/ para "
        "reativar esta guarda"
    ),
)


@PENDENTE_DE_FIGURAS_E2
def test_figuras_da_e2_estao_no_manifesto_e_hasheadas():
    """Figura publicada sem hash é figura que ninguém consegue auditar."""
    manifesto = json.loads(
        (MANIFESTOS / "detectabilidade.json").read_text(encoding="utf-8")
    )
    listados = set(manifesto["output_artifacts"])
    faltando = [
        nome
        for nome in FIGURAS_E2
        if f"resultados/detectabilidade/{nome}" not in listados
    ]
    assert not faltando, f"figuras da E2 fora do manifesto: {faltando}"


PENDENTE_DE_REGENERACAO_E2 = pytest.mark.xfail(
    _artefato_e2_predata_o_codigo(),
    strict=True,
    reason=(
        "resultados/detectabilidade/ foi gerado antes da guarda de domínio da "
        "Weibull; rode `python -m src.ml.campanha_detectabilidade` e commite "
        "resultados/ para reativar esta guarda"
    ),
)


@PENDENTE_DE_REGENERACAO_E2
def test_e2_publicada_nao_traz_percentil_fora_do_dominio():
    """Guarda sobre o ARTEFATO publicado, não sobre a função que o gera.

    `a` é fração da assinatura nominal e vive em `[0,1]`; percentil paramétrico
    fora disso não é magnitude de detecção, é extrapolação além da maior
    severidade injetável. O teste unitário cobre o portão; este cobre o que
    efetivamente foi para o repositório.
    """
    caminho = DETECTABILIDADE / "detectabilidade_resumo.csv"
    assert caminho.is_file(), "resumo da E2 ausente"
    with caminho.open("r", encoding="utf-8", newline="") as stream:
        linhas = list(csv.DictReader(stream))
    assert linhas, "resumo da E2 vazio"

    fora = [
        f"{linha['injection_id']}/{linha['model']}: {coluna}={valor}"
        for linha in linhas
        for coluna, valor in linha.items()
        if coluna.endswith("_parametrico")
        and valor
        and not 0.0 <= float(valor) <= MAXIMO_A_PUBLICAVEL
    ]
    assert not fora, (
        f"percentil paramétrico publicado fora de [0, {MAXIMO_A_PUBLICAVEL:g}]: {fora}"
    )
