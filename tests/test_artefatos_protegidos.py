"""Guardas da publicação científica canônica."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.auditar_resultados import auditar_publicacao
from src.core.config import RAIZ_PROJETO
from src.ml.proveniencia import funcao_de_hash_para

RAIZ = Path(RAIZ_PROJETO)
RESULTADOS = RAIZ / "resultados"
MANIFESTOS = RESULTADOS / "manifestos"

MANIFESTOS_CANONICOS = (
    MANIFESTOS / "comparacao_autoencoders.json",
    MANIFESTOS / "confiabilidade_componentes.json",
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


def test_resultados_contem_somente_as_tres_pastas_canonicas():
    entradas = {item.name for item in RESULTADOS.iterdir()}
    assert entradas == {"comparacao", "confiabilidade", "manifestos"}


def test_manifestos_contem_somente_as_duas_execucoes_canonicas():
    assert {item.name for item in MANIFESTOS.iterdir()} == {
        "comparacao_autoencoders.json",
        "confiabilidade_componentes.json",
    }


def test_auditoria_canonica_aprova_publicacao():
    relatorio = auditar_publicacao(RAIZ)
    assert relatorio["ok"], "\n".join(relatorio["errors"])
    assert relatorio["manifests"] == 2
    assert relatorio["artifacts"] == 30
