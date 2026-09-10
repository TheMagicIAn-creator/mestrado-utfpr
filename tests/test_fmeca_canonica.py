"""Guardas de coerência entre a FMECA documentada e a publicada.

POR QUE ESTA GUARDA EXISTE
==========================
Quando a FMECA passou de 3 itens sem escore para 6 itens validados por
`NPR = S * O * D`, a frase antiga sobreviveu em quatro lugares e precisou de
quatro correções separadas: `docs/mapa_de_resultados.md`,
`src/webapp/scientific_context.py`, `docs/reproducibilidade.md` e o prompt do
agente. Em todos, o texto continuava afirmando escore nulo enquanto o artefato
publicava valor — e a CI passava, porque nada ligava um ao outro.

O que estas guardas ligam é o mesmo que o projeto já liga entre artefato e
código: a **fonte única** (`docs/fmeca.md`) contra o **contrato publicado**
(`resultados/confiabilidade/metodologia.json`). Mudar o escopo em um sem mudar
no outro passa a reprovar.

O QUE ELAS NÃO FAZEM
====================
Não varrem prosa livre atrás de frases proibidas. Tentei: a varredura por
janela de linhas confunde negação ("nada aqui é nulo"), a própria guarda de
regressão em `tests/test_webapp.py` e planos de execução históricos, que
registram legitimamente o que valia na data em que foram escritos. Guarda com
falso positivo é guarda que alguém desliga.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

from src.core.config import RAIZ_PROJETO

RAIZ = Path(RAIZ_PROJETO)
DOCUMENTO = RAIZ / "docs" / "fmeca.md"
CONTRATO = RAIZ / "resultados" / "confiabilidade" / "metodologia.json"

# Textos que instruem uma pessoa ou o modelo sobre o escopo vigente. Planos de
# execução datados ficam de fora de propósito: são registro histórico.
TEXTOS_INSTRUCIONAIS = (
    RAIZ / "docs" / "fmeca.md",
    RAIZ / "CLAUDE.md",
    RAIZ / "src" / "conhecimento" / "agente.py",
)

_ESCOPO_DECLARADO = re.compile(r"escopo de \**(\d+) itens", re.IGNORECASE)


def _sem_acentos(texto: str) -> str:
    decomposto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn").casefold()


@pytest.fixture(scope="module")
def contrato() -> dict:
    return json.loads(CONTRATO.read_text(encoding="utf-8"))["fmeca"]


@pytest.fixture(scope="module")
def tabela() -> list[dict]:
    """Linhas de dado da tabela de escopo em `docs/fmeca.md`."""
    linhas = []
    for linha in DOCUMENTO.read_text(encoding="utf-8").splitlines():
        if not linha.startswith("|"):
            continue
        celulas = [celula.strip() for celula in linha.strip("|").split("|")]
        if len(celulas) < 6 or not celulas[2].isdigit():
            continue  # cabeçalho, separador ou outra tabela do documento
        linhas.append(
            {
                "item": celulas[0].replace("**", ""),
                "severity": int(celulas[2]),
                "occurrence": int(celulas[3]),
                "detectability": int(celulas[4]),
                "npr": int(celulas[5]),
            }
        )
    return linhas


def test_tabela_do_documento_bate_com_o_contrato_publicado(contrato, tabela):
    """A fonte única e o artefato descrevem a mesma FMECA, item a item."""
    componentes = contrato["components"]
    assert len(tabela) == len(componentes), (
        f"docs/fmeca.md lista {len(tabela)} itens e o contrato publica "
        f"{len(componentes)}"
    )

    chave = ("severity", "occurrence", "detectability", "npr")
    do_documento = sorted(tuple(linha[campo] for campo in chave) for linha in tabela)
    do_contrato = sorted(
        tuple(componente[campo] for campo in chave) for componente in componentes
    )
    assert do_documento == do_contrato, (
        "escores S/O/D/NPR divergem entre docs/fmeca.md e o contrato publicado"
    )

    # O nome do documento pode ser mais curto que o do contrato, nunca outro
    # item: "Sistema/circuito de controle" vale por "... do inversor".
    nomes = [_sem_acentos(item["component_name"]) for item in componentes]
    for linha in tabela:
        alvo = _sem_acentos(linha["item"])
        assert any(
            nome.startswith(alvo) or alvo.startswith(nome) for nome in nomes
        ), f"item de docs/fmeca.md ausente do contrato: {linha['item']}"


def test_documento_e_contrato_recalculam_o_npr_da_mesma_forma(contrato, tabela):
    """`NPR = S * O * D` vale nos dois lados; nenhum traz valor fabricado."""
    for linha in tabela:
        esperado = linha["severity"] * linha["occurrence"] * linha["detectability"]
        assert linha["npr"] == esperado, f"NPR inconsistente em {linha['item']}"
    for componente in contrato["components"]:
        esperado = (
            componente["severity"]
            * componente["occurrence"]
            * componente["detectability"]
        )
        assert componente["npr"] == esperado, (
            f"NPR inconsistente em {componente['component_id']}"
        )


def test_textos_instrucionais_declaram_o_escopo_vigente(contrato):
    """Toda contagem de itens escrita à mão bate com a publicada.

    Foi por uma contagem defasada — "3 itens", quando já eram 6 — que a
    afirmação falsa entrou no contexto autoritativo do LLM.
    """
    esperado = len(contrato["components"])
    divergentes = []
    for path in TEXTOS_INSTRUCIONAIS:
        texto = path.read_text(encoding="utf-8")
        for declarado in _ESCOPO_DECLARADO.findall(texto):
            if int(declarado) != esperado:
                divergentes.append(f"{path.relative_to(RAIZ).as_posix()}: {declarado}")
    assert not divergentes, (
        f"escopo da FMECA declarado fora do contrato ({esperado} itens): {divergentes}"
    )


def test_contrato_nao_afirma_bloqueio_enquanto_os_escores_valem(contrato):
    """Coerência interna: item com escore não convive com FMECA bloqueada.

    Guarda o sentido inverso do mesmo erro — o artefato declarando indisponível
    aquilo que ele mesmo publica preenchido.
    """
    preenchidos = [
        componente
        for componente in contrato["components"]
        if componente.get("npr") is not None
    ]
    if not preenchidos:
        return
    assert contrato["status"] == "validated"
    assert contrato["calculation_enabled"] is True
    assert contrato["formula"] == "NPR = S * O * D"
    for componente in preenchidos:
        assert componente["status"] == "validated", componente["component_id"]
        assert componente["calculation_enabled"] is True, componente["component_id"]
