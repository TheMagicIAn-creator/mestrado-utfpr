"""Nota superada não pode chegar ao LLM com cara de nota vigente.

POR QUE ESTE TESTE EXISTE
=========================
O vault guarda notas curadas de fases anteriores do projeto. Algumas delas
afirmam, como resultado, coisas que já não valem: os macro-códigos removidos,
o escore localizado sobre o conjunto Paderborn, a detectabilidade calculada
sobre o recorte Contator AC / IGBT / Fusível AC — que não é a FMECA vigente.

Havia dois furos, e o segundo era o pior.

**Furo 1 — o vocabulário derivou.** Três notas de `notas/Cerebro/Resultados/`
estavam marcadas `status: superseded`, em inglês. O código penalizava apenas
`{"rascunho", "superado"}`. As três eram tratadas como ATIVAS.

**Furo 2 — a marca não viajava.** A penalidade de -0,18 mexia no *ranking*,
mas o cabeçalho montado para o LLM mostrava origem, tipo, confiança, evidência
e data — nunca o status. Uma nota superada que sobrevivesse ao ranking chegava
indistinguível de uma vigente.

Na fase de escrita da dissertação isso é caro: perguntado "o que eu tenho de
resultado sobre Weibull?", o agente tinha material curado para responder
errado com confiança.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.conhecimento.consultas_obsidian import (
    STATUS_NAO_VIGENTES,
    status_normalizado,
)

RAIZ = Path(__file__).resolve().parents[1]


# ── o vocabulário ──────────────────────────────────────────────────────────

@pytest.mark.leve
@pytest.mark.parametrize(
    "status", ["superado", "superseded", "rascunho", "draft", "obsoleto", "historico"]
)
def test_os_dois_idiomas_contam_como_nao_vigente(status):
    assert status in STATUS_NAO_VIGENTES


@pytest.mark.leve
def test_ativo_continua_vigente():
    assert "ativo" not in STATUS_NAO_VIGENTES
    assert status_normalizado({}) == "ativo"
    assert status_normalizado({"status": "  ATIVO "}) == "ativo"
    assert status_normalizado({"status": ""}) == "ativo"


@pytest.mark.leve
def test_o_status_e_normalizado_antes_da_comparacao():
    """`Superseded` com maiúscula não pode escapar da marcação."""
    assert status_normalizado({"status": "Superseded"}) in STATUS_NAO_VIGENTES


# ── a marca tem de viajar até o LLM ────────────────────────────────────────

@pytest.mark.leve
def test_o_cabecalho_do_contexto_carrega_status_e_aviso():
    """Guarda estrutural sobre o texto que vai para o modelo.

    Sem `status=` no cabeçalho, a penalidade de ranking é a única defesa — e
    ela só reordena, não avisa.
    """
    fonte = (
        RAIZ / "src/conhecimento/consultas_obsidian.py"
    ).read_text(encoding="utf-8")

    assert "status={status}" in fonte, (
        "o cabeçalho do contexto Obsidian voltou a omitir o status da nota"
    )
    assert "SUPERADA" in fonte, "o aviso explícito de nota superada sumiu"


# ── as notas do vault ──────────────────────────────────────────────────────

# Notas curadas cujo conteúdo descreve estado anterior do projeto. Cada uma
# afirma, como resultado ou decisão vigente, algo que já não vale.
NOTAS_HISTORICAS = (
    "notas/Cerebro/Decisoes/Macro-códigos de comparação.md",
    "notas/Cerebro/Resultados/Correção do escore — antes e depois.md",
    "notas/Cerebro/Resultados/Detectabilidade E2 e ajuste Weibull auditado.md",
    "notas/Cerebro/Resultados/Estimativas Exploratórias de RUL e Análise de Weibull (E2).md",
    "notas/Cerebro/Resultados/Resultados Exploratórios de Confiabilidade Weibull e RUL (E2).md",
    "notas/Cerebro/Resultados/Resultados Exploratórios de RUL e Ajuste Weibull (Evidência E2).md",
)


@pytest.mark.leve
@pytest.mark.parametrize("relativo", NOTAS_HISTORICAS)
def test_nota_historica_esta_marcada_como_nao_vigente(relativo):
    caminho = RAIZ / relativo
    if not caminho.is_file():
        pytest.skip(f"nota ausente: {relativo}")

    cabecalho = caminho.read_text(encoding="utf-8").split("---", 2)[1]
    status = next(
        (
            linha.split(":", 1)[1].strip().strip('"').lower()
            for linha in cabecalho.splitlines()
            if linha.strip().lower().startswith("status:")
        ),
        "ativo",
    )
    assert status in STATUS_NAO_VIGENTES, (
        f"{relativo} voltou a ser 'ativo'; ela descreve estado anterior do "
        "projeto e o agente pode citá-la como corrente"
    )
