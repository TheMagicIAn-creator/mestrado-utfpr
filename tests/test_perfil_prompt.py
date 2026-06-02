"""
Sprint 1 — integridade acadêmica.

Garante que:
- o PERFIL_COMPACTO (identidade ESTÁTICA injetada no prompt) NÃO contém
  resultados numéricos experimentais (limiar, AUC, F1, SMD, recall) que
  ficariam desatualizados após novo treino ou exclusão de artefatos;
- o parâmetro `perfil` REALMENTE entra no prompt final (não é mais ignorado).
"""

import re

from src.conhecimento.agente import PERFIL_COMPACTO, _montar_prompt

_ORC = {
    "contexto_chars": 4000,
    "max_prompt_chars": 30000,
    "sessao_chars": 1000,
    "anexos_chars": 4000,
}


def test_perfil_compacto_sem_metricas_hardcoded():
    txt = PERFIL_COMPACTO.lower()
    proibidos = [
        r"auc\s*=\s*\d",
        r"f1\s*=?\s*0[.,]\d",
        r"smd\s*=\s*\d",
        r"limiar\s*p99\s*=\s*\d",
        r"recall\s*=?\s*1[.,]0",
        r"baseline\s*=\s*0[.,]30",
    ]
    achados = [p for p in proibidos if re.search(p, txt)]
    assert not achados, f"PERFIL_COMPACTO contém métrica hardcoded: {achados}"


def test_perfil_entra_no_prompt():
    marcador = "MARCADOR_PERFIL_UNICO_4711"
    prompt = _montar_prompt(
        "pergunta de teste", "", "", _ORC,
        consultar_literatura=False, perfil=marcador,
    )
    assert marcador in prompt, "o parâmetro perfil deve entrar no prompt final"


def test_perfil_default_usa_compacto():
    prompt = _montar_prompt("oi", "", "", _ORC, consultar_literatura=False)
    assert "Al IAdo PV" in prompt  # identidade estática default presente
