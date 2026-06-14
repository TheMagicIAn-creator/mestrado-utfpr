"""
Sprint 4 — logging estruturado (item 10.1).

get_logger escreve em logs/al_iado_pv.log (arquivo rotativo, não versionado).
"""

import logging

from src.core.logs import ARQUIVO_LOG, get_logger


def test_get_logger_escreve_no_arquivo():
    log = get_logger("teste")
    assert log.name == "al_iado_pv.teste"

    log.error("mensagem estruturada de teste 4711")
    for h in logging.getLogger("al_iado_pv").handlers:
        h.flush()

    assert ARQUIVO_LOG.exists()
    conteudo = ARQUIVO_LOG.read_text(encoding="utf-8")
    assert "4711" in conteudo
    assert "ERROR" in conteudo
    assert "al_iado_pv.teste" in conteudo


def test_limpar_simbolos_tira_emoji_preserva_texto():
    from src.core.logs import limpar_simbolos

    assert limpar_simbolos("✅ ok") == " ok"
    assert "⚡" not in limpar_simbolos("⚡ ANÁLISE")
    assert "🔄" not in limpar_simbolos("🔄 motor")
    # acentos, box-drawing e setas de texto são preservados
    s = limpar_simbolos("ANÁLISE — Mín 1 ═══ → fim")
    assert "ANÁLISE" in s and "Mín" in s and "═" in s and "→" in s


def test_log_no_arquivo_nao_tem_emoji():
    log = get_logger("teste_emoji")
    log.error("status ✅ pronto ⚡ 🔄 fim 8842")
    for h in logging.getLogger("al_iado_pv").handlers:
        h.flush()
    conteudo = ARQUIVO_LOG.read_text(encoding="utf-8")
    linha = [ln for ln in conteudo.splitlines() if "8842" in ln][-1]
    assert "✅" not in linha and "⚡" not in linha and "🔄" not in linha
    assert "status" in linha and "pronto" in linha and "fim" in linha
