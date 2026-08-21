"""Contrato de logging: console padrão e arquivo rotativo opt-in."""

import logging

from src.core.logs import (
    ARQUIVO_LOG,
    adaptar_logger_como_print,
    configurar_logging,
    get_logger,
)


class _LoggerEspiao:
    def __init__(self):
        self.mensagens_info = []
        self.mensagens_debug = []

    def info(self, mensagem):
        self.mensagens_info.append(mensagem)

    def debug(self, mensagem):
        self.mensagens_debug.append(mensagem)


def test_get_logger_configura_console_por_padrao():
    log = get_logger("teste")
    assert log.name == "al_iado_pv.teste"
    handlers = logging.getLogger("al_iado_pv").handlers
    assert any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.handlers.RotatingFileHandler)
        for handler in handlers
    )


def test_arquivo_rotativo_e_opt_in():
    configurar_logging(arquivo=True)
    log = get_logger("teste_arquivo")

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


def test_log_opt_in_nao_tem_emoji():
    configurar_logging(arquivo=True)
    log = get_logger("teste_emoji")
    log.error("status ✅ pronto ⚡ 🔄 fim 8842")
    for h in logging.getLogger("al_iado_pv").handlers:
        h.flush()
    conteudo = ARQUIVO_LOG.read_text(encoding="utf-8")
    linha = [ln for ln in conteudo.splitlines() if "8842" in ln][-1]
    assert "✅" not in linha and "⚡" not in linha and "🔄" not in linha
    assert "status" in linha and "pronto" in linha and "fim" in linha


def test_adaptador_print_preserva_contrato_dos_scripts_ml():
    logger = _LoggerEspiao()
    log = adaptar_logger_como_print(logger)

    log("epoca", 3, sep="=", end="\r", flush=True)
    log("\rprogresso 50%")
    log("   ")

    assert logger.mensagens_info == ["epoca=3"]
    assert logger.mensagens_debug == ["progresso 50%"]
