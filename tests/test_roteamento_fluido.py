"""Roteamento FLUIDO: guardas críticas → LLM semântico → cascata (fallback)."""

from __future__ import annotations

import src.conhecimento.ferramentas as fr


class LLMFalso:
    """LLM de teste: devolve a decisão programada e conta as chamadas."""

    def __init__(self, ferramenta=None, usar=True, quebrar=False):
        self.ferramenta, self.usar, self.quebrar = ferramenta, usar, quebrar
        self.chamadas = 0

    def invoke(self, _mensagens):
        self.chamadas += 1
        if self.quebrar:
            raise RuntimeError("LLM indisponível")

        class R:
            content = (
                '{"usar_ferramenta": %s, "ferramenta": %s}'
                % ("true" if self.usar else "false",
                   f'"{self.ferramenta}"' if self.ferramenta else "null")
            )
        return R()


# ── 1. guardas críticas decidem ANTES do LLM (nem o consultam) ───────────────

def test_declaracao_de_memoria_nao_chega_ao_llm():
    llm = LLMFalso("rodar_pipeline_completo")
    d = fr.decidir_acao("Lembre-se: decidimos que a falha injetada no pipeline "
                        "é a do contator", llm)
    assert d["usar_ferramenta"] is False
    assert llm.chamadas == 0, "guarda negativa deve decidir sem gastar LLM"


def test_pedido_de_codigo_nao_chega_ao_llm():
    llm = LLMFalso("consultar_resultados")
    d = fr.decidir_acao("escreva um código em python para plotar a curva de TTF", llm)
    assert d["usar_ferramenta"] is False
    assert llm.chamadas == 0


def test_limpeza_e_destrutiva_e_nao_depende_do_llm():
    llm = LLMFalso("consultar_resultados")
    d = fr.decidir_acao("apague os resultados a partir do autoencoder", llm)
    assert d == {"usar_ferramenta": True, "ferramenta": "limpar_resultados_ml"}
    assert llm.chamadas == 0, "destrutivo não pode depender do LLM"


# ── 2. o LLM roteia o resto (fluidez) ────────────────────────────────────────

def test_llm_decide_quando_nao_ha_guarda():
    llm = LLMFalso("consultar_status_pipeline")
    d = fr.decidir_acao("me diz aí como andam as etapas", llm)
    assert d == {"usar_ferramenta": True, "ferramenta": "consultar_status_pipeline"}
    assert llm.chamadas == 1


def test_llm_pode_decidir_que_nao_usa_ferramenta():
    llm = LLMFalso(None, usar=False)
    d = fr.decidir_acao("o que significa o beta da Weibull ser maior que 1?", llm)
    assert d["usar_ferramenta"] is False
    assert llm.chamadas == 1


def test_frase_torta_e_roteada_pelo_llm_nao_por_palavra_chave():
    # sem gatilho literal de "rodar"; só o LLM entenderia
    llm = LLMFalso("rodar_weibull")
    d = fr.decidir_acao("bora fechar a parte de confiabilidade agora?", llm)
    assert d["ferramenta"] == "rodar_weibull"


def test_ferramenta_inexistente_do_llm_e_ignorada():
    llm = LLMFalso("ferramenta_que_nao_existe")
    d = fr.decidir_acao("faz aquilo lá do gráfico", llm)
    # não confia cegamente: cai para a cascata/negativa, nunca inventa ferramenta
    assert d["ferramenta"] in (None, *fr._DESPACHO)


# ── 3. cascata é a rede de segurança quando o LLM cai ────────────────────────

def test_cascata_assume_se_o_llm_quebra():
    llm = LLMFalso(quebrar=True)
    d = fr.decidir_acao("rode o pipeline completo", llm)
    assert d["usar_ferramenta"] is True
    assert d["ferramenta"] == "rodar_pipeline_completo"


def test_sem_llm_ainda_funciona():
    d = fr.decidir_acao("mostre os resultados", None)
    assert d["usar_ferramenta"] is True


def test_guardas_criticas_sao_poucas():
    # a fluidez depende de a lista de guardas ser curta e justificada
    import inspect
    fonte = inspect.getsource(fr._guardas_criticas)
    assert fonte.count("return {") <= 6
