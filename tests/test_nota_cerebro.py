"""Ferramenta de REGISTRO no cérebro — o agente escrevendo no vault."""

from __future__ import annotations

import src.conhecimento.ferramentas as fr
from src.conhecimento.nota_cerebro import registrar_nota_cerebro


def test_cria_nota_com_frontmatter_e_tags_validas(tmp_path):
    r = registrar_nota_cerebro(
        titulo="Resultado da comparação", conteudo="O método proposto venceu.",
        tipo="resultado", tags=["comparacao-literatura", "igbt"],
        nivel_evidencia="E2", fonte="resultados/comparacao/comparacao_autoencoders.json",
        pasta_base=tmp_path,
    )
    assert r["ok"]
    txt = (tmp_path / "Resultados" / "Resultado da comparação.md").read_text(encoding="utf-8")
    assert txt.startswith("---")
    assert "tipo: resultado" in txt and "nivel_evidencia: E2" in txt
    assert "comparacao-literatura" in txt and "igbt" in txt
    assert "cerebro" in txt                      # tag estrutural sempre presente
    assert "[[00 - Painel do cerebro]]" in txt   # conectada ao hub
    assert "resultados/comparacao/comparacao_autoencoders.json" in txt


def test_descarta_tag_fora_da_taxonomia(tmp_path):
    r = registrar_nota_cerebro("T", "corpo", tags=["igbt", "tag-inventada"],
                               pasta_base=tmp_path)
    assert "igbt" in r["tags"]
    assert "tag-inventada" in r["descartadas"]
    assert "tag-inventada" not in r["tags"]


def test_tipo_define_a_subpasta(tmp_path):
    registrar_nota_cerebro("Conceito X", "c", tipo="conceito", pasta_base=tmp_path)
    registrar_nota_cerebro("Decisao Y", "c", tipo="decisao", pasta_base=tmp_path)
    assert (tmp_path / "Conceitos" / "Conceito X.md").exists()
    assert (tmp_path / "Decisoes" / "Decisao Y.md").exists()


def test_titulo_vira_h1_quando_o_corpo_nao_tem(tmp_path):
    registrar_nota_cerebro("Meu titulo", "só o corpo", pasta_base=tmp_path)
    txt = (tmp_path / "Meu titulo.md").read_text(encoding="utf-8")
    assert "# Meu titulo" in txt


def test_recusa_titulo_ou_conteudo_vazio(tmp_path):
    assert not registrar_nota_cerebro("", "corpo", pasta_base=tmp_path)["ok"]
    assert not registrar_nota_cerebro("t", "  ", pasta_base=tmp_path)["ok"]


def test_nome_de_arquivo_seguro(tmp_path):
    from pathlib import Path
    r = registrar_nota_cerebro('Titulo com / e : proibidos', "c", pasta_base=tmp_path)
    nome = Path(r["caminho"]).name
    assert r["ok"] and "/" not in nome and ":" not in nome


# ── roteamento: o agente escolhe a ferramenta certa ──────────────────────────

def test_roteia_pedido_de_registro():
    assert fr._quer_registrar_no_cerebro("guarde esse resultado no cérebro")
    assert fr._quer_registrar_no_cerebro("registre essa decisão no vault")
    assert fr._quer_registrar_no_cerebro("anote isso no obsidian")


def test_nao_confunde_com_memoria_nem_com_consulta():
    # memória validada (preferência) NÃO é registro de nota curada
    assert not fr._quer_registrar_no_cerebro("lembre que prefiro gráficos escuros")
    # consulta ao vault é RAG, não escrita
    assert not fr._quer_registrar_no_cerebro("o que tem no cérebro sobre FMECA?")
    # limpeza continua sendo limpeza
    assert not fr._quer_registrar_no_cerebro("apague os resultados do weibull")


def test_ferramenta_registrada_no_roteador():
    assert "registrar_no_cerebro" in fr._DESPACHO
    nomes = {e["name"] for e in fr.ESPEC_FERRAMENTAS}
    assert "registrar_no_cerebro" in nomes


def test_pede_ajuda_quando_nao_consegue_redigir():
    # sem LLM e sem contexto, não há como compor a nota — pede ao pesquisador
    r = fr.registrar_no_cerebro(pergunta="guarde no cérebro")
    assert not r["ok"] and "não foi criada" in r["mensagem"].lower()


# ── redação automática: "guarde ESSE resultado" (pedido dêitico) ─────────────

class _LLMRedator:
    """LLM de teste que devolve a nota pronta em JSON."""

    def __init__(self, titulo="Resultado da comparação"):
        self.titulo, self.viu_contexto = titulo, None

    def invoke(self, entrada):
        self.viu_contexto = str(entrada)

        class R:
            content = (
                '{"titulo": "%s", "conteudo": "AUC 0.978 vs 0.909 no IGBT.",'
                ' "tipo": "resultado", "tags": ["igbt", "comparacao-literatura"],'
                ' "nivel_evidencia": "E2"}' % self.titulo
            )
        return R()


def test_llm_redige_a_nota_a_partir_do_contexto(tmp_path, monkeypatch):
    import src.conhecimento.nota_cerebro as nc
    monkeypatch.setattr(nc, "PASTA_CEREBRO_OBSIDIAN", tmp_path)

    llm = _LLMRedator()
    r = fr.registrar_no_cerebro(
        pergunta="guarde esse resultado no cérebro",
        llm=llm,
        contexto="Rodolfo: rode a comparação\nAl IAdo PV: AUC 0.978 vs 0.909 no IGBT.",
    )
    assert r["ok"], r["mensagem"]
    assert "Resultado da comparação" in r["mensagem"]
    # o LLM precisa ter recebido o contexto (senão escreveria sobre o tema errado)
    assert "0.978" in llm.viu_contexto
    nota = (tmp_path / "Resultados" / "Resultado da comparação.md").read_text(encoding="utf-8")
    assert "AUC 0.978" in nota and "nivel_evidencia: E2" in nota


def test_sem_contexto_nem_llm_pede_ajuda():
    r = fr.registrar_no_cerebro(pergunta="guarde no cérebro", llm=None, contexto="")
    assert not r["ok"] and "NÃO foi criada" in r["mensagem"]


def test_executar_ferramenta_repassa_llm_e_contexto(tmp_path, monkeypatch):
    """A ferramenta só funciona se o despacho REPASSAR llm/contexto."""
    import src.conhecimento.nota_cerebro as nc
    monkeypatch.setattr(nc, "PASTA_CEREBRO_OBSIDIAN", tmp_path)

    r = fr.executar_ferramenta(
        "registrar_no_cerebro",
        pergunta="guarde esse resultado no cérebro",
        llm=_LLMRedator("Nota via despacho"),
        contexto="Al IAdo PV: censura do IGBT caiu de 70% para 13%.",
    )
    assert r["ok"], r["mensagem"]
    assert (tmp_path / "Resultados" / "Nota via despacho.md").exists()


def test_ferramentas_antigas_nao_quebram_com_os_novos_parametros():
    """Só quem declara llm/contexto recebe — as demais seguem intactas."""
    r = fr.executar_ferramenta("consultar_status_pipeline", pergunta="status",
                               llm=_LLMRedator(), contexto="qualquer coisa")
    assert isinstance(r, dict) and "mensagem" in r


# ── falha NUNCA pode ser reescrita pelo LLM ──────────────────────────────────

def test_falha_e_reportada_literalmente(monkeypatch):
    """Erro reescrito pelo LLM escondia que a acao nao aconteceu."""
    class LLMInventor:
        def invoke(self, _m):
            class R:
                content = "Estou alinhado com o escopo do seu mestrado..."
            return R()

    falha = {"ok": False, "mensagem": "⚠️ **A nota NÃO foi criada.** Motivo X.",
             "resposta_pronta": True}
    saida = fr.comentar_resultado("guarde no cérebro", falha, "", LLMInventor())
    assert "NÃO foi criada" in saida
    assert "alinhado com o escopo" not in saida


def test_falha_literal_mesmo_em_pedido_autoral():
    class LLMInventor:
        def invoke(self, _m):
            class R:
                content = "Na minha opinião, tudo certo!"
            return R()

    falha = {"ok": False, "mensagem": "⚠️ Falhou por tal motivo."}
    saida = fr.comentar_resultado("na sua opinião, guarde isso", falha, "", LLMInventor())
    assert "Falhou" in saida


def test_sem_contexto_explica_o_motivo_real():
    r = fr.registrar_no_cerebro(pergunta="guarde esse resultado no cérebro",
                                llm=None, contexto="")
    assert not r["ok"]
    assert "histórico está vazio" in r["mensagem"]
    assert "NÃO foi criada" in r["mensagem"]


def test_confirmacao_de_acao_e_literal_mesmo_em_pedido_autoral():
    """O LLM trocava 'Nota criada em X.md' por uma analise do tema."""
    class LLMAnalista:
        def invoke(self, _m):
            class R:
                content = ("A análise dos resultados de RUL consolida a "
                           "viabilidade metodológica da dissertação...")
            return R()

    ok = {"ok": True, "acao_executada": True, "resposta_pronta": True,
          "mensagem": "Nota criada no cérebro: **Resultado RUL** (`x.md`)."}
    for pedido in ("guarde esse resultado no cérebro",
                   "na sua opinião, guarde esse resultado"):
        saida = fr.comentar_resultado(pedido, ok, "", LLMAnalista())
        assert "Nota criada no cérebro" in saida, pedido
        assert "viabilidade metodológica" not in saida, pedido


def test_registro_marca_acao_executada(tmp_path, monkeypatch):
    import src.conhecimento.nota_cerebro as nc
    monkeypatch.setattr(nc, "PASTA_CEREBRO_OBSIDIAN", tmp_path)
    r = fr.registrar_no_cerebro(
        pergunta="guarde no cérebro", llm=_LLMRedator("N"),
        contexto="Al IAdo PV: censura caiu de 70% para 13%.")
    assert r["ok"] and r.get("acao_executada") is True
