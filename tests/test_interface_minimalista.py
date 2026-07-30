"""Disciplina visual da interface (repaginação minimalista, 2026-07).

A interface não tinha teste algum: cada ajuste no Streamlit era verificado
abrindo o app. Aqui um duplo de teste registra as chamadas de UI, o que
permite afirmar coisas como "a tela inicial tem no máximo um bloco de texto"
sem precisar do pacote instalado.

O que estes testes protegem (as regras que motivaram a repaginação):
  - a tela inicial NÃO volta a abrir com parágrafo de instruções;
  - a barra lateral NÃO volta a empilhar caixas coloridas e métricas grandes
    para dizer que está tudo bem;
  - mas a FALHA continua ganhando espaço — inclusive quando um alvo da
    persistência falha e outro tem sucesso no mesmo turno.
"""

from __future__ import annotations

import sys
import types

import pytest


# ── duplo de teste do Streamlit ──────────────────────────────────────────────

class _Rerun(Exception):
    """Emula o st.rerun(), que interrompe o script no Streamlit real."""


class _Registro:
    def __init__(self):
        self.chamadas: list[tuple[str, tuple, dict]] = []

    def nomes(self) -> list[str]:
        return [nome for nome, _a, _k in self.chamadas]

    def rotulos(self, sufixo: str) -> list[str]:
        return [str(a[0]) for nome, a, _k in self.chamadas
                if nome.endswith(sufixo) and a]

    def html(self) -> str:
        return " ".join(str(a[0]) for nome, a, _k in self.chamadas
                        if nome.endswith("markdown") and a)


class _Ctx:
    """Coluna, expander, sidebar, popover, status — todos delegam ao registro."""

    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def __getattr__(self, nome):
        return getattr(self._st, nome, None) or self._st._fn(f"col.{nome}")


class _Estado(dict):
    def __getattr__(self, chave):
        return self.get(chave)

    def __setattr__(self, chave, valor):
        self[chave] = valor


class _StreamlitFalso(types.ModuleType):
    def __init__(self, botao_verdadeiro: str | None = None):
        super().__init__("streamlit")
        self.registro = _Registro()
        self.botao_verdadeiro = botao_verdadeiro
        self.session_state = _Estado()
        self.sidebar = _Ctx(self)

    def _fn(self, nome):
        def chamada(*args, **kwargs):
            self.registro.chamadas.append((nome, args, kwargs))
            curto = nome.split(".")[-1]
            if curto == "rerun":
                raise _Rerun
            if curto in {"button", "download_button"}:
                return bool(args) and args[0] == self.botao_verdadeiro
            if curto == "columns":
                spec = args[0] if args else 1
                n = spec if isinstance(spec, int) else len(spec)
                return [_Ctx(self) for _ in range(n)]
            if curto in {"expander", "popover", "spinner", "status",
                         "container", "empty", "chat_message", "form",
                         "sidebar"}:
                return _Ctx(self)
            return None
        return chamada

    def __getattr__(self, nome):
        if nome.startswith("_"):
            raise AttributeError(nome)
        return self._fn(nome)


_STUBS = {
    "src.conhecimento.agente": {
        "_saudacao_pelo_horario": lambda: "Boa tarde",
        "remover_bloco_fontes_llm": lambda t: t,
    },
    "src.ml.pipeline": {
        "capacidade_recalculo_pipeline": lambda: {"disponivel": True},
        "NOMES_ETAPAS": {}, "estado_pipeline": lambda: {},
        "estado_resultados_publicados": lambda: {},
    },
    "src.conhecimento.obsidian": {"contar_notas_indexadas": lambda _c: 248},
}


class _Colecao:
    def __init__(self, n=39):
        self._n = n

    def count(self):
        return self._n


class _Equipe:
    class memoria:
        @staticmethod
        def contar():
            return 12


@pytest.fixture
def ui(request):
    """(modulo_da_interface, streamlit_falso) com sys.modules restaurado."""
    botao = getattr(request, "param", None)
    st = _StreamlitFalso(botao_verdadeiro=botao)

    lc = types.ModuleType("langchain_core")
    msgs = types.ModuleType("langchain_core.messages")
    msgs.HumanMessage = lambda **kw: kw
    lc.messages = msgs

    alvos = {"streamlit": st, "langchain_core": lc,
             "langchain_core.messages": msgs,
             "src.interface.streamlit_app": None,
             # Trocados pelos helpers _persistencia()/_ferramentas() dentro dos
             # testes. Precisam ser restaurados: outros módulos de teste
             # importam os reais (test_roteamento_fluido, test_nota_cerebro,
             # test_persistencia_nuvem) e receberiam o dublê.
             "src.conhecimento.ferramentas": sys.modules.get(
                 "src.conhecimento.ferramentas"),
             "src.conhecimento.persistencia_nuvem": sys.modules.get(
                 "src.conhecimento.persistencia_nuvem")}
    for nome, attrs in _STUBS.items():
        mod = types.ModuleType(nome)
        mod.__dict__.update(attrs)
        alvos[nome] = mod

    anteriores = {nome: sys.modules.get(nome) for nome in alvos}
    for nome, mod in alvos.items():
        if mod is None:
            sys.modules.pop(nome, None)
        else:
            sys.modules[nome] = mod

    import importlib

    app = importlib.import_module("src.interface.streamlit_app")
    try:
        yield app, st
    finally:
        for nome, anterior in anteriores.items():
            if anterior is None:
                sys.modules.pop(nome, None)
            else:
                sys.modules[nome] = anterior


def _persistencia(diag):
    mod = types.ModuleType("src.conhecimento.persistencia_nuvem")
    mod.diagnostico = lambda: diag
    sys.modules["src.conhecimento.persistencia_nuvem"] = mod


_SAUDAVEL = {"ativa": True, "resumo": "", "detalhe": "",
             "por_alvo": {"memoria": {"rotulo": "memória", "estado": "ok"}}}


# ── tela inicial ─────────────────────────────────────────────────────────────

def test_tela_inicial_nao_tem_paragrafo_de_instrucoes(ui):
    """Antes abria com um st.info explicativo + 5 exemplos em lista."""
    app, st = ui
    app.renderizar_boas_vindas()
    nomes = st.registro.nomes()
    assert "info" not in nomes, "a tela inicial não deve abrir com caixa de aviso"
    # exatamente um bloco de texto: o herói
    assert nomes.count("markdown") == 1
    assert len(st.registro.html()) < 400, "o herói deve caber em duas linhas"


def test_tela_inicial_oferece_tres_atalhos(ui):
    app, st = ui
    app.renderizar_boas_vindas()
    assert len(st.registro.rotulos("button")) == 3


@pytest.mark.parametrize("ui", ["Weibull e RUL"], indirect=True)
def test_atalho_envia_a_pergunta_completa_nao_o_rotulo(ui):
    """O botão mostra 3 palavras, mas o agente recebe o pedido inteiro."""
    app, st = ui
    with pytest.raises(_Rerun):
        app.renderizar_boas_vindas()
    pendente = st.session_state["pergunta_pendente"]
    assert pendente != "Weibull e RUL"
    assert "Weibull" in pendente and len(pendente) > 25


# ── barra lateral ────────────────────────────────────────────────────────────

def _sidebar(app, st, *, equipe=_Equipe(), colecao=None, mensagens=()):
    st.session_state["equipe"] = equipe
    st.session_state["mensagens"] = list(mensagens)
    col = colecao if colecao is not None else _Colecao()
    app.renderizar_sidebar(None, col, col, col, [])


def test_barra_lateral_saudavel_e_uma_lista_de_linhas(ui):
    """Saúde não pede caixa colorida, métrica grande nem divisor."""
    app, st = ui
    _persistencia(_SAUDAVEL)
    _sidebar(app, st)
    nomes = set(st.registro.nomes())
    assert not (nomes & {"success", "warning", "error", "metric", "divider"}), nomes
    assert st.registro.rotulos("expander") == ["Documentos", "Manutenção", "Diagnóstico"]


def test_limpar_conversa_so_aparece_com_conversa(ui):
    app, st = ui
    _persistencia(_SAUDAVEL)
    _sidebar(app, st)
    assert "Limpar conversa" not in st.registro.rotulos("button")

    st.registro.chamadas.clear()
    _sidebar(app, st, mensagens=[{"role": "user", "content": "oi"}])
    assert "Limpar conversa" in st.registro.rotulos("button")


def test_equipe_desconectada_ganha_ponto_vermelho_e_botao(ui):
    app, st = ui
    _persistencia(_SAUDAVEL)
    st.session_state["erro_equipe"] = "GOOGLE_API_KEY ausente"
    _sidebar(app, st, equipe=None)
    assert app._CORES_ESTADO["erro"] in st.registro.html()
    assert "Ativar equipe" in st.registro.rotulos("button")


def test_base_vazia_oferece_indexacao(ui):
    app, st = ui
    _persistencia(_SAUDAVEL)
    _sidebar(app, st, colecao=_Colecao(0))
    assert "Indexar literatura" in st.registro.rotulos("button")


# ── persistência: o silêncio já custou uma memória ───────────────────────────

def test_falha_em_um_alvo_nao_e_mascarada_pelo_alvo_que_funcionou(ui):
    app, _st = ui
    _persistencia({"ativa": True, "resumo": "", "detalhe": "", "por_alvo": {
        "memoria": {"rotulo": "memória", "estado": "ok"},
        "sessao": {"rotulo": "sessão", "estado": "erro", "detalhe": "403"},
    }})
    nivel, rotulo, detalhe = app._estado_persistencia()
    assert nivel == "erro" and "FALHANDO" in rotulo
    assert "sessão" in detalhe and "403" in detalhe


def test_persistencia_desligada_explica_a_consequencia(ui):
    app, _st = ui
    _persistencia({"ativa": False, "resumo": "sem token",
                   "detalhe": "Defina GITHUB_TOKEN."})
    nivel, _rotulo, detalhe = app._estado_persistencia()
    assert nivel == "alerta"
    assert "reboot" in detalhe.lower()


def test_diagnostico_quebrado_nao_derruba_a_barra(ui):
    app, _st = ui
    mod = types.ModuleType("src.conhecimento.persistencia_nuvem")

    def explode():
        raise RuntimeError("token inválido")

    mod.diagnostico = explode
    sys.modules["src.conhecimento.persistencia_nuvem"] = mod
    nivel, rotulo, detalhe = app._estado_persistencia()
    assert nivel == "alerta" and "indisponível" in rotulo
    assert detalhe == "RuntimeError"


# ── o corpo é só a conversa ──────────────────────────────────────────────────

def test_sem_cabecalho_no_corpo_da_pagina(ui):
    app, _st = ui
    assert not hasattr(app, "renderizar_topo"), (
        "o cabeçalho duplicava a identidade da barra lateral"
    )


# ── animação de espera ───────────────────────────────────────────────────────

class _Placeholder:
    """Registra o que foi escrito, na ordem, com os kwargs de cada escrita."""

    def __init__(self):
        self.escritas: list[tuple[str, dict]] = []
        self.limpezas = 0

    def markdown(self, corpo, **kwargs):
        self.escritas.append((str(corpo), kwargs))

    def empty(self):
        self.limpezas += 1


class _LLMStream:
    def __init__(self, pedacos=("Olá", " Rodolfo"), erro=None):
        self.pedacos, self.erro = pedacos, erro

    def stream(self, _mensagens):
        if self.erro:
            raise self.erro
        for p in self.pedacos:
            yield type("C", (), {"content": p})()


@pytest.fixture
def _sem_pausa(ui, monkeypatch):
    """Zera a cadência de digitação para o teste não dormir."""
    app, _st = ui
    monkeypatch.setattr(app, "VELOCIDADE_DIGITACAO", 0.0)
    return app


def test_brilho_aparece_antes_do_primeiro_token(_sem_pausa):
    app = _sem_pausa
    ph = _Placeholder()
    app.stream_resposta_limpa("prompt", _LLMStream(), ph, "")

    primeira, kwargs = ph.escritas[0]
    assert "alp-pensando" in primeira, "a espera deve começar com o brilho"
    assert kwargs.get("unsafe_allow_html") is True


def test_primeiro_token_substitui_o_brilho(_sem_pausa):
    app = _sem_pausa
    ph = _Placeholder()
    final = app.stream_resposta_limpa("prompt", _LLMStream(), ph, "")

    assert "alp-pensando" not in ph.escritas[-1][0]
    assert "Olá" in final and "Rodolfo" in final
    # o brilho aparece uma única vez, no início
    assert sum(1 for corpo, _k in ph.escritas if "alp-pensando" in corpo) == 1


def test_texto_do_modelo_nunca_e_renderizado_como_html(_sem_pausa):
    """O brilho é string nossa; a saída do LLM segue em Markdown escapado.

    Habilitar HTML no streaming abriria a resposta do modelo para injeção —
    o brilho não pode custar isso.
    """
    app = _sem_pausa
    ph = _Placeholder()
    app.stream_resposta_limpa("prompt", _LLMStream(("<script>alert(1)</script>",)),
                              ph, "")
    for corpo, kwargs in ph.escritas:
        if kwargs.get("unsafe_allow_html"):
            assert "alp-pensando" in corpo, (
                f"HTML habilitado para conteúdo não-nosso: {corpo[:60]!r}"
            )


def test_html_de_espera_e_montado_a_partir_de_string_nossa(ui):
    app, _st = ui
    assert "alp-pensando" in app._html_pensando()
    assert app._html_pensando("Interpretando o pedido").endswith("…</span>")


def test_animacao_respeita_prefers_reduced_motion(ui):
    app, _st = ui
    css = app._CSS_MINIMO
    assert "@keyframes alp-brilho" in css
    assert "prefers-reduced-motion" in css


def _ferramentas(decidir, progressos=(), erro=None):
    """Dublê do módulo de ferramentas.

    `progressos` são as mensagens que a ferramenta emitiria enquanto trabalha;
    `erro` faz a execução levantar, para checar a limpeza do balão de espera.
    """
    mod = types.ModuleType("src.conhecimento.ferramentas")
    mod.decidir_acao = decidir

    def _processar(**kw):
        for msg in progressos:
            kw["progresso"](msg)
        if erro:
            raise erro
        return {"resposta": "pronto", "resultado": {"ok": True, "imagens": []}}

    mod.processar_com_ferramentas = _processar
    sys.modules["src.conhecimento.ferramentas"] = mod


def test_roteamento_mostra_brilho_e_o_descarta(ui):
    """Sem o descarte, o balão de espera ficaria pulsando sob a resposta."""
    app, st = ui
    _ferramentas(lambda _p, _llm: {"usar_ferramenta": False, "ferramenta": None})
    st.session_state["mensagens"] = []

    app.responder_com_ferramenta("e aí?", "", object())
    nomes = st.registro.nomes()
    assert "alp-pensando" in st.registro.html(), "faltou o brilho na espera"
    assert nomes.count("empty") >= 2, "o balão de espera não foi descartado"


def test_brilho_e_descartado_mesmo_se_o_roteamento_falhar(ui):
    app, st = ui

    def explode(_p, _llm):
        raise RuntimeError("LLM fora do ar")

    _ferramentas(explode)
    st.session_state["mensagens"] = []

    with pytest.raises(RuntimeError):
        app.responder_com_ferramenta("e aí?", "", object())
    assert st.registro.nomes().count("empty") >= 2, (
        "o finally deve limpar a espera também no caminho de erro"
    )


# ── a máquina não aparece ────────────────────────────────────────────────────

_USA_FERRAMENTA = lambda _p, _llm: {"usar_ferramenta": True, "ferramenta": "x"}  # noqa: E731


def test_sem_caixa_de_execucao(ui):
    """Era st.status("Executando solicitação...") com o log rolando dentro."""
    app, st = ui
    _ferramentas(_USA_FERRAMENTA, progressos=["Lendo artefatos de resultado..."])
    st.session_state["mensagens"] = []

    app.responder_com_ferramenta("mostre os resultados", "", object())
    nomes = st.registro.nomes()
    assert "status" not in nomes, "a caixa de execução não deve mais existir"
    assert not any("Executando solicitação" in str(a)
                   for _n, a, _k in st.registro.chamadas)


def test_progresso_vira_texto_do_brilho(ui):
    """A informação não se perde: cada etapa aparece na linha pulsante."""
    app, st = ui
    _ferramentas(_USA_FERRAMENTA,
                 progressos=["Treinando o classificador PV Farms (CC)...",
                             "Lendo artefatos de resultado..."])
    st.session_state["mensagens"] = []

    app.responder_com_ferramenta("treine o classificador", "", object())
    html = st.registro.html()
    assert "Treinando o classificador PV Farms (CC)" in html
    assert "Lendo artefatos de resultado" in html
    # cada mensagem é renderizada COMO brilho, não como texto solto
    for corpo, kwargs in [(a[0], k) for n, a, k in st.registro.chamadas
                          if n.endswith("markdown") and a]:
        if "Treinando o classificador" in str(corpo):
            assert "alp-pensando" in str(corpo) and kwargs.get("unsafe_allow_html")


def test_nome_interno_da_ferramenta_nao_vai_para_a_tela(ui):
    app, st = ui
    _ferramentas(_USA_FERRAMENTA, progressos=["Lendo status do pipeline..."])
    st.session_state["mensagens"] = []

    app.responder_com_ferramenta("como está o pipeline?", "", object())
    assert "Acionando ferramenta" not in st.registro.html()


def test_brilho_sai_de_cena_se_a_ferramenta_levantar(ui):
    """Sem o finally, o balão pulsaria para sempre acima do erro."""
    app, st = ui
    _ferramentas(_USA_FERRAMENTA, progressos=["Rodando..."],
                 erro=RuntimeError("pipeline quebrou"))
    st.session_state["mensagens"] = []

    with pytest.raises(RuntimeError):
        app.responder_com_ferramenta("rode o pipeline", "", object())
    assert st.registro.nomes().count("empty") >= 2


def test_progresso_vazio_nao_apaga_o_rotulo(ui):
    app, st = ui
    _ferramentas(_USA_FERRAMENTA, progressos=["   ", "..."])
    st.session_state["mensagens"] = []

    app.responder_com_ferramenta("faça algo", "", object())
    assert "Trabalhando nisso" in st.registro.html()
