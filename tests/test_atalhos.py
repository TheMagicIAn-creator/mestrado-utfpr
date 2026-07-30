"""Registro único dos atalhos determinísticos.

Antes eram cinco `if` espalhados por dois arquivos e três funções — exportar
conversa e cofre de trechos no meio de `renderizar_chat`, inventário do vault,
cronologia e saudação dentro de `responder_com_rag`. Acrescentar um sexto
significava enfiar mais um `if` no meio de uma função de render.

O que estes testes protegem: a ORDEM (que é a precedência), o isolamento de
falhas (um atalho quebrado não derruba a conversa) e o contrato de devolver
None para o pedido seguir o fluxo normal.
"""

from __future__ import annotations

import src.conhecimento.atalhos as at
from src.conhecimento.atalhos import Resposta, resolver_atalho


# ── mecânica do registro ─────────────────────────────────────────────────────

def test_devolve_none_quando_nenhum_atalho_reconhece():
    assert resolver_atalho("explique o beta da Weibull", {}) is None


def test_pergunta_vazia_nao_aciona_nada():
    for vazia in ("", "   ", "\n"):
        assert resolver_atalho(vazia, {}) is None


def test_primeiro_acerto_vence(monkeypatch):
    monkeypatch.setattr(at, "ATALHOS", (
        ("nunca", lambda _p, _c: None),
        ("primeiro", lambda _p, _c: Resposta("A")),
        ("segundo", lambda _p, _c: Resposta("B")),
    ))
    r = resolver_atalho("qualquer coisa", {})
    assert r.texto == "A" and r.origem == "primeiro"


def test_atalho_que_levanta_e_pulado_nao_propagado(monkeypatch):
    """Defeito num caminho secundário não pode derrubar a conversa inteira."""
    def explode(_p, _c):
        raise RuntimeError("índice corrompido")

    monkeypatch.setattr(at, "ATALHOS", (
        ("quebrado", explode),
        ("bom", lambda _p, _c: Resposta("segui em frente")),
    ))
    r = resolver_atalho("pergunta", {})
    assert r is not None and r.texto == "segui em frente"


def test_todos_quebrados_devolve_none(monkeypatch):
    def explode(_p, _c):
        raise RuntimeError("x")

    monkeypatch.setattr(at, "ATALHOS", (("a", explode), ("b", explode)))
    assert resolver_atalho("pergunta", {}) is None


def test_origem_identifica_qual_atalho_respondeu(monkeypatch):
    monkeypatch.setattr(at, "ATALHOS", (
        ("inventario_vault", lambda _p, _c: Resposta("26 memórias")),
    ))
    assert resolver_atalho("quantas memórias?", {}).origem == "inventario_vault"


# ── ordem declarada ──────────────────────────────────────────────────────────

def test_ordem_de_precedencia_e_a_documentada():
    nomes = [nome for nome, _fn in at.ATALHOS]
    assert nomes == [
        "exportar_conversa",
        "cofre_de_trechos",
        "inventario_vault",
        "cronologia_vault",
        "interacao_simples",
    ]


def test_inventario_vem_antes_da_cronologia():
    """'as últimas memórias' casa nos dois; contar é a resposta certa."""
    nomes = [nome for nome, _fn in at.ATALHOS]
    assert nomes.index("inventario_vault") < nomes.index("cronologia_vault")


# ── atalhos que dependem de contexto ─────────────────────────────────────────

def test_sem_colecao_o_inventario_nao_quebra():
    """Na inicialização o Obsidian pode não estar disponível."""
    assert at._inventario_do_vault("quantas memórias consolidadas?", {}) is None
    assert at._cronologia_do_vault("qual foi a primeira sessão?", {}) is None


def test_exportar_conversa_devolve_anexo():
    mensagens = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "olá"},
    ]
    r = at._exportar_conversa("exporte a conversa", {"mensagens": mensagens})
    assert r is not None
    assert r.anexo_txt and r.anexo_txt["file_name"].endswith(".txt")
    assert "oi" in r.anexo_txt["data"]
    assert "1 troca" in r.texto


def test_exportar_conversa_ignora_pedido_de_outra_coisa():
    assert at._exportar_conversa("rode o pipeline", {"mensagens": []}) is None


# ── cofre de trechos: a ordem interna também importa ─────────────────────────

def test_recuperar_vem_antes_de_salvar(monkeypatch):
    """'guardei' contém 'guarde' — inverter faria recuperar virar salvar."""
    from src.conhecimento import snippets as snp

    chamadas = []
    monkeypatch.setattr(snp, "quer_recuperar_snippet", lambda p: "guardei" in p)
    monkeypatch.setattr(snp, "recuperar_snippet",
                        lambda p: chamadas.append("recuperar") or {"rotulo": "s1"})
    monkeypatch.setattr(snp, "carregar_snippets", lambda: [{"rotulo": "s1"}])
    monkeypatch.setattr(snp, "formatar_snippet_para_chat",
                        lambda reg, total: "aqui está o trecho")
    monkeypatch.setattr(snp, "quer_salvar_snippet",
                        lambda p: chamadas.append("salvar") or True)

    r = at._cofre_de_trechos("me manda o script que guardei", {})
    assert r.texto == "aqui está o trecho"
    assert chamadas == ["recuperar"], "salvar não pode ser consultado antes"


def test_falha_ao_salvar_e_reportada_nao_engolida(monkeypatch):
    from src.conhecimento import snippets as snp

    monkeypatch.setattr(snp, "quer_recuperar_snippet", lambda _p: False)
    monkeypatch.setattr(snp, "quer_listar_snippets", lambda _p: False)
    monkeypatch.setattr(snp, "quer_salvar_snippet", lambda _p: True)
    monkeypatch.setattr(snp, "ultimo_bloco_codigo",
                        lambda _p, _m: {"codigo": "x = 1", "linguagem": "python"})

    def falha(*_a, **_k):
        raise OSError("disco cheio")

    monkeypatch.setattr(snp, "salvar_snippet", falha)
    r = at._cofre_de_trechos("guarde este script", {})
    assert "não consegui guardar" in r.texto.lower()
    assert "OSError" in r.texto


def test_a_interface_nao_chama_os_detectores_por_fora():
    """Sem isto, os `if` voltam a se espalhar pelo render — foi como chegamos aqui.

    A interface deve conhecer UMA porta (`resolver_atalho`). Se um detector
    reaparecer em streamlit_app.py, o registro deixou de ser o lugar único.
    """
    from pathlib import Path

    fonte = Path("src/interface/streamlit_app.py").read_text(encoding="utf-8")
    assert "resolver_atalho" in fonte, "a interface precisa usar o registro"
    for detector in (
        "quer_exportar_conversa",
        "responder_inventario_vault",
        "responder_consulta_cronologica",
        "resposta_interacao_simples",
        "quer_salvar_snippet",
        "quer_recuperar_snippet",
        "_tratar_snippet",
    ):
        assert detector not in fonte, (
            f"{detector} voltou para a interface — deve viver em atalhos.py"
        )


def test_sem_bloco_de_codigo_orienta_em_vez_de_falhar(monkeypatch):
    from src.conhecimento import snippets as snp

    monkeypatch.setattr(snp, "quer_recuperar_snippet", lambda _p: False)
    monkeypatch.setattr(snp, "quer_listar_snippets", lambda _p: False)
    monkeypatch.setattr(snp, "quer_salvar_snippet", lambda _p: True)
    monkeypatch.setattr(snp, "ultimo_bloco_codigo", lambda _p, _m: None)

    r = at._cofre_de_trechos("guarde este script", {"mensagens": []})
    assert "não achei nenhum bloco" in r.texto.lower()
