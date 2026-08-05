"""
atalhos.py — Al IAdo PV

Registro ÚNICO e ordenado dos atalhos determinísticos: pedidos que têm resposta
exata e por isso não devem passar por LLM nenhum — nem pelo roteador de
ferramentas, nem pelo RAG.

Antes existiam cinco desses espalhados por dois arquivos e três funções:
exportar conversa e cofre de trechos no meio de `renderizar_chat`; inventário
do vault, consulta cronológica e saudação dentro de `responder_com_rag`.
Acrescentar um sexto significava enfiar mais um `if` no meio de uma função de
render — foi assim que a lista virou o emaranhado de gatilhos reclamado pelo
pesquisador.

Aqui a lista é uma lista. Cada atalho é uma função `(pergunta, ctx) -> Resposta
| None` que devolve None quando não é o caso dela; `resolver_atalho` percorre
na ordem e devolve o primeiro acerto. Detectar e responder ficam juntos porque
já era assim que as funções existiam (`responder_inventario_vault` já
retornava None) — separar em detectar/responder duplicaria o trabalho.

POR QUE DETERMINÍSTICO, e não roteado pelo LLM:
  - exportar conversa  → o LLM não cria arquivo; alucinaria "gerei o .txt";
  - cofre de trechos   → o recall precisa ser IDÊNTICO ao salvo, e o LLM
                         reescreveria o código do zero;
  - inventário         → contagem é fato, e busca semântica devolve amostra
                         (foi o que respondeu "4" para um vault de 26);
  - cronologia         → ordem por metadado, não por similaridade;
  - saudação           → "oi" não precisa acordar o modelo pesado.

ORDEM IMPORTA e é a da lista. Recuperar trecho vem antes de salvar porque
"guardei" contém "guarde".

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos

_logger = get_logger("conhecimento.atalhos")


@dataclass
class Resposta:
    """O que um atalho devolve. `anexo_txt` vira botão de download no chat."""

    texto: str
    anexo_txt: dict[str, str] | None = None
    origem: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


# ── atalhos ──────────────────────────────────────────────────────────────────

def _exportar_conversa(pergunta: str, ctx: dict) -> Resposta | None:
    from src.core.conversa_export import (
        montar_transcricao,
        nome_arquivo_conversa,
        quer_exportar_conversa,
    )
    from src.core.tempo import agora_local

    if not quer_exportar_conversa(pergunta):
        return None

    mensagens = ctx.get("mensagens") or []
    agora = agora_local()
    transcricao = montar_transcricao(
        mensagens,
        exportado_em=f"{agora:%d/%m/%Y às %H:%M} (America/Sao_Paulo)",
    )
    nome_arq = nome_arquivo_conversa(f"{agora:%Y-%m-%d_%H-%M}")
    trocas = sum(1 for m in mensagens if m.get("role") == "user")
    texto = (
        f"📄 Preparei o histórico completo desta conversa "
        f"({trocas} {'troca' if trocas == 1 else 'trocas'}) em **{nome_arq}**. "
        "Clique no botão abaixo para baixar — o texto traz cada mensagem na íntegra."
    )
    return Resposta(texto, {"data": transcricao, "file_name": nome_arq})


def _cofre_de_trechos(pergunta: str, ctx: dict) -> Resposta | None:
    from src.conhecimento import snippets as snp

    # RECUPERAR antes de SALVAR: 'guardei' contém 'guarde'.
    if snp.quer_recuperar_snippet(pergunta):
        registro = snp.recuperar_snippet(pergunta)
        if registro is None:
            return Resposta(
                "Não encontrei nenhum trecho salvo ainda. Depois que o código "
                "aparecer no chat, diga *'guarde este script'* que eu guardo "
                "idêntico para recuperar quando quiser."
            )
        return Resposta(
            snp.formatar_snippet_para_chat(registro, total=len(snp.carregar_snippets()))
        )

    if snp.quer_listar_snippets(pergunta):
        return Resposta(snp.formatar_lista_snippets(snp.carregar_snippets()))

    if not snp.quer_salvar_snippet(pergunta):
        return None

    bloco = snp.ultimo_bloco_codigo(pergunta, ctx.get("mensagens"))
    if not bloco:
        return Resposta(
            "Não achei nenhum bloco de código para guardar. Cole o código "
            "(entre ``` ```), ou peça primeiro que eu escreva o script e "
            "depois diga *'guarde este script'*."
        )
    try:
        reg = snp.salvar_snippet(bloco["codigo"], linguagem=bloco.get("linguagem", ""))
    except Exception as exc:  # noqa: BLE001 - a falha é reportada, não escondida
        return Resposta(f"Não consegui guardar o trecho ({type(exc).__name__}).")

    # Persiste no Git para sobreviver a reboot/consolidação. Best-effort: se
    # falhar, o trecho continua salvo em disco e a resposta segue verdadeira.
    try:
        from src.conhecimento.persistencia_nuvem import (
            persistencia_ativa,
            persistir_arquivo,
        )

        if persistencia_ativa():
            persistir_arquivo(
                snp.ARQUIVO_SNIPPETS,
                mensagem=f"chore(snippet): guarda trecho {reg['rotulo']}",
                alvo="snippet",
            )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "snippet salvo localmente, mas não persistido na nuvem: %s",
            mascarar_segredos(str(exc)),
        )

    n = len(reg["codigo"].splitlines())
    return Resposta(
        f"✅ Guardei o trecho como **{reg['rotulo']}** "
        f"({reg.get('linguagem') or 'texto'}, {n} linhas), **idêntico**. "
        f"Para recuperá-lo exatamente assim (mesmo após reboot), peça: "
        f"*'me manda o script {reg['rotulo']} que salvei'*."
    )


def _inventario_do_vault(pergunta: str, ctx: dict) -> Resposta | None:
    colecao = ctx.get("colecao_obsidian")
    if colecao is None:
        return None
    from src.conhecimento.obsidian import responder_inventario_vault

    texto = responder_inventario_vault(colecao, pergunta)
    return Resposta(texto) if texto else None


def _cronologia_do_vault(pergunta: str, ctx: dict) -> Resposta | None:
    colecao = ctx.get("colecao_obsidian")
    if colecao is None:
        return None
    from src.conhecimento.obsidian import responder_consulta_cronologica

    texto = responder_consulta_cronologica(colecao, pergunta)
    return Resposta(texto) if texto else None


def _interacao_simples(pergunta: str, _ctx: dict) -> Resposta | None:
    from src.conhecimento.agente import resposta_interacao_simples

    texto = resposta_interacao_simples(pergunta)
    return Resposta(texto) if texto else None


# A ordem é a precedência. Mudar a ordem muda o comportamento.
ATALHOS: tuple[tuple[str, Callable[[str, dict], Resposta | None]], ...] = (
    ("exportar_conversa", _exportar_conversa),
    ("cofre_de_trechos", _cofre_de_trechos),
    ("inventario_vault", _inventario_do_vault),
    ("cronologia_vault", _cronologia_do_vault),
    ("interacao_simples", _interacao_simples),
)


def resolver_atalho(pergunta: str, ctx: dict | None = None) -> Resposta | None:
    """Primeiro atalho que reconhecer o pedido, ou None para seguir o fluxo.

    Um atalho que levanta é PULADO, não propagado: defeito em um caminho
    secundário não pode derrubar a conversa inteira — o pedido segue para o
    roteador de ferramentas e para o RAG, que respondem de outro jeito.
    """
    if not (pergunta or "").strip():
        return None
    ctx = ctx or {}
    for nome, atalho in ATALHOS:
        try:
            resposta = atalho(pergunta, ctx)
        except Exception:  # noqa: BLE001
            continue
        if resposta is not None:
            resposta.origem = nome
            return resposta
    return None
