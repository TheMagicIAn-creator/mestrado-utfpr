"""
conversa_export.py — Al IAdo PV

Exportação do histórico da conversa do chat para um arquivo .txt baixável.

Lógica PURA (stdlib apenas, sem Streamlit): detecção da intenção e montagem do
transcrito completo. O botão de download em si é responsabilidade da interface
(src/interface/streamlit_app.py), que tem acesso a `st.session_state` e ao
`st.download_button`. Manter a lógica aqui a torna testável sem carregar a UI.

Motivação: sem isto, um pedido como "gere um .txt do histórico" ia ao LLM, que
NÃO tem como criar arquivos — então ele apenas *afirmava* ter gerado (alucinação)
e não trazia o conteúdo. Agora a interface intercepta o pedido, monta o
transcrito real e oferece o download, sem passar pelo modelo.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import re
import unicodedata


def _normalizar(texto: str) -> str:
    base = unicodedata.normalize("NFKD", str(texto or "").lower())
    sem_acentos = "".join(c for c in base if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem_acentos).strip()


# Alvos que indicam "a conversa/o histórico" como objeto do pedido.
_ALVOS = (
    "conversa", "historico", "transcricao", "transcript", "chat", "dialogo",
    "bate papo", "bate-papo", "nossa troca", "essa sessao", "esta sessao",
    "conversation", "history",
)

# Verbos/substantivos que indicam querer um ARQUIVO/exportação para baixar.
_ACOES_ARQUIVO = (
    "arquivo", "txt", ".txt", ".md", "documento", "download", "baixar",
    "baixe", "exporte", "exportar", "exporta", "salve em", "salvar em",
    "gere um arquivo", "gerar um arquivo", "gera um arquivo", "gere o arquivo",
    "em arquivo", "para arquivo", "file", "export",
)


def quer_exportar_conversa(pergunta: str) -> bool:
    """True quando o pedido é para EXPORTAR a conversa em arquivo baixável.

    Exige os dois sinais juntos: falar da conversa/histórico E pedir arquivo/
    exportação — assim "qual o histórico do inversor?" (só alvo) e "gere um
    arquivo de resultados" (só ação) não disparam por engano.
    """
    txt = _normalizar(pergunta)
    tem_alvo = any(a in txt for a in _ALVOS)
    tem_acao = any(a in txt for a in _ACOES_ARQUIVO)
    return tem_alvo and tem_acao


_ROTULOS = {"user": "Rodolfo", "assistant": "Al IAdo PV"}


def montar_transcricao(mensagens, *, exportado_em: str = "") -> str:
    """Monta o transcrito COMPLETO da conversa a partir das mensagens do chat.

    `mensagens` é a lista de {role, content, imagens?} de `st.session_state`.
    `exportado_em` é um carimbo de data/hora já formatado (a interface passa o
    horário de São Paulo) — mantido como parâmetro para esta função ser pura.
    """
    mensagens = list(mensagens or [])
    trocas = sum(1 for m in mensagens if m.get("role") == "user")

    linhas = [
        "=" * 64,
        "HISTÓRICO DA CONVERSA — Al IAdo PV (assistente do mestrado, UTFPR)",
    ]
    if exportado_em:
        linhas.append(f"Exportado em: {exportado_em}")
    linhas.append(f"Total de trocas (perguntas do pesquisador): {trocas}")
    linhas.append("=" * 64)
    linhas.append("")

    if not mensagens:
        linhas.append("(Ainda não há mensagens nesta conversa para exportar.)")
        return "\n".join(linhas) + "\n"

    for msg in mensagens:
        quem = _ROTULOS.get(msg.get("role"), str(msg.get("role", "?")))
        conteudo = str(msg.get("content", "")).strip()
        linhas.append(f"{quem}:")
        linhas.append(conteudo if conteudo else "(sem conteúdo)")
        for img in (msg.get("imagens") or []):
            legenda = img.get("caption") or img.get("path") or "imagem"
            linhas.append(f"    [imagem exibida: {legenda}]")
        linhas.append("")
        linhas.append("-" * 64)
        linhas.append("")

    return "\n".join(linhas) + "\n"


def nome_arquivo_conversa(carimbo: str) -> str:
    """Nome de arquivo seguro para o .txt, a partir de um carimbo já formatado
    tipo '2026-07-21_14-30'. Sem depender de relógio (a interface passa a data).
    """
    seguro = re.sub(r"[^0-9A-Za-z_-]", "", carimbo.replace(" ", "_")) or "conversa"
    return f"conversa_al_iado_{seguro}.txt"
