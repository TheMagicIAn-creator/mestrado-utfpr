"""
citacao_guarda.py — Al IAdo PV

Trava ESTRUTURAL contra citação sem lastro (defesa além do prompt).

Motivação: só instruir o LLM ("não invente página") não segura 100% — sob
pressão ele fabrica citações confiantes, sobretudo de NORMAS técnicas que não
estão indexadas (ex.: "IEC 60812:2018, Cláusula 7.3.3, p. 27" com aspas da
norma). Esta função compara a RESPOSTA com as fontes REALMENTE recuperadas e,
em dois casos de alta precisão, devolve um aviso a ser anexado à resposta:

  1) A resposta cita uma norma (IEC/ISO/IEEE/ABNT/NBR/...) que NÃO aparece nas
     fontes recuperadas → páginas/cláusulas dessa norma não são verificáveis.
  2) NENHUMA fonte foi recuperada, mas a resposta cita "p. N" → citação sem
     ancoragem na base.

É conservadora de propósito (não tenta casar toda página, o que geraria falso
positivo): mira só os padrões de fabricação mais graves. Lógica pura, testável.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import re

# Normas técnicas: quase nunca indexadas (PDFs pagos). Citar com página/cláusula
# sem a fonte no rodapé é fabricação.
_PADRAO_NORMA = re.compile(
    r"\b(?:IEC|ISO|IEEE|ABNT|NBR|ASTM|DIN|EN|MIL-STD|MIL-HDBK|SAE|API)\s*[-:]?\s*\d{2,6}",
    re.I,
)
# Página de citação: exige separador real ("p. 27", "pág 27", "pagina 27",
# "page 27"). NÃO casa "p99"/"p95" (percentil) nem "p1"/"p2" (pontos) — que são
# notação técnica, não referência de página. Evita falso positivo do guard.
_PADRAO_PAGINA = re.compile(
    r"\b(?:p\.\s*|p[áa]gs?\.?\s+|p[áa]ginas?\s+|pages?\s+)\d+", re.I
)


def _norm(texto: str) -> str:
    return re.sub(r"[\s\-:]", "", str(texto or "")).upper()


def _rotulo_curto(entrada: str) -> str:
    """Extrai 'Autor (ano) - ... - p. N' de uma entrada de citacao, sem o trecho."""
    s = str(entrada or "")
    for sep in (" — trecho", " - trecho", "— trecho", "- trecho", "trecho:", "Trecho"):
        i = s.find(sep)
        if i != -1:
            s = s[:i]
            break
    return re.sub(r"\s+", " ", s).strip(" —-:·")


def montar_restricao_fontes(citacoes) -> str:
    """Bloco de instrucao que RESTRINGE o que o LLM pode citar ao MESMO conjunto
    que vai para o rodape — corrige a incoerencia em que a prosa cita uma fonte
    que o rodape (filtrado pelo auditor apos montar o contexto) nao mostra.

    Sem fontes utilizaveis -> proibe qualquer citacao (mata a fabricacao de
    normas/paginas quando a busca vem vazia ou off-topic).
    """
    valores = list(citacoes.values()) if isinstance(citacoes, dict) else list(citacoes or [])
    rotulos = [r for r in (_rotulo_curto(v) for v in valores) if r]

    if not rotulos:
        return (
            "FONTES DISPONIVEIS PARA CITAR NESTA RESPOSTA: NENHUMA. A busca nao "
            "trouxe fonte utilizavel para esta pergunta. NAO cite autor, ano, "
            "pagina, clausula ou norma, e NAO use aspas de nenhuma fonte. Diga com "
            "franqueza que nao localizou uma fonte on-topic e ofereca refazer a "
            "busca (ou buscar na web). Qualquer citacao aqui seria invencao."
        )

    linhas = "\n".join(f"- {r}" for r in rotulos)
    return (
        "FONTES QUE VOCE PODE CITAR (use EXATAMENTE estas, com a pagina indicada; "
        "NAO cite nenhuma outra fonte, pagina, clausula ou norma, mesmo que "
        "apareca no contexto acima ou que voce 'conheca' de fora):\n"
        f"{linhas}\n"
        "Esta lista e so para seu controle — NAO a reproduza como secao de "
        "referencias. Se nenhuma delas sustentar uma afirmacao, diga que a busca "
        "nao trouxe fonte para aquele ponto, em vez de preencher com pagina inventada."
    )


def _texto_das_citacoes(citacoes) -> str:
    if not citacoes:
        return ""
    valores = citacoes.values() if isinstance(citacoes, dict) else citacoes
    return " ".join(str(v) for v in (valores or []))


def alerta_citacao_infundada(resposta: str, citacoes) -> str:
    """Retorna um aviso Markdown (ou '' quando não há nada a sinalizar).

    O aviso deve ser ANEXADO à resposta exibida/armazenada, para o pesquisador
    nunca tomar uma citação fabricada como verificada.
    """
    resposta = str(resposta or "")
    if not resposta.strip():
        return ""
    fontes_txt = _texto_das_citacoes(citacoes)
    fontes_norm = _norm(fontes_txt)
    alertas = []

    # 1) Normas citadas que NÃO aparecem nas fontes recuperadas.
    normas = {m.group(0) for m in _PADRAO_NORMA.finditer(resposta)}
    infundadas = sorted({re.sub(r"\s+", " ", n).strip()
                         for n in normas if _norm(n) not in fontes_norm})
    if infundadas:
        alertas.append(
            "Normas técnicas citadas acima (" + ", ".join(infundadas[:3])
            + ") NÃO constam nas fontes recuperadas desta busca — trate suas "
            "páginas/cláusulas como NÃO verificadas (podem ter sido inventadas)."
        )

    # 2) Sem fonte recuperada, mas há citação com página → não ancorado.
    if not fontes_txt.strip() and _PADRAO_PAGINA.search(resposta):
        alertas.append(
            "Nenhuma fonte foi recuperada para esta pergunta, mas o texto cita "
            "páginas — essas citações não estão ancoradas na base. Confirme na "
            "fonte primária antes de usar."
        )

    if not alertas:
        return ""
    return "\n\n> ⚠️ **Verificação de citações:** " + " ".join(alertas)
