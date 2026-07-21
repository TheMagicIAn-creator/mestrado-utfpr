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
_PADRAO_PAGINA = re.compile(r"\bp(?:á|a)?g?\.?\s*\d+", re.I)


def _norm(texto: str) -> str:
    return re.sub(r"[\s\-:]", "", str(texto or "")).upper()


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
