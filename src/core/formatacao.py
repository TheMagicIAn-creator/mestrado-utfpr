"""
formatacao.py - Al IAdo PV
Formatação canônica de números e tabelas Markdown do chat.

Ponto único de verdade: TODA tabela e TODO número exibidos ao usuário
passam por aqui, para que resultados de módulos diferentes cheguem ao
chat com a mesma cara. Política de casas decimais:

- métricas 0–1 (AUC, F1, recall, precision) → 3 casas
- limiares e erros de reconstrução          → 4 casas
- parâmetros físicos/Weibull (eta, MTTF)    → 1 casa
- percentuais                               → 1 casa + "%"
- p-valores                                 → p=0.0014 ou p<0.0001
"""

from __future__ import annotations


def fmt_num(valor, casas: int = 3) -> str:
    """Número com casas fixas; qualquer coisa não numérica vira '-'."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return "-"
    return f"{valor:.{casas}f}"


def fmt_metrica(valor) -> str:
    """Métricas 0–1 (AUC, F1, recall...): 3 casas."""
    return fmt_num(valor, 3)


def fmt_limiar(valor) -> str:
    """Limiares e erros de reconstrução: 4 casas."""
    return fmt_num(valor, 4)


def fmt_fisico(valor) -> str:
    """Grandezas físicas/Weibull (eta, MTTF, B10): 1 casa."""
    return fmt_num(valor, 1)


def fmt_pct(valor, casas: int = 1) -> str:
    """Percentual já em escala 0–100."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return "-"
    return f"{valor:.{casas}f}%"


def fmt_pvalor(p) -> str:
    """Convenção acadêmica: p<0.0001 abaixo do limiar de exibição."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return "-"
    if p < 0.0001:
        return "p<0.0001"
    return f"p={p:.4f}"


def tabela_markdown(cabecalhos: list[str], linhas: list[list],
                    alinhamentos: list[str] | None = None) -> str:
    """
    Tabela Markdown uniforme.

    ``alinhamentos``: lista com "e" (esquerda) ou "d" (direita) por coluna.
    Se omitida, a primeira coluna alinha à esquerda e as demais à direita
    (padrão para tabelas nome-da-linha + valores numéricos).
    Valores None viram '-'; números NÃO são reformatados aqui — formate
    antes com os fmt_* para manter a política de casas decimais.
    """
    n = len(cabecalhos)
    if alinhamentos is None:
        alinhamentos = ["e"] + ["d"] * (n - 1)
    marcas = {"e": "---", "d": "---:"}
    sep = "|" + "|".join(marcas.get(a, "---") for a in alinhamentos) + "|"

    partes = ["| " + " | ".join(str(c) for c in cabecalhos) + " |", sep]
    for linha in linhas:
        celulas = ["-" if v is None else str(v) for v in linha]
        celulas += ["-"] * (n - len(celulas))  # completa linhas curtas
        partes.append("| " + " | ".join(celulas[:n]) + " |")
    return "\n".join(partes) + "\n"
