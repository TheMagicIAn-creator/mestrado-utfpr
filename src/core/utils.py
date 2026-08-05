"""
utils.py — Al IAdo PV
Funções utilitárias compartilhadas entre os módulos.

Autor: Rodolfo Torres (UTFPR)
"""

import sys
from pathlib import Path


def to_project_relative_path(caminho) -> str:
    """
    Converte um caminho (absoluto ou relativo) em caminho RELATIVO à raiz do
    projeto, em formato POSIX (com '/'), para gravar em JSONs portáveis.

    Caminhos fora da árvore do projeto são mantidos como estão (raro). Use
    `resolve_project_path` para reconstruir o caminho absoluto na interface.
    """
    from src.core.config import RAIZ_PROJETO

    try:
        p = Path(caminho).resolve()
    except (OSError, ValueError):
        return str(caminho)
    try:
        return p.relative_to(Path(RAIZ_PROJETO).resolve()).as_posix()
    except ValueError:
        # fora do projeto — devolve POSIX sem forçar relatividade
        return p.as_posix()


def resolve_project_path(caminho_relativo) -> Path:
    """
    Resolve um caminho relativo ao projeto para absoluto (uso na interface).
    Caminho já absoluto é devolvido como Path inalterado.
    """
    from src.core.config import RAIZ_PROJETO

    p = Path(caminho_relativo)
    return p if p.is_absolute() else (Path(RAIZ_PROJETO) / p)


def configurar_saida_utf8() -> bool:
    """
    Torna stdout/stderr à prova de emoji no Windows.

    O console do Windows (PyCharm, cmd) usa cp1252 por padrão, que NÃO
    codifica emoji (🟢🔵✅🤖…). Qualquer `print()` com emoji estoura
    `UnicodeEncodeError` e, quando isso acontece dentro da inicialização
    do provedor/agente, derruba a subida do app.

    A correção reconfigura os fluxos para UTF-8 com `errors="replace"`:
    além de exibir os emojis quando o terminal suporta, garante que um
    caractere não-codificável NUNCA mais derrube um `print()` (vira um
    caractere de substituição em vez de exceção).

    Idempotente e silenciosa: se o fluxo não suportar `reconfigure`
    (ex.: já encapsulado), apenas ignora.
    """
    configurados = 0
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
            configurados += 1
        except (AttributeError, OSError, ValueError):
            # Fluxo sem reconfigure (ou já configurado) — segue sem travar.
            continue
    return configurados == 2


def parsear_nome_arquivo(nome_arquivo: str) -> dict:
    """
    Parseia o nome do arquivo no padrão autor_titulo_ano.pdf
    e retorna um dicionário com os campos formatados.

    Exemplos:
      carpinetti-l_fmea-ingles-failure-mode-effect_2016.pdf
      → autor : "Carpinetti L"
      → titulo: "Fmea Ingles Failure Mode Effect"
      → ano   : "2016"
      → citacao: "Carpinetti L (2016) — Fmea Ingles Failure Mode Effect"

      autor-desconhecido_analise-confiabilidade_0000.pdf
      → autor : "Autor Desconhecido"
      → titulo: "Analise Confiabilidade"
      → ano   : "s.d."
      → citacao: "Autor Desconhecido (s.d.) — Analise Confiabilidade"
    """

    # Remove a extensão .pdf
    nome = nome_arquivo.replace(".pdf", "")

    # Divide pelo underscore
    partes = nome.split("_")

    # Identifica o ano — parte numérica de 4 dígitos
    ano            = "s.d."
    partes_validas = []

    for parte in partes:
        if parte.isdigit() and len(parte) == 4:
            if parte != "0000":
                ano = parte
            # 0000 = ano desconhecido → mantém "s.d."
        else:
            partes_validas.append(parte)

    # Primeira parte válida = autor
    # Restante = título
    if len(partes_validas) >= 2:
        autor_raw = partes_validas[0]
        titulo_raw = " ".join(partes_validas[1:])
    elif len(partes_validas) == 1:
        autor_raw  = partes_validas[0]
        titulo_raw = ""
    else:
        autor_raw  = nome
        titulo_raw = ""

    # Formata: troca hífen por espaço e aplica Title Case
    autor  = autor_raw.replace("-", " ").title()
    titulo = titulo_raw.replace("-", " ").title()

    # Monta a citação formatada
    if titulo:
        citacao = f"{autor} ({ano}) — {titulo}"
    else:
        citacao = f"{autor} ({ano})"

    return {
        "autor"  : autor,
        "titulo" : titulo,
        "ano"    : ano,
        "citacao": citacao
    }
