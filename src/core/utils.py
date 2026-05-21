"""
utils.py — Al IAdo PV
Funções utilitárias compartilhadas entre os módulos.

Autor: Rodolfo Torres (UTFPR)
"""


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