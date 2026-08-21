"""
nota_cerebro.py — Al IAdo PV

Permite ao agente REGISTRAR conhecimento curado no vault Obsidian — a peça que
faltava para ele usar o vault como repositório, e não só como fonte de leitura.

Antes, o agente só escrevia no vault de forma automática (sessões, memória
validada, consolidações). Não havia como pedir "guarde isto no cérebro" e ter
uma nota curada, tagueada e conectada ao painel. Esta ferramenta fecha o ciclo:

    ENTRADA (indexação) → CONSULTA (RAG) → SAÍDA (resposta) → **REGISTRO** (aqui)

A nota criada entra na coleção ``obsidian_pv`` na sincronização do turno
seguinte, ficando pesquisável como conhecimento curado.

Disciplina preservada (o registro NÃO afrouxa as regras do projeto):
  - nota curada nunca vira citação bibliográfica;
  - métrica continua vindo de artefato — o texto pode CITAR um número, mas o
    campo `nivel_evidencia` o qualifica (E1/E2/projeto);
  - as tags são validadas contra a taxonomia (os nós comuns da dissertação).

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

from src.core.config import PASTA_CEREBRO_OBSIDIAN

# Nós comuns da dissertação — ver notas/Cerebro/00 - Painel do cerebro.md
TAGS_VALIDAS = {
    "fmea", "fmeca", "rcm", "manutencao", "confiabilidade", "weibull-rul",
    "inversor-pv", "contator-ac", "igbt", "fusivel-ac", "autoencoder",
    "deteccao-anomalia", "escore-localizado", "machine-learning",
    "sinais-eletricos", "paderborn", "evidencia-e2", "comparacao-literatura",
    "metodologia", "cerebro", "vault", "dataset", "resultado",
}

TIPOS = {"conceito", "decisao", "resultado", "contexto", "hipotese", "experimento"}
NIVEIS = {"projeto", "E1", "E2", "literatura"}

# tipo → subpasta do Cerebro
SUBPASTA = {
    "conceito": "Conceitos",
    "decisao": "Decisoes",
    "resultado": "Resultados",
    "hipotese": "Conceitos",
    "experimento": "Resultados",
    "contexto": "",
}


def _sanitizar_titulo(titulo: str) -> str:
    """Título → nome de arquivo seguro, preservando acentos legíveis."""
    limpo = re.sub(r'[<>:"/\\|?*\n\r\t]', "", titulo).strip(" .")
    return (limpo or "Nota sem titulo")[:120]


def _normalizar_tag(tag: str) -> str:
    t = unicodedata.normalize("NFKD", str(tag).strip().lstrip("#").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9-]+", "-", t).strip("-")


def registrar_nota_cerebro(titulo: str, conteudo: str, tipo: str = "contexto",
                           tags: list[str] | None = None,
                           nivel_evidencia: str = "projeto",
                           fonte: str = "",
                           pasta_base: Path | None = None) -> dict:
    """Cria (ou atualiza) uma nota curada no `Cerebro/` do vault.

    titulo   : vira o nome do arquivo e o H1.
    conteudo : corpo em Markdown (sem frontmatter — ele é montado aqui).
    tipo     : conceito | decisao | resultado | contexto | hipotese | experimento.
    tags     : nós comuns da dissertação; inválidas são descartadas com aviso.
    nivel_evidencia : projeto | E1 | E2 | literatura.
    fonte    : arquivo/artefato de origem (ex.: resultados/comparacao/...json).

    Retorna {"ok", "caminho", "tags", "descartadas", "mensagem"}.
    """
    if not titulo or not titulo.strip():
        return {"ok": False, "mensagem": "Título vazio — a nota precisa de um título."}
    if not conteudo or not conteudo.strip():
        return {"ok": False, "mensagem": "Conteúdo vazio — nada a registrar."}

    tipo = (tipo or "contexto").strip().lower()
    if tipo not in TIPOS:
        tipo = "contexto"
    if nivel_evidencia not in NIVEIS:
        nivel_evidencia = "projeto"

    pedidas = [_normalizar_tag(t) for t in (tags or []) if str(t).strip()]
    validas = [t for t in dict.fromkeys(pedidas) if t in TAGS_VALIDAS]
    descartadas = [t for t in dict.fromkeys(pedidas) if t not in TAGS_VALIDAS]
    if "cerebro" not in validas:
        validas.insert(0, "cerebro")

    base = Path(pasta_base) if pasta_base else PASTA_CEREBRO_OBSIDIAN
    destino = base / SUBPASTA.get(tipo, "")
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{_sanitizar_titulo(titulo)}.md"

    corpo = conteudo.strip()
    if not corpo.lstrip().startswith("#"):
        corpo = f"# {titulo.strip()}\n\n{corpo}"

    linha_fonte = f"\nFonte: `{fonte}`\n" if fonte else ""
    frontmatter = (
        "---\n"
        "al_iado: true\n"
        f'titulo: "{titulo.strip()}"\n'
        f"tipo: {tipo}\n"
        "status: ativo\n"
        "confianca: media\n"
        f"nivel_evidencia: {nivel_evidencia}\n"
        f"registrado_em: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"tags: [{', '.join(validas)}]\n"
        "---\n\n"
    )
    rodape = (
        f"{linha_fonte}\n## Conexões\n\n- [[00 - Painel do cerebro]]\n\n"
        "> Nota curada registrada pelo Al IAdo PV. Não é citação bibliográfica;\n"
        "> métricas devem ser conferidas nos artefatos de `resultados/`.\n"
    )
    existia = caminho.exists()
    caminho.write_text(frontmatter + corpo + "\n" + rodape, encoding="utf-8")

    aviso = (f" Tags ignoradas (fora da taxonomia): {', '.join(descartadas)}."
             if descartadas else "")
    return {
        "ok": True,
        "caminho": str(caminho),
        "tags": validas,
        "descartadas": descartadas,
        "mensagem": (
            f"{'Nota atualizada' if existia else 'Nota criada'} no cérebro: "
            f"**{titulo.strip()}** (`{caminho.name}`), tipo *{tipo}*, "
            f"evidência *{nivel_evidencia}*, tags: {', '.join(validas)}.{aviso} "
            "Ela entra na busca do agente na próxima sincronização."
        ),
    }
