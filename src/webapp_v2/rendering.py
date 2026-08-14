"""Renderizacao segura das respostas academicas do agente."""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

_MARKDOWN = MarkdownIt(
    "commonmark",
    {
        "html": False,
        "linkify": False,
        "typographer": False,
    },
).enable(["table", "strikethrough"])


def render_agent_markdown(texto: str) -> str:
    """Converte Markdown em HTML sem aceitar HTML bruto fornecido pelo modelo."""
    texto = re.sub(r"<br\s*/?>", " ", str(texto or ""), flags=re.IGNORECASE)
    return _MARKDOWN.render(texto)
