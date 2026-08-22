"""Renderizacao segura das respostas academicas do agente."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

from markdown_it import MarkdownIt

_MARKDOWN = MarkdownIt(
    "commonmark",
    {
        "html": False,
        "linkify": False,
        "typographer": False,
    },
).enable(["table", "strikethrough"])

MAX_RENDER_ITEMS = 100
MAX_RENDER_CHARS = 20_000
_MATH_EXPRESSION = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?!\$)[^\n$]+(?<!\\)\$)",
    flags=re.DOTALL,
)


def render_agent_markdown(texto: str) -> str:
    """Converte Markdown em HTML sem aceitar HTML bruto fornecido pelo modelo."""
    texto = re.sub(r"<br\s*/?>", " ", str(texto or ""), flags=re.IGNORECASE)
    expressions: dict[str, str] = {}

    def protect(match: re.Match) -> str:
        token = f"ALIADOMATHTOKEN{len(expressions):04d}END"
        expressions[token] = html.escape(match.group(0), quote=False)
        return token

    protected = _MATH_EXPRESSION.sub(protect, texto)
    rendered = _MARKDOWN.render(protected)
    for token, expression in expressions.items():
        rendered = rendered.replace(token, expression)
    return rendered


def render_agent_messages(items: Sequence[dict]) -> list[dict[str, str]]:
    """Renderiza um lote identificado sem aceitar HTML fornecido pelo cliente."""
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise ValueError("messages deve ser uma lista")
    if len(items) > MAX_RENDER_ITEMS:
        raise ValueError(f"No máximo {MAX_RENDER_ITEMS} mensagens por lote")

    rendered = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Cada mensagem deve ser um objeto")
        identifier = str(item.get("id", index))[:80]
        content = str(item.get("content") or "")
        if len(content) > MAX_RENDER_CHARS:
            raise ValueError(
                f"A mensagem {identifier} excede {MAX_RENDER_CHARS} caracteres"
            )
        rendered.append(
            {"id": identifier, "html": render_agent_markdown(content)}
        )
    return rendered
