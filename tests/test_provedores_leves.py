from __future__ import annotations

from types import SimpleNamespace

from src.conhecimento.provedores import GeminiLeve, GroqLeve


class _Mensagem:
    def __init__(self, content):
        self.content = content


def test_gemini_invoke_e_stream_preservam_contrato():
    class Models:
        def generate_content(self, **kwargs):
            assert kwargs["contents"] == "prompt"
            return SimpleNamespace(text="resposta")

        def generate_content_stream(self, **kwargs):
            assert kwargs["contents"] == "prompt"
            return iter([SimpleNamespace(text="res"), SimpleNamespace(text="posta")])

    cliente = SimpleNamespace(models=Models())
    llm = GeminiLeve("chave", "modelo", client=cliente)

    assert llm.invoke([_Mensagem("prompt")]).content == "resposta"
    assert "".join(item.content for item in llm.stream([_Mensagem("prompt")])) == "resposta"


def test_groq_remove_imagem_de_provedor_textual():
    chamadas = []

    class Completions:
        def create(self, **kwargs):
            chamadas.append(kwargs)
            if kwargs.get("stream"):
                delta = SimpleNamespace(content="fluxo")
                return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])
            mensagem = SimpleNamespace(content="resposta")
            return SimpleNamespace(choices=[SimpleNamespace(message=mensagem)])

    cliente = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    llm = GroqLeve("chave", "modelo", client=cliente)
    conteudo = [
        {"type": "text", "text": "pergunta"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]

    assert llm.invoke([_Mensagem(conteudo)]).content == "resposta"
    assert next(llm.stream([_Mensagem(conteudo)])).content == "fluxo"
    assert chamadas[0]["messages"] == [{"role": "user", "content": "pergunta"}]
