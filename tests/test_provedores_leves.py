from __future__ import annotations

from types import SimpleNamespace

from src.conhecimento.provedores import GeminiLeve, GroqLeve


class _Mensagem:
    def __init__(self, content):
        self.content = content


def test_gemini_invoke_e_stream_preservam_contrato():
    chamadas = []

    class Models:
        def generate_content(self, **kwargs):
            chamadas.append(kwargs)
            assert kwargs["contents"] == "prompt"
            return SimpleNamespace(text="resposta")

        def generate_content_stream(self, **kwargs):
            chamadas.append(kwargs)
            assert kwargs["contents"] == "prompt"
            return iter([SimpleNamespace(text="res"), SimpleNamespace(text="posta")])

    cliente = SimpleNamespace(models=Models())
    llm = GeminiLeve("chave", "modelo", client=cliente)

    assert llm.invoke([_Mensagem("prompt")]).content == "resposta"
    assert "".join(item.content for item in llm.stream([_Mensagem("prompt")])) == "resposta"
    assert all(chamada["config"]["max_output_tokens"] == 8192 for chamada in chamadas)


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


def test_groq_invoke_json_limita_saida_e_exige_objeto():
    chamadas = []

    class Completions:
        def create(self, **kwargs):
            chamadas.append(kwargs)
            mensagem = SimpleNamespace(content='{"status": "aprovado"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=mensagem)])

    cliente = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    llm = GroqLeve("chave", "modelo", client=cliente)

    assert llm.invoke_json([_Mensagem("audite")], max_tokens=321) == {
        "status": "aprovado"
    }
    assert chamadas[0]["temperature"] == 0.0
    assert chamadas[0]["max_completion_tokens"] == 321
    assert chamadas[0]["response_format"] == {"type": "json_object"}
