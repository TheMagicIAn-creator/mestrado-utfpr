from __future__ import annotations

from types import SimpleNamespace

from src.conhecimento.provedores import GeminiLeve


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


def test_gemini_invoke_json_forca_json_e_limita_saida():
    """O papel de auditor (antes Groq) agora roda no GeminiLeve.invoke_json:
    temperatura 0, mime-type JSON e teto de tokens do parametro."""
    chamadas = []

    class Models:
        def generate_content(self, **kwargs):
            chamadas.append(kwargs)
            return SimpleNamespace(text='{"status": "aprovado"}')

    cliente = SimpleNamespace(models=Models())
    llm = GeminiLeve("chave", "modelo", client=cliente)

    assert llm.invoke_json([_Mensagem("audite")], max_tokens=321) == {
        "status": "aprovado"
    }
    assert chamadas[0]["config"]["temperature"] == 0.0
    assert chamadas[0]["config"]["max_output_tokens"] == 321
    assert chamadas[0]["config"]["response_mime_type"] == "application/json"


def test_gemini_invoke_json_rejeita_nao_objeto():
    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="[1, 2, 3]")

    cliente = SimpleNamespace(models=Models())
    llm = GeminiLeve("chave", "modelo", client=cliente)

    try:
        llm.invoke_json([_Mensagem("audite")])
    except ValueError:
        return
    raise AssertionError("invoke_json deveria rejeitar JSON que nao e objeto")
