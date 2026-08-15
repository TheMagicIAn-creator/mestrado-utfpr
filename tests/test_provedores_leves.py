from __future__ import annotations

from types import SimpleNamespace

import src.conhecimento.provedores as pv
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


def test_texto_da_resposta_normaliza_blocos_textuais():
    resposta = SimpleNamespace(content=[
        {"type": "text", "text": "primeira parte"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "output_text", "text": " e segunda"},
    ])

    assert pv.texto_da_resposta(resposta) == "primeira parte e segunda"


def test_texto_da_resposta_aceita_objetos_de_bloco():
    resposta = SimpleNamespace(content=[
        SimpleNamespace(text="A"),
        SimpleNamespace(content={"type": "text", "text": "B"}),
    ])

    assert pv.texto_da_resposta(resposta) == "AB"


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


def test_gemini_3_remove_sampling_e_controla_thinking():
    chamadas = []

    class Models:
        def generate_content(self, **kwargs):
            chamadas.append(kwargs)
            return SimpleNamespace(text="ok")

    llm = GeminiLeve(
        "chave",
        "gemini-3.6-flash",
        client=SimpleNamespace(models=Models()),
        thinking_level="low",
    )

    assert llm.invoke([_Mensagem("analise")]).content == "ok"
    config = chamadas[0]["config"]
    assert "temperature" not in config
    assert config["thinking_config"] == {"thinking_level": "low"}


def test_gemini_cai_para_fallback_quando_modelo_indisponivel():
    """404 'no longer available' num modelo deve cair para o próximo candidato,
    e o modelo que funcionar vira o novo self.model (sem repetir o 404)."""
    usados = []

    class Models:
        def generate_content(self, **kwargs):
            usados.append(kwargs["model"])
            if kwargs["model"] == "gemini-pro-latest":
                raise RuntimeError(
                    "404 NOT_FOUND: model gemini-pro-latest is no longer available"
                )
            return SimpleNamespace(text="ok")

    cliente = SimpleNamespace(models=Models())
    llm = GeminiLeve("chave", "gemini-pro-latest", client=cliente,
                     fallbacks=("gemini-flash-latest",))

    assert llm.invoke([_Mensagem("oi")]).content == "ok"
    assert usados == ["gemini-pro-latest", "gemini-flash-latest"]
    assert llm.model == "gemini-flash-latest"  # fixou o que funcionou
    # Segunda chamada não repete o modelo morto.
    usados.clear()
    assert llm.invoke([_Mensagem("de novo")]).content == "ok"
    assert usados == ["gemini-flash-latest"]


def test_gemini_retenta_em_503_e_depois_cai_para_fallback(monkeypatch):
    """503 (alta demanda) retenta o mesmo modelo e, esgotando, cai p/ fallback."""
    monkeypatch.setattr(pv, "_dormir", lambda s: None)  # sem esperar de verdade
    tentativas = []

    class Models:
        def generate_content(self, **kwargs):
            tentativas.append(kwargs["model"])
            if kwargs["model"] == "gemini-pro-latest":
                raise RuntimeError("503 UNAVAILABLE: currently experiencing high demand")
            return SimpleNamespace(text="ok")

    llm = GeminiLeve("chave", "gemini-pro-latest",
                     client=SimpleNamespace(models=Models()),
                     fallbacks=("gemini-flash-latest",))
    assert llm.invoke([_Mensagem("q")]).content == "ok"
    # pro tentado (1 + retentativas) vezes antes de cair; flash resolve.
    assert tentativas.count("gemini-pro-latest") >= 2
    assert tentativas[-1] == "gemini-flash-latest"
    assert llm.model == "gemini-flash-latest"


def test_gemini_503_transitorio_que_se_resolve_no_retry(monkeypatch):
    """Se o 503 passar (spike temporário), o retry no MESMO modelo resolve."""
    monkeypatch.setattr(pv, "_dormir", lambda s: None)
    estado = {"n": 0}

    class Models:
        def generate_content(self, **kwargs):
            estado["n"] += 1
            if estado["n"] == 1:
                raise RuntimeError("503 UNAVAILABLE: high demand, try again later")
            return SimpleNamespace(text="ok")

    llm = GeminiLeve("chave", "gemini-flash-latest",
                     client=SimpleNamespace(models=Models()))
    assert llm.invoke([_Mensagem("q")]).content == "ok"
    assert estado["n"] == 2  # 1 falha + 1 sucesso, mesmo modelo
    assert llm.model == "gemini-flash-latest"


def test_gemini_erro_que_nao_e_de_modelo_nao_faz_fallback():
    """Erro comum (ex.: 429/500) NÃO deve mascarar-se de troca de modelo."""
    class Models:
        def generate_content(self, **kwargs):
            raise RuntimeError("500 internal error")

    cliente = SimpleNamespace(models=Models())
    llm = GeminiLeve("chave", "gemini-pro-latest", client=cliente,
                     fallbacks=("gemini-flash-latest",))
    try:
        llm.invoke([_Mensagem("oi")])
    except RuntimeError as e:
        assert "500" in str(e)
        return
    raise AssertionError("erro nao-de-modelo deveria propagar")


def test_conversa_e_auditor_tem_modelo_alternativo_de_verdade(monkeypatch):
    """Fallback deve incluir um modelo de POOL DIFERENTE (flash-lite), senão um
    503 do Flash só re-bateria no mesmo modelo lotado."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.delenv("AL_IADO_GEMINI_MODEL", raising=False)
    conversa, _ = pv.inicializar_provedor("1")
    auditor, _ = pv.inicializar_provedor("2")
    assert "gemini-3.5-flash-lite" in conversa._candidatos()
    assert "gemini-3.5-flash-lite" in auditor._candidatos()
    # o alternativo vem por último (último recurso), não na frente.
    assert conversa._candidatos()[-1] == "gemini-3.5-flash-lite"


def test_503_persistente_no_flash_escapa_para_flash_lite(monkeypatch):
    """Cenário real do usuário: 503 no gemini-flash-latest deve cair para o
    gemini-flash-lite-latest e responder, em vez de estourar erro."""
    monkeypatch.setattr(pv, "_dormir", lambda s: None)
    usados = []

    class Models:
        def generate_content(self, **kwargs):
            usados.append(kwargs["model"])
            if kwargs["model"] == "gemini-flash-latest":
                raise RuntimeError("503 UNAVAILABLE: experiencing high demand")
            return SimpleNamespace(text="resposta do lite")

    llm = GeminiLeve("chave", "gemini-flash-latest",
                     client=SimpleNamespace(models=Models()),
                     fallbacks=("gemini-flash-lite-latest",))
    assert llm.invoke([_Mensagem("q")]).content == "resposta do lite"
    assert usados[-1] == "gemini-flash-lite-latest"
    assert llm.model == "gemini-flash-lite-latest"


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
