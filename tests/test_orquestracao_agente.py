from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from src.conhecimento import ferramentas
from src.conhecimento.agente_interacao import resposta_interacao_simples
from src.conhecimento.roteamento_ferramentas import decidir_acao
from src.ml import resultados
from src.webapp.agent_adapter import AgentAdapter, _Componentes


class _LibrarySpy:
    def __init__(self):
        self.queued = []

    def queue_pdf(self, filename, data, metadata=None):
        self.queued.append((filename, data, metadata))
        return {"job_id": "job-library-1", "state": "queued"}


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _rag_adapter(monkeypatch, library=None):
    captured = {}

    def fake_question(**kwargs):
        captured.update(kwargs)
        names = ", ".join(item["nome"] for item in kwargs["anexos"])
        return f"Conteúdo lido de {names}."

    monkeypatch.setattr("src.conhecimento.agente.perguntar", fake_question)
    adapter = AgentAdapter(library_service=library)
    adapter._components = _Componentes(
        perfil="perfil",
        modelo_embeddings=object(),
        literatura=object(),
        sessoes=object(),
        obsidian=object(),
        indice_lexical=object(),
        llm=None,
        auditor=SimpleNamespace(),
        modo_consulta=True,
    )
    adapter._state = "pronto"
    monkeypatch.setattr(adapter, "_schedule_postprocessing", lambda *_args: None)
    return adapter, captured


@pytest.mark.parametrize(
    "message",
    (
        "Oi",
        "Oii",
        "Oiii",
        "Oii, Aliado",
        "Oii, Aliado.. Quanto tempo",
        "Quanto tempo!",
        "Tudo bem?",
    ),
)
def test_cumprimentos_puros_sao_locais_sem_rag(message):
    assert resposta_interacao_simples(message)


def test_cumprimento_com_pergunta_academica_nao_e_interceptado():
    assert resposta_interacao_simples("Oi, explique a FMECA") is None


@pytest.mark.parametrize(
    ("filename", "content"),
    (("nota.txt", b"conteudo TXT"), ("nota.md", b"# Conteudo MD")),
)
def test_txt_e_md_anexados_sao_lidos_pelo_agente(monkeypatch, filename, content):
    adapter, captured = _rag_adapter(monkeypatch)

    response = adapter.answer(
        "Leia este arquivo",
        attachments=[(filename, content)],
        session_id="sessao_leitura_anexo",
    )

    assert response["route"] == "rag"
    assert response["answer"] == f"Conteúdo lido de {filename}."
    assert captured["anexos"][0]["texto"]


def test_adicionar_pdf_aciona_ferramenta_da_biblioteca_sem_rag():
    library = _LibrarySpy()

    def fail_answerer(*_args):
        raise AssertionError("RAG/answerer não deveria ser consultado")

    adapter = AgentAdapter(answerer=fail_answerer, library_service=library)
    response = adapter.answer(
        "Adicione este arquivo à biblioteca",
        attachments=[("referencia.pdf", _pdf_bytes())],
        session_id="sessao_importacao_pdf",
        library_write_allowed=True,
    )

    assert response["route"] == "tool"
    assert response["library_jobs"][0]["job_id"] == "job-library-1"
    assert [item[0] for item in library.queued] == ["referencia.pdf"]


def test_leitura_de_pdf_permanece_efemera(monkeypatch):
    library = _LibrarySpy()
    adapter, captured = _rag_adapter(monkeypatch, library)

    response = adapter.answer(
        "Leia este arquivo",
        attachments=[("referencia.pdf", _pdf_bytes())],
        session_id="sessao_pdf_efemero",
        library_write_allowed=True,
    )

    assert response["route"] == "rag"
    assert captured["anexos"][0]["nome"] == "referencia.pdf"
    assert library.queued == []


def test_anexo_nao_desativa_tool_calling(monkeypatch):
    adapter, _captured = _rag_adapter(monkeypatch)
    calls = []

    def fake_process(**kwargs):
        calls.append(kwargs)
        return {
            "resultado": {"imagens": [], "jobs": []},
            "resposta": "Pipeline acionado com o anexo presente.",
        }

    monkeypatch.setattr(ferramentas, "processar_com_ferramentas", fake_process)
    response = adapter.answer(
        "rode o pipeline",
        attachments=[("observacao.txt", b"contexto adicional")],
        session_id="sessao_pipeline_com_anexo",
    )

    assert response["route"] == "tool"
    assert response["answer"] == "Pipeline acionado com o anexo presente."
    assert calls[0]["decisao"]["ferramenta"] == "executar_pipeline_cientifico"
    assert calls[0]["anexos"] == [("observacao.txt", b"contexto adicional")]


def test_regressao_sessao_20260901_limpeza_multiturno_sem_rag(monkeypatch):
    calls = []

    def fake_cleaner(progresso=None, pergunta="", *, etapas=None, confirmado=False):
        del progresso, pergunta
        calls.append((tuple(etapas or ()), confirmado))
        return {
            "ok": True,
            "mensagem": (
                "Foram removidas ambas as publicações."
                if confirmado
                else "Confirma excluir ambas? Responda `confirmar`."
            ),
        }

    monkeypatch.setattr(ferramentas, "limpar_resultados_ml", fake_cleaner)

    def fail_answerer(*_args):
        raise AssertionError("RAG/answerer não deveria ser consultado")

    adapter = AgentAdapter(answerer=fail_answerer)
    greeting = AgentAdapter().answer(
        "Oii, Aliado.. Quanto tempo",
        session_id="sessao_problematica",
    )
    first = adapter.answer(
        "apague todos os resultados",
        session_id="sessao_problematica",
    )
    second = adapter.answer(
        "comparação e confiabilidade",
        session_id="sessao_problematica",
    )
    foreign = adapter.answer("confirmar", session_id="sessao_diferente")
    third = adapter.answer("confirmar", session_id="sessao_problematica")

    assert greeting["route"] == "local"
    assert "Quais resultados" in first["answer"]
    assert "Confirma excluir ambas" in second["answer"]
    assert "Não há uma exclusão pendente" in foreign["answer"]
    assert "Foram removidas ambas" in third["answer"]
    assert calls == [(("comparacao", "confiabilidade"), False), (("comparacao", "confiabilidade"), True)]


def test_cancelar_remove_acao_pendente_sem_executar(monkeypatch):
    executed = []

    def fake_cleaner(progresso=None, pergunta="", *, etapas=None, confirmado=False):
        del progresso, pergunta
        if confirmado:
            executed.append(tuple(etapas or ()))
        return {"ok": True, "mensagem": "Confirma excluir?"}

    monkeypatch.setattr(ferramentas, "limpar_resultados_ml", fake_cleaner)
    adapter = AgentAdapter(answerer=lambda *_args: "não deveria chegar aqui")
    adapter.answer("apague resultados da comparação", session_id="sessao_cancelar")
    response = adapter.answer("cancelar", session_id="sessao_cancelar")
    after = adapter.answer("confirmar", session_id="sessao_cancelar")

    assert "cancelada" in response["answer"]
    assert "Não há uma exclusão pendente" in after["answer"]
    assert executed == []


@pytest.mark.parametrize(
    ("message", "tool"),
    (
        ("mostre os resultados", "consultar_resultados"),
        ("status do pipeline", "consultar_status_pipeline"),
        ("rode o pipeline", "executar_pipeline_cientifico"),
        ("recalcule o pipeline", "executar_pipeline_cientifico"),
    ),
)
def test_consulta_status_execucao_e_recalculo_tem_rotas_distintas(message, tool):
    assert decidir_acao(message)["ferramenta"] == tool


def test_pipeline_comum_nao_forca_e_recalculo_forca(monkeypatch):
    forces = []

    def fake_pipeline(_stage, *, force=False, progresso=None):
        del progresso
        forces.append(force)
        return ["comparação concluída", "confiabilidade concluída"]

    monkeypatch.setattr(ferramentas, "executar_pipeline_ml", fake_pipeline)
    monkeypatch.setattr(
        ferramentas,
        "resumir_resultados",
        lambda *_args, **kwargs: {
            "mensagem": f"Operação: {kwargs['operacao']}",
            "imagens": [],
        },
    )

    run = ferramentas.executar_pipeline_cientifico(pergunta="rode o pipeline")
    recalc = ferramentas.executar_pipeline_cientifico(pergunta="refaça o pipeline do zero")

    assert forces == [False, True]
    assert "executado" in run["mensagem"]
    assert "recalculado" in recalc["mensagem"]


def test_resultado_consultado_exibe_proveniencia():
    response = resultados.resumir_resultados(
        "mostre os resultados da comparação",
        incluir_imagens=False,
    )
    text = response["mensagem"]

    assert "## Proveniência da resposta" in text
    assert "Operação: **consultado**" in text
    assert "comparacao_autoencoders.json" in text
    assert "seed de referência=42" in text
    assert any(state in text for state in ("ready", "stale", "pending"))
