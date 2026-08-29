from __future__ import annotations

from types import SimpleNamespace

from src.conhecimento.cliente_llm import RouterLLMFacade
from src.conhecimento.contratos_llm import LLMResult, TaskType
from src.conhecimento.memoria_persistente import MemoriaPersistente
from src.conhecimento.multiagente import (
    AgenteAuditor,
    AgenteAuditorGemini,
    criar_equipe_agentes,
    filtrar_citacoes_auditadas,
    RelatorioAuditoria,
)


class _LLMJson:
    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.chamadas = []

    def invoke_json(self, mensagens, max_tokens=700):
        self.chamadas.append((mensagens, max_tokens))
        return self.respostas.pop(0)


class _LLMGemini:
    def invoke(self, mensagens):
        return SimpleNamespace(content="resposta")

    def stream(self, mensagens):
        yield SimpleNamespace(content="res")
        yield SimpleNamespace(content="posta")


def test_auditor_audita_pacote_compacto_sem_responder_pergunta(tmp_path):
    llm = _LLMJson([{
        "status": "com_ressalvas",
        "restricoes": ["Falta o terceiro autor."],
        "orientacao": "Compare somente os dois autores cobertos.",
        "fontes_utilizaveis": ["F1", "F2"],
    }])
    auditor = AgenteAuditorGemini(llm, MemoriaPersistente(tmp_path / "m.json"))
    citacoes = {f"c{i}": "Fonte " + ("x" * 1200) for i in range(12)}

    resultado = auditor.auditar_evidencias("Compare os autores.", citacoes)

    assert resultado.status == "com_ressalvas"
    assert resultado.fontes_utilizaveis == ("F1", "F2")
    prompt = llm.chamadas[0][0][0]["content"]
    assert len(prompt) < 8000
    assert "F8:" in prompt
    assert "F9:" not in prompt


def test_auditor_so_avalia_memoria_com_gatilho_explicito(tmp_path):
    llm = _LLMJson([])
    auditor = AgenteAuditorGemini(llm, MemoriaPersistente(tmp_path / "m.json"))

    resultado = auditor.aprender_da_interacao(
        "Explique FMEA.", "FMEA e uma analise..."
    )

    assert resultado.avaliou is False
    assert llm.chamadas == []


def test_auditor_aprova_e_persiste_preferencia_do_pesquisador(tmp_path):
    llm = _LLMJson([{
        "salvar": True,
        "motivo": "Preferencia duravel.",
        "candidatos": [{
            "tipo": "preferencia",
            "escopo": "conversa",
            "conteudo": "Prefere respostas objetivas em portugues.",
            "evidencia_usuario": "Daqui em diante, prefiro respostas objetivas em portugues.",
            "confianca": 0.96,
        }],
    }])
    memoria = MemoriaPersistente(tmp_path / "m.json")
    auditor = AgenteAuditorGemini(llm, memoria)

    resultado = auditor.aprender_da_interacao(
        "Daqui em diante, prefiro respostas objetivas em portugues.",
        "Entendido.",
    )

    assert resultado.avaliou is True
    assert resultado.salvas == 1
    assert memoria.contar() == 1


def test_consolidacao_automatica_persiste_decisao_sem_gatilho(tmp_path):
    """Consolidacao automatica: extrai decisao metodologica de um transcript
    inteiro SEM o pesquisador ter usado gatilho ("lembre/decidi")."""
    llm = _LLMJson([{
        "salvar": True,
        "motivo": "Decisao metodologica declarada na sessao.",
        "candidatos": [{
            "tipo": "decisao_metodologica",
            "escopo": "ml",
            "conteudo": "A primeira falha a injetar e o Contator AC (NPR=315).",
            "evidencia_usuario": "vamos comecar a injecao pelo Contator AC",
            "confianca": 0.93,
        }],
    }])
    memoria = MemoriaPersistente(tmp_path / "m.json")
    auditor = AgenteAuditorGemini(llm, memoria)

    transcrito = (
        "## Interacao 1\n🔬 Voce: qual falha injetamos primeiro?\n"
        "🤖 Agente: pela FMECA, o Contator AC tem o maior NPR.\n"
        "## Interacao 2\n🔬 Voce: entao vamos comecar a injecao pelo Contator AC.\n"
    )
    resultado = auditor.consolidar_memoria_das_sessoes(transcrito)

    assert resultado.avaliou is True
    assert resultado.salvas == 1
    assert memoria.contar() == 1
    assert "Contator AC" in memoria.listar()[0]["conteudo"]


def test_consolidacao_automatica_ignora_sessao_sem_nada_duravel(tmp_path):
    llm = _LLMJson([{"salvar": False, "motivo": "So duvidas pontuais."}])
    memoria = MemoriaPersistente(tmp_path / "m.json")
    auditor = AgenteAuditorGemini(llm, memoria)

    resultado = auditor.consolidar_memoria_das_sessoes("🔬 Voce: o que e FMEA?")

    assert resultado.avaliou is True
    assert resultado.salvas == 0
    assert memoria.contar() == 0


def test_gemini_recebe_memoria_validada_e_parecer_do_auditor(tmp_path):
    memoria = MemoriaPersistente(tmp_path / "m.json")
    memoria.registrar(
        {
            "tipo": "preferencia",
            "escopo": "conversa",
            "conteudo": "Prefere respostas objetivas em portugues.",
            "evidencia_usuario": "Prefiro respostas objetivas em portugues.",
        },
        origem="teste",
        validado_por="Gemini Flash",
        confianca=0.9,
    )
    auditor_llm = _LLMJson([{
        "status": "aprovado",
        "restricoes": [],
        "orientacao": "Use F1.",
        "fontes_utilizaveis": ["F1"],
    }])
    equipe = criar_equipe_agentes(
        memoria=memoria,
        llm_gemini=_LLMGemini(),
        llm_auditor=auditor_llm,
    )
    auditoria = equipe.auditoria.auditar_evidencias("Resposta objetiva", {"a": "Fonte"})
    prompt = equipe.conversa.contextualizar_prompt(
        "PROMPT BASE", "Resposta objetiva", auditoria
    )

    assert "MEMORIA VALIDADA ENTRE SESSOES" in prompt
    assert "Prefere respostas objetivas" in prompt
    assert "PARECER DO AUDITOR" in prompt
    assert "Status da auditoria: aprovado" in prompt


def test_rodape_mantem_apenas_fontes_aprovadas_pelo_auditor():
    citacoes = {"a": "Stender", "b": "Francisti", "c": "Torres"}
    auditoria = RelatorioAuditoria(
        status="aprovado",
        fontes_utilizaveis=("F1", "F3"),
    )

    assert filtrar_citacoes_auditadas(citacoes, auditoria) == {
        "a": "Stender",
        "c": "Torres",
    }


def test_correcao_pode_superar_memoria_anterior(tmp_path):
    memoria = MemoriaPersistente(tmp_path / "m.json")
    anterior = memoria.registrar(
        {
            "tipo": "preferencia",
            "escopo": "conversa",
            "conteudo": "Prefere respostas curtas em portugues.",
            "evidencia_usuario": "Prefiro respostas curtas em portugues.",
        },
        origem="teste",
        validado_por="Gemini Flash",
        confianca=0.9,
    ).item
    llm = _LLMJson([{
        "salvar": True,
        "motivo": "Correcao explicita.",
        "candidatos": [{
            "tipo": "correcao",
            "escopo": "conversa",
            "conteudo": "Prefere respostas detalhadas em portugues.",
            "evidencia_usuario": "Corrigindo: agora prefiro respostas detalhadas em portugues.",
            "substitui_id": anterior["id"],
            "confianca": 0.95,
        }],
    }])
    auditor = AgenteAuditorGemini(llm, memoria)

    resultado = auditor.aprender_da_interacao(
        "Corrigindo: agora prefiro respostas detalhadas em portugues.",
        "Entendido.",
    )

    assert resultado.salvas == 1
    assert memoria.contar() == 1
    assert memoria.listar()[0]["conteudo"].startswith("Prefere respostas detalhadas")
    assert anterior["id"] in llm.chamadas[0][0][0]["content"]


def test_alias_legado_aponta_para_classe_neutra():
    assert AgenteAuditorGemini is AgenteAuditor


def test_fabrica_padrao_compartilha_router_entre_papeis(tmp_path):
    router = SimpleNamespace()
    equipe = criar_equipe_agentes(
        memoria=MemoriaPersistente(tmp_path / "m.json"),
        router=router,
    )
    assert isinstance(equipe.conversa.llm, RouterLLMFacade)
    assert isinstance(equipe.auditoria.llm, RouterLLMFacade)
    assert equipe.conversa.llm.router is router
    assert equipe.auditoria.llm.router is router
    assert "Gemini" not in " ".join(equipe.nomes.values())


def test_auditoria_e_memoria_declaram_tarefas_distintas_ao_router(tmp_path):
    class Router:
        def __init__(self):
            self.requests = []

        def execute(self, request):
            self.requests.append(request)
            if request.task_type == TaskType.EVIDENCE_AUDIT:
                payload = {
                    "status": "aprovado",
                    "restricoes": [],
                    "orientacao": "Use F1.",
                    "fontes_utilizaveis": ["F1"],
                }
            else:
                payload = {"salvar": False, "motivo": "Nada durável", "candidatos": []}
            import json

            return LLMResult(
                content=json.dumps(payload),
                provider="fake",
                model="modelo",
                task_type=request.task_type,
                structured_data=payload,
            )

    router = Router()
    auditor = AgenteAuditor(
        RouterLLMFacade(router),
        MemoriaPersistente(tmp_path / "m.json"),
    )
    auditor.auditar_evidencias("Explique", {"fonte": "trecho"})
    auditor.consolidar_memoria_das_sessoes("Pesquisador: decisão durável")
    assert [request.task_type for request in router.requests] == [
        TaskType.EVIDENCE_AUDIT,
        TaskType.MEMORY_CONSOLIDATION,
    ]
