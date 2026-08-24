"""Tool calling do ALIAdo para ciência, literatura e memória."""

from __future__ import annotations

import inspect
import json
import re

from src.conhecimento.ferramentas_academicas import (
    buscar_na_web,
    comparar_abordagens_ml,
    consultar_comparacao_autoencoders,
    consultar_datasets,
    listar_base_bibliografica,
)
from src.conhecimento.provedores import texto_da_resposta
from src.conhecimento.intencoes_ferramentas import (
    _quer_registrar_no_cerebro,
    _quer_resposta_autoral,
)
from src.core.config import RAIZ_PROJETO
from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos
from src.core.texto import normalizar_sem_acentos
from src.ml.pipeline import (
    NOMES_ETAPAS,
    ORDEM_ETAPAS_ML,
    artefatos_a_partir,
    capacidade_recalculo_pipeline,
    estado_pipeline,
    executar_etapa,
    executar_pipeline_ml,
    limpar_artefatos,
)
from src.ml.resultados import resumir_resultados


LOGGER = get_logger("conhecimento.ferramentas")

ESPEC_FERRAMENTAS = [
    {
        "name": "executar_comparacao_autoencoders",
        "description": (
            "Treina e avalia Autoencoder Denso e AE-LSTM no GPVS-Faults, "
            "publicando a comparação experimental E3. Use somente em pedido explícito "
            "de executar, recalcular ou retreinar."
        ),
    },
    {
        "name": "gerar_confiabilidade",
        "description": (
            "Regenera os cenários bibliográficos de R(t), F(t), f(t) e h(t). "
            "Não treina ML e não estima Weibull físico."
        ),
    },
    {
        "name": "executar_pipeline_cientifico",
        "description": (
            "Executa em ordem a comparação Denso versus AE-LSTM e a publicação "
            "de confiabilidade física."
        ),
    },
    {
        "name": "consultar_resultados",
        "description": (
            "Consulta as métricas, tabelas e figuras canônicas E3 e de "
            "confiabilidade sem recalcular."
        ),
    },
    {
        "name": "consultar_comparacao_autoencoders",
        "description": (
            "Compara o Autoencoder Denso e o AE-LSTM nos contratos publicados, "
            "incluindo AUC-PR e seus IC95%. Nunca treina."
        ),
    },
    {
        "name": "consultar_status_pipeline",
        "description": "Mostra integridade e disponibilidade das duas publicações canônicas.",
    },
    {
        "name": "limpar_resultados_ml",
        "description": (
            "Remove, após confirmação literal, somente a publicação de comparação "
            "ou de confiabilidade escolhida."
        ),
    },
    {
        "name": "registrar_no_cerebro",
        "description": (
            "Registra uma decisão, conceito ou resultado curado no vault Obsidian "
            "quando o pesquisador pedir explicitamente para guardar ou documentar."
        ),
    },
    {
        "name": "buscar_web",
        "description": "Pesquisa fatos externos que não estejam na literatura indexada.",
    },
    {
        "name": "listar_base_bibliografica",
        "description": "Lista deterministicamente todo o catálogo bibliográfico indexado.",
    },
    {
        "name": "consultar_datasets",
        "description": "Explica o GPVS-Faults, único dataset ativo, e seu papel na E3.",
    },
    {
        "name": "comparar_abordagens_ml",
        "description": (
            "Delimita Denso, AE-LSTM, evidência E3 e confiabilidade física."
        ),
    },
]


def consultar_status_pipeline(progresso=None, pergunta: str = "") -> dict:
    if progresso:
        progresso("Auditando as publicações canônicas...")
    capacity = capacidade_recalculo_pipeline()
    states = estado_pipeline()
    labels = {"ready": "pronto", "stale": "desatualizado", "pending": "pendente"}
    lines = ["## Estado do pipeline científico", ""]
    for key in ORDEM_ETAPAS_ML:
        state = states[key]
        detail = "; ".join(state.get("motivos", []))
        suffix = f": {detail}" if detail else ""
        lines.append(f"- **{NOMES_ETAPAS[key]}:** {labels[state['estado']]}{suffix}")
    lines.extend(
        [
            "",
            f"Modo: **{capacity['modo']}**.",
            (
                "Os 16 CSVs GPVS estão disponíveis para novo treinamento."
                if capacity["disponivel"]
                else "Sem dados brutos neste ambiente; os resultados publicados continuam consultáveis."
            ),
        ]
    )
    return {
        "ok": True,
        "etapa": "Status científico",
        "mensagem": "\n".join(lines),
        "imagens": [],
        "resposta_pronta": True,
    }


def consultar_resultados(progresso=None, pergunta: str = "") -> dict:
    if progresso:
        progresso("Lendo os contratos científicos publicados...")
    return resumir_resultados(pergunta)


def _run_stage(stage: str, progresso=None, pergunta: str = "") -> dict:
    force = any(
        term in normalizar_sem_acentos(pergunta).lower()
        for term in ("force", "forcar", "recalcular", "refazer", "do zero")
    )
    result = executar_etapa(stage, force=force, progresso=progresso)
    if not result["ok"]:
        return {
            "ok": False,
            "etapa": result["etapa"],
            "mensagem": result["mensagem"],
            "imagens": [],
            "resposta_pronta": True,
        }
    summary = resumir_resultados(
        "e3 denso lstm" if stage == "comparacao" else "confiabilidade"
    )
    return {
        "ok": True,
        "etapa": result["etapa"],
        "mensagem": f"{result['mensagem']}\n\n{summary['mensagem']}",
        "imagens": summary["imagens"],
        "resposta_pronta": True,
        "acao_executada": True,
    }


def executar_comparacao_autoencoders(progresso=None, pergunta: str = "") -> dict:
    return _run_stage("comparacao", progresso, pergunta)


def gerar_confiabilidade(progresso=None, pergunta: str = "") -> dict:
    return _run_stage("confiabilidade", progresso, pergunta)


def executar_pipeline_cientifico(progresso=None, pergunta: str = "") -> dict:
    force = "recal" in normalizar_sem_acentos(pergunta).lower()
    results = executar_pipeline_ml("comparacao", force=force, progresso=progresso)
    ok = all(not item.startswith("ERRO") for item in results)
    summary = resumir_resultados(pergunta)
    message = "## Execução do pipeline científico\n\n" + "\n".join(
        f"- {item}" for item in results
    )
    if ok:
        message += f"\n\n{summary['mensagem']}"
    return {
        "ok": ok,
        "etapa": "Pipeline científico",
        "mensagem": message,
        "imagens": summary["imagens"] if ok else [],
        "resposta_pronta": True,
        "acao_executada": ok,
    }


def _selected_stage(question: str) -> str | None:
    text = normalizar_sem_acentos(question).lower()
    if any(term in text for term in ("confiabilidade", "taxa de falha", "r(t)", "h(t)")):
        return "confiabilidade"
    if any(term in text for term in ("comparacao", "autoencoder", "denso", "lstm", "e3")):
        return "comparacao"
    return None


def limpar_resultados_ml(progresso=None, pergunta: str = "") -> dict:
    stage = _selected_stage(pergunta)
    if stage is None:
        return {
            "ok": False,
            "etapa": "Limpeza de resultados",
            "mensagem": "Indique `comparacao` ou `confiabilidade`.",
            "imagens": [],
            "resposta_pronta": True,
        }
    token = f"CONFIRMAR LIMPEZA {stage.upper()}"
    if normalizar_sem_acentos(token).lower() not in normalizar_sem_acentos(pergunta).lower():
        existing = [path for path in artefatos_a_partir(stage) if path.is_file()]
        return {
            "ok": True,
            "etapa": "Limpeza de resultados",
            "mensagem": (
                f"A operação removerá {len(existing)} arquivo(s) de "
                f"**{NOMES_ETAPAS[stage]}**. A ação é irreversível. "
                f"Para confirmar, escreva `{token}`."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }
    if progresso:
        progresso(f"Removendo {NOMES_ETAPAS[stage]}...")
    removed = limpar_artefatos(stage)
    return {
        "ok": True,
        "etapa": "Limpeza de resultados",
        "mensagem": f"Foram removidos {len(removed)} arquivo(s) de {NOMES_ETAPAS[stage]}.",
        "imagens": [],
        "resposta_pronta": True,
        "acao_executada": True,
    }


def _draft_note(question: str, context: str, llm) -> dict | None:
    if llm is None or not context.strip():
        return None
    from src.conhecimento.nota_cerebro import TAGS_VALIDAS, TIPOS

    prompt = f"""Redija uma nota curada para o vault acadêmico.

Conversa recente:
---
{context[-6000:]}
---
Pedido: {question}

Não invente números ou fontes. Responda apenas JSON válido:
{{"titulo":"...","conteudo":"...","tipo":"{'|'.join(sorted(TIPOS))}",
"tags":["uma ou mais de: {', '.join(sorted(TAGS_VALIDAS))}"],
"nivel_evidencia":"projeto|E1|E2|E3|literatura"}}
"""
    try:
        raw = texto_da_resposta(llm.invoke(prompt))
        cleaned = re.sub(r"```json?\s*", "", str(raw)).replace("```", "").strip()
        value = json.loads(cleaned)
    except Exception as exc:
        LOGGER.warning("Falha ao redigir nota: %s", mascarar_segredos(str(exc)))
        return None
    return value if value.get("titulo") and value.get("conteudo") else None


def registrar_no_cerebro(
    progresso=None,
    pergunta: str = "",
    titulo: str = "",
    conteudo: str = "",
    tipo: str = "contexto",
    tags: list | None = None,
    nivel_evidencia: str = "projeto",
    fonte: str = "",
    llm=None,
    contexto: str = "",
) -> dict:
    from src.conhecimento.nota_cerebro import registrar_nota_cerebro

    if not titulo.strip() or not conteudo.strip():
        drafted = _draft_note(pergunta, contexto, llm)
        if drafted:
            titulo = drafted.get("titulo", titulo)
            conteudo = drafted.get("conteudo", conteudo)
            tipo = drafted.get("tipo", tipo)
            tags = drafted.get("tags", tags)
            nivel_evidencia = drafted.get("nivel_evidencia", nivel_evidencia)
    if not titulo.strip() or not conteudo.strip():
        return {
            "ok": False,
            "etapa": "Registro no cérebro",
            "mensagem": (
                "**A nota NÃO foi criada.** O histórico está vazio ou não há "
                "contexto suficiente para identificar o conteúdo. Informe o título "
                "e o resultado ou decisão que deve ser guardado."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }
    if progresso:
        progresso(f"Registrando nota: {titulo}...")
    result = registrar_nota_cerebro(
        titulo=titulo,
        conteudo=conteudo,
        tipo=tipo,
        tags=tags,
        nivel_evidencia=nivel_evidencia,
        fonte=fonte,
    )
    return {
        "ok": result["ok"],
        "etapa": "Registro no cérebro",
        "mensagem": result["mensagem"],
        "imagens": [],
        "resposta_pronta": True,
        "acao_executada": result["ok"],
    }


_DESPACHO = {
    "executar_comparacao_autoencoders": executar_comparacao_autoencoders,
    "gerar_confiabilidade": gerar_confiabilidade,
    "executar_pipeline_cientifico": executar_pipeline_cientifico,
    "consultar_resultados": consultar_resultados,
    "consultar_comparacao_autoencoders": consultar_comparacao_autoencoders,
    "consultar_status_pipeline": consultar_status_pipeline,
    "limpar_resultados_ml": limpar_resultados_ml,
    "registrar_no_cerebro": registrar_no_cerebro,
    "buscar_web": buscar_na_web,
    "listar_base_bibliografica": listar_base_bibliografica,
    "consultar_datasets": consultar_datasets,
    "comparar_abordagens_ml": comparar_abordagens_ml,
}


def executar_ferramenta(
    nome: str,
    progresso=None,
    pergunta: str = "",
    llm=None,
    contexto: str = "",
) -> dict:
    function = _DESPACHO.get(nome)
    if function is None:
        return {
            "ok": False,
            "etapa": nome,
            "mensagem": f"Ferramenta desconhecida: {nome}",
            "imagens": [],
            "resposta_pronta": True,
        }
    parameters = inspect.signature(function).parameters
    extras = {}
    if "llm" in parameters:
        extras["llm"] = llm
    if "contexto" in parameters:
        extras["contexto"] = contexto
    return function(progresso=progresso, pergunta=pergunta, **extras)


from src.conhecimento.roteamento_ferramentas import (  # noqa: E402
    _corrigir_descricao_visual,
    _dados_sao_inventario,
    _decisao_rapida,
    _e_pergunta,
    _etapa_mais_avancada_mencionada,
    _guardas_criticas,
    _quer_codigo_snippet,
    _rotear_por_llm,
    comentar_resultado,
    decidir_acao,
    processar_com_ferramentas,
)


__all__ = [
    "ESPEC_FERRAMENTAS",
    "comentar_resultado",
    "consultar_resultados",
    "consultar_status_pipeline",
    "decidir_acao",
    "executar_ferramenta",
    "processar_com_ferramentas",
]
