"""Roteamento enxuto das ferramentas canônicas do ALIAdo."""

from __future__ import annotations

import json
import re

from src.conhecimento.intencoes_ferramentas import (
    _parece_pedido_de_ferramenta,
    _quer_catalogo,
    _quer_comparar_abordagens,
    _quer_consultar_datasets,
    _quer_limpar,
    _quer_literatura_tematica,
    _quer_registrar_no_cerebro,
    _quer_resposta_autoral,
    _quer_status,
)
from src.conhecimento.provedores import texto_da_resposta
from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos
from src.core.texto import normalizar_sem_acentos


LOGGER = get_logger("conhecimento.roteamento_ferramentas")

_ETAPA_ORDEM = (
    ("confiabilidade fisica", "gerar_confiabilidade"),
    ("taxa de falha", "gerar_confiabilidade"),
    ("r(t)", "gerar_confiabilidade"),
    ("h(t)", "gerar_confiabilidade"),
    ("autoencoder", "executar_comparacao_autoencoders"),
    ("denso", "executar_comparacao_autoencoders"),
    ("lstm", "executar_comparacao_autoencoders"),
    ("e2", "executar_comparacao_autoencoders"),
    ("e3", "executar_comparacao_autoencoders"),
    ("fmeca", "executar_comparacao_autoencoders"),
    ("weibull", "executar_comparacao_autoencoders"),
)


def _text(value: str) -> str:
    return normalizar_sem_acentos(value or "").lower()


def _e_pergunta(pergunta: str) -> bool:
    text = _text(pergunta).strip()
    if not text:
        return False
    return "?" in pergunta or text.split()[0] in {
        "qual",
        "quais",
        "como",
        "por",
        "porque",
        "quando",
        "onde",
        "o",
        "a",
    }


def _quer_codigo_snippet(pergunta: str) -> bool:
    text = _text(pergunta)
    code = any(term in text for term in ("codigo", "script", "funcao python", "exemplo python"))
    author = any(term in text for term in ("escreva", "gere", "crie", "mostre", "como"))
    return code and author


def _etapa_mais_avancada_mencionada(pergunta: str) -> str | None:
    text = _text(pergunta)
    for term, tool in _ETAPA_ORDEM:
        if term in text:
            return tool
    return None


def _explicit_execution(text: str) -> bool:
    return any(
        term in text
        for term in (
            "execute",
            "executar",
            "rode",
            "rodar",
            "recalcule",
            "recalcular",
            "retreine",
            "retreinar",
            "regenere",
            "regenerar",
            "refaca",
            "refazer",
        )
    )


def _asks_results(text: str) -> bool:
    result_term = any(
        term in text
        for term in (
            "resultado",
            "metrica",
            "grafico",
            "figura",
            "auc",
            "smd",
            "matriz",
            "curva",
            "taxa de falha",
            "confiabilidade",
        )
    )
    request_term = any(
        term in text
        for term in ("mostre", "mostrar", "consulte", "consultar", "compare", "qual", "interprete")
    )
    return result_term and request_term


def _decisao_rapida(pergunta: str) -> dict | None:
    text = _text(pergunta)
    if _quer_codigo_snippet(pergunta) or _quer_literatura_tematica(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}
    if text.startswith(("lembre-se", "lembre se", "lembre que")):
        return {"usar_ferramenta": False, "ferramenta": None}
    if _quer_registrar_no_cerebro(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "registrar_no_cerebro"}
    if _quer_catalogo(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "listar_base_bibliografica"}
    if any(term in text for term in ("buscar na web", "pesquisar na web", "procure na internet")):
        return {"usar_ferramenta": True, "ferramenta": "buscar_web"}
    if _quer_status(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "consultar_status_pipeline"}
    if _quer_limpar(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "limpar_resultados_ml"}
    if _quer_consultar_datasets(pergunta) and not _explicit_execution(text):
        return {"usar_ferramenta": True, "ferramenta": "consultar_datasets"}

    if _explicit_execution(text):
        if any(term in text for term in ("pipeline", "tudo", "ambas", "completo")):
            return {"usar_ferramenta": True, "ferramenta": "executar_pipeline_cientifico"}
        stage_tool = _etapa_mais_avancada_mencionada(pergunta)
        return {
            "usar_ferramenta": True,
            "ferramenta": stage_tool or "executar_comparacao_autoencoders",
        }

    compare_models = "compar" in text and any(term in text for term in ("denso", "lstm", "modelo"))
    if compare_models:
        return {"usar_ferramenta": True, "ferramenta": "consultar_comparacao_autoencoders"}
    if _quer_comparar_abordagens(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "comparar_abordagens_ml"}
    if _asks_results(text):
        return {"usar_ferramenta": True, "ferramenta": "consultar_resultados"}
    return None


def _guardas_criticas(pergunta: str, decisao: dict | None = None) -> dict | None:
    """Impede que pedidos autorais ou conceituais executem cálculo pesado."""

    if _quer_codigo_snippet(pergunta) or _quer_literatura_tematica(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}
    if _quer_resposta_autoral(pergunta) and not _explicit_execution(_text(pergunta)):
        if decisao and str(decisao.get("ferramenta", "")).startswith("executar_"):
            return {"usar_ferramenta": False, "ferramenta": None}
    return decisao


def _specifications() -> list[dict]:
    from src.conhecimento.ferramentas import ESPEC_FERRAMENTAS

    return ESPEC_FERRAMENTAS


def _rotear_por_llm(pergunta: str, llm) -> dict:
    if llm is None:
        return {"usar_ferramenta": False, "ferramenta": None}
    names = [item["name"] for item in _specifications()]
    prompt = (
        "Escolha uma ferramenta somente se a solicitação exigir ação ou leitura "
        "determinística. Responda apenas JSON: "
        '{"usar_ferramenta":true|false,"ferramenta":"nome|null"}.\n'
        f"Ferramentas: {', '.join(names)}\nSolicitação: {pergunta}"
    )
    try:
        raw = texto_da_resposta(llm.invoke(prompt))
        cleaned = re.sub(r"```json?\s*", "", str(raw)).replace("```", "").strip()
        decision = json.loads(cleaned)
    except Exception as exc:
        LOGGER.debug("Roteamento por LLM indisponível: %s", mascarar_segredos(str(exc)))
        return {"usar_ferramenta": False, "ferramenta": None}
    tool = decision.get("ferramenta")
    if tool not in names:
        return {"usar_ferramenta": False, "ferramenta": None}
    return {"usar_ferramenta": bool(decision.get("usar_ferramenta")), "ferramenta": tool}


def decidir_acao(pergunta: str, llm=None) -> dict:
    fast = _decisao_rapida(pergunta)
    if fast is not None:
        return _guardas_criticas(pergunta, fast) or fast
    decision = _rotear_por_llm(pergunta, llm)
    return _guardas_criticas(pergunta, decision) or {
        "usar_ferramenta": False,
        "ferramenta": None,
    }


def _corrigir_descricao_visual(texto: str, imagens=None) -> str:
    """Evita afirmar que uma figura foi exibida quando ela só foi anexada."""

    if not imagens:
        return texto
    corrected = re.sub(
        r"(?i)como (?:vemos|visto) (?:no|na) gr[aá]fico",
        "como indicado nos dados da figura anexada",
        texto,
    )
    paragraphs = corrected.split("\n\n")
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if not (
            re.search(r"(?i)gr[aá]ficos? (?:mostra|mostram|exibe|exibem)", paragraph)
            and any(
                term in normalizar_sem_acentos(paragraph).lower()
                for term in (
                    "distribuicoes de scores",
                    "curvas roc",
                    "ao longo do tempo",
                )
            )
        )
    ]
    captions = [
        normalizar_sem_acentos(str(item.get("caption", ""))).lower()
        for item in imagens
    ]
    if any("comparacao por pontos" in caption for caption in captions):
        paragraphs.append("A figura apresenta a comparação das métricas por pontos.")
    if any("anomalias detectadas" in caption for caption in captions):
        paragraphs.append("A figura apresenta contagens de detecções e a cobertura percentual.")
    return "\n\n".join(paragraphs)


def _dados_sao_inventario(resultado) -> bool:
    if isinstance(resultado, dict):
        if resultado.get("etapa") in {"Base bibliográfica", "Dataset canônico"}:
            return True
        resultado = resultado.get("mensagem", "")
    text = str(resultado)
    return "|---" in text or text.count("\n-") >= 12


def comentar_resultado(pergunta: str, resultado: dict, perfil: str, llm) -> str:
    message = str(resultado.get("mensagem", ""))
    authorial = _quer_resposta_autoral(pergunta)
    if not resultado.get("ok", True) or resultado.get("acao_executada") or llm is None:
        return message
    if (resultado.get("resposta_pronta") or _dados_sao_inventario(resultado)) and not authorial:
        return message
    images = resultado.get("imagens") or []
    visual_inventory = "\n".join(
        f"- {item.get('caption', item.get('path', 'figura sem legenda'))}"
        for item in images
    ) or "- nenhuma figura anexada"
    prompt = f"""Responda ao pesquisador em português usando apenas o resultado abaixo.
Não invente métricas nem suprima ressalvas metodológicas. Perfil: {perfil}.

inventário autoritativo das figuras anexadas:
{visual_inventory}
Não descreva elementos visuais ausentes. As legendas não autorizam inferir
distribuições de scores, curvas ROC ou séries temporais que não estejam nomeadas.

Pergunta: {pergunta}

Resultado determinístico:
{message}
"""
    try:
        try:
            from langchain_core.messages import HumanMessage

            request = [HumanMessage(content=prompt)]
        except ImportError:
            request = prompt
        answer = texto_da_resposta(llm.invoke(request))
        return _corrigir_descricao_visual(answer, resultado.get("imagens"))
    except Exception as exc:
        LOGGER.debug("Comentário por LLM indisponível: %s", mascarar_segredos(str(exc)))
        return message


def processar_com_ferramentas(
    pergunta: str,
    perfil: str,
    llm,
    progresso=None,
    decisao: dict | None = None,
    contexto: str = "",
) -> dict:
    decision = decisao or decidir_acao(pergunta, llm)
    if not decision["usar_ferramenta"]:
        return {
            "usou_ferramenta": False,
            "ferramenta": None,
            "resultado": None,
            "resposta": None,
        }
    tool = decision["ferramenta"]
    LOGGER.info("ferramenta acionada: %s", tool)
    from src.conhecimento.ferramentas import executar_ferramenta

    result = executar_ferramenta(
        tool,
        progresso=progresso,
        pergunta=pergunta,
        llm=llm,
        contexto=contexto,
    )
    return {
        "usou_ferramenta": True,
        "ferramenta": tool,
        "resultado": result,
        "resposta": comentar_resultado(pergunta, result, perfil, llm),
    }


__all__ = [
    "_corrigir_descricao_visual",
    "_dados_sao_inventario",
    "_decisao_rapida",
    "_e_pergunta",
    "_etapa_mais_avancada_mencionada",
    "_guardas_criticas",
    "_quer_codigo_snippet",
    "_rotear_por_llm",
    "comentar_resultado",
    "decidir_acao",
    "processar_com_ferramentas",
]
