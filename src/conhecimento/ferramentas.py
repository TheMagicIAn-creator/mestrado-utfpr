"""
ferramentas.py - Al IAdo PV
Camada de tool calling para o chat.

As ferramentas sao a ponte entre linguagem natural e pipeline de ML:
o usuario pede no prompt, a ferramenta executa/consulta, e o resultado
volta na propria conversa.
"""

from __future__ import annotations

import json
import re
import unicodedata

from src.ml.pipeline import (
    NOMES_ETAPAS,
    executar_etapa,
    executar_pipeline_ml,
    pipeline_status,
)
from src.ml.resultados import resumir_resultados


ESPEC_FERRAMENTAS = [
    {
        "name": "rodar_features_ca",
        "description": (
            "Extrai features eletricas CA do dataset de Paderborn. Use quando "
            "o usuario pedir para extrair features ou preparar dados."
        ),
    },
    {
        "name": "rodar_autoencoder",
        "description": (
            "Treina o Autoencoder de normalidade. Depende das features CA. Use "
            "quando o usuario pedir treinamento do detector de anomalias."
        ),
    },
    {
        "name": "rodar_injecao_falhas",
        "description": (
            "Executa injecao de falhas sinteticas fundamentadas no FMEA. Use "
            "quando o usuario pedir simulacao ou injecao de falhas."
        ),
    },
    {
        "name": "rodar_validacao",
        "description": (
            "Calcula metricas formais: AUC-ROC, F1, Recall, Precision. Use "
            "quando o usuario pedir validacao ou avaliacao do detector."
        ),
    },
    {
        "name": "rodar_weibull",
        "description": (
            "Executa analise de Weibull e RUL. Use quando o usuario pedir "
            "confiabilidade, MTTF, B10, RUL ou vida util remanescente."
        ),
    },
    {
        "name": "rodar_pipeline_completo",
        "description": (
            "Executa o pipeline completo em ordem: features, autoencoder, "
            "injecao de falhas, validacao e Weibull."
        ),
    },
    {
        "name": "consultar_resultados",
        "description": (
            "Mostra resultados ja existentes do pipeline: limiar, SMD, AUC, "
            "F1, Weibull, RUL e graficos quando solicitados."
        ),
    },
    {
        "name": "consultar_status_pipeline",
        "description": (
            "Mostra quais etapas do pipeline ja estao prontas e quais ainda "
            "estao pendentes."
        ),
    },
]


_STAGE_BY_TOOL = {
    "rodar_features_ca": "features_ca",
    "rodar_autoencoder": "autoencoder",
    "rodar_injecao_falhas": "injecao_falhas",
    "rodar_validacao": "validacao",
    "rodar_weibull": "rul_weibull",
}


def _normalizar(texto: str) -> str:
    texto = texto.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _deve_forcar(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    return any(t in txt for t in (
        "refazer", "regerar", "regenerar", "rodar de novo",
        "executar de novo", "do zero", "apagar", "recalcular",
    ))


def _quer_status(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    return "status" in txt or "pendente" in txt or "falt" in txt


def _parece_pedido_de_ferramenta(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    termos_ml = (
        "pipeline", "feature", "features", "autoencoder", "falha", "falhas",
        "validacao", "weibull", "rul", "mttf", "auc", "f1", "recall",
        "precision", "limiar", "anomalia", "confiabilidade", "grafico",
        "graficos", "resultado", "resultados", "smd", "b10",
    )
    termos_acao = (
        "rodar", "rode", "executar", "execute", "treinar", "treine",
        "gerar", "gere", "refazer", "regerar", "calcular", "calcule",
        "validar", "valide", "injetar", "injete", "estimar", "estime",
        "mostrar", "mostre", "consultar", "consulte", "ver", "status",
        "quais", "qual", "quanto",
    )
    return any(t in txt for t in termos_ml) and any(t in txt for t in termos_acao)


def consultar_status_pipeline(progresso=None, pergunta: str = "") -> dict:
    if progresso:
        progresso("Lendo status do pipeline...")

    status = pipeline_status()
    linhas = ["## Status do pipeline de ML\n"]
    for key, pronto in status.items():
        estado = "pronto" if pronto else "pendente"
        linhas.append(f"- **{NOMES_ETAPAS[key]}**: {estado}")

    return {
        "ok": True,
        "etapa": "Status do pipeline",
        "mensagem": "\n".join(linhas),
        "imagens": [],
        "resposta_pronta": True,
    }


def consultar_resultados(progresso=None, pergunta: str = "") -> dict:
    if progresso:
        progresso("Lendo artefatos de resultado...")
    return resumir_resultados(pergunta)


def _resultado_pos_execucao(stage_key: str, pergunta: str) -> dict:
    if stage_key == "features_ca":
        return {
            "mensagem": (
                "Features CA extraidas e prontas para o Autoencoder. "
                "A proxima etapa metodologica e treinar o modelo de normalidade."
            ),
            "imagens": [],
        }
    foco = {
        "autoencoder": "autoencoder",
        "injecao_falhas": "injecao falhas smd",
        "validacao": "validacao auc f1 recall",
        "rul_weibull": "weibull rul mttf b10",
    }[stage_key]
    return resumir_resultados(f"{pergunta} {foco}")


def _rodar_stage(stage_key: str, progresso=None, pergunta: str = "") -> dict:
    force = _deve_forcar(pergunta)
    res = executar_etapa(stage_key, force=force, progresso=progresso)

    if not res["ok"]:
        return {
            "ok": False,
            "etapa": res["etapa"],
            "mensagem": res["mensagem"],
            "imagens": [],
            "resposta_pronta": True,
        }

    resumo = _resultado_pos_execucao(stage_key, pergunta)
    mensagem = f"{res['mensagem']}\n\n{resumo['mensagem']}"
    return {
        "ok": True,
        "etapa": res["etapa"],
        "mensagem": mensagem,
        "imagens": resumo.get("imagens", []),
        "resposta_pronta": True,
    }


def rodar_features_ca(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("features_ca", progresso, pergunta)


def rodar_autoencoder(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("autoencoder", progresso, pergunta)


def rodar_injecao_falhas(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("injecao_falhas", progresso, pergunta)


def rodar_validacao(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("validacao", progresso, pergunta)


def rodar_weibull(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("rul_weibull", progresso, pergunta)


def rodar_pipeline_completo(progresso=None, pergunta: str = "") -> dict:
    force = _deve_forcar(pergunta)
    resultados = executar_pipeline_ml("features_ca", force=force, progresso=progresso)
    ok = all(not r.startswith("ERRO") for r in resultados)
    resumo = resumir_resultados(pergunta)
    mensagem = "## Execucao do pipeline\n\n" + "\n".join(f"- {r}" for r in resultados)
    if ok:
        mensagem += "\n\n" + resumo["mensagem"]
    return {
        "ok": ok,
        "etapa": "Pipeline completo",
        "mensagem": mensagem,
        "imagens": resumo.get("imagens", []),
        "resposta_pronta": True,
    }


_DESPACHO = {
    "rodar_features_ca": rodar_features_ca,
    "rodar_autoencoder": rodar_autoencoder,
    "rodar_injecao_falhas": rodar_injecao_falhas,
    "rodar_validacao": rodar_validacao,
    "rodar_weibull": rodar_weibull,
    "rodar_pipeline_completo": rodar_pipeline_completo,
    "consultar_resultados": consultar_resultados,
    "consultar_status_pipeline": consultar_status_pipeline,
}


def executar_ferramenta(nome: str, progresso=None, pergunta: str = "") -> dict:
    funcao = _DESPACHO.get(nome)
    if funcao is None:
        return {
            "ok": False,
            "etapa": nome,
            "mensagem": f"Ferramenta desconhecida: {nome}",
            "imagens": [],
            "resposta_pronta": True,
        }
    return funcao(progresso=progresso, pergunta=pergunta)


def _decisao_rapida(pergunta: str) -> dict | None:
    txt = _normalizar(pergunta)
    if not _parece_pedido_de_ferramenta(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}
    if _quer_status(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "consultar_status_pipeline"}
    if any(t in txt for t in ("rodar", "rode", "execut", "trein", "treine", "gerar", "calcular", "calcule", "validar", "valide", "injetar", "injete", "refazer", "regerar", "estimar", "estime")):
        if "pipeline" in txt or "tudo" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_pipeline_completo"}
        if "autoencoder" in txt or "detector" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_autoencoder"}
        if "weibull" in txt or "rul" in txt or "mttf" in txt or "b10" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_weibull"}
        if "valid" in txt or "auc" in txt or "f1" in txt or "recall" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_validacao"}
        if "injet" in txt or "falha" in txt or "smd" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_injecao_falhas"}
        if "feature" in txt or "sinais" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_features_ca"}
    if any(t in txt for t in ("resultado", "resultados", "mostrar", "mostre", "grafico", "graficos", "auc", "f1", "mttf", "b10", "smd", "limiar")):
        if not any(t in txt for t in ("rodar", "execut", "trein", "gerar", "calcular", "validar", "injetar", "refazer", "regerar")):
            return {"usar_ferramenta": True, "ferramenta": "consultar_resultados"}
    return None


def decidir_acao(pergunta: str, llm) -> dict:
    """
    Decide se a pergunta deve acionar uma ferramenta.
    Usa regras rapidas primeiro e LLM apenas quando a intencao e ambigua.
    """
    decisao = _decisao_rapida(pergunta)
    if decisao is not None:
        return decisao

    catalogo = "\n".join(f"- {f['name']}: {f['description']}" for f in ESPEC_FERRAMENTAS)
    prompt = f"""Voce roteia pedidos para ferramentas de ML.

Ferramentas:
{catalogo}

Mensagem do usuario:
"{pergunta}"

Responda apenas JSON valido:
{{"usar_ferramenta": true/false, "ferramenta": "nome_ou_null"}}
"""
    try:
        from langchain_core.messages import HumanMessage

        resposta = llm.invoke([HumanMessage(content=prompt)]).content
        limpo = re.sub(r"```json?\n?", "", resposta.strip()).replace("```", "").strip()
        dados = json.loads(limpo)
        nomes = {f["name"] for f in ESPEC_FERRAMENTAS}
        ferramenta = dados.get("ferramenta")
        if dados.get("usar_ferramenta") and ferramenta in nomes:
            return {"usar_ferramenta": True, "ferramenta": ferramenta}
    except Exception:
        pass
    return {"usar_ferramenta": False, "ferramenta": None}


def comentar_resultado(pergunta: str, resultado: dict, perfil: str, llm) -> str:
    if resultado.get("resposta_pronta"):
        return resultado.get("mensagem", "")

    status = "SUCESSO" if resultado.get("ok") else "FALHA"
    prompt = f"""{perfil}

Rodolfo pediu: "{pergunta}"

Resultado tecnico ({status}):
{resultado.get('mensagem', 'sem detalhes')}

Explique de forma natural, humana e tecnicamente precisa. Nao invente numeros."""
    try:
        from langchain_core.messages import HumanMessage

        return llm.invoke([HumanMessage(content=prompt)]).content
    except Exception:
        return resultado.get("mensagem", "")


def processar_com_ferramentas(pergunta: str,
                              perfil: str,
                              llm,
                              progresso=None,
                              decisao: dict | None = None) -> dict:
    if decisao is None:
        decisao = decidir_acao(pergunta, llm)

    if not decisao["usar_ferramenta"]:
        return {
            "usou_ferramenta": False,
            "ferramenta": None,
            "resultado": None,
            "resposta": None,
        }

    ferramenta = decisao["ferramenta"]
    if progresso:
        progresso(f"Acionando ferramenta: {ferramenta}")

    resultado = executar_ferramenta(
        ferramenta,
        progresso=progresso,
        pergunta=pergunta,
    )
    resposta = comentar_resultado(pergunta, resultado, perfil, llm)

    return {
        "usou_ferramenta": True,
        "ferramenta": ferramenta,
        "resultado": resultado,
        "resposta": resposta,
    }
