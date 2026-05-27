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

from src.core.config import RAIZ_PROJETO
from src.ml.pipeline import (
    NOMES_ETAPAS,
    ORDEM_ETAPAS_ML,
    artefatos_a_partir,
    executar_etapa,
    executar_pipeline_ml,
    limpar_artefatos,
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
    {
        "name": "limpar_resultados_ml",
        "description": (
            "Apaga artefatos/resultados de uma etapa do pipeline e de todas as "
            "etapas posteriores, para permitir recalculo com outro modelo ou "
            "parametrizacao."
        ),
    },
    {
        "name": "buscar_web",
        "description": (
            "Busca rapida na Wikipedia/DuckDuckGo para lookups factuais que "
            "estao FORA da literatura indexada (datas, biografias, definicoes "
            "amplas, eventos, normas tecnicas, padroes). Use quando o usuario "
            "perguntar algo factual sobre o mundo que a base local nao cobre."
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
        "refazer", "refaca", "refaça", "regerar", "regere", "regenerar",
        "rodar de novo", "executar de novo", "do zero", "apagar",
        "recalcular", "recalculo", "recalcule", "recalcula",
    ))


def _quer_status(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    return "status" in txt or "pendente" in txt or "falt" in txt


def _quer_limpar(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    termos = (
        "apagar", "apague", "limpar", "limpe", "zerar", "zere",
        "remover", "remova", "excluir", "exclua", "deletar", "delete",
    )
    return any(t in txt for t in termos)


# Termos que SOZINHOS já implicam a intenção de mexer com o pipeline,
# mesmo sem aparecer "pipeline" ou outro termo de ML explicitamente.
_TERMOS_PIPELINE_IMPLICITO = (
    "recalcular", "recalculo", "recalcule", "recalcula",
    "refazer", "refaca", "refaça", "regerar", "regere", "regenerar",
    "rodar de novo", "executar de novo", "do zero", "rode tudo",
    "rodar tudo", "executar tudo", "pipeline completo",
)


def _parece_pedido_de_ferramenta(pergunta: str) -> bool:
    txt = _normalizar(pergunta)

    # Atalho: "recalcule", "refaça", "do zero" etc. → sempre é pipeline.
    if any(t in txt for t in _TERMOS_PIPELINE_IMPLICITO):
        return True

    termos_ml = (
        "pipeline", "feature", "features", "autoencoder", "falha", "falhas",
        "validacao", "weibull", "rul", "mttf", "auc", "f1", "recall",
        "precision", "limiar", "anomalia", "confiabilidade", "grafico",
        "graficos", "resultado", "resultados", "artefato", "artefatos",
        "modelo", "modelos", "smd", "b10", "detector", "deteccao",
        "treinamento", "metricas", "roc", "imagem", "imagens", "figura",
        "figuras", "curva", "curvas", "plot", "plots", "visualizacao",
        "visualizacoes", "matriz", "matrizes", "heatmap", "tabela",
    )
    termos_acao = (
        "rodar", "rode", "roda", "executar", "execute", "executa",
        "treinar", "treine", "treina", "gerar", "gere", "gera",
        "refazer", "refaca", "regerar", "regere", "calcular", "calcule",
        "calcula", "validar", "valide", "valida", "injetar", "injete",
        "injeta", "estimar", "estime", "estima", "mostrar", "mostre",
        "mostra", "consultar", "consulte", "consulta", "ver", "vejo",
        "status", "quais", "qual", "quanto", "apagar", "apague", "limpar",
        "limpe", "zerar", "zere", "remover", "remova", "excluir", "exclua",
        "deletar", "delete", "fazer", "faca", "faça", "cade", "onde",
        "tem", "existe", "existem", "ha", "há",
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


def _etapa_por_pergunta(pergunta: str) -> str | None:
    txt = _normalizar(pergunta)
    if "pipeline" in txt or "tudo" in txt or "todos" in txt or "do zero" in txt:
        return "features_ca"
    if "feature" in txt or "sinais" in txt or "dados processados" in txt:
        return "features_ca"
    if "autoencoder" in txt or "detector" in txt or "limiar" in txt:
        return "autoencoder"
    if "injec" in txt or "falha" in txt or "smd" in txt:
        return "injecao_falhas"
    if "valid" in txt or "auc" in txt or "f1" in txt or "recall" in txt:
        return "validacao"
    if "weibull" in txt or "rul" in txt or "mttf" in txt or "b10" in txt:
        return "rul_weibull"
    return None


def limpar_resultados_ml(progresso=None, pergunta: str = "") -> dict:
    etapa = _etapa_por_pergunta(pergunta)
    if etapa is None:
        opcoes = ", ".join(NOMES_ETAPAS[key] for key in ORDEM_ETAPAS_ML)
        return {
            "ok": False,
            "etapa": "Limpeza de resultados",
            "mensagem": (
                "Diga a partir de qual etapa devo apagar os artefatos. "
                f"Opcoes: {opcoes}."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }

    if progresso:
        progresso(f"Apagando artefatos a partir de: {NOMES_ETAPAS[etapa]}...")

    alvos = artefatos_a_partir(etapa)
    removidos = limpar_artefatos(etapa)
    if removidos:
        linhas = "\n".join(
            f"- {path.relative_to(RAIZ_PROJETO)}"
            for path in removidos
        )
        detalhe = f"\n\nArquivos removidos:\n{linhas}"
    else:
        detalhe = "\n\nNao havia arquivos existentes para remover nessa selecao."

    return {
        "ok": True,
        "etapa": "Limpeza de resultados",
        "mensagem": (
            f"Resultados apagados a partir de **{NOMES_ETAPAS[etapa]}**. "
            f"As etapas seguintes tambem foram invalidadas para recalculo. "
            f"Artefatos verificados: {len(alvos)}."
            f"{detalhe}\n\n"
            "Quando quiser recalcular, peca pelo chat: "
            f"\"rode {NOMES_ETAPAS[etapa]}\" ou \"rode o pipeline completo\"."
        ),
        "imagens": [],
        "resposta_pronta": True,
    }


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


def buscar_na_web(progresso=None, pergunta: str = "") -> dict:
    """Adapta src.conhecimento.web_search.buscar_web para o formato de ferramenta."""
    from src.conhecimento.web_search import buscar_web

    if progresso:
        progresso(f"Pesquisando na web: '{pergunta[:60]}'...")

    termo = (pergunta or "").strip()
    # Remove gatilhos de comando para deixar só o termo
    for gat in (
        "buscar na web", "pesquisar na web", "pesquise na web", "busque na web",
        "buscar online", "pesquisar online", "procure na internet",
        "procure online", "na internet", "na web", "buscar", "pesquisar",
        "procurar", "google", "googlar",
    ):
        termo = re.sub(rf"\b{gat}\b", "", termo, flags=re.IGNORECASE)
    termo = termo.strip(" ,.;?!:")

    if not termo:
        return {
            "ok": False,
            "etapa": "Busca na web",
            "mensagem": "Me diga o que quer pesquisar (ex.: 'pesquise na web sobre IEC 61724').",
            "imagens": [],
            "resposta_pronta": True,
        }

    out = buscar_web(termo)
    return {
        "ok": bool(out["ok"]),
        "etapa": "Busca na web",
        "mensagem": out["mensagem"],
        "imagens": [],
        "resposta_pronta": False,  # passa pelo LLM para integrar com o contexto
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
    "limpar_resultados_ml": limpar_resultados_ml,
    "buscar_web": buscar_na_web,
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


# Mapeamento "ordem cronológica" do pipeline → nome da ferramenta.
# Quando o usuário menciona várias etapas, pegamos a MAIS AVANÇADA — assim
# 'auto_deps=True' garante que tudo até ela rode em ordem.
_ETAPA_ORDEM = [
    ("feature", "rodar_features_ca"),
    ("sinais", "rodar_features_ca"),
    ("autoencoder", "rodar_autoencoder"),
    ("detector", "rodar_autoencoder"),
    ("limiar", "rodar_autoencoder"),
    ("injec", "rodar_injecao_falhas"),
    ("injet", "rodar_injecao_falhas"),
    ("falha sint", "rodar_injecao_falhas"),
    ("smd", "rodar_injecao_falhas"),
    ("valid", "rodar_validacao"),
    ("auc", "rodar_validacao"),
    ("f1", "rodar_validacao"),
    ("recall", "rodar_validacao"),
    ("precision", "rodar_validacao"),
    ("roc", "rodar_validacao"),
    ("weibull", "rodar_weibull"),
    ("rul", "rodar_weibull"),
    ("mttf", "rodar_weibull"),
    ("b10", "rodar_weibull"),
    ("confiabilidade", "rodar_weibull"),
]


def _etapa_mais_avancada_mencionada(txt_normalizado: str) -> str | None:
    """
    Retorna o nome da ferramenta correspondente à etapa MAIS AVANÇADA
    mencionada no texto, ou None se nenhuma for mencionada.
    """
    mais_avancada = None
    for chave, ferramenta in _ETAPA_ORDEM:
        if chave in txt_normalizado:
            mais_avancada = ferramenta
    return mais_avancada


_GATILHOS_WEB = (
    "buscar na web", "pesquisar na web", "pesquise na web", "busque na web",
    "buscar online", "pesquisar online", "procure na internet",
    "procure online", "googlar", "no google", "na internet",
    "consulte a wikipedia", "consultar a wikipedia", "wikipedia",
    "qual a definicao oficial", "norma iec", "norma iso", "norma abnt",
)


def _decisao_rapida(pergunta: str) -> dict | None:
    txt = _normalizar(pergunta)

    # Busca na web — atalho prioritário quando gatilho explícito aparece
    if any(g in txt for g in _GATILHOS_WEB):
        return {"usar_ferramenta": True, "ferramenta": "buscar_web"}

    if not _parece_pedido_de_ferramenta(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}

    # Limpeza explícita ("apague", "limpe...") tem prioridade sobre tudo.
    if _quer_limpar(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "limpar_resultados_ml"}

    # Status do pipeline ("o que está pendente?")
    if _quer_status(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "consultar_status_pipeline"}

    # "Recalcule", "refaça", "rode tudo de novo", "do zero" → pipeline completo.
    if any(t in txt for t in _TERMOS_PIPELINE_IMPLICITO):
        # Quando há ETAPAS específicas mencionadas, roteia para a mais AVANÇADA
        # (com auto_deps=True a etapa puxa todas anteriores que faltarem).
        etapa_mais_avancada = _etapa_mais_avancada_mencionada(txt)
        if etapa_mais_avancada:
            return {"usar_ferramenta": True, "ferramenta": etapa_mais_avancada}
        # Sem etapa específica, recalcular = pipeline inteiro.
        return {"usar_ferramenta": True, "ferramenta": "rodar_pipeline_completo"}

    termos_executar = (
        "rodar", "rode", "roda", "execut", "trein", "treine", "treina",
        "gerar", "gere", "gera", "calcular", "calcule", "calcula",
        "validar", "valide", "valida", "injetar", "injete", "injeta",
        "estimar", "estime", "estima", "fazer", "faca", "faça",
    )
    if any(t in txt for t in termos_executar):
        if "pipeline" in txt or "tudo" in txt or "todos" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_pipeline_completo"}
        # Roteia para a etapa mais avançada mencionada (auto_deps roda o resto)
        etapa_mais_avancada = _etapa_mais_avancada_mencionada(txt)
        if etapa_mais_avancada:
            return {"usar_ferramenta": True, "ferramenta": etapa_mais_avancada}
        # Pedido genérico de "gere os resultados" — interpreta como pipeline.
        if "resultado" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_pipeline_completo"}

    # Consulta passiva ("mostre", "quais foram...", "cadê as imagens?")
    termos_consulta = (
        "resultado", "resultados", "mostrar", "mostre", "mostra", "grafico",
        "graficos", "auc", "f1", "mttf", "b10", "smd", "limiar",
        "metrica", "metricas", "imagem", "imagens", "figura", "figuras",
        "curva", "curvas", "plot", "plots", "visualizacao", "matriz",
        "heatmap", "roc", "tabela",
    )
    termos_acao_ativa = (
        "rodar", "execut", "trein", "gerar", "gere", "calcular",
        "validar", "injetar", "refazer", "regerar", "recalc",
    )
    if any(t in txt for t in termos_consulta):
        if not any(t in txt for t in termos_acao_ativa):
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
