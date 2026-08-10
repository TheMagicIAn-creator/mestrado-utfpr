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
import shutil

from src.core.config import RAIZ_PROJETO
from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos
from src.core.texto import normalizar_sem_acentos as _normalizar
from src.conhecimento.provedores import texto_da_resposta
from src.ml.pipeline import (
    NOMES_ETAPAS,
    ORDEM_ETAPAS_ML,
    artefatos_a_partir,
    capacidade_recalculo_pipeline,
    estado_pipeline,
    estado_resultados_publicados,
    executar_etapa,
    executar_pipeline_ml,
    limpar_artefatos,
)
from src.ml.resultados import resumir_resultados

_logger = get_logger("conhecimento.ferramentas")

ESPEC_FERRAMENTAS = [
    {
        "name": "rodar_features_ca",
        "description": (
            "Extrai as 24 features eletricas do GPVS-Faults experimental. "
            "O nome da ferramenta e mantido por compatibilidade. Use quando o "
            "usuario pedir para extrair features ou preparar dados."
        ),
    },
    {
        "name": "rodar_autoencoder",
        "description": (
            "Treina o Autoencoder de normalidade em F0L/F0M do GPVS-Faults. Use "
            "quando o usuario pedir treinamento do detector de anomalias."
        ),
    },
    {
        "name": "rodar_injecao_falhas",
        "description": (
            "Executa injecao de falhas sinteticas fundamentadas na FMECA. Use "
            "quando o usuario pedir simulacao ou injecao de falhas."
        ),
    },
    {
        "name": "rodar_validacao",
        "description": (
            "Calcula a validacao E2 orientada pela FMECA e a validacao E3 nos "
            "14 ensaios reais F1L-F7M. Use ao avaliar o detector."
        ),
    },
    {
        "name": "rodar_weibull",
        "description": (
            "Executa Weibull exploratoria da magnitude de detectabilidade E2. "
            "Nao estima RUL, MTTF ou vida fisica. Use em pedidos de Weibull ou "
            "detectabilidade."
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
            "sensibilidade, especificidade e Weibull de detectabilidade."
        ),
    },
    {
        "name": "consultar_comparacao_macro",
        "description": (
            "Compara o METODO PROPOSTO (Autoencoder denso + MSE p99) "
            "com a LITERATURA (AE-LSTM temporal do Ibrahim 2022) por AUC e por "
            "SMD, lendo a comparacao ja publicada. Nunca treina. Use sempre "
            "que o usuario pedir para comparar o metodo dele com a literatura, "
            "com o Ibrahim ou com o AE-LSTM: 'compare meu metodo com a "
            "literatura', 'como estou frente ao Ibrahim', 'meu detector e "
            "melhor que o AE-LSTM?', 'mostre a comparacao', 'qual o SMD'."
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
        "name": "registrar_no_cerebro",
        "description": (
            "REGISTRA conhecimento curado como nota no Cerebro/ do vault "
            "Obsidian (conceito, decisao metodologica, resultado validado, "
            "hipotese). Use quando o pesquisador pedir para GUARDAR, REGISTRAR, "
            "ANOTAR ou DOCUMENTAR algo no cerebro/vault/Obsidian — ex.: "
            "'guarde esse resultado no cerebro', 'registre essa decisao', "
            "'anote isso no vault'. Voce deve REDIGIR o texto da nota em "
            "Markdown e passar: titulo, conteudo, tipo (conceito|decisao|"
            "resultado|contexto|hipotese|experimento), tags (nos comuns da "
            "dissertacao: fmea, fmeca, rcm, manutencao, confiabilidade, "
            "weibull-rul, inversor-pv, contator-ac, igbt, fusivel-ac, "
            "autoencoder, deteccao-anomalia, escore-localizado, "
            "machine-learning, sinais-eletricos, paderborn, evidencia-e2, "
            "comparacao-literatura, metodologia), nivel_evidencia "
            "(projeto|E1|E2|literatura) e fonte (artefato de origem, se "
            "houver). NAO use para memorizar preferencia do usuario (isso e a "
            "memoria validada) nem para consultar notas existentes (isso e RAG)."
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
    {
        "name": "listar_base_bibliografica",
        "description": (
            "Lista o CATALOGO COMPLETO da literatura indexada (todos os "
            "documentos da base, agrupados por tema). Use quando o usuario "
            "pedir o INVENTARIO inteiro: 'liste todas as referencias', 'o que "
            "voce tem indexado', 'mostre a base bibliografica completa', "
            "'quantos artigos voce tem', 'todas as 39'. NAO use para busca "
            "tematica ('artigos sobre anomalias') — isso e RAG."
        ),
    },
    {
        "name": "consultar_datasets",
        "description": (
            "Explica o GPVS-Faults, unico dataset do pipeline principal, e "
            "separa candidatos e experimentos legados. Distingue E2, E3 de "
            "bancada e os requisitos ausentes para Weibull fisico."
        ),
    },
    {
        "name": "comparar_abordagens_ml",
        "description": (
            "Explica e compara as abordagens de ML do projeto: supervisionada "
            "(classifica falhas CC conhecidas), nao supervisionada (aprende "
            "normalidade e detecta anomalia CA) e sintetica orientada pela FMECA. "
            "Use quando o usuario pedir a diferenca entre supervisionado e nao "
            "supervisionado, ou comparar as abordagens."
        ),
    },
    {
        "name": "treinar_classificador_pv",
        "description": (
            "Treina e salva o classificador supervisionado PV Farms (falhas CC). "
            "Use quando o usuario pedir para treinar/retreinar o classificador PV."
        ),
    },
    {
        "name": "avaliar_classificador_pv",
        "description": (
            "Mostra metricas, melhor modelo e limitacoes do classificador PV "
            "Farms ja treinado. Use ao pedir desempenho/metricas do classificador."
        ),
    },
    {
        "name": "classificar_amostra_pv",
        "description": (
            "Classifica uma amostra PV Farms enviada como JSON. Use quando o "
            "usuario pedir para classificar uma amostra. Valida colunas e avisa "
            "que e dominio CC (nao diagnostica falhas CA)."
        ),
    },
]


_STAGE_BY_TOOL = {
    "rodar_features_ca": "features_gpvs",
    "rodar_autoencoder": "autoencoder",
    "rodar_injecao_falhas": "injecao_falhas",
    "rodar_validacao": "validacao",
    "rodar_weibull": "rul_weibull",
}


from src.conhecimento.intencoes_ferramentas import (
    _deve_forcar,
    _quer_status,
    _TERMOS_BIBLIO,
    _TERMOS_TOTALIDADE,
    _GATILHOS_CATALOGO_FORTE,
    _QUALIFICADORES_TOPICO,
    _quer_catalogo,
    _AUTORES_EXP,
    _AUTORES_CITAVEIS,
    _VERBOS_RODAR_EXP,
    _experimentos_alvo,
    _quer_rodar_experimento,
    _quer_catalogo_experimentos,
    _quer_limpar_experimentos,
    _quer_literatura_tematica,
    _quer_consultar_datasets,
    _quer_comparar_abordagens,
    _quer_classificador_pv,
    _quer_resposta_autoral,
    _quer_comparar_auc_experimentos,
    _quer_consultar_resultados_experimentos,
    _quer_registrar_no_cerebro,
    _quer_limpar,
    _TERMOS_PIPELINE_IMPLICITO,
    _parece_pedido_de_ferramenta,
)


def consultar_status_pipeline(progresso=None, pergunta: str = "") -> dict:
    if progresso:
        progresso("Lendo status do pipeline...")

    capacidade = capacidade_recalculo_pipeline()
    if not capacidade["disponivel"]:
        publicados = estado_resultados_publicados()
        linhas = [
            "## Pipeline de ML — modo de consulta\n",
            "O site não contém os CSVs brutos do GPVS-Faults, portanto não "
            "executa treinamento na nuvem. Ele consulta a última execução "
            "local publicada no repositório.\n",
        ]
        for key in ORDEM_ETAPAS_ML:
            info = publicados[key]
            marcador = "✅ disponível" if info["disponivel"] else "⬜ ausente"
            linhas.append(f"- **{NOMES_ETAPAS[key]}**: {marcador}")
        linhas.append(
            "\n_Esses itens são artefatos recalculados no PC e publicados para "
            "consulta; não representam uma nova execução no servidor._"
        )
        return {
            "ok": True,
            "etapa": "Status do pipeline",
            "mensagem": "\n".join(linhas),
            "imagens": [],
            "resposta_pronta": True,
        }

    estados = estado_pipeline()
    rotulo = {"ready": "✅ pronto", "stale": "⚠️ desatualizado (stale)",
              "pending": "⬜ pendente"}
    linhas = ["## Status do pipeline de ML\n"]
    for key in ORDEM_ETAPAS_ML:
        info = estados[key]
        txt = rotulo.get(info["estado"], info["estado"])
        if info["estado"] == "stale" and info.get("motivos"):
            txt += f" — {', '.join(info['motivos'])}"
        linhas.append(f"- **{NOMES_ETAPAS[key]}**: {txt}")
    linhas.append(
        "\n_stale = artefato existe mas o código, os parâmetros ou um artefato "
        "anterior mudaram; recalcule sob comando para revalidar._"
    )

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
    if any(t in txt for t in (
        "pipeline", "tudo", "todos", "do zero", "everything", "all",
        "from scratch", "todo", "todos", "desde cero", "tout", "tous",
        "depuis zero", "depuis zéro",
    )):
        return "features_gpvs"
    if any(t in txt for t in (
        "feature", "features", "sinais", "dados processados",
        "signals", "processed data", "senales", "señales", "datos procesados",
        "signaux", "donnees traitees", "données traitées",
    )):
        return "features_gpvs"
    if any(t in txt for t in ("autoencoder", "detector", "limiar", "threshold", "umbral", "seuil")):
        return "autoencoder"
    if any(t in txt for t in ("injec", "falha", "failure", "fault", "falla", "defaillance", "smd")):
        return "injecao_falhas"
    if any(t in txt for t in ("valid", "auc", "f1", "recall")):
        return "validacao"
    if any(t in txt for t in ("weibull", "rul", "mttf", "b10", "reliability", "confiabilidad", "fiabilite")):
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

    # ── Confirmação em DUAS ETAPAS (item 10.2) ───────────────────
    # Nenhuma exclusão ocorre sem o token explícito na mensagem do usuário.
    token = f"CONFIRMAR LIMPEZA {etapa.upper()}"
    if _normalizar(token) not in _normalizar(pergunta):
        existentes = [p for p in artefatos_a_partir(etapa) if p.exists()]
        return {
            "ok": True,
            "etapa": "Limpeza de resultados",
            "mensagem": (
                f"⚠️ Isso vai **apagar {len(existentes)} artefato(s)** a partir de "
                f"**{NOMES_ETAPAS[etapa]}** e invalidar as etapas seguintes. "
                f"A ação é irreversível.\n\n"
                f"Para confirmar, escreva exatamente:\n\n`{token}`"
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
        "acao_executada": True,   # apagou artefato: o fato e literal
    }


def _resultado_pos_execucao(stage_key: str, pergunta: str) -> dict:
    if stage_key == "features_gpvs":
        return {
            "mensagem": (
                "Features GPVS extraidas e prontas para o Autoencoder. "
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
    if not capacidade_recalculo_pipeline()["disponivel"]:
        resumo = _resultado_pos_execucao(stage_key, pergunta)
        return {
            "ok": True,
            "etapa": NOMES_ETAPAS[stage_key],
            "mensagem": (
                "## Cálculo indisponível neste ambiente\n\n"
                "Os 16 CSVs brutos do GPVS-Faults não são publicados no Streamlit "
                "Cloud. Por isso, o site não pode retreinar esta etapa. Abaixo "
                "está a última execução local publicada.\n\n"
                + resumo["mensagem"]
            ),
            "imagens": resumo.get("imagens", []),
            "resposta_pronta": True,
        }

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
    return _rodar_stage("features_gpvs", progresso, pergunta)


def rodar_autoencoder(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("autoencoder", progresso, pergunta)


def rodar_injecao_falhas(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("injecao_falhas", progresso, pergunta)


def rodar_validacao(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("validacao", progresso, pergunta)


def rodar_weibull(progresso=None, pergunta: str = "") -> dict:
    return _rodar_stage("rul_weibull", progresso, pergunta)


def rodar_pipeline_completo(progresso=None, pergunta: str = "") -> dict:
    if not capacidade_recalculo_pipeline()["disponivel"]:
        status = consultar_status_pipeline(pergunta=pergunta)
        return {
            "ok": True,
            "etapa": "Pipeline completo",
            "mensagem": (
                "## Cálculo indisponível neste ambiente\n\n"
                "O pipeline pesado só pode ser recalculado no PC que contém "
                "os 16 CSVs em `dados/brutos/gpvs/csv/CSV_Files/`. O site está em modo de "
                "consulta e preserva a última execução local publicada.\n\n"
                + status["mensagem"]
                + "\n\nConsulte uma parte por vez, por exemplo: `mostre os "
                  "resultados de validação`, `compare os experimentos de "
                  "anomalia` ou `interprete a análise de Weibull`."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }

    force = _deve_forcar(pergunta)
    resultados = executar_pipeline_ml("features_gpvs", force=force, progresso=progresso)
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


def _redigir_nota_com_llm(pergunta: str, contexto: str, llm) -> dict | None:
    """Pede ao LLM que REDIJA a nota a partir do que acabou de ser discutido.

    O pedido do pesquisador costuma ser dêitico ("guarde ESSE resultado"), então
    o conteúdo só existe no contexto da conversa — não na frase. Retorna None se
    o LLM falhar (o chamador então pede título/conteúdo explicitamente).
    """
    if llm is None or not str(contexto).strip():
        return None
    from src.conhecimento.nota_cerebro import TAGS_VALIDAS, TIPOS

    prompt = f"""Voce redige uma NOTA CURADA para o vault Obsidian do mestrado.

Conversa recente (o pesquisador quer guardar algo DAQUI):
---
{str(contexto)[-6000:]}
---

Pedido: "{pergunta}"

Escreva a nota em Markdown. Regras:
- Registre o que foi efetivamente discutido/obtido — NAO invente numero nem fonte.
- Se houver metricas, cite-as exatamente como apareceram.
- Seja conciso e util para consulta futura (o pesquisador vai reler isto meses depois).
- Marque ressalvas de evidencia quando os dados forem sinteticos (E2).

Responda APENAS JSON valido:
{{"titulo": "...", "conteudo": "markdown da nota",
  "tipo": "um de: {'|'.join(sorted(TIPOS))}",
  "tags": ["escolha entre: {', '.join(sorted(TAGS_VALIDAS))}"],
  "nivel_evidencia": "projeto|E1|E2|literatura"}}
"""
    try:
        try:
            from langchain_core.messages import HumanMessage

            entrada = [HumanMessage(content=prompt)]
        except ImportError:
            entrada = prompt
        bruto = texto_da_resposta(llm.invoke(entrada))
        limpo = re.sub(r"```json?\n?", "", str(bruto).strip()).replace("```", "").strip()
        dados = json.loads(limpo)
        if dados.get("titulo") and dados.get("conteudo"):
            return dados
    except Exception as exc:
        _logger.warning(
            "LLM não estruturou a nota; usando entrada explícita: %s",
            mascarar_segredos(str(exc)),
        )
    return None


def registrar_no_cerebro(progresso=None, pergunta: str = "",
                         titulo: str = "", conteudo: str = "",
                         tipo: str = "contexto", tags: list | None = None,
                         nivel_evidencia: str = "projeto",
                         fonte: str = "", llm=None, contexto: str = "") -> dict:
    """Registra conhecimento curado como NOTA no `Cerebro/` do vault Obsidian.

    É a peça que faltava para o agente usar o vault como REPOSITÓRIO (e não só
    como leitura): fecha o ciclo entrada → consulta → saída → **registro**.

    Sem `titulo`/`conteudo` explícitos, orienta o LLM a fornecê-los — o texto da
    nota é redigido por ele, não extraído por regex da pergunta.
    """
    from src.conhecimento.nota_cerebro import (
        TAGS_VALIDAS, TIPOS, registrar_nota_cerebro,
    )

    # Sem título/conteúdo explícitos, o LLM REDIGE a nota a partir da conversa —
    # é o caso normal, já que o pedido costuma ser "guarde ESSE resultado".
    if not titulo.strip() or not conteudo.strip():
        if progresso:
            progresso("Redigindo a nota a partir da conversa...")
        redigida = _redigir_nota_com_llm(pergunta, contexto, llm)
        if redigida:
            titulo = redigida.get("titulo", "") or titulo
            conteudo = redigida.get("conteudo", "") or conteudo
            tipo = redigida.get("tipo", tipo) or tipo
            tags = redigida.get("tags", tags)
            nivel_evidencia = redigida.get("nivel_evidencia", nivel_evidencia)

    if not titulo.strip() or not conteudo.strip():
        # Distingue os dois motivos: sem conversa anterior (app recém-aberto,
        # histórico vazio) x falha ao redigir. Antes, os dois davam a mesma
        # mensagem vaga, e o pesquisador não sabia o que fazer.
        sem_contexto = not str(contexto).strip()
        motivo = (
            "Não há conversa anterior nesta sessão para eu saber o que é "
            "\"esse resultado\" — o app foi reaberto e o histórico está vazio."
            if sem_contexto else
            "Não consegui redigir a nota a partir da conversa."
        )
        return {
            "ok": False,
            "etapa": "Registro no cérebro",
            "mensagem": (
                f"⚠️ **A nota NÃO foi criada.** {motivo}\n\n"
                "Peça de novo dizendo **o que** guardar — por exemplo:\n"
                "> _\"registre no cérebro: no IGBT o método proposto detecta a "
                "partir da severidade 0,50 contra 1,00 do AE-LSTM (AUC 0,978 vs "
                "0,909). Evidência E2.\"_\n\n"
                "Ou rode a etapa primeiro (ex.: \"mostre o Weibull\") e então "
                "peça para guardar — aí eu leio o resultado da própria conversa."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }

    if progresso:
        progresso(f"Registrando nota no cérebro: {titulo}...")
    res = registrar_nota_cerebro(
        titulo=titulo, conteudo=conteudo, tipo=tipo, tags=tags,
        nivel_evidencia=nivel_evidencia, fonte=fonte,
    )
    return {
        "ok": res["ok"],
        "etapa": "Registro no cérebro",
        "mensagem": res["mensagem"],
        "imagens": [],
        "resposta_pronta": True,
        # CONFIRMAÇÃO DE AÇÃO EXECUTADA é literal (ver comentar_resultado).
        # Sem isto, o LLM substituía "Nota criada em X.md" por uma análise do
        # tema e o pesquisador não sabia se o arquivo existia.
        "acao_executada": True,
    }


from src.conhecimento.ferramentas_academicas import (
    buscar_na_web,
    listar_base_bibliografica,
    consultar_comparacao_macro,
    listar_experimentos_artigos,
    limpar_experimentos_artigos,
    _md_experimento_legacy,
    _md_experimento,
    rodar_experimento_artigo,
    _contar_linhas,
    consultar_datasets,
    comparar_abordagens_ml,
    treinar_classificador_pv,
    avaliar_classificador_pv,
    classificar_amostra_pv,
)


_DESPACHO = {
    "rodar_features_ca": rodar_features_ca,
    "rodar_autoencoder": rodar_autoencoder,
    "rodar_injecao_falhas": rodar_injecao_falhas,
    "rodar_validacao": rodar_validacao,
    "rodar_weibull": rodar_weibull,
    "rodar_pipeline_completo": rodar_pipeline_completo,
    "consultar_resultados": consultar_resultados,
    "consultar_status_pipeline": consultar_status_pipeline,
    "consultar_datasets": consultar_datasets,
    "comparar_abordagens_ml": comparar_abordagens_ml,
    "treinar_classificador_pv": treinar_classificador_pv,
    "avaliar_classificador_pv": avaliar_classificador_pv,
    "classificar_amostra_pv": classificar_amostra_pv,
    "limpar_resultados_ml": limpar_resultados_ml,
    "buscar_web": buscar_na_web,
    "listar_base_bibliografica": listar_base_bibliografica,
    "consultar_comparacao_macro": consultar_comparacao_macro,
    "registrar_no_cerebro": registrar_no_cerebro,
}


def executar_ferramenta(nome: str, progresso=None, pergunta: str = "",
                        llm=None, contexto: str = "") -> dict:
    """Executa a ferramenta `nome`.

    `llm` e `contexto` (últimas trocas da conversa) são REPASSADOS apenas às
    ferramentas que os aceitam — hoje `registrar_no_cerebro`, que precisa
    REDIGIR a nota a partir do que acabou de ser discutido. Sem isso, ela era
    chamada sem título/conteúdo e só podia pedir os dados de volta, e o agente
    acabava *descrevendo* a nota no chat em vez de gravá-la.
    """
    import inspect

    funcao = _DESPACHO.get(nome)
    if funcao is None:
        return {
            "ok": False,
            "etapa": nome,
            "mensagem": f"Ferramenta desconhecida: {nome}",
            "imagens": [],
            "resposta_pronta": True,
        }
    extras = {}
    try:
        aceita = inspect.signature(funcao).parameters
        if "llm" in aceita:
            extras["llm"] = llm
        if "contexto" in aceita:
            extras["contexto"] = contexto
    except (TypeError, ValueError):
        pass
    return funcao(progresso=progresso, pergunta=pergunta, **extras)

from src.conhecimento.roteamento_ferramentas import (
    _ETAPA_ORDEM,
    _quer_codigo_snippet,
    _etapa_mais_avancada_mencionada,
    _GATILHOS_WEB,
    _quer_resposta_discursiva_sem_ferramenta,
    _GATILHOS_DECLARACAO_MEMORIA,
    _e_declaracao_memoria,
    _INTERROGATIVOS,
    _e_pergunta,
    _decisao_rapida,
    _guardas_criticas,
    _rotear_por_llm,
    decidir_acao,
    _corrigir_descricao_visual,
    _dados_sao_inventario,
    comentar_resultado,
    processar_com_ferramentas,
)
