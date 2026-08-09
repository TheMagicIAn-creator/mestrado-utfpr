"""Roteamento, decisao e comentario das ferramentas do agente."""

from __future__ import annotations

import json
import re

from src.conhecimento.intencoes_ferramentas import (
    _TERMOS_PIPELINE_IMPLICITO,
    _parece_pedido_de_ferramenta,
    _quer_catalogo,
    _quer_catalogo_experimentos,
    _quer_classificador_pv,
    _quer_comparar_abordagens,
    _quer_comparar_auc_experimentos,
    _quer_consultar_datasets,
    _quer_consultar_resultados_experimentos,
    _quer_limpar,
    _quer_literatura_tematica,
    _quer_registrar_no_cerebro,
    _quer_resposta_autoral,
    _quer_rodar_experimento,
    _quer_status,
)
from src.conhecimento.provedores import texto_da_resposta
from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos
from src.core.texto import normalizar_sem_acentos as _normalizar

_logger = get_logger("conhecimento.roteamento_ferramentas")


def _especificacoes_ferramentas() -> list[dict]:
    from src.conhecimento.ferramentas import ESPEC_FERRAMENTAS

    return ESPEC_FERRAMENTAS

# Mapeamento "ordem cronológica" do pipeline → nome da ferramenta.
# Quando o usuário menciona várias etapas, pegamos a MAIS AVANÇADA — assim
# 'auto_deps=True' garante que tudo até ela rode em ordem.
_ETAPA_ORDEM = [
    ("feature", "rodar_features_ca"),
    ("features", "rodar_features_ca"),
    ("sinais", "rodar_features_ca"),
    ("signals", "rodar_features_ca"),
    ("senales", "rodar_features_ca"),
    ("señales", "rodar_features_ca"),
    ("signaux", "rodar_features_ca"),
    ("autoencoder", "rodar_autoencoder"),
    ("detector", "rodar_autoencoder"),
    ("limiar", "rodar_autoencoder"),
    ("threshold", "rodar_autoencoder"),
    ("umbral", "rodar_autoencoder"),
    ("seuil", "rodar_autoencoder"),
    ("injec", "rodar_injecao_falhas"),
    ("injet", "rodar_injecao_falhas"),
    ("falha sint", "rodar_injecao_falhas"),
    ("synthetic fault", "rodar_injecao_falhas"),
    ("synthetic failure", "rodar_injecao_falhas"),
    ("falla sintet", "rodar_injecao_falhas"),
    ("defaillance synt", "rodar_injecao_falhas"),
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
    ("reliability", "rodar_weibull"),
    ("confiabilidad", "rodar_weibull"),
    ("fiabilite", "rodar_weibull"),
]


def _quer_codigo_snippet(pergunta: str) -> bool:
    """
    Rodolfo pediu para o agente ESCREVER código — um trecho Python, um script,
    "como plotar ..." — e NÃO para rodar o pipeline nem devolver artefatos já
    gerados. Nesse caso quem responde é o LLM (que redige o código); nenhuma
    ferramenta de execução ou de consulta de resultados deve interceptar.

    Bug corrigido: "gere um código de um gráfico da TTF/distribuição" caía em
    `termos_executar`/`consultar_resultados` e o agente trazia as figuras já
    criadas, em vez de escrever o código pedido.

    Exemplos que passam a ir para o LLM:
      "gere um código para plotar a distribuição do erro do autoencoder"
      "me dê o script em Python do gráfico da TTF"
      "como plotar a curva de Weibull no matplotlib?"
      "escreva a função que desenha o histograma do erro de reconstrução"
    """
    txt = _normalizar(pergunta)

    # Sinais fortes e inequívocos de pedido de código, isoladamente suficientes.
    # "code"/"script"/"snippet" exigem fronteira de palavra: "code" é substring
    # de "autoencoder", "decode" etc. — usar \b evita esses falsos positivos.
    if "codigo" in txt or "pseudocodigo" in txt or "pseudo codigo" in txt:
        return True
    if any(re.search(rf"\b{t}\b", txt) for t in ("code", "script", "snippet")):
        return True
    if any(t in txt for t in (
        "matplotlib", "seaborn", "plotly", "pyplot", "plt.", "sns.",
        "ggplot", "bokeh",
    )):
        return True

    # "escreva/implemente a função/classe/método ..." é pedido de código. Gate
    # em SUBSTANTIVOS de código (função/classe/...), nunca em palavras de plot
    # como "distribuição/curva" — senão "escreva um resumo da distribuição"
    # (discursivo) seria confundido com código.
    verbo_escrever = any(t in txt for t in (
        "escreva", "escreve", "escrever", "implemente", "implementar",
        "programe", "programar", "codar", "coda ",
    ))
    if verbo_escrever and any(t in txt for t in (
        "funcao", "classe", "metodo", "rotina", "trecho",
    )):
        return True
    if any(t in txt for t in (
        "como plot", "como faco para plot", "como desenh", "como trac",
        "how to plot", "como gerar o grafico", "como fazer o grafico",
        "como criar o grafico", "como monto o grafico", "como plotar",
        "como ploto", "como codar", "como programar",
    )):
        return True

    # "... em Python ..." combinado com um pedido de visualização também é
    # pedido de código (o usuário quer o programa, não o artefato pronto).
    tem_python = any(t in txt for t in (
        "em python", "in python", "com python", "usando python",
        "no matplotlib", "em py ", "codigo python", "funcao python",
    ))
    quer_plot = any(t in txt for t in (
        "grafico", "graficos", "plot", "plotar", "curva", "distribuicao",
        "histograma", "figura", "chart", "ttf", "weibull", "roc",
        "dispersao", "scatter", "boxplot", "heatmap",
    ))
    return tem_python and quer_plot


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
    "search the web", "web search", "search online", "look up online",
    "google it", "on the internet", "official definition", "iec standard",
    "iso standard",
    "buscar en la web", "buscar online", "busca en internet",
    "consulta wikipedia", "definicion oficial", "norma iec", "norma iso",
    "rechercher sur le web", "recherche en ligne", "chercher sur internet",
    "consulte wikipedia", "definition officielle", "norme iec", "norme iso",
)


def _quer_resposta_discursiva_sem_ferramenta(pergunta: str) -> bool:
    """
    Conversa tecnica/conceitual deve ir para RAG/LLM, nao para execucao.

    Ex.: "Faca uma revisao bibliografica curta sobre RUL" contem RUL, mas a
    intencao e escrever uma revisao, nao rodar Weibull. O mesmo vale para
    "Explique FMEA com base no projeto".
    """
    txt = _normalizar(pergunta)

    if _quer_limpar(pergunta) or _quer_rodar_experimento(pergunta):
        return False
    if any(t in txt for t in _TERMOS_PIPELINE_IMPLICITO):
        return False

    pede_resultado = any(t in txt for t in (
        "resultado", "resultados", "artefato", "artefatos", "grafico",
        "graficos", "matriz", "matrizes", "imagem", "imagens", "metricas",
        "metrica", "auc", "f1", "recall", "precision", "status",
        "result", "results", "artifact", "artifacts", "chart", "charts",
        "matrix", "metrics",
    ))
    if pede_resultado:
        return False

    if any(t in txt for t in (
        "revisao bibliografica", "revisao da literatura", "estado da arte",
        "levantamento bibliografico", "referencial teorico",
        "literature review", "state of the art", "survey",
        "revision bibliografica", "estado del arte",
        "revue bibliographique", "etat de l art",
    )):
        return True

    verbos_discursivos = (
        "explique", "explica", "fale", "descreva", "resuma", "discuta",
        "analise", "o que e", "o que eh", "qual a diferenca",
        "explain", "describe", "summarize", "discuss",
        "explica", "describe", "resume", "discute",
        "expliquez", "decrivez", "decrire", "resumer", "discutez",
    )
    conceitos = (
        "fmea", "fmeca", "rul", "weibull", "mttf", "b10", "autoencoder",
        "anomalia", "anomalias", "rcm", "npr", "paderborn", "inversor",
        "confiabilidade", "manutencao", "eletronica de potencia",
        "reliability", "maintenance", "inverter", "anomaly",
        "confiabilidad", "mantenimiento", "fiabilite", "onduleur",
    )
    return any(v in txt for v in verbos_discursivos) and any(c in txt for c in conceitos)


_GATILHOS_DECLARACAO_MEMORIA = (
    "lembre", "lembra", "lembrar", "memorize", "memoriza", "anote", "anota",
    "considere que", "corrigindo", "correcao", "decidi", "decidimos",
    "combinamos", "a partir de agora", "daqui em diante", "de agora em diante",
    "prefiro", "quero que voce lembre",
)


def _e_declaracao_memoria(pergunta: str) -> bool:
    """Declaração para o agente LEMBRAR ('Lembre-se: ...', 'Decidimos que ...',
    'Corrigindo: ...'). Deve ir ao LLM — que responde e o auditor memoriza —
    NUNCA virar comando de pipeline só porque menciona 'pipeline'/'injetar'."""
    txt = _normalizar(pergunta)
    return any(re.match(rf"\s*{re.escape(g)}\b", txt) for g in _GATILHOS_DECLARACAO_MEMORIA)


_INTERROGATIVOS = (
    "qual", "quais", "quanto", "quantos", "quanta", "quantas",
    "o que", "por que", "porque", "pq", "como", "quando", "onde",
    "cade", "quem", "sera que",
    "what", "which", "how", "when", "where", "who", "why",
)


def _e_pergunta(pergunta: str) -> bool:
    """Pergunta (recall/consulta), não comando. Detecta '?' no texto original ou
    palavra interrogativa no início. Uma PERGUNTA nunca deve DISPARAR execução do
    pipeline (ex.: 'Qual falha decidimos injetar?' não é 'injete a falha')."""
    if "?" in (pergunta or ""):
        return True
    txt = _normalizar(pergunta)
    return any(re.match(rf"\s*{re.escape(w)}\b", txt) for w in _INTERROGATIVOS)


def _decisao_rapida(pergunta: str) -> dict | None:
    txt = _normalizar(pergunta)

    # Busca na web — atalho prioritário quando gatilho explícito aparece
    if any(g in txt for g in _GATILHOS_WEB):
        return {"usar_ferramenta": True, "ferramenta": "buscar_web"}

    # Declaração de memória ("Lembre-se: decidimos que...") → LLM. Sem isto,
    # "injetada"/"pipeline" na frase faziam a declaração virar comando de
    # pipeline, e a memória nunca era criada.
    if _e_declaracao_memoria(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}

    # Pedido de CÓDIGO ("gere um código do gráfico da TTF", "como plotar ...")
    # vai para o LLM, que ESCREVE o código — nunca para execução do pipeline
    # nem para devolver os artefatos já criados.
    if _quer_codigo_snippet(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}

    if _quer_literatura_tematica(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}

    # Comparação com a literatura — os macro-códigos são a fonte única desde
    # que o framework por artigo (protocolo E1) foi aposentado. Qualquer
    # variação do pedido cai na comparação PUBLICADA; nada aqui treina.
    if (_quer_comparar_auc_experimentos(pergunta)
            or _quer_rodar_experimento(pergunta)
            or _quer_catalogo_experimentos(pergunta)
            or _quer_consultar_resultados_experimentos(pergunta)):
        return {"usar_ferramenta": True, "ferramenta": "consultar_comparacao_macro"}

    # Datasets do projeto (Paderborn CA / PV Farms CC) — explicação determinística.
    if _quer_consultar_datasets(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "consultar_datasets"}

    # Comparação das abordagens de ML (supervisionado x anomalia x sintético).
    if _quer_comparar_abordagens(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "comparar_abordagens_ml"}

    # Classificador supervisionado PV Farms (CC): treinar / avaliar / classificar.
    _acao_clf = _quer_classificador_pv(pergunta)
    if _acao_clf:
        return {"usar_ferramenta": True, "ferramenta": _acao_clf}

    # Catálogo da literatura — pedido pelo INVENTÁRIO inteiro ("liste todas as
    # referências", "o que você tem indexado", "quantos artigos", "as 39").
    # Atende direto, sem RAG, para nunca truncar nem inventar a lista.
    if _quer_catalogo(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "listar_base_bibliografica"}

    # Status é consulta operacional, então deve vencer antes do guard genérico.
    if _quer_status(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "consultar_status_pipeline"}

    if _quer_resposta_discursiva_sem_ferramenta(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}

    if not _parece_pedido_de_ferramenta(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}

    # Registro no cérebro: "guarde/registre/anote isso no cérebro/vault".
    # Vem antes da limpeza porque "registre" nao e destrutivo e nao deve cair
    # em nenhum outro roteamento.
    if _quer_registrar_no_cerebro(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "registrar_no_cerebro"}

    # Limpeza explícita ("apague", "limpe...") tem prioridade sobre tudo.
    if _quer_limpar(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "limpar_resultados_ml"}

    # "Recalcule", "refaça", "rode tudo de novo", "do zero" → pipeline completo.
    # Perguntas ("qual falha decidimos injetar?") nunca EXECUTAM o pipeline.
    if not _e_pergunta(pergunta) and any(t in txt for t in _TERMOS_PIPELINE_IMPLICITO):
        # Quando há ETAPAS específicas mencionadas, roteia para a mais AVANÇADA
        # (com auto_deps=True a etapa puxa todas anteriores que faltarem).
        etapa_mais_avancada = _etapa_mais_avancada_mencionada(txt)
        if etapa_mais_avancada:
            return {"usar_ferramenta": True, "ferramenta": etapa_mais_avancada}
        # Sem etapa específica, recalcular = pipeline inteiro.
        return {"usar_ferramenta": True, "ferramenta": "rodar_pipeline_completo"}

    # Consulta passiva ("mostre", "quais foram...", "cadê as imagens?")
    termos_consulta = (
        "resultado", "resultados", "mostrar", "mostre", "mostra", "grafico",
        "graficos", "auc", "f1", "mttf", "b10", "smd", "limiar",
        "metrica", "metricas", "imagem", "imagens", "figura", "figuras",
        "curva", "curvas", "plot", "plots", "visualizacao", "matriz",
        "heatmap", "roc",
        "result", "results", "show", "display", "chart", "charts",
        "threshold", "metric", "metrics", "image", "images", "figure",
        "figures", "curve", "curves", "visualization", "matrix",
        "resultado", "resultados", "muestra", "mostrar", "grafico",
        "graficos", "umbral", "metrica", "metricas", "imagen", "imagenes",
        "figura", "curva", "visualizacion", "matriz",
        "resultat", "resultats", "montre", "affiche", "graphique",
        "seuil", "metrique", "metriques", "image", "figure", "courbe",
        "visualisation", "matrice",
    )
    # NOTA: "tabela"/"table"/"tabla"/"tableau" foram REMOVIDOS dos gatilhos de
    # consulta — eram ambíguos demais ("quais as tabelas de S/O/D da FMECA?" é
    # conceitual, não um pedido do artefato de resultados). Para ver a tabela de
    # resultados, "mostre os resultados"/"matriz"/"métricas" continuam valendo.
    termos_acao_ativa = (
        "rodar", "execut", "trein", "gerar", "gere", "calcular",
        "injetar", "refazer", "regerar", "recalc",
        "run", "execut", "train", "generate", "calculate", "inject",
        "rerun", "recompute",
        "ejecut", "entren", "gener", "calcul", "inyect",
        "execut", "entraîn", "entrain", "gener", "génér", "calcul", "inject",
    )
    termos_validacao_ativa = ("validar", "valide", "valida")
    # Verbos de GERAÇÃO (fazer/plotar/escrever). Fronteira de palavra para não
    # pegar "geral", "gerenciar" etc. Sem isto, "gera um gráfico da ttf" caía no
    # despejo de resultados só por conter "gráfico" — o LLM deve escrever o
    # código/conversar, não devolver o artefato salvo.
    termos_geracao_ativa = (
        "gera", "gere", "gerar", "plota", "plote", "plotar",
        "desenha", "desenhe", "desenhar", "traca", "trace", "tracar",
        "escreve", "escreva", "escrever", "coda", "code", "codar",
        "monta", "monte", "montar", "cria", "crie", "criar",
    )
    tem_acao_ativa = (
        any(t in txt for t in termos_acao_ativa)
        or any(re.search(rf"\b{re.escape(t)}\b", txt)
               for t in termos_validacao_ativa + termos_geracao_ativa)
    )
    if any(t in txt for t in termos_consulta):
        if not tem_acao_ativa:
            return {"usar_ferramenta": True, "ferramenta": "consultar_resultados"}

    termos_executar = (
        "rodar", "rode", "roda", "execut", "trein", "treine", "treina",
        "gerar", "gere", "gera", "calcular", "calcule", "calcula",
        "validar", "valide", "valida", "injetar", "injete", "injeta",
        "estimar", "estime", "estima", "fazer", "faca", "faça",
        "run", "train", "generate", "calculate", "validate", "inject",
        "estimate", "make", "execute",
        "ejecut", "entren", "gener", "calcul", "valid", "inyect", "estim",
        "execut", "exécut", "entrain", "entraîn", "gener", "génér",
        "calcul", "valid", "inject", "estim",
    )
    # Perguntas ("qual falha decidimos injetar?") não EXECUTAM — só comandos.
    if not _e_pergunta(pergunta) and any(t in txt for t in termos_executar):
        if "pipeline" in txt or "tudo" in txt or "todos" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_pipeline_completo"}
        # Roteia para a etapa mais avançada mencionada (auto_deps roda o resto)
        etapa_mais_avancada = _etapa_mais_avancada_mencionada(txt)
        if etapa_mais_avancada:
            return {"usar_ferramenta": True, "ferramenta": etapa_mais_avancada}
        # Pedido genérico de "gere os resultados" — interpreta como pipeline.
        if "resultado" in txt:
            return {"usar_ferramenta": True, "ferramenta": "rodar_pipeline_completo"}

    # Pergunta conceitual/recall SEM referência a artefato de resultado vai
    # DEFINITIVAMENTE ao LLM — impede o roteador-LLM (fallback) de despejar
    # consultar_resultados (gráficos, às vezes velhos) numa pergunta de conceito.
    _NOUNS_RESULTADO = (
        "resultado", "resultados", "grafico", "graficos", "figura", "figuras",
        "imagem", "imagens", "matriz", "matrizes", "metrica", "metricas",
        "auc", "f1", "roc", "curva", "curvas", "heatmap", "limiar", "ttf",
        "smd", "b10", "mttf", "artefato", "artefatos", "plot",
    )
    if _e_pergunta(pergunta) and not any(t in txt for t in _NOUNS_RESULTADO):
        return {"usar_ferramenta": False, "ferramenta": None}

    return None


def _guardas_criticas(pergunta: str) -> dict | None:
    """Regras que DEVEM decidir antes do LLM — onde errar é caro ou irreversível.

    São poucas de propósito. Todo o resto vai para o roteamento semântico, que
    entende paráfrase e frase torta. Duas famílias:

    (a) NEGATIVAS — impedem o uso de ferramenta. Sem elas, uma declaração de
        memória ("Lembre-se: decidimos injetar...") virava execução de pipeline
        e a memória nunca era criada; e um pedido de CÓDIGO devolvia artefato
        salvo em vez de o LLM escrever o código.
    (b) DESTRUTIVAS — apagar artefatos. Um falso positivo do LLM aqui custa
        recálculo; a palavra-chave explícita é mais segura.
    """
    if _e_declaracao_memoria(pergunta):          # (a) declaração → memória
        return {"usar_ferramenta": False, "ferramenta": None}
    if _quer_codigo_snippet(pergunta):           # (a) "escreva um código..."
        return {"usar_ferramenta": False, "ferramenta": None}
    if _quer_literatura_tematica(pergunta):      # (a) busca bibliográfica → RAG
        return {"usar_ferramenta": False, "ferramenta": None}
    if _quer_limpar(pergunta):                   # (b) destrutivo
        return {"usar_ferramenta": True, "ferramenta": "limpar_resultados_ml"}
    return None


def _rotear_por_llm(pergunta: str, llm) -> dict | None:
    """Roteamento SEMÂNTICO: o LLM escolhe a ferramenta pelo catálogo.

    Retorna None se o LLM falhar/estiver indisponível — aí o chamador cai na
    cascata de palavras-chave, que segue existindo como rede de segurança.
    """
    if llm is None:
        return None
    catalogo = "\n".join(
        f"- {f['name']}: {f['description']}"
        for f in _especificacoes_ferramentas()
    )
    prompt = f"""Voce roteia pedidos do pesquisador para ferramentas de ML.

Ferramentas disponiveis:
{catalogo}

Regras:
- Escolha a ferramenta SO se o pedido realmente exigir executar/consultar algo.
- Pergunta conceitual, pedido de explicacao, opiniao ou redacao => sem ferramenta.
- Na duvida entre conversar e acionar ferramenta, prefira SEM ferramenta.

Mensagem do usuario:
"{pergunta}"

Responda apenas JSON valido:
{{"usar_ferramenta": true/false, "ferramenta": "nome_ou_null"}}
"""
    try:
        # HumanMessage quando o langchain estiver disponível; senão manda o
        # texto puro (a maioria dos wrappers aceita). Sem isto, a ausência do
        # pacote derrubava o roteamento semântico INTEIRO em silêncio.
        try:
            from langchain_core.messages import HumanMessage

            entrada = [HumanMessage(content=prompt)]
        except ImportError:
            entrada = prompt

        resposta = texto_da_resposta(llm.invoke(entrada))
        limpo = re.sub(r"```json?\n?", "", str(resposta).strip()).replace("```", "").strip()
        dados = json.loads(limpo)
        nomes = {f["name"] for f in _especificacoes_ferramentas()}
        ferramenta = dados.get("ferramenta")
        if dados.get("usar_ferramenta") and ferramenta in nomes:
            return {"usar_ferramenta": True, "ferramenta": ferramenta}
        if dados.get("usar_ferramenta") is False:
            return {"usar_ferramenta": False, "ferramenta": None}
    except Exception as exc:
        _logger.warning(
            "roteamento semântico falhou; usando regras locais: %s",
            mascarar_segredos(str(exc)),
        )
    return None


def decidir_acao(pergunta: str, llm) -> dict:
    """Decide se a pergunta deve acionar uma ferramenta.

    Ordem (fluidez primeiro, segurança onde importa):
      1. GUARDAS CRÍTICAS por palavra-chave — negativas e destrutivas;
      2. ROTEAMENTO SEMÂNTICO pelo LLM — entende paráfrase, frase torta, pedido
         indireto. É o caminho principal;
      3. CASCATA de palavras-chave — rede de segurança quando o LLM falha ou
         está indisponível (offline, cota, erro de rede).

    Antes, a cascata de 27 gatilhos vinha PRIMEIRO e capturava quase tudo — o
    roteamento semântico existia mas quase nunca era alcançado, e o agente
    parecia "de gatilho". A inversão custa uma chamada de LLM por mensagem
    (decisão consciente: latência trocada por fluidez).
    """
    guarda = _guardas_criticas(pergunta)
    if guarda is not None:
        return guarda

    semantico = _rotear_por_llm(pergunta, llm)
    if semantico is not None:
        return semantico

    decisao = _decisao_rapida(pergunta)
    if decisao is not None:
        return decisao

    return {"usar_ferramenta": False, "ferramenta": None}


def _corrigir_descricao_visual(resposta: str, imagens: list[dict] | None) -> str:
    """Remove afirmações visuais que não correspondem aos artefatos exibidos."""
    imagens = imagens or []
    if not resposta or not imagens:
        return resposta

    inventario = " ".join(
        f"{imagem.get('caption', '')} {imagem.get('path', '')}"
        for imagem in imagens
    )
    inventario_norm = _normalizar(inventario)
    # Quando o pedido realmente inclui esse tipo de figura, a interpretação é
    # permitida. A proteção atua apenas nos conjuntos de métricas/contagens.
    if any(termo in inventario_norm for termo in (
        "score", "curva roc", "temporal", "limiar", "distribuicao",
    )):
        return resposta

    termos_nao_suportados = (
        "distribuicao de score",
        "distribuicoes de score",
        "curva roc",
        "curvas roc",
        "ao longo do tempo",
        "separacao entre classes",
        "separacao entre os dados",
        "localizacao das deteccoes",
    )
    partes = re.split(r"(\n\s*\n)", resposta)
    filtradas: list[str] = []
    removeu = False
    for parte in partes:
        normalizada = _normalizar(parte)
        descreve_figura = "grafico" in normalizada or "figura" in normalizada
        incompativel = descreve_figura and any(
            termo in normalizada for termo in termos_nao_suportados
        )
        if incompativel:
            removeu = True
            continue
        filtradas.append(parte)

    if not removeu:
        return resposta

    corrigida = "".join(filtradas).strip()
    descricoes = []
    if "comparacao por pontos" in inventario_norm:
        descricoes.append("a comparação das métricas por pontos")
    if "anomalias detectadas" in inventario_norm:
        descricoes.append("as contagens de detecções e a cobertura percentual")
    if descricoes:
        corrigida += (
            "\n\nOs gráficos exibidos abaixo mostram "
            + " e ".join(descricoes)
            + ", organizadas por artigo."
        )
    return corrigida


def _dados_sao_inventario(mensagem: str) -> bool:
    """True quando a mensagem é tabela/inventário longo.

    Nesses casos o texto cru é preservado: parafrasear uma tabela de métricas
    ou um catálogo de 39 referências arrisca truncar ou distorcer. Para
    respostas curtas (status, confirmações), o LLM pode falar à vontade.
    """
    if not mensagem:
        return False
    if "|---" in mensagem or "| --- " in mensagem:      # tabela markdown
        return True
    itens = sum(1 for l in mensagem.splitlines() if l.lstrip().startswith(("- ", "* ", "|")))
    return itens > 8 or len(mensagem) > 1800


def comentar_resultado(pergunta: str, resultado: dict, perfil: str, llm) -> str:
    """Transforma o resultado de uma ferramenta em RESPOSTA.

    Por padrão o LLM fala — é ele quem dá voz, contexto e leitura técnica. O
    texto cru só passa direto quando parafrasear seria arriscado (tabela de
    métricas, catálogo longo) ou quando a ferramenta exige literalidade.

    Antes, `resposta_pronta` (36 ferramentas!) silenciava o LLM e o agente
    despejava o texto da ferramenta — soava robótico mesmo com o roteamento
    correto. Agora `resposta_pronta` significa "os dados são autoritativos",
    não "não fale".
    """
    autoral = _quer_resposta_autoral(pergunta)
    mensagem = resultado.get("mensagem", "")

    # FATO DE EXECUÇÃO é reportado LITERALMENTE, sempre — mesmo em pedido
    # autoral. Vale para os dois lados:
    #   - FALHA: deixar o LLM reescrever o erro produzia texto plausível que
    #     escondia o problema; o pesquisador não sabia que a ação não ocorreu.
    #   - CONFIRMAÇÃO de ação (nota criada, artefato apagado): o LLM trocava
    #     "Nota criada em X.md" por uma análise do tema, e o pesquisador ficava
    #     sem saber se o arquivo existia.
    # Opinião se dá sobre DADOS; sobre o que aconteceu, vale o fato.
    if mensagem and (resultado.get("ok") is False
                     or resultado.get("acao_executada")):
        return mensagem

    # Literalidade obrigatória: só quando a ferramenta força OU quando os dados
    # são um inventário/tabela — e nunca quando o pedido é autoral.
    if not autoral and (resultado.get("forcar_resposta_direta")
                        or (resultado.get("resposta_pronta")
                            and _dados_sao_inventario(mensagem))):
        return mensagem
    if llm is None:
        return mensagem

    status = "SUCESSO" if resultado.get("ok") else "FALHA"
    perfil_txt = (perfil or "").strip()[:4000]
    legendas_visuais = []
    for imagem in resultado.get("imagens") or []:
        legenda = str(imagem.get("caption", "")).strip()
        if not legenda:
            continue
        grupo = str(imagem.get("grupo", "")).strip()
        tipo = str(imagem.get("tipo", "")).strip()
        qualificadores = ", ".join(item for item in (grupo, tipo) if item)
        sufixo = f" ({qualificadores})" if qualificadores else ""
        legendas_visuais.append(f"- {legenda}{sufixo}")
    inventario_visual = (
        "ARTEFATOS VISUAIS QUE SERÃO EXIBIDOS (inventário autoritativo):\n"
        + "\n".join(legendas_visuais)
        if legendas_visuais
        else "ARTEFATOS VISUAIS QUE SERÃO EXIBIDOS: nenhum."
    )
    prompt = f"""{perfil_txt}

Rodolfo pediu: "{pergunta}"

Resultado técnico ({status}) — use como EVIDÊNCIA, NÃO copie a tabela crua:
{resultado.get('mensagem', 'sem detalhes')}

{inventario_visual}

Responda como o Al IAdo PV, no papel de coorientador: INTERPRETE os números,
priorize o que importa para a dissertação, aponte ressalvas (ajuste estatístico
rejeitado, detecção nula, evidência E1/E2) e diga o que aquilo SIGNIFICA para o
trabalho. Não invente números — cite só os que estão na evidência.
NORMAS TÉCNICAS (IEC/ISO/ABNT/IEEE/NBR): se o pedido for sobre uma norma e a
EVIDÊNCIA acima não trouxer o número EXATO de cláusula/seção/página, NÃO os
invente. Diga com franqueza que não tem a norma indexada nem uma fonte
verificável para a localização exata, e recomende consultar a norma oficial
(ex.: webstore da IEC) ou uma fonte que o Rodolfo já tenha. Você pode explicar o
CONCEITO (ex.: NPR = S×O×D) sem atribuir cláusula/página inventada. Nunca
escreva "Clause X.Y" ou "p. N" de uma norma que não esteja na evidência. Se um número
tiver ressalva na evidência (ex.: KS rejeitado, SMD não detectada), NÃO o
apresente como conclusivo. Escolha a forma que melhor atende ao pedido: prosa,
lista, ranking ou tabela específica. Não despeje sempre a mesma tabela completa;
se o pesquisador pedir uma métrica ou contagem, mostre apenas as colunas úteis.
Em comparações, não omita modelos relevantes presentes na evidência. Português
brasileiro natural, salvo se Rodolfo escreveu claramente em outro idioma.
Não repita o pedido, não use encorajamento genérico e mantenha a extensão
proporcional. Se o resultado mencionar imagens, elas serão renderizadas no chat:
    não diga que não pode vê-las. Comece diretamente pela análise, sem
    vocativos como "Prezado Rodolfo". Descreva cada gráfico somente pelo que a
    evidência e sua legenda dizem; não atribua distribuições, limiares ou curvas a
    uma figura de métricas, contagens ou cobertura. Uma "comparação por pontos"
    compara métricas dos modelos; um gráfico de "anomalias detectadas" compara
    contagens e cobertura. Nenhum dos dois mostra distribuição de scores,
    limiares ou separação amostra a amostra, salvo se isso estiver explicitamente
    nomeado no inventário. Em Weibull/RUL, preserve
    rigorosamente o rótulo:
"RUL restrita" é Kaplan-Meier não paramétrica; não a chame de paramétrica. NPR
é criticidade FMECA e não causa a frequência de eventos simulados."""
    try:
        try:
            from langchain_core.messages import HumanMessage

            entrada = [HumanMessage(content=prompt)]
        except ImportError:      # sem langchain: manda texto puro
            entrada = prompt
        resposta = texto_da_resposta(llm.invoke(entrada))
        return _corrigir_descricao_visual(resposta, resultado.get("imagens"))
    except Exception:
        return resultado.get("mensagem", "")


def processar_com_ferramentas(pergunta: str,
                              perfil: str,
                              llm,
                              progresso=None,
                              decisao: dict | None = None,
                              contexto: str = "") -> dict:
    """`contexto` são as últimas trocas da conversa — necessário para pedidos
    dêiticos ("guarde ESSE resultado"), em que o conteúdo não está na frase."""
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
    # O NOME da ferramenta é detalhe de implementação: vai para o log, não
    # para a tela. Cada ferramenta emite o próprio progresso em linguagem de
    # gente ("Treinando o classificador PV Farms (CC)..."), e é isso que o
    # pesquisador deve ler enquanto espera.
    _logger.info("ferramenta acionada: %s", ferramenta)

    from src.conhecimento.ferramentas import executar_ferramenta

    resultado = executar_ferramenta(
        ferramenta,
        progresso=progresso,
        pergunta=pergunta,
        llm=llm,
        contexto=contexto,
    )
    resposta = comentar_resultado(pergunta, resultado, perfil, llm)

    return {
        "usou_ferramenta": True,
        "ferramenta": ferramenta,
        "resultado": resultado,
        "resposta": resposta,
    }
