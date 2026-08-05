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
import unicodedata

from src.core.config import RAIZ_PROJETO
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
            "Executa injecao de falhas sinteticas fundamentadas na FMECA. Use "
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
        "name": "consultar_comparacao_macro",
        "description": (
            "Compara o METODO PROPOSTO (Autoencoder denso + escore localizado) "
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
            "Explica os datasets do projeto (Paderborn e PV Farms): finalidade, "
            "rotulos, arquivos, nº de linhas, features, dominio CA ou CC e "
            "limitacoes — lendo as contagens dinamicamente. Use quando o usuario "
            "perguntar sobre os datasets, os dados, ou a diferenca entre "
            "Paderborn e PV Farms."
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


# --- Catalogo da literatura (inventario completo, NAO e RAG tematico) -------
# Termos que se referem ao acervo bibliografico em si.
_TERMOS_BIBLIO = (
    "referencia", "referencias", "fonte", "fontes", "artigo", "artigos",
    "documento", "documentos", "obra", "obras", "literatura", "biblio",
    "bibliografia", "bibliografica", "acervo", "papers", "paper",
    "source", "sources", "reference", "references", "literature", "library",
    "bibliography", "article", "articles", "documents",
    "fuente", "fuentes", "referencia", "referencias", "literatura",
    "bibliografia", "articulo", "articulos", "documentos",
    "source", "sources", "reference", "references", "litterature",
    "littérature", "bibliographie", "article", "articles", "documents",
)
# Termos que indicam INTENCAO de totalidade/inventario (e nao busca por tema).
_TERMOS_TOTALIDADE = (
    "todas", "todos", "toda", "todo", "tudo", "completa", "completo",
    "completas", "completos", "inteira", "inteiro", "lista", "liste",
    "listar", "catalogo", "inventario", "quantos", "quantas", "quais",
    "que voce tem", "que voce possui", "disponiveis", "disponivel",
    "all", "complete", "full", "entire", "list", "catalog", "catalogue",
    "inventory", "how many", "available", "what do you have",
    "todas", "todos", "completa", "completo", "lista", "catalogo",
    "inventario", "cuantos", "cuantas", "disponibles",
    "tous", "toutes", "complete", "complet", "liste", "catalogue",
    "inventaire", "combien", "disponibles",
)
# Gatilhos fortes: sozinhos ja bastam para pedir o catalogo inteiro.
_GATILHOS_CATALOGO_FORTE = (
    "base bibliografica", "base de conhecimento", "literatura indexada",
    "literatura completa", "toda a literatura", "toda literatura",
    "catalogo da base", "as 39", "todas as 39", "todos os 39",
    "39 referencias", "39 artigos", "39 documentos", "39 obras",
    "knowledge base", "indexed literature", "complete literature",
    "full bibliography", "all references", "all papers", "all documents",
    "base de conocimiento", "literatura indexada", "bibliografia completa",
    "todas las referencias", "todos los articulos",
    "base de connaissances", "litterature indexee", "littérature indexée",
    "bibliographie complete", "toutes les references", "tous les articles",
    "indexad",  # "o que voce tem indexado", "o que esta indexado"
)
# Qualificadores de TOPICO: se aparecem, e busca tematica (RAG), nunca catalogo.
_QUALIFICADORES_TOPICO = (
    "sobre", "a respeito", "acerca", "referente a", "referentes a",
    "que tratam de", "que trata de", "que falam de", "que fala de",
    "relacionad", "do tema", "no tema", "a cerca",
    "about", "regarding", "related to", "concerning",
    "sobre", "acerca", "relacionad", "referente a",
    "sur", "concernant", "relatif", "relative",
)


def _quer_catalogo(pergunta: str) -> bool:
    """
    True quando o pedido e pelo INVENTARIO inteiro da base bibliografica
    (lista completa), e nao por uma busca tematica na literatura.

    Ex. True : "liste todas as referencias", "o que voce tem indexado",
               "quantos artigos voce tem", "quais documentos voce tem",
               "mostre a base bibliografica".
    Ex. False: "cite artigos sobre anomalias", "quais artigos sobre falhas
               CA?", "o que a literatura diz sobre Weibull?" (vao para o RAG).
    """
    txt = _normalizar(pergunta)
    if any(t in txt for t in (
        "resultado", "resultados", "metrica", "metricas", "grafico",
        "graficos", "matriz", "matrizes", "replicacao", "replicacoes",
        "replicado", "replicados", "artefato", "artefatos", "proveniencia",
    )):
        return False
    # Qualificador de topico => busca tematica; deixa para o RAG.
    if any(q in txt for q in _QUALIFICADORES_TOPICO):
        return False
    if any(g in txt for g in _GATILHOS_CATALOGO_FORTE):
        return True
    tem_biblio = any(t in txt for t in _TERMOS_BIBLIO)
    tem_total = any(t in txt for t in _TERMOS_TOTALIDADE)
    return tem_biblio and tem_total


# --- Experimentos de ML por artigo-base ------------------------------------
# Sobrenome citado -> chave do experimento no registry.
_AUTORES_EXP = {
    "ibrahim": "ibrahim",
}

# Autores CITÁVEIS na literatura indexada (superset dos de experimento):
# usados como guarda para NÃO desviar uma pergunta autoral ("o que o Stender
# diz...") para a ferramenta de catálogo de datasets. Stender/Ahirwar/etc.
# não são mais experimentos executáveis, mas seguem sendo papers indexados.
_AUTORES_CITAVEIS = set(_AUTORES_EXP) | {
    "stender", "ahirwar", "ghoneim", "sharma", "cristaldi", "golnas",
    "voss", "torres",
}
_VERBOS_RODAR_EXP = (
    "rode", "rodar", "roda", "execut", "teste", "testar", "testa",
    "treine", "treinar", "treina", "rodar os modelos",
    "run", "execute", "test", "train",
    "ejecute", "ejecuta", "ejecutar", "prueba", "probar", "entrena",
    "entrenar",
    "executez", "executer", "exécuter", "tester", "entraine", "entrainer",
    "entraîner",
)


def _experimentos_alvo(pergunta: str) -> list[str]:
    """Quais experimentos o usuário citou (por autor, tarefa, ou 'todos')."""
    txt = _normalizar(pergunta)
    alvos = [k for nome, k in _AUTORES_EXP.items() if nome in txt]
    if alvos:
        return alvos
    if any(t in txt for t in ("anomalia", "anomalias", "anomaly", "anomalies", "anomalie", "anomalies")):
        return ["ibrahim"]
    if any(t in txt for t in ("todos", "tudo", "compare", "comparar", "todas", "all", "todos", "todas", "tous", "toutes")):
        return ["ibrahim"]
    return []


def _quer_rodar_experimento(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    tem_verbo = any(v in txt for v in _VERBOS_RODAR_EXP)
    if any(t in txt for t in ("experimento", "experiment", "experimento", "experience", "expérience")):
        return tem_verbo
    # "teste os modelos do ahirwar" (sem a palavra 'experimento')
    tem_autor = any(a in txt for a in _AUTORES_EXP)
    tem_modelos = any(t in txt for t in ("modelo", "modelos", "model", "models", "modele", "modeles", "modèle", "modèles"))
    return tem_autor and tem_verbo and tem_modelos


def _quer_catalogo_experimentos(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    if not any(t in txt for t in ("experimento", "experiment", "experimento", "experience", "expérience")):
        return False
    if _quer_rodar_experimento(pergunta):
        return False
    if _quer_consultar_resultados_experimentos(pergunta):
        return False
    consulta = any(t in txt for t in (
        "quais", "que ", "liste", "lista", "listar", "mostre", "mostra",
        "disponiveis", "disponivel", "existem", "tem ", "status", "quantos",
        "which", "what", "list", "show", "available", "exist", "how many",
        "cuales", "que ", "lista", "listar", "muestra", "disponibles",
        "quels", "quelles", "quoi", "liste", "montre", "disponibles",
    ))
    return consulta


def _quer_limpar_experimentos(pergunta: str) -> bool:
    """Pedido destrutivo voltado aos experimentos/artigos, nao ao pipeline AE."""
    txt = _normalizar(pergunta)
    if not _quer_limpar(pergunta):
        return False
    return (
        any(t in txt for t in ("experimento", "experimentos", "benchmark", "benchmarks"))
        or any(autor in txt for autor in _AUTORES_EXP)
    )


def _quer_literatura_tematica(pergunta: str) -> bool:
    """
    Pedido de busca/sintese bibliografica sobre um tema.

    Deve ir para RAG, nao para ferramentas. Sem esse retorno explicito, o LLM
    roteador pode confundir "cite artigos sobre anomalias" com o catalogo de
    experimentos por artigo-base.
    """
    txt = _normalizar(pergunta)
    if _quer_catalogo(pergunta):
        return False
    if any(t in txt for t in (
        "experimento", "experimentos", "resultado", "resultados",
        "artefato", "artefatos", "metricas", "metrica",
    )):
        return False

    termos_biblio = (
        "literatura", "artigo", "artigos", "paper", "papers", "fonte",
        "fontes", "referencia", "referencias", "bibliografia",
        "bibliografica", "bibliografico", "autor", "autores",
        "literature", "source", "sources", "reference", "references",
        "bibliography", "author", "authors",
    )
    gatilhos_tematicos = (
        "cite", "citar", "cita", "liste referencias", "liste artigos",
        "quais autores", "quais artigos", "artigos sobre", "papers sobre",
        "sobre", "segundo a literatura", "com base na literatura",
        "revisao bibliografica", "estado da arte", "o que a bibliografia diz",
        "cite papers", "cite sources", "papers about", "articles about",
        "literature review", "state of the art",
    )
    return any(t in txt for t in termos_biblio) and any(g in txt for g in gatilhos_tematicos)


def _quer_consultar_datasets(pergunta: str) -> bool:
    """Pergunta sobre os datasets do projeto (Paderborn / PV Farms)."""
    txt = _normalizar(pergunta)
    if _quer_rodar_experimento(pergunta):
        return False
    # Pergunta que cita um autor da literatura ("o que o Stender diz...") é
    # consulta bibliográfica → RAG, não catálogo de datasets.
    if any(autor in txt for autor in _AUTORES_CITAVEIS):
        return False
    if any(t in txt for t in (
        "resultado", "resultados", "replicacao", "replicacoes",
        "replicado", "replicados", "artigo", "artigos", "paper", "papers",
    )):
        return False
    tem_dataset = any(t in txt for t in (
        "dataset", "datasets", "paderborn", "pv farms", "pv-farms",
        "dados do projeto", "conjunto de dados", "conjuntos de dados",
    ))
    if not tem_dataset:
        return False
    consulta = any(t in txt for t in (
        "qual", "quais", "que ", "o que", "explique", "explica", "descreva",
        "diferenca", "diferença", "sobre", "mostre", "fale", "compare",
        "para que", "serve", "quantos", "quantas",
    ))
    return consulta


def _quer_comparar_abordagens(pergunta: str) -> bool:
    """Pergunta sobre supervisionado x não supervisionado x sintético."""
    txt = _normalizar(pergunta)
    if _quer_rodar_experimento(pergunta):
        return False
    tem_abordagem = any(t in txt for t in (
        "abordagem", "abordagens", "supervisionad", "nao supervision",
        "não supervision",
    ))
    contexto = any(t in txt for t in (
        "supervision", "anomalia", "classificacao", "classificação",
        "deteccao", "detecção", "diferenca", "diferença", "compare",
        "comparar", "versus", " vs ",
    ))
    return tem_abordagem and contexto


def _quer_classificador_pv(pergunta: str) -> str | None:
    """Roteia ações do classificador supervisionado PV Farms (CC)."""
    txt = _normalizar(pergunta)
    quer_classificar = ("classifique" in txt or "classificar" in txt) and "amostra" in txt
    tem_clf = "classificador" in txt
    if not (tem_clf or quer_classificar):
        return None
    if quer_classificar:
        return "classificar_amostra_pv"
    if any(t in txt for t in ("treine", "treinar", "treina", "retreine")):
        return "treinar_classificador_pv"
    return "avaliar_classificador_pv"


def _quer_resposta_autoral(pergunta: str) -> bool:
    """Perguntas que precisam de interpretacao do agente, nao so tabela pronta."""
    txt = _normalizar(pergunta)
    termos = (
        "na sua opiniao", "sua opiniao", "opine", "parecer", "interprete",
        "interpretar", "explique", "explica", "como se eu fosse apresentar",
        "apresentar", "orientadora", "confiavel", "confiaveis", "recomende",
        "recomendar", "qual escolher", "escolher", "reforca", "reforcam",
        "sustenta", "sustentam", "discuta", "analise", "analisa",
        "compare", "comparar", "compara", "separe", "separar",
        "origem", "metodologia", "dados usados", "recalculado",
        "recalculados", "replicacao", "replicacoes",
        "o que isso significa", "implicacao", "implicacoes",
        "in your opinion", "your opinion", "interpret", "explain", "present",
        "advisor", "supervisor", "reliable", "recommend", "choose",
        "support", "supports", "discuss", "analyze", "analyse",
        "what does this mean", "implication", "implications",
        "en tu opinion", "tu opinion", "interpreta", "explica", "presentar",
        "orientadora", "confiable", "recomienda", "elegir", "refuerza",
        "sustenta", "discute", "analiza", "que significa",
        "a ton avis", "votre avis", "interprete", "explique", "presenter",
        "directrice", "fiable", "recommande", "choisir", "soutient",
        "discute", "analyse", "qu est ce que cela signifie",
    )
    return any(t in txt for t in termos)


def _quer_comparar_auc_experimentos(pergunta: str) -> bool:
    """
    Pedido de COMPARAÇÃO dos experimentos de anomalia por AUC.

    Distingue de 'rode o experimento' (rodar_experimento_artigo) e de 'compare
    as abordagens de ML' (comparar_abordagens_ml). Só dispara quando o verbo de
    comparação acompanha experimentos ou anomalia, sem verbo de execução.

    Ex. True : "compare os experimentos de anomalia", "comparar por AUC",
               "qual o melhor modelo de anomalia", "analise os experimentos".
    Ex. False: "rode o experimento e compare", "compare as abordagens de ML".
    """
    txt = _normalizar(pergunta)
    # Confronto direto com o concorrente, nomeando-o. Vem ANTES do desvio por
    # "rodar" porque desde a aposentadoria do framework E1 não há mais o que
    # rodar por aqui: qualquer forma do pedido lê a comparação publicada.
    # "meu detector e melhor que o AE-LSTM?" escapava de todos os padrões.
    if any(t in txt for t in ("ae-lstm", "ae lstm", "aelstm", "ibrahim")):
        return True
    if _quer_rodar_experimento(pergunta):
        return False
    # Pedidos compostos de apresentação devem ler os artefatos já calculados.
    # A ferramenta de banco comum gera somente a comparação AUC e, por isso,
    # não consegue cumprir matrizes, contagens ou formatos gráficos pedidos.
    if any(t in txt for t in (
        "anomalias detectadas", "quantas anomalias", "matriz", "matrizes",
        "grafico", "graficos", "por pontos", "dot plot", "barras",
        "cada artigo", "proprio grupo", "próprio grupo",
    )):
        return False
    tem_compare = any(t in txt for t in (
        "compare", "comparar", "comparacao", "comparacao",
        "analise", "analisa", "qual o melhor", "qual e o melhor",
        "melhor modelo",
    ))
    if not tem_compare:
        return False
    tem_alvo = any(t in txt for t in (
        "experimento", "experimentos",
        "anomalia", "anomalias",
        "protocolo", "protocolos",
        "auc", "por auc",
        # comparação do método proposto com a literatura (mesma ferramenta)
        "literatura", "meu metodo", "metodo proposto", "meu autoencoder",
        "com os artigos", "estado da arte",
    ))
    return tem_alvo


def _quer_consultar_resultados_experimentos(pergunta: str) -> bool:
    """Consulta aos artefatos ja gerados dos experimentos por artigo."""
    txt = _normalizar(pergunta)
    tem_exp = (
        any(t in txt for t in ("experimento", "experiment", "experimento", "experience", "expérience"))
        or any(autor in txt for autor in _AUTORES_EXP)
        or any(t in txt for t in (
            "modelo", "modelos", "model", "models", "modele", "modeles",
            "anomalia", "anomalias", "anomaly", "anomalies", "anomalie",
        ))
    )
    tem_resultado = any(t in txt for t in (
        "resultado", "resultados", "metrica", "metricas", "f1", "auc",
        "recall", "precision", "matriz", "grafico", "graficos",
        "anomalias detectadas", "detectaram", "detectou",
        "result", "results", "metric", "metrics", "matrix", "confusion",
        "chart", "charts", "plot", "plots", "detected anomalies",
        "resultado", "resultados", "metrica", "metricas", "matriz",
        "grafico", "graficos", "anomalias detectadas",
        "resultat", "resultats", "résultat", "résultats", "metrique",
        "métrique", "matrice", "graphique", "anomalies detectees",
    ))
    return tem_exp and (tem_resultado or _quer_resposta_autoral(pergunta))


def _quer_registrar_no_cerebro(pergunta: str) -> bool:
    """Pedido explicito de GRAVAR conhecimento curado no vault.

    Exige verbo de registro E destino (cerebro/vault/obsidian/nota) — evita
    confundir com a memoria validada ("lembre que prefiro...") ou com consulta.
    """
    txt = _normalizar(pergunta)
    verbos = ("registre", "registrar", "guarde", "guardar", "anote", "anotar",
              "documente", "documentar", "salve como nota", "crie uma nota",
              "criar nota", "adicione ao cerebro", "registra", "anota")
    destinos = ("cerebro", "vault", "obsidian", "nota curada", "nas notas",
                "no cerebro", "como nota")
    return any(v in txt for v in verbos) and any(d in txt for d in destinos)


def _quer_limpar(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    termos = (
        "apagar", "apague", "limpar", "limpe", "zerar", "zere",
        "remover", "remova", "excluir", "exclua", "deletar", "delete",
        "clear", "clean", "remove", "reset", "erase",
        "borrar", "limpiar", "eliminar", "reiniciar",
        "effacer", "supprimer", "nettoyer", "reinitialiser", "réinitialiser",
    )
    return any(t in txt for t in termos)


# Termos que SOZINHOS já implicam a intenção de mexer com o pipeline,
# mesmo sem aparecer "pipeline" ou outro termo de ML explicitamente.
_TERMOS_PIPELINE_IMPLICITO = (
    "recalcular", "recalculo", "recalcule", "recalcula",
    "refazer", "refaca", "refaça", "regerar", "regere", "regenerar",
    "rodar de novo", "executar de novo", "do zero", "rode tudo",
    "rodar tudo", "executar tudo", "pipeline completo",
    "recalculate", "recompute", "rerun", "run again", "from scratch",
    "run everything", "full pipeline",
    "ejecutar de nuevo", "desde cero", "ejecutar todo",
    "recalculer", "relancer", "executer a nouveau", "exécuter à nouveau",
    "depuis zero", "depuis zéro", "pipeline complet",
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
        "fault", "failure", "validation", "threshold", "anomaly",
        "reliability", "chart", "charts", "result", "results", "artifact",
        "artifacts", "model", "models", "detection", "training", "metrics",
        "image", "images", "figure", "figures", "curve", "curves",
        "visualization", "matrix", "table",
        "falla", "fallas", "validacion", "umbral", "confiabilidad",
        "deteccion", "entrenamiento", "imagen", "imagenes", "tabla",
        "defaillance", "seuil", "anomalie", "fiabilite", "graphique",
        "resultat", "resultats", "modele", "modeles", "entrainement",
        "metriques", "courbe", "matrice", "tableau",
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
        "run", "train", "generate", "calculate", "validate", "inject",
        "estimate", "show", "display", "consult", "see", "which", "what",
        "clear", "remove", "make", "where", "available",
        "ejecutar", "entrenar", "generar", "inyectar", "mostrar",
        "muestra", "cuales", "cual", "limpiar", "eliminar", "hacer", "donde",
        "executer", "exécuter", "entrainer", "entraîner", "generer", "générer",
        "calculer", "injecter", "montrer", "afficher", "consulter", "voir",
        "quels", "quel", "combien", "effacer", "supprimer", "faire", "ou", "où",
    )
    return any(t in txt for t in termos_ml) and any(t in txt for t in termos_acao)


def consultar_status_pipeline(progresso=None, pergunta: str = "") -> dict:
    if progresso:
        progresso("Lendo status do pipeline...")

    capacidade = capacidade_recalculo_pipeline()
    if not capacidade["disponivel"]:
        publicados = estado_resultados_publicados()
        linhas = [
            "## Pipeline de ML — modo de consulta\n",
            "O site não contém o dataset bruto de Paderborn, portanto não "
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
        return "features_ca"
    if any(t in txt for t in (
        "feature", "features", "sinais", "dados processados",
        "signals", "processed data", "senales", "señales", "datos procesados",
        "signaux", "donnees traitees", "données traitées",
    )):
        return "features_ca"
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
    if not capacidade_recalculo_pipeline()["disponivel"]:
        resumo = _resultado_pos_execucao(stage_key, pergunta)
        return {
            "ok": True,
            "etapa": NOMES_ETAPAS[stage_key],
            "mensagem": (
                "## Cálculo indisponível neste ambiente\n\n"
                "O dataset bruto de Paderborn não é publicado no Streamlit "
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
    if not capacidade_recalculo_pipeline()["disponivel"]:
        status = consultar_status_pipeline(pergunta=pergunta)
        return {
            "ok": True,
            "etapa": "Pipeline completo",
            "mensagem": (
                "## Cálculo indisponível neste ambiente\n\n"
                "O pipeline pesado só pode ser recalculado no PC que contém "
                "`dados/brutos/Inverter_Data_Set.csv`. O site está em modo de "
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
    except Exception:
        pass
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


def listar_base_bibliografica(progresso=None, pergunta: str = "") -> dict:
    """
    Devolve o catálogo COMPLETO da literatura indexada (todos os documentos,
    agrupados por tema). Lê os metadados do ChromaDB diretamente — NÃO usa RAG
    — então a lista é determinística e nunca trunca nem inventa referências.
    """
    if progresso:
        progresso("Lendo o catálogo completo da base de conhecimento...")

    try:
        import chromadb

        from src.conhecimento.agente import catalogo_literatura
        from src.core.config import NOME_COLECAO, PASTA_CHROMADB

        cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = cliente.get_collection(NOME_COLECAO)
        texto = catalogo_literatura(colecao)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "etapa": "Base bibliográfica",
            "mensagem": (
                "Não consegui ler o catálogo da base de conhecimento agora "
                f"({exc}). Verifique se o ChromaDB foi construído."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }

    return {
        "ok": True,
        "etapa": "Base bibliográfica",
        "mensagem": texto,
        "imagens": [],
        "resposta_pronta": True,  # texto determinístico — não passa pelo LLM
    }


def consultar_comparacao_macro(progresso=None, pergunta: str = "") -> dict:
    """Comparação vigente: método proposto × AE-LSTM do Ibrahim, por AUC e SMD.

    Lê `resultados/macro/` — a FONTE ÚNICA de resultado de anomalia desde que os
    macro-códigos substituíram o framework por artigo. Não treina nem recalcula:
    só apresenta o que está publicado, então funciona também na nuvem.

    Existe porque essa pasta era inalcançável pelo chat: `resultados/experimentos/`
    foi deletada em `9fe0322` e nenhuma ferramenta lia `resultados/macro/`. Pedir
    "compare meu método com a literatura" caía num caminho morto.
    """
    import json
    from pathlib import Path

    from src.core.config import RAIZ_PROJETO

    if progresso:
        progresso("Lendo a comparação publicada (proposto × Ibrahim)...")

    pasta = Path(RAIZ_PROJETO) / "resultados" / "macro"
    tabela = pasta / "comparacao_tabela.md"
    dados = pasta / "comparacao_resultado.json"
    if not tabela.is_file() or not dados.is_file():
        return {
            "ok": False, "etapa": "Comparação com a literatura",
            "mensagem": (
                "Ainda não há comparação publicada em `resultados/macro/`. "
                "Rode no PC: `python -m src.ml.macro_comparar`."
            ),
            "imagens": [], "resposta_pronta": True, "forcar_resposta_direta": True,
        }

    try:
        metodos = json.loads(dados.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "ok": False, "etapa": "Comparação com a literatura",
            "mensagem": f"A comparação publicada está ilegível: {exc}",
            "imagens": [], "resposta_pronta": True, "forcar_resposta_direta": True,
        }

    # Protocolo: sai do próprio artefato, não de constante escrita aqui.
    protocolo = [
        f"- **{m.get('nome', '?')}** — limiar no percentil "
        f"{m.get('percentil', '?')}, FP {m.get('fp_pct', 0):.1f}%, "
        f"{m.get('n_calib', '?')} janelas de calibração e "
        f"{m.get('n_aval', '?')} de avaliação"
        for m in metodos
    ]

    msg = (
        "## Método proposto × literatura\n\n"
        + tabela.read_text(encoding="utf-8").strip()
        + "\n\n**Protocolo de cada método**\n"
        + "\n".join(protocolo)
        + "\n\n**Como ler**\n"
        "- **SMD** é a menor severidade em que o método detecta a falha em ≥95% "
        "das janelas, com o falso positivo travado em 10%. **Menor é melhor** — "
        "é o *pickup* do detector.\n"
        "- Em severidade 1,0 todos saturam em 100%: é o SMD que discrimina.\n\n"
        "**Ressalvas** — evidência **E2** (falha sintética injetada no sinal, "
        "fundamentada na FMECA): mostra que o detector responde à assinatura "
        "elétrica esperada, **não** desempenho em campo. Amostra pequena "
        f"({metodos[0].get('n_aval', '?')} janelas), então os valores são "
        "consistentes, não precisos. A grade de severidade é discreta "
        "(0,05…1,0): um SMD de 0,50 significa \"falhou em 0,3, passou em 0,5\"."
    )

    imagens = []
    for arquivo, legenda in (
        ("comparacao_deteccao_severidade.png", "Detecção por severidade — proposto × Ibrahim"),
        ("proposto_deteccao_severidade.png", "Método proposto — detecção por severidade"),
        ("ibrahim_deteccao_severidade.png", "Ibrahim (AE-LSTM) — detecção por severidade"),
    ):
        caminho = pasta / arquivo
        if caminho.is_file():
            imagens.append({
                "path": str(caminho), "caption": legenda,
                "group": "Comparação com a literatura", "inline": False,
            })

    return {
        "ok": True, "etapa": "Comparação com a literatura",
        "mensagem": msg, "imagens": imagens, "resposta_pronta": True,
    }


def listar_experimentos_artigos(progresso=None, pergunta: str = "") -> dict:
    """Catálogo dos experimentos de ML por artigo-base + status dos modelos."""
    if progresso:
        progresso("Lendo o catálogo de experimentos por artigo...")
    try:
        from src.ml.experimentos_artigos import catalogo_experimentos_md

        msg = catalogo_experimentos_md()
        msg += (
            "\n\nPara rodar, peça por exemplo: \"rode o experimento do Ibrahim\" "
            "ou use a barra lateral (🧪 Experimentos por artigo)."
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "etapa": "Experimentos por artigo",
            "mensagem": f"Não consegui ler o catálogo de experimentos: {exc}",
            "imagens": [], "resposta_pronta": True,
        }
    return {
        "ok": True, "etapa": "Experimentos por artigo",
        "mensagem": msg, "imagens": [], "resposta_pronta": True,
    }


def limpar_experimentos_artigos(progresso=None, pergunta: str = "") -> dict:
    """Apaga artefatos dos experimentos por artigo mediante confirmacao."""
    from pathlib import Path

    from src.ml.experimentos_artigos import ORDEM_EXPERIMENTOS, PASTA_EXPERIMENTOS

    alvos = _experimentos_alvo(pergunta) or list(ORDEM_EXPERIMENTOS)
    alvos = list(dict.fromkeys(alvos))
    rotulo = (
        "TODOS"
        if len(alvos) >= len(ORDEM_EXPERIMENTOS)
        else " ".join(k.upper() for k in alvos)
    )
    token = (
        "CONFIRMAR LIMPEZA EXPERIMENTOS"
        if rotulo == "TODOS" else
        f"CONFIRMAR LIMPEZA EXPERIMENTOS {rotulo}"
    )

    pasta_base = Path(PASTA_EXPERIMENTOS).resolve()
    pastas = []
    for key in alvos:
        p = (pasta_base / key).resolve()
        if pasta_base in p.parents:
            pastas.append(p)

    existentes = [p for p in pastas if p.exists()]
    n_arquivos = sum(
        1 for pasta in existentes for item in pasta.rglob("*") if item.is_file()
    )

    if _normalizar(token) not in _normalizar(pergunta):
        nomes = ", ".join(alvos)
        return {
            "ok": True,
            "etapa": "Limpeza de experimentos",
            "mensagem": (
                f"Isso vai apagar os artefatos dos experimentos: **{nomes}**.\n\n"
                f"Diretorios encontrados: {len(existentes)} | arquivos: {n_arquivos}.\n"
                "A acao e irreversivel e nao apaga dados brutos nem literatura.\n\n"
                f"Para confirmar, escreva exatamente:\n\n`{token}`"
            ),
            "imagens": [],
            "resposta_pronta": True,
        }

    if progresso:
        progresso("Apagando artefatos dos experimentos por artigo...")

    removidos = []
    for pasta in existentes:
        shutil.rmtree(pasta)
        removidos.append(pasta)

    if removidos:
        detalhe = "\n".join(f"- {p.relative_to(RAIZ_PROJETO)}" for p in removidos)
        detalhe = f"\n\nDiretorios removidos:\n{detalhe}"
    else:
        detalhe = "\n\nNao havia diretorios de experimento para remover."

    return {
        "ok": True,
        "etapa": "Limpeza de experimentos",
        "mensagem": (
            "Experimentos por artigo apagados. Os dados brutos permanecem "
            "intactos; quando quiser comparar novamente, peca para rodar o "
            "experimento do autor desejado ou todos os experimentos."
            f"{detalhe}"
        ),
        "imagens": [],
        "resposta_pronta": True,
    }


def _md_experimento_legacy(res: dict) -> tuple[str, list[dict]]:
    """Markdown + imagens de um resultado de experimento."""
    if not res.get("ok"):
        ref = res.get("referencia", res.get("experimento", "experimento"))
        return f"### {ref}\nNão executado — {res.get('mensagem', 'sem modelos disponíveis')}.", []

    mp = res["metrica_principal"]
    linhas = [
        f"### {res['referencia']} — {res['dataset']} ({res['tarefa']})",
        f"| Modelo | {mp} | demais |",
        "|---|---:|---|",
    ]
    for nome, m in res["modelos"].items():
        if not m.get("disponivel", True):
            linhas.append(f"| {nome} | — | _{m.get('motivo', 'indisponível')}_ |")
            continue
        principal = m.get(mp)
        outras = ", ".join(
            f"{k}={v:.3f}" for k, v in m.items()
            if isinstance(v, (int, float)) and k not in (mp, "disponivel")
        )
        linhas.append(f"| {nome} | {principal:.4f} | {outras} |")
    linhas.append(
        f"\n**Melhor: {res['melhor_modelo']}** ({mp}={res['melhor_valor']:.4f}). "
        f"Salvo em `resultados/experimentos/{res['experimento']}/`."
    )
    imagens = []
    graf = res.get("grafico")
    if graf:
        from src.core.utils import resolve_project_path
        graf_abs = resolve_project_path(graf)  # relativo→absoluto (na interface)
        if graf_abs.exists():
            imagens.append({"path": str(graf_abs), "caption": f"{res['referencia']} — comparação"})
    return "\n".join(linhas), imagens


def _md_experimento(res: dict) -> tuple[str, list[dict]]:
    """Markdown + imagens no schema padronizado dos experimentos."""
    if not res.get("ok"):
        ref = res.get("referencia", res.get("experimento", "experimento"))
        return f"### {ref}\nNao executado - {res.get('mensagem', 'sem modelos disponiveis')}.", []

    mp = res["metrica_principal"]
    linhas = [
        f"### {res['referencia']} - {res['dataset']} ({res['tarefa']})",
        "| Modelo | Accuracy | Precision | Recall | F1 | AUC | Specificity | Anomalias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nome, m in res["modelos"].items():
        if not m.get("disponivel", True):
            linhas.append(f"| {nome} (_{m.get('motivo', 'indisponivel')}_) | - | - | - | - | - | - | - |")
            continue
        valores = []
        for chave in ("accuracy", "precision", "recall", "f1", "auc", "specificity"):
            valor = m.get(chave)
            valores.append(f"{valor:.3f}" if isinstance(valor, (int, float)) else "-")
        anomalias = m.get("anomalias_detectadas", "-")
        linhas.append(
            f"| {nome} | {valores[0]} | {valores[1]} | {valores[2]} | "
            f"{valores[3]} | {valores[4]} | {valores[5]} | {anomalias} |"
        )
    linhas.append(
        f"\n**Melhor: {res['melhor_modelo']}** ({mp}={res['melhor_valor']:.4f}). "
        f"Salvo em `resultados/experimentos/{res['experimento']}/`."
    )

    # Bloco de METODOLOGIA do protocolo por artigo (split temporal, injeção
    # FMECA e a regra de decisão de cada modelo) — rastreabilidade na resposta.
    met = res.get("metodologia")
    if met:
        linhas.append(f"\n**Protocolo do artigo** (`{met.get('protocolo', '?')}`):")
        sp = met.get("split", {})
        if sp:
            partes = [f"treino={sp.get('treino')}", f"teste={sp.get('teste')}"]
            if sp.get("val"):
                partes.insert(1, f"val={sp.get('val')}")
            linhas.append(
                f"- Split {sp.get('tipo', '?')} (purga={sp.get('purga_janelas')}): "
                f"{', '.join(partes)} janelas.")
        inj = met.get("injecao", {})
        if inj:
            linhas.append(
                f"- Injeção: {inj.get('tipo', '?')} — famílias FMECA "
                f"{', '.join(inj.get('falhas', []))} (severidade {inj.get('severidade')}).")
        for modelo, regra in (met.get("decisoes") or {}).items():
            linhas.append(f"- Decisão {modelo}: {regra}.")
        for nota in met.get("fidelidade", []):
            linhas.append(f"- _{nota}_")

    # Detecção por família de falha FMECA (quando o protocolo reporta)
    com_falhas = {
        nome: m["deteccao_por_falha"]
        for nome, m in res["modelos"].items()
        if isinstance(m, dict) and m.get("deteccao_por_falha")
    }
    if com_falhas:
        linhas.append("\n**Detecção por família de falha (recall):**")
        linhas.append("| Modelo | Contator AC (NPR 315) | IGBT (NPR 90) | Fusível AC (NPR 30) |")
        linhas.append("|---|---:|---:|---:|")
        for nome, det in com_falhas.items():
            def _pct(v):
                return f"{v:.0%}" if isinstance(v, (int, float)) else "—"
            linhas.append(
                f"| {nome} | {_pct(det.get('contator_ac'))} | "
                f"{_pct(det.get('igbt'))} | {_pct(det.get('fusivel_ac'))} |")

    from src.core.utils import resolve_project_path

    imagens = []
    for graf in res.get("graficos", []) or [res.get("grafico")]:
        if not graf:
            continue
        graf_abs = resolve_project_path(graf)  # relativo→absoluto (na interface)
        if graf_abs.exists():
            imagens.append({"path": str(graf_abs), "caption": f"{res['referencia']} - experimento"})
    return "\n".join(linhas), imagens


def rodar_experimento_artigo(progresso=None, pergunta: str = "") -> dict:
    """Roda um ou mais experimentos por artigo e devolve a comparação."""
    if not capacidade_recalculo_pipeline()["disponivel"]:
        resumo = resumir_resultados(pergunta)
        return {
            "ok": True,
            "etapa": "Experimentos por artigo",
            "mensagem": (
                "## Experimento indisponível neste ambiente\n\n"
                "Os experimentos exigem os dados locais de Paderborn. O site "
                "não os recalcula, mas pode consultar os resultados publicados.\n\n"
                + resumo["mensagem"]
            ),
            "imagens": resumo.get("imagens", []),
            "resposta_pronta": True,
        }

    from src.ml.experimentos_artigos import catalogo_experimentos_md
    # 10.4 — isola cargas pesadas (torch) em subprocesso para
    # que um segfault/conflito de OpenMP não derrube o app. Cai para in-process
    # se o subprocesso não puder ser lançado.
    from src.ml.exec_experimento_isolado import (
        executar_experimento_isolado as executar_experimento,
    )

    alvos = _experimentos_alvo(pergunta)
    if not alvos:
        return {
            "ok": True, "etapa": "Experimentos por artigo",
            "mensagem": (
                "Diga qual experimento rodar (por autor). Ex.: \"rode o "
                "experimento do Ibrahim\" ou \"compare os experimentos de "
                "anomalia\".\n\n" + catalogo_experimentos_md()
            ),
            "imagens": [], "resposta_pronta": True,
        }

    blocos, imagens = [], []
    for key in alvos:
        if progresso:
            progresso(f"Rodando experimento: {key}...")
        try:
            res = executar_experimento(key, progresso=progresso)
        except Exception as exc:  # noqa: BLE001
            res = {"experimento": key, "ok": False, "mensagem": str(exc)}
        md, imgs = _md_experimento(res)
        blocos.append(md)
        imagens.extend(imgs)

    cabecalho = (
        "## Experimentos por artigo — resultados\n"
        if len(alvos) > 1 else ""
    )
    return {
        "ok": True, "etapa": "Experimentos por artigo",
        "mensagem": cabecalho + "\n\n".join(blocos),
        "imagens": imagens, "resposta_pronta": True,
    }


def _contar_linhas(caminho) -> int:
    """Conta linhas de um arquivo grande sem carregá-lo na memória."""
    total = 0
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            total += bloco.count(b"\n")
    return total


def consultar_datasets(progresso=None, pergunta: str = "") -> dict:
    """Explica os datasets do projeto lendo contagens DINAMICAMENTE (sem hardcode)."""
    if progresso:
        progresso("Lendo metadados dos datasets...")
    from pathlib import Path

    from src.core.config import RAIZ_PROJETO

    base = Path(RAIZ_PROJETO) / "dados" / "brutos"
    linhas = ["## Datasets do projeto\n"]

    # PV Farms (supervisionado, falhas CC)
    try:
        import pandas as pd

        tr = pd.read_csv(base / "train_data.csv", sep=";")
        te = pd.read_csv(base / "test_data.csv", sep=";")
        n_feat = tr.shape[1] - (1 if "class" in tr.columns else 0)
        classes = sorted(tr["class"].unique().tolist()) if "class" in tr.columns else []
        linhas.append(
            f"### PV Farms — classificação supervisionada (domínio **CC**)\n"
            f"- Arquivos: `dados/brutos/train_data.csv`, `test_data.csv`\n"
            f"- Treino: {len(tr)} linhas | Teste: {len(te)} linhas | {n_feat} features\n"
            f"- Classes: {classes} (Normal, F1 string, F2 string-terra, F3 string-string)\n"
            f"- Uso: **classificação supervisionada de falhas CC conhecidas**.\n"
            f"- Limitação: NÃO diagnostica falhas CA do inversor."
        )
    except Exception as exc:  # noqa: BLE001
        linhas.append(f"### PV Farms — não foi possível ler ({exc})")

    # Paderborn (saudável, anomalia CA)
    pad = base / "Inverter_Data_Set.csv"
    if pad.exists():
        try:
            n_rows = max(0, _contar_linhas(pad) - 1)  # menos cabeçalho
        except Exception:  # noqa: BLE001
            n_rows = "?"
        linhas.append(
            f"\n### Paderborn — detecção de anomalia (domínio **CA**)\n"
            f"- Arquivo: `dados/brutos/Inverter_Data_Set.csv`\n"
            f"- Amostras: {n_rows} (inversor IGBT trifásico **SAUDÁVEL**, 10 kHz)\n"
            f"- Uso: **treinar o modelo de normalidade** (Autoencoder); como é "
            f"saudável, a validação de anomalia usa falhas sintéticas (E2).\n"
            f"- Limitação: sem rótulos reais de falha."
        )
    else:
        linhas.append("\n### Paderborn — arquivo não encontrado localmente.")

    linhas.append(
        "\n**Separação de domínio:** os dois NÃO se fundem. PV Farms = falhas "
        "**CC** conhecidas (supervisionado); Paderborn = anomalia **CA** por "
        "modelagem de normalidade. O uso combinado é conceitual/arquitetural."
    )
    return {
        "ok": True, "etapa": "Datasets do projeto",
        "mensagem": "\n".join(linhas), "imagens": [], "resposta_pronta": True,
    }


def comparar_abordagens_ml(progresso=None, pergunta: str = "") -> dict:
    """Compara supervisionado x não supervisionado x sintético (FMECA), com rigor."""
    msg = (
        "## Abordagens de ML na dissertação\n\n"
        "| Abordagem | O que faz | Rótulos? | No projeto |\n"
        "|---|---|---|---|\n"
        "| **Supervisionada** | classifica falhas CONHECIDAS | sim | PV Farms (**CC**): RF, AdaBoost, LogReg, Naive Bayes, CN2 |\n"
        "| **Não supervisionada** | aprende a NORMALIDADE e detecta desvios | não | Paderborn (**CA**): Autoencoder denso e AE-LSTM do Ibrahim |\n"
        "| **Sintética (FMECA)** | valida assinaturas CA modeladas | ground truth sintético | injeção de falhas no Paderborn (**E2**) |\n\n"
        "**Rigor:**\n"
        "- O não supervisionado DETECTA anomalia, mas NÃO garante diagnóstico "
        "causal da falha.\n"
        "- A validação sintética (E2) depende de calibração física (ex.: o ruído "
        "de sensor é um proxy).\n"
        "- PV Farms (CC) e Paderborn (CA) NÃO se fundem: o classificador PV Farms "
        "não diagnostica falhas CA do inversor, nem transfere suas métricas ao "
        "pipeline CA.\n"
        "- Cada experimento por artigo segue o PROTOCOLO do próprio artigo "
        "(Shewhart 3σ, contaminação a priori, p99 do treino congelado, "
        "PPO em validação temporal, voto majoritário) — por isso o F1 não é "
        "comparável entre protocolos; compare métodos pelo AUC."
    )
    return {
        "ok": True, "etapa": "Abordagens de ML",
        "mensagem": msg, "imagens": [],
        "resposta_pronta": True, "forcar_resposta_direta": True,
    }


def treinar_classificador_pv(progresso=None, pergunta: str = "") -> dict:
    """Treina e salva o classificador supervisionado PV Farms (CC)."""
    if progresso:
        progresso("Treinando o classificador PV Farms (CC)...")
    try:
        from src.ml.classificador_pv_infer import AVISO_DOMINIO, treinar_e_salvar

        r = treinar_e_salvar()
        m = r["metricas"]
        msg = (
            f"Classificador PV Farms (**CC**) treinado e salvo. "
            f"{r['n_features']} features, classes {r['classes']}.\n\n"
            f"F1={m.get('f1', 0):.3f} · MCC={m.get('mcc', 0):.3f} · "
            f"balanced_acc={m.get('balanced_accuracy', 0):.3f}. Evidência **E1**.\n\n"
            f"{AVISO_DOMINIO}"
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "etapa": "Classificador PV Farms",
                "mensagem": f"Não consegui treinar: {exc}", "imagens": [],
                "resposta_pronta": True}
    return {"ok": True, "etapa": "Classificador PV Farms", "mensagem": msg,
            "imagens": [], "resposta_pronta": True}


def avaliar_classificador_pv(progresso=None, pergunta: str = "") -> dict:
    """Mostra métricas + limitações do classificador PV Farms já treinado."""
    import json
    from pathlib import Path

    from src.core.config import RAIZ_PROJETO
    from src.ml.classificador_pv_infer import AVISO_DOMINIO

    arq = Path(RAIZ_PROJETO) / "resultados" / "classificacao_pv" / "metricas.json"
    if not arq.exists():
        return {"ok": True, "etapa": "Classificador PV Farms",
                "mensagem": "Classificador ainda não treinado. Peça: \"treine o "
                "classificador PV Farms\".", "imagens": [], "resposta_pronta": True}
    m = json.loads(arq.read_text(encoding="utf-8"))
    msg = (
        "## Classificador PV Farms (CC)\n"
        f"- Modelo: {m.get('modelo', 'Random Forest')} · evidência **E1**\n"
        f"- Acurácia: {m.get('accuracy', 0):.3f} · F1: {m.get('f1', 0):.3f} · "
        f"MCC: {m.get('mcc', 0):.3f} · balanced_acc: {m.get('balanced_accuracy', 0):.3f}\n"
        f"- Specificity ({m.get('specificity_tipo', '-')}): {m.get('specificity', 0):.3f}\n\n"
        f"{AVISO_DOMINIO}"
    )
    return {"ok": True, "etapa": "Classificador PV Farms", "mensagem": msg,
            "imagens": [], "resposta_pronta": True}


def classificar_amostra_pv(progresso=None, pergunta: str = "") -> dict:
    """Classifica uma amostra PV Farms enviada como JSON na mensagem."""
    import json
    import re

    from src.ml.classificador_pv_infer import AVISO_DOMINIO, classificar

    achado = re.search(r"\{.*\}", pergunta or "", re.S)
    if not achado:
        return {"ok": True, "etapa": "Classificação PV Farms",
                "mensagem": "Envie a amostra como JSON, ex.: "
                "`classifique a amostra {\"feature_0\": 1.2, ...}`.\n\n" + AVISO_DOMINIO,
                "imagens": [], "resposta_pronta": True}
    try:
        amostra = json.loads(achado.group(0))
    except Exception:
        return {"ok": True, "etapa": "Classificação PV Farms",
                "mensagem": "JSON inválido. Use {\"coluna\": valor, ...}.\n\n" + AVISO_DOMINIO,
                "imagens": [], "resposta_pronta": True}
    r = classificar(amostra)
    if not r.get("ok"):
        msg = f"Não classifiquei: {r.get('erro')}\n\n{r.get('aviso', AVISO_DOMINIO)}"
    else:
        msg = (f"Classe prevista: **{r['classe_nome']}** "
               f"(probabilidade {r['probabilidade']:.2f}) — domínio CC.\n\n"
               f"{r['aviso']} (importância de feature ≠ causalidade.)")
    return {"ok": True, "etapa": "Classificação PV Farms", "mensagem": msg,
            "imagens": [], "resposta_pronta": True}


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
    catalogo = "\n".join(f"- {f['name']}: {f['description']}" for f in ESPEC_FERRAMENTAS)
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
        nomes = {f["name"] for f in ESPEC_FERRAMENTAS}
        ferramenta = dados.get("ferramenta")
        if dados.get("usar_ferramenta") and ferramenta in nomes:
            return {"usar_ferramenta": True, "ferramenta": ferramenta}
        if dados.get("usar_ferramenta") is False:
            return {"usar_ferramenta": False, "ferramenta": None}
    except Exception:
        pass
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
    try:
        from src.core.logs import get_logger

        get_logger(__name__).info("ferramenta acionada: %s", ferramenta)
    except Exception:  # noqa: BLE001 - log nunca pode derrubar a execução
        pass

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
