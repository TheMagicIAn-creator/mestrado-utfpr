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
from src.ml.pipeline import (
    NOMES_ETAPAS,
    ORDEM_ETAPAS_ML,
    artefatos_a_partir,
    estado_pipeline,
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
        "name": "listar_experimentos_artigos",
        "description": (
            "Lista os EXPERIMENTOS de ML por artigo-base (Ghoneim, Francisti, "
            "Ibrahim, Sharma, Ahirwar, Stender) e o status de cada modelo. Use "
            "quando o usuario perguntar quais experimentos existem, quais "
            "modelos rodam, ou o que da para testar com base nos artigos."
        ),
    },
    {
        "name": "limpar_experimentos_artigos",
        "description": (
            "Apaga artefatos/resultados dos experimentos por artigo-base "
            "(Ghoneim, Francisti, Ibrahim, Sharma, Ahirwar), mediante "
            "confirmacao explicita. Use quando o usuario pedir para apagar, "
            "limpar, excluir ou resetar experimentos por artigo."
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
            "normalidade e detecta anomalia CA) e sintetica orientada pelo FMEA. "
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
    {
        "name": "rodar_experimento_artigo",
        "description": (
            "Treina e avalia os modelos de ML de um ou mais artigos-base e "
            "compara os resultados (AUC/F1 ou acuracia). Use quando o usuario "
            "pedir para RODAR/TESTAR um experimento de um artigo: 'rode o "
            "experimento do Ghoneim', 'teste os modelos do Sharma'. Tarefa "
            "pesada (treina modelos). Para comparar resultados ja gerados, use "
            "consultar_resultados."
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
    "ghoneim": "ghoneim",
    "francisti": "francisti",
    "ibrahim": "ibrahim",
    "sharma": "sharma",
    "ahirwar": "ahirwar",
    "stender": "stender",
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
        return ["francisti", "ibrahim", "sharma", "ahirwar"]
    if any(t in txt for t in ("classificacao", "supervision", "classification", "clasificacion", "classification")):
        return ["ghoneim"]
    if any(t in txt for t in ("todos", "tudo", "compare", "comparar", "todas", "all", "todos", "todas", "tous", "toutes")):
        return ["ghoneim", "francisti", "ibrahim", "sharma", "ahirwar"]
    return []


def _quer_rodar_experimento(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    tem_verbo = any(v in txt for v in _VERBOS_RODAR_EXP)
    if any(t in txt for t in ("experimento", "experiment", "experimento", "experience", "expérience")):
        return tem_verbo
    # "teste os modelos do sharma" (sem a palavra 'experimento')
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
    if any(autor in txt for autor in _AUTORES_EXP):
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


def listar_experimentos_artigos(progresso=None, pergunta: str = "") -> dict:
    """Catálogo dos experimentos de ML por artigo-base + status dos modelos."""
    if progresso:
        progresso("Lendo o catálogo de experimentos por artigo...")
    try:
        from src.ml.experimentos_artigos import catalogo_experimentos_md

        msg = catalogo_experimentos_md()
        msg += (
            "\n\nPara rodar, peça por exemplo: \"rode o experimento do Ghoneim\" "
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

    alvos = _experimentos_alvo(pergunta) or [
        key for key in ORDEM_EXPERIMENTOS if key != "stender"
    ]
    alvos = list(dict.fromkeys(alvos))
    rotulo = (
        "TODOS"
        if len(alvos) >= len([k for k in ORDEM_EXPERIMENTOS if k != "stender"])
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
    # FMEA e a regra de decisão de cada modelo) — rastreabilidade na resposta.
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
                f"- Injeção: {inj.get('tipo', '?')} — famílias FMEA "
                f"{', '.join(inj.get('falhas', []))} (severidade {inj.get('severidade')}).")
        for modelo, regra in (met.get("decisoes") or {}).items():
            linhas.append(f"- Decisão {modelo}: {regra}.")
        for nota in met.get("fidelidade", []):
            linhas.append(f"- _{nota}_")

    # Detecção por família de falha FMEA (quando o protocolo reporta)
    com_falhas = {
        nome: m["deteccao_por_falha"]
        for nome, m in res["modelos"].items()
        if isinstance(m, dict) and m.get("deteccao_por_falha")
    }
    if com_falhas:
        linhas.append("\n**Detecção por família de falha (recall):**")
        linhas.append("| Modelo | LCL (NPR 210) | Desbalanceamento (NPR 150) | Sensor |")
        linhas.append("|---|---:|---:|---:|")
        for nome, det in com_falhas.items():
            def _pct(v):
                return f"{v:.0%}" if isinstance(v, (int, float)) else "—"
            linhas.append(
                f"| {nome} | {_pct(det.get('lcl'))} | "
                f"{_pct(det.get('desbalanceamento'))} | {_pct(det.get('sensor'))} |")

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
    from src.ml.experimentos_artigos import catalogo_experimentos_md
    # 10.4 — isola cargas pesadas (Orange/RL/torch/prophet) em subprocesso para
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
                "experimento do Ghoneim\" ou \"compare os experimentos de "
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
    """Compara supervisionado x não supervisionado x sintético (FMEA), com rigor."""
    msg = (
        "## Abordagens de ML na dissertação\n\n"
        "| Abordagem | O que faz | Rótulos? | No projeto |\n"
        "|---|---|---|---|\n"
        "| **Supervisionada** | classifica falhas CONHECIDAS | sim | PV Farms (**CC**): RF, AdaBoost, LogReg, Naive Bayes, CN2 |\n"
        "| **Não supervisionada** | aprende a NORMALIDADE e detecta desvios | não | Paderborn (**CA**): Autoencoder, Isolation Forest |\n"
        "| **Sintética (FMEA)** | valida assinaturas CA modeladas | ground truth sintético | injeção de falhas no Paderborn (**E2**) |\n\n"
        "**Rigor:**\n"
        "- O não supervisionado DETECTA anomalia, mas NÃO garante diagnóstico "
        "causal da falha.\n"
        "- A validação sintética (E2) depende de calibração física (ex.: o ruído "
        "de sensor é um proxy).\n"
        "- PV Farms (CC) e Paderborn (CA) NÃO se fundem: o classificador PV Farms "
        "não diagnostica falhas CA do inversor, nem transfere suas métricas ao "
        "pipeline CA."
    )
    return {
        "ok": True, "etapa": "Abordagens de ML",
        "mensagem": msg, "imagens": [], "resposta_pronta": True,
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
    "limpar_experimentos_artigos": limpar_experimentos_artigos,
    "buscar_web": buscar_na_web,
    "listar_base_bibliografica": listar_base_bibliografica,
    "listar_experimentos_artigos": listar_experimentos_artigos,
    "rodar_experimento_artigo": rodar_experimento_artigo,
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


def _decisao_rapida(pergunta: str) -> dict | None:
    txt = _normalizar(pergunta)

    # Busca na web — atalho prioritário quando gatilho explícito aparece
    if any(g in txt for g in _GATILHOS_WEB):
        return {"usar_ferramenta": True, "ferramenta": "buscar_web"}

    if _quer_limpar_experimentos(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "limpar_experimentos_artigos"}

    if _quer_literatura_tematica(pergunta):
        return {"usar_ferramenta": False, "ferramenta": None}

    # Experimentos de ML por artigo — checados ANTES do catálogo de literatura,
    # pois "experimento" é um sinal mais específico que "artigo" genérico.
    # RODAR tem prioridade sobre LISTAR.
    if _quer_rodar_experimento(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "rodar_experimento_artigo"}
    if _quer_consultar_resultados_experimentos(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "consultar_resultados"}
    if _quer_catalogo_experimentos(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "listar_experimentos_artigos"}

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

    # Limpeza explícita ("apague", "limpe...") tem prioridade sobre tudo.
    if _quer_limpar(pergunta):
        return {"usar_ferramenta": True, "ferramenta": "limpar_resultados_ml"}

    # "Recalcule", "refaça", "rode tudo de novo", "do zero" → pipeline completo.
    if any(t in txt for t in _TERMOS_PIPELINE_IMPLICITO):
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
        "heatmap", "roc", "tabela",
        "result", "results", "show", "display", "chart", "charts",
        "threshold", "metric", "metrics", "image", "images", "figure",
        "figures", "curve", "curves", "visualization", "matrix", "table",
        "resultado", "resultados", "muestra", "mostrar", "grafico",
        "graficos", "umbral", "metrica", "metricas", "imagen", "imagenes",
        "figura", "curva", "visualizacion", "matriz", "tabla",
        "resultat", "resultats", "montre", "affiche", "graphique",
        "seuil", "metrique", "metriques", "image", "figure", "courbe",
        "visualisation", "matrice", "tableau",
    )
    termos_acao_ativa = (
        "rodar", "execut", "trein", "gerar", "gere", "calcular",
        "injetar", "refazer", "regerar", "recalc",
        "run", "execut", "train", "generate", "calculate", "inject",
        "rerun", "recompute",
        "ejecut", "entren", "gener", "calcul", "inyect",
        "execut", "entraîn", "entrain", "gener", "génér", "calcul", "inject",
    )
    termos_validacao_ativa = ("validar", "valide", "valida")
    tem_acao_ativa = (
        any(t in txt for t in termos_acao_ativa)
        or any(re.search(rf"\b{re.escape(t)}\b", txt) for t in termos_validacao_ativa)
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
    if resultado.get("resposta_pronta") and not _quer_resposta_autoral(pergunta):
        return resultado.get("mensagem", "")

    status = "SUCESSO" if resultado.get("ok") else "FALHA"
    prompt = f"""Voce e o Al IAdo PV, pesquisador tecnico do mestrado do Rodolfo.
Responda em portugues brasileiro natural por padrao. Se Rodolfo perguntou claramente
em ingles, espanhol ou frances, voce pode responder no mesmo idioma.
Use os resultados abaixo como evidencia. Nao invente numeros.
Nao devolva apenas a tabela: interprete, priorize, compare e diga o que isso significa para a dissertacao.
Distinga dados locais, metodologia dos artigos e falhas sinteticas quando isso afetar a interpretacao.

Rodolfo pediu: "{pergunta}"

Resultado tecnico ({status}):
{resultado.get('mensagem', 'sem detalhes')}

Explique de forma natural, humana e tecnicamente precisa."""
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
