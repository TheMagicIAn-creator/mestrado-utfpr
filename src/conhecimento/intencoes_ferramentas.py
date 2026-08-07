"""Deteccao deterministica de intencoes relacionadas a ferramentas."""

from __future__ import annotations

import re

# Importa da FOLHA (src.core.texto), nao de src.conhecimento.ferramentas.
# `ferramentas` importa deste modulo, entao pegar `_normalizar` de la fechava um
# ciclo: importar `intencoes_ferramentas` primeiro quebrava com ImportError de
# modulo parcialmente inicializado. `_normalizar` sempre foi um mero alias de
# `normalizar_sem_acentos` (ferramentas.py:19), entao o ciclo nao comprava nada.
from src.core.texto import normalizar_sem_acentos as _normalizar

# Verbo de recomputação + marcador de repetição. A lista fechada anterior exigia
# o INFINITIVO ("rodar de novo") e não reconhecia o imperativo, que é como se
# fala: "rode o pipeline de novo", "retreine o autoencoder", "execute
# novamente" — nenhum dos três ativava force. Sem force a etapa READY é PULADA,
# e a resposta imprimia a tabela de resultados logo abaixo de "já está pronto",
# o que lê como execução fresca. Combinado com o determinismo do treino (mesma
# semente ⇒ mesmos números), não havia como distinguir SKIP de recálculo real
# olhando os arquivos. Ver docs/auditoria_total_src.md §2.
_VERBO_RECOMPUTAR = re.compile(
    r"\b(?:re)?(?:rod|exec|calcul|faz|fac|faç|trein|ger|process|avali|atualiz)\w*",
    re.IGNORECASE,
)
_MARCA_REPETICAO = re.compile(
    r"\b(?:de novo|novamente|outra vez|mais uma vez|do zero|de novo tudo|tudo de novo)\b",
    re.IGNORECASE,
)
# Termos que já significam recomputação por si sós, sem precisar de marcador.
_TERMOS_FORCA_DIRETA = (
    "refazer", "refaca", "refaça", "regerar", "regere", "regenerar",
    "recalcular", "recalculo", "recalcule", "recalcula", "recomputar",
    "recompute", "retreinar", "retreine", "retreina", "reexecutar",
    "reexecute", "reprocessar", "reprocesse", "do zero", "apagar", "forcar",
    "forçar", "force",
)


def _deve_forcar(pergunta: str) -> bool:
    """A pergunta pede recálculo REAL, e não leitura do artefato pronto?

    Falso-negativo aqui é grave: o pesquisador acredita ter retreinado sem ter.
    Por isso a detecção cobre flexão verbal em vez de casar substrings fixas.
    """
    txt = _normalizar(pergunta)
    if any(t in txt for t in _TERMOS_FORCA_DIRETA):
        return True
    return bool(_VERBO_RECOMPUTAR.search(txt) and _MARCA_REPETICAO.search(txt))


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
