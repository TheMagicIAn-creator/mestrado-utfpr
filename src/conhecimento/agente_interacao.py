"""Interacao leve, intencoes, citacoes e utilitarios de consulta."""

from __future__ import annotations

from src.conhecimento.agente import (
    FUSO_PADRAO,
    ORCAMENTOS_RAG,
    _logger,
    agora_local,
    hashlib,
    os,
    re,
)
from src.core.identidade import nome_pesquisador
from src.core.tempo import saudacao_periodo

def _normalizar_texto(texto: str) -> str:
    import re
    import unicodedata

    texto = texto.lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def pedido_sem_literatura(pergunta: str) -> bool:
    """
    True quando o pesquisador proibe explicitamente literatura/fontes.

    Importante: a palavra "literatura" por si so costuma acionar RAG. Sem este
    guard, prompts como "Nao use literatura; explique FMEA com base no projeto"
    acabam recebendo rodape de fontes, contrariando a instrucao principal.
    """
    txt = _normalizar_texto(pergunta or "")
    alvos = (
        "literatura", "fontes", "fonte", "referencias", "referencia",
        "artigos", "artigo", "papers", "paper", "bibliografia",
        "literature", "sources", "source", "references", "reference",
        "bibliography", "fuentes", "referencias", "bibliografia",
        "litterature", "littérature", "bibliographie",
    )
    negacoes_fortes = (
        "nao use", "nao consulte", "nao buscar", "nao busque", "nao cite",
        "dispense", "ignore", "do not use", "dont use", "do not cite",
        "no use", "no consultes", "ne pas utiliser",
    )
    if any(n in txt for n in negacoes_fortes) and any(a in txt for a in alvos):
        return True
    # Preposicoes amplas so negam a literatura quando aparecem imediatamente
    # ligadas ao alvo. "cite a fonte sem inventar" deve continuar consultando.
    negacoes_diretas = (
        "sem literatura", "sem fontes", "sem fonte", "sem referencias",
        "sem artigos", "without literature", "without sources",
        "without references", "sin literatura", "sin fuentes",
        "sin referencias", "sans litterature", "sans sources",
        "sans references",
    )
    if any(frase in txt for frase in negacoes_diretas):
        return True
    return any(t in txt for t in (
        "somente com base no projeto",
        "apenas com base no projeto",
        "so com base no projeto",
        "com base no projeto apenas",
        "use somente o projeto",
        "use apenas o projeto",
        "only based on the project",
        "solo con base en el proyecto",
        "seulement sur la base du projet",
    ))


def _saudacao_pelo_horario() -> str:
    """Retorna 'Bom dia', 'Boa tarde' ou 'Boa noite' conforme a hora atual."""
    return saudacao_periodo(agora_local())


def resposta_interacao_simples(pergunta: str, historico=None) -> str | None:
    """
    Responde localmente a mensagens puramente conversacionais sem acionar RAG/LLM.
    Cobre cumprimentos, despedidas, agradecimentos, reações casuais e correções
    de horário — tudo o que não justifica busca na literatura.

    Perguntas sociais curtas sao reconhecidas antes dos guards interrogativos.
    Mensagens tecnicas ou longas continuam seguindo para ferramentas/RAG.
    """
    pergunta_original = pergunta or ""
    txt = _normalizar_texto(pergunta_original).strip()
    termos = [t for t in txt.split() if t]

    if not termos:
        return None

    perguntas_sociais = {
        "tudo bem",
        "tudo certo",
        "como vai",
        "como esta",
        "oi tudo bem",
        "ola tudo bem",
        "opa tudo bem",
        "bom dia",
        "boa tarde",
        "boa noite",
        "tudo bem aliado",
        "como voce esta",
        "quanto tempo",
    }
    saudacao_oi = re.fullmatch(
        r"oi+(?: aliado)?(?: quanto tempo)?",
        txt,
    )
    eh_pergunta_social = txt in perguntas_sociais or bool(saudacao_oi)

    # Perguntas sociais como "tudo bem?" nao justificam inicializar RAG/LLM.
    if "?" in pergunta_original and not eh_pergunta_social:
        return None

    # Guard 2: palavras interrogativas → é pergunta mesmo sem '?'.
    PALAVRAS_INTERROGATIVAS = {
        "que", "qual", "quais", "quanto", "quantos", "quanta", "quantas",
        "onde", "cade", "quando", "como", "por", "porque", "pq", "porquê",
        "poderia", "poderias", "consegue", "consegues", "pode",
        "what", "which", "where", "when", "how", "why", "can", "could",
        "que", "cual", "cuales", "donde", "cuando", "como", "por", "puede",
        "quoi", "quel", "quels", "quelle", "quelles", "ou", "quand",
        "comment", "pourquoi", "peux", "pouvez",
    }
    if any(t in PALAVRAS_INTERROGATIVAS for t in termos) and not eh_pergunta_social:
        return None

    # Guard 3: termos técnicos → RAG/ferramentas resolvem.
    TERMOS_PESQUISA = {
        "fmea", "fmeca", "npr", "rpn", "weibull", "rul", "mttf", "b10",
        "autoencoder", "inversor", "inversores", "fotovoltaico", "fotovoltaica",
        "pv", "falha", "falhas", "pipeline", "feature", "features", "validacao",
        "auc", "f1", "recall", "precision", "dataset", "paderborn", "rcm",
        "modelo", "algoritmo", "dissertacao", "mestrado", "metodologia",
        "confiabilidade", "lcl", "igbt", "thd", "fft", "rms", "anomalia",
        "smd", "ceamazon", "filtro", "sensor", "harmonicos", "deteccao",
        "literatura", "artigo", "paper", "tese", "tcc", "metrica", "metricas",
        "resultado", "resultados", "limiar", "baseline", "ml", "imagem",
        "imagens", "grafico", "graficos", "figura", "figuras", "curva",
        "curvas", "plot", "roc", "matriz", "tabela", "internet", "web",
        "wikipedia", "google", "pesquise", "pesquisar", "busque", "buscar",
        "hora", "horas", "data",
        "fault", "failure", "failures", "anomaly", "anomalies", "reliability",
        "maintenance", "inverter", "photovoltaic", "dataset", "paper",
        "source", "sources", "reference", "references", "metrics", "results",
        "chart", "charts", "figure", "figures", "confusion", "matrix",
        "falla", "fallas", "anomalia", "anomalias", "confiabilidad",
        "mantenimiento", "inversor", "fotovoltaico", "articulo", "fuente",
        "referencia", "metricas", "resultados", "grafico", "matriz",
        "defaillance", "defaillances", "anomalie", "fiabilite", "onduleur",
        "photovoltaique", "article", "source", "reference", "resultats",
    }
    if any(t in TERMOS_PESQUISA for t in termos):
        return None

    # Guard 4: mensagem comprida geralmente carrega intenção técnica.
    if len(termos) > 12:
        return None

    saudacao_h = _saudacao_pelo_horario()

    saudacoes_genericas = {
        "oi", "ola", "opa", "salve", "eai", "eae", "hey", "alo",
        "olá", "fala", "fala ai", "fala cara", "tudo bem", "tudo certo",
        "como vai", "como esta", "hi", "hello", "hola", "buenas",
        "bonjour", "salut",
        "quanto tempo",
    }
    agradecimentos = {
        "obrigado", "obrigada", "valeu", "thanks", "grato", "grata",
        "agradeco", "agradeço", "obg", "vlw", "thank", "gracias", "merci",
    }
    despedidas = {
        "tchau", "ate", "ateh", "ate mais", "falou", "ate logo",
        "ate amanha", "ate breve", "vou indo", "ate depois",
    }
    reacoes_curtas = {
        "kkk", "kk", "rs", "rsrs", "haha", "hahaha", "hehe", "hehehe",
        "legal", "show", "massa", "top", "bacana", "blz", "beleza",
        "ok", "okay", "certo", "entendi", "entendido", "perfeito",
        "ótimo", "otimo", "boa", "bom", "fechou", "combinado",
    }

    tem_bomdia = "bom dia" in txt or "bomdia" in txt
    tem_boatarde = "boa tarde" in txt or "boatarde" in txt
    tem_boanoite = "boa noite" in txt or "boanoite" in txt
    def contem_frase(frases: set[str]) -> bool:
        return any(txt == frase or txt.startswith(f"{frase} ") for frase in frases)

    tem_saudacao_gen = bool(saudacao_oi) or contem_frase(saudacoes_genericas)
    tem_agradecimento = contem_frase(agradecimentos)
    tem_despedida = contem_frase(despedidas) or txt.startswith("ate ")
    tem_reacao = contem_frase(reacoes_curtas)
    nome = nome_pesquisador()
    conversa_em_andamento = bool(historico)

    # ── Correção de horário (ex.: "Tá de noite cara kkk") ─────
    fala_de_noite = "de noite" in txt or "ta noite" in txt or "esta noite" in txt
    fala_de_tarde = "de tarde" in txt or "ta tarde" in txt or "esta tarde" in txt
    fala_de_dia = "de dia" in txt or "ta dia" in txt or "esta dia" in txt
    if fala_de_noite or fala_de_tarde or fala_de_dia:
        return (
            f"Boa correção. Eu estava no automático. "
            f"**{saudacao_h}**, {nome}. Como seguimos?"
        )

    # ── Cumprimentos específicos por período ──────────────────
    if tem_bomdia:
        if saudacao_h == "Bom dia":
            return f"Bom dia, {nome}! Como seguimos com a pesquisa?"
        return (
            f"Saudação anotada, mas aqui já é **{saudacao_h.lower()}**. "
            "De todo modo, estou por aqui."
        )
    if tem_boatarde:
        if saudacao_h == "Boa tarde":
            return f"Boa tarde, {nome}! Em que parte da pesquisa trabalhamos agora?"
        return f"Aqui na verdade é **{saudacao_h.lower()}**, mas estou à disposição."
    if tem_boanoite:
        if saudacao_h == "Boa noite":
            return f"Boa noite, {nome}! Estou por aqui. Como seguimos?"
        return f"Aqui ainda é **{saudacao_h.lower()}**, mas seja bem-vindo!"

    # ── Saudações genéricas ───────────────────────────────────
    if tem_saudacao_gen:
        if conversa_em_andamento:
            if txt in {"tudo bem", "tudo certo", "como vai", "como esta", "como voce esta"}:
                return f"Tudo bem por aqui, {nome}. E com você?"
            return f"Oi, {nome}. Como seguimos?"
        return (
            f"{saudacao_h}, {nome}! Tudo bem por aqui. "
            "Em que parte da pesquisa trabalhamos hoje?"
        )

    # ── Despedidas ────────────────────────────────────────────
    if tem_despedida:
        return (
            f"Até mais, {nome}! Quando voltar, retomamos de onde paramos. "
            "Boa pesquisa."
        )

    # ── Agradecimentos ────────────────────────────────────────
    if tem_agradecimento:
        return "Disponha. Seguimos lapidando o mestrado com calma e rigor."

    # ── Reações curtas ────────────────────────────────────────
    if tem_reacao and len(termos) <= 5:
        return "Perfeito. Quando quiser continuar, é só mandar a próxima pergunta."

    return None


def _orcamento_rag(nome_provedor: str | None = None) -> dict:
    nome = (nome_provedor or "").lower()
    if any(marca in nome for marca in ("router", "gemini", "google")):
        orcamento = ORCAMENTOS_RAG["amplo"].copy()
    else:
        orcamento = ORCAMENTOS_RAG["padrao"].copy()
    for chave in orcamento:
        env = os.getenv(f"AL_IADO_RAG_{chave.upper()}")
        if env:
            try:
                orcamento[chave] = max(1, int(env))
            except ValueError:
                pass
    return orcamento


def _limitar_texto(texto: str, limite: int) -> str:
    if limite <= 0 or len(texto) <= limite:
        return texto
    corte = texto[:limite].rsplit(" ", 1)[0].strip()
    return corte + "\n[trecho encurtado para caber no limite de inferência]"


def _tokens_busca(pergunta: str) -> list[str]:
    stopwords = {
        "a", "as", "o", "os", "de", "da", "do", "das", "dos", "e", "em",
        "no", "na", "nos", "nas", "um", "uma", "para", "por", "sobre",
        "fale", "explique", "quais", "qual", "como", "que", "com",
        "the", "and", "for", "with", "about", "from", "what", "which", "how",
        "show", "explain", "compare", "give", "me",
        "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "sobre",
        "con", "para", "por", "que", "cual", "cuales", "como", "explique",
        "le", "la", "les", "des", "du", "un", "une", "avec", "pour", "sur",
        "quel", "quels", "quelle", "quelles", "comment", "expliquez",
    }
    termos = [
        t for t in _normalizar_texto(pergunta).split()
        if len(t) > 2 and t not in stopwords
    ]
    extras = []
    # Mapa multilingue PT<->EN<->ES<->FR. Quando a pergunta usa um termo em
    # qualquer idioma suportado, injetamos variantes tecnicas para que papers
    # em ingles e notas em portugues sejam recuperados pelo reranker lexico.
    mapa = {
        "fmea": [
            "failure", "mode", "effects", "analysis", "fmeca", "npr", "rpn",
            "modos", "falhas", "efeitos", "analise", "fallas", "efectos",
            "defaillance", "effets",
        ],
        "fmeca": [
            "fmea", "criticidade", "criticality", "npr", "rpn",
            "modos", "falhas", "efeitos", "analise", "tecnicas",
            "criticidad", "criticite",
        ],
        "npr": ["rpn", "fmea", "criticidade", "criticality"],
        "rpn": ["npr", "fmea", "risk", "priority"],
        "weibull": ["confiabilidade", "reliability", "fiabilidade", "fiabilite", "rul", "mttf", "b10"],
        "autoencoder": ["anomalia", "anomaly", "anomalia", "anomalie", "reconstrucao", "reconstruction", "detector"],
        "inversor": ["fotovoltaico", "pv", "converter", "inverter", "photovoltaic", "ondulador", "onduleur"],
        "inverter": ["inversor", "photovoltaic", "pv", "converter", "ondulador", "onduleur"],
        "ondulador": ["inversor", "inverter", "pv", "fotovoltaico"],
        "onduleur": ["inversor", "inverter", "pv", "photovoltaique"],
        "fotovoltaico": ["pv", "photovoltaic", "solar", "inverter", "fotovoltaica", "photovoltaique"],
        "photovoltaic": ["fotovoltaico", "pv", "solar", "inverter"],
        "manutencao": ["maintenance", "mantenimiento", "preventive", "predictive", "preditiva"],
        "maintenance": ["manutencao", "mantenimiento", "predictive", "reliability"],
        "mantenimiento": ["manutencao", "maintenance", "predictivo", "confiabilidad"],
        "preditiva": ["predictive", "predictivo", "manutencao", "maintenance", "prognosis"],
        "predictive": ["preditiva", "predictivo", "maintenance", "prognosis"],
        "confiabilidade": ["reliability", "confiabilidad", "fiabilite", "rcm", "weibull", "mttf"],
        "reliability": ["confiabilidade", "confiabilidad", "fiabilite", "rcm", "weibull"],
        "confiabilidad": ["confiabilidade", "reliability", "weibull"],
        "fiabilite": ["confiabilidade", "reliability", "weibull"],
        "rcm": ["reliability", "centered", "maintenance", "manutencao", "centrada", "centrado"],
        "anomalia": ["anomaly", "anomalie", "outlier", "detection", "deteccao"],
        "anomaly": ["anomalia", "anomalie", "outlier", "detection"],
        "anomalie": ["anomalia", "anomaly", "detection"],
        "deteccao": ["detection", "deteccion", "detection", "anomalia", "anomaly", "diagnosis"],
        "detection": ["deteccao", "deteccion", "anomalia", "anomaly", "diagnosis"],
        "deteccion": ["deteccao", "detection", "anomalia"],
        "falha": ["failure", "fault", "falla", "defaillance", "defect", "falhas"],
        "falhas": ["failure", "fault", "failures", "fallas", "defaillances", "modes"],
        "fault": ["falha", "failure", "falla", "defaillance", "diagnosis"],
        "failure": ["falha", "fault", "falla", "defaillance", "mode"],
        "falla": ["falha", "fault", "failure", "modo"],
        "defaillance": ["falha", "fault", "failure"],
        "lcl": ["filter", "filtro", "filtre", "passive", "harmonic", "harmonico"],
        "igbt": ["transistor", "switching", "power", "semiconductor"],
        "rul": ["remaining", "useful", "life", "vida", "util", "weibull", "mttf"],
        "ml": ["machine", "learning", "aprendizado", "aprendizaje", "algorithm"],
        "machine": ["learning", "ml", "algorithm", "aprendizado", "aprendizaje"],
        "learning": ["machine", "ml", "aprendizado", "aprendizaje"],
    }
    for termo in termos:
        extras.extend(mapa.get(termo, []))
    return list(dict.fromkeys(termos + extras))


# Autores/instituicoes presentes na base — qualquer mencao a um deles
# forca consulta a literatura, mesmo sem palavras como "fonte" ou
# "artigo". Inclui nomes proprios curtos (NASA) e siglas comuns. A
# lista oficial e materializada via autores_indexados(colecao) na
# primeira chamada, lendo o ChromaDB; este fallback cobre quando a
# colecao nao esta disponivel.
AUTORES_INDEXADOS_FALLBACK = {
    "nasa", "administration",
    "torres",
    "lafraia",
    "carpinetti",
    "sakurada",
    "muqauwim",
    "frontin",
    "moura",
    "eletrica",
    "stewart",
    "gonzalez",
    "tekalp",
    "oppenheim",
    "smith",
    "diniz",
    "grewal",
    "ahirwar",
    "francisti",
    "ghoneim",
    "ibrahim",
    "marangis",
    "narayanan",
    "puc-rio", "pucrio", "puc",
    "risi",
    "sharma",
    "silva",
    "xavier",
    "cristaldi",
    "dhople",
    "joshi",
    "karim",
    "monteiro",
    "pahwa",
    "patil",
    "shuttleworth",
    "stender",
    "voss",
    # Datasets/instituicoes — tambem ativam consulta a literatura
    "paderborn",
    "ceamazon",
    "ufpa",
    "utfpr",
    "ieee",
    "iec",
    "abnt",
    "iso",
    "mil-hdbk", "milhdbk",
}

_AUTORES_CACHE: set[str] = set()
_AUTOR_CANONICO_CACHE: dict[str, set[str]] = {}
# Mapa: autor canonico (ex.: 'Grewal') → lista de arquivos desse autor
# (necessario porque ha autores com varios papers, e where={"autor": X}
# pode trazer todos os chunks de um arquivo e nenhum do outro quando o
# limit nao cobre o primeiro).
_AUTOR_ARQUIVOS_CACHE: dict[str, set[str]] = {}


def autores_indexados(colecao=None) -> set[str]:
    """
    Retorna o conjunto de autores presentes no ChromaDB (campo 'autor'
    do metadado), em minusculas e normalizado. Em caso de erro ou colecao
    nao fornecida, usa o fallback hardcoded.

    Tambem popula _AUTOR_CANONICO_CACHE, que mapeia cada token normalizado
    (incluindo sub-tokens de autores compostos como 'Puc Rio' → 'puc' e
    'rio') para o conjunto de formas canonicas do metadado autor.
    """
    global _AUTORES_CACHE, _AUTOR_CANONICO_CACHE, _AUTOR_ARQUIVOS_CACHE
    if _AUTORES_CACHE:
        return _AUTORES_CACHE
    nomes: set[str] = set(AUTORES_INDEXADOS_FALLBACK)
    canonicos: dict[str, set[str]] = {}
    autor_arquivos: dict[str, set[str]] = {}
    if colecao is not None:
        try:
            offset, lote = 0, 500
            while True:
                r = colecao.get(limit=lote, offset=offset, include=["metadatas"])
                metas = r.get("metadatas", [])
                if not metas:
                    break
                for m in metas:
                    autor_raw = str(m.get("autor", "")).strip()
                    arquivo = str(m.get("arquivo", "")).lower()
                    autor_norm = _normalizar_texto(autor_raw).strip()
                    if autor_raw and arquivo:
                        autor_arquivos.setdefault(autor_raw, set()).add(arquivo)
                    if autor_norm:
                        nomes.add(autor_norm)
                        # Indexa o autor completo e cada sub-token
                        canonicos.setdefault(autor_norm, set()).add(autor_raw)
                        for sub in autor_norm.split():
                            if len(sub) > 2:
                                nomes.add(sub)
                                canonicos.setdefault(sub, set()).add(autor_raw)
                    if "_" in arquivo:
                        primeiro = arquivo.split("_", 1)[0]
                        primeiro_norm = _normalizar_texto(primeiro).strip()
                        if primeiro_norm and len(primeiro_norm) > 2:
                            nomes.add(primeiro_norm)
                            for sub in primeiro_norm.split():
                                if len(sub) > 2:
                                    nomes.add(sub)
                                    if autor_raw:
                                        canonicos.setdefault(sub, set()).add(autor_raw)
                if len(metas) < lote:
                    break
                offset += lote
        except Exception as exc:
            _logger.warning("não foi possível enumerar autores no ChromaDB: %s", exc)
    _AUTORES_CACHE = nomes
    _AUTOR_CANONICO_CACHE = canonicos
    _AUTOR_ARQUIVOS_CACHE = autor_arquivos
    return nomes


def arquivos_do_autor(autor_canonico: str, colecao=None) -> set[str]:
    """Retorna o conjunto de arquivos (filenames) atribuidos a um autor canonico."""
    if not _AUTOR_ARQUIVOS_CACHE:
        autores_indexados(colecao)
    return _AUTOR_ARQUIVOS_CACHE.get(autor_canonico, set())


def autores_canonicos_para(token: str, colecao=None) -> set[str]:
    """
    Dado um token (sobrenome lowercase, ex.: 'puc', 'grewal'), retorna o
    conjunto de formas canonicas do metadado autor que cobrem esse token —
    ex.: 'puc' → {'Puc Rio'}, 'grewal' → {'Grewal'}.
    """
    if not _AUTOR_CANONICO_CACHE:
        autores_indexados(colecao)  # popula cache
    return _AUTOR_CANONICO_CACHE.get(_normalizar_texto(token).strip(), set())


def deve_consultar_literatura(pergunta: str, colecao=None) -> bool:
    """
    Mantem a biblioteca disponível para toda pergunta não social do projeto.

    A decisão deixa de depender de listas de palavras-chave ou sobrenomes. O
    ranking híbrido seleciona os trechos pertinentes e o Evidence Guard impede
    citações sem lastro. O pesquisador ainda pode proibir explicitamente o uso
    de literatura em uma pergunta.
    """
    return bool(str(pergunta or "").strip()) and not pedido_sem_literatura(pergunta)


def _espera_retry_429(erro: str, tentativa: int) -> int:
    """
    Tempo de espera para retry após HTTP 429: respeita o 'retry in N' do
    provedor quando presente; senão, backoff exponencial. Sempre soma um
    JITTER aleatório (0-5 s) para evitar que múltiplos clientes re-tentem em
    sincronia (thundering herd). Teto de 120 s.
    """
    import random
    import re as _re

    match = _re.search(r"retry in (\d+)", erro or "")
    base = int(match.group(1)) + 5 if match else min(120, 2 ** (tentativa + 3))
    return min(120, base + random.randint(0, 5))


def formatar_referencias_markdown(citacoes: dict | list | tuple | set) -> str:
    """Formata referencias como lista Markdown, deduplicando e ignorando vazios."""
    valores = citacoes.values() if isinstance(citacoes, dict) else (citacoes or [])
    vistos = []
    for valor in valores:
        if not valor:
            continue
        item = str(valor).strip()
        if item and item not in vistos:
            vistos.append(item)
    return "\n".join(f"- {item}" for item in vistos)


def _formatar_intervalo_paginas(paginas) -> str:
    """
    Comprime um conjunto de números de página em uma string compacta de
    intervalos: {3} → "3"; {3,4,5,8} → "3–5, 8". Usa travessão (en dash) para
    intervalos. Ignora valores nulos/zero/negativos. Retorna "" se vazio.
    """
    try:
        nums = sorted({int(p) for p in paginas if p not in (None, "") and int(p) > 0})
    except (TypeError, ValueError):
        return ""
    if not nums:
        return ""
    grupos: list[tuple[int, int]] = []
    ini = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            grupos.append((ini, prev))
            ini = prev = n
    grupos.append((ini, prev))
    partes = [str(a) if a == b else f"{a}–{b}" for a, b in grupos]
    return ", ".join(partes)


def _paginas_do_intervalo(pagina_inicio, pagina_fim=None) -> list[int]:
    """Normaliza metadados de pagina em uma lista inclusiva e ordenada."""
    try:
        inicio = int(pagina_inicio)
    except (TypeError, ValueError):
        return []
    if inicio <= 0:
        return []
    try:
        fim = int(pagina_fim) if pagina_fim not in (None, "") else inicio
    except (TypeError, ValueError):
        fim = inicio
    if fim < inicio:
        fim = inicio
    return list(range(inicio, fim + 1))


def _rotulo_paginas_meta(meta: dict) -> str:
    """Formata a pagina fisica e, quando houver, o rotulo interno do PDF."""
    paginas = _paginas_do_intervalo(
        meta.get("pagina_inicio"),
        meta.get("pagina_fim"),
    )
    intervalo = _formatar_intervalo_paginas(paginas)
    if not intervalo:
        return ""

    rotulo_pdf = str(meta.get("pagina_rotulo") or "").strip()
    rotulos_fisicos = {str(p) for p in paginas}
    if rotulo_pdf and rotulo_pdf not in rotulos_fisicos and rotulo_pdf != intervalo:
        return f"p. {intervalo} (rotulo PDF: {rotulo_pdf})"
    return f"p. {intervalo}"


def _limpar_trecho_citacao(texto: str, limite: int = 280) -> str:
    """Normaliza e encurta excertos exibidos na lista de fontes."""
    texto = re.sub(r"\s+", " ", str(texto or "")).strip()
    texto = texto.replace('"', "'")
    if not texto:
        return ""
    limite = max(120, int(limite))
    if len(texto) <= limite:
        return texto
    corte = max(
        texto.rfind(".", 0, limite),
        texto.rfind(";", 0, limite),
        texto.rfind(":", 0, limite),
        texto.rfind("?", 0, limite),
        texto.rfind("!", 0, limite),
    )
    if corte < int(limite * 0.55):
        corte = texto.rfind(" ", 0, limite)
    if corte < int(limite * 0.55):
        corte = limite
    return texto[:corte].strip().rstrip(",;:") + "..."


def _trecho_relevante(doc: str, pergunta: str, meta: dict | None = None, limite: int = 280) -> str:
    """
    Seleciona um excerto curto do chunk que tenha maior sobreposicao lexical
    com a pergunta. Fallback: trecho auditavel gravado no indice ou inicio do
    chunk. A lista final de fontes usa esse excerto para auditoria rapida.
    """
    meta = meta or {}
    texto = re.sub(r"\s+", " ", str(doc or "")).strip()
    if not texto:
        return _limpar_trecho_citacao(meta.get("trecho", ""), limite)

    termos = [t for t in _tokens_busca(pergunta or "") if len(t) >= 4]
    termos = list(dict.fromkeys(termos))[:24]

    melhor = ""
    melhor_score = 0
    sentencas = re.split(r"(?<=[.!?])\s+", texto)
    for sentenca in sentencas:
        sentenca = sentenca.strip()
        if len(sentenca) < 40:
            continue
        norm = _normalizar_texto(sentenca)
        score = sum(1 for termo in termos if termo in norm)
        if score > melhor_score:
            melhor = sentenca
            melhor_score = score

    if melhor_score > 0:
        return _limpar_trecho_citacao(melhor, limite)

    trecho_meta = meta.get("trecho")
    if trecho_meta:
        return _limpar_trecho_citacao(trecho_meta, limite)
    return _limpar_trecho_citacao(texto, limite)


def _chave_citacao(meta: dict, doc: str) -> str:
    """Identidade estavel da fonte usada: arquivo + pagina + hash do chunk."""
    arquivo = str(meta.get("arquivo") or meta.get("citacao") or "fonte")
    p_ini = str(meta.get("pagina_inicio") or "")
    p_fim = str(meta.get("pagina_fim") or p_ini)
    sha = str(meta.get("chunk_sha256") or meta.get("chunk_sha1") or "").strip()
    if not sha:
        sha = hashlib.sha256(
            str(doc or "").encode("utf-8", errors="ignore")
        ).hexdigest()
    return f"{arquivo}|{p_ini}|{p_fim}|{sha[:16]}"


def _entrada_citacao(meta: dict, doc: str, pergunta: str) -> str:
    """Monta a fonte final com pagina e trecho exatamente do chunk recuperado."""
    arquivo = str(meta.get("arquivo") or "").strip()
    base = str(meta.get("citacao") or arquivo or "Fonte sem identificacao").strip()
    pagina = _rotulo_paginas_meta(meta)
    trecho = _trecho_relevante(doc, pergunta, meta)

    partes = [base]
    if pagina:
        partes.append(pagina)
    if trecho:
        partes.append(f'trecho: "{trecho}"')
    return " — ".join(partes)


def remover_bloco_fontes_llm(texto: str) -> str:
    """
    Remove qualquer secao terminal de 'Referencias', 'Bibliografia',
    '📚 Fontes' etc. que o LLM tenha gerado por conta propria. Evita o
    duplo bloco quando a interface anexa a lista oficial de citacoes.

    Heuristica anti-falso-positivo: so corta se o cabecalho for SEGUIDO
    por uma lista (linha comecando com '-', '*', '1.', etc.) ou pelo
    fim do texto. Assim, uma menção em prosa do tipo
    "📚 Fontes do paragrafo anterior estavam ok." nao e cortada.

    Detecta cabecalhos como:
      - "## Referencias", "### Referencias"
      - "**Referencias:**", "**Referências bibliográficas**"
      - "📚 Fontes:", "📚 **Fontes consultadas:**"
      - "Referencias:" no inicio de linha
    Apaga do cabecalho ate o final do texto (e separadores '---' que o
    LLM as vezes coloca logo antes).
    """
    if not texto:
        return texto

    import re

    # Palavra-chave do cabecalho — fontes/referencias/bibliografia (com
    # qualificadores opcionais como 'consultadas' ou 'bibliograficas').
    _palavra = (
        r"(?:refer[eê]ncias?(?:\s+bibliogr[áa]ficas?)?"
        r"|bibliografia"
        r"|fontes?(?:\s+consultadas?)?)"
    )
    padroes_cabecalho = [
        # Headers Markdown (##, ###, etc.)
        rf"(?im)^\s*#{{1,6}}\s*{_palavra}\b[^\n]*$",
        # Negrito/italico com colon em qualquer lado: **Refs:**, **Refs**:, **Refs**
        rf"(?im)^\s*\*+\s*{_palavra}\s*:?\s*\*+\s*:?\s*$",
        # Plain text com colon obrigatorio: REFERÊNCIAS:, Bibliografia:
        rf"(?im)^\s*{_palavra}\s*:\s*$",
        # 📚 — regex generoso; _eh_bloco_real filtra os falsos positivos
        # (linhas em prosa que apenas mencionam 'Fontes' sem lista logo abaixo).
        rf"(?im)^\s*📚[^\n]*{_palavra}[^\n]*$",
    ]

    def _eh_bloco_real(start: int, end: int) -> bool:
        """Confirma que o cabecalho e seguido por lista ou fim de texto."""
        apos = texto[end:].lstrip("\n").lstrip(" \t")
        if not apos:
            return True  # fim de texto — header solto conta como bloco
        primeira = apos.split("\n", 1)[0].strip()
        if not primeira:
            return True
        return (
            primeira.startswith(("-", "*", "•"))
            or bool(re.match(r"^\d+[.)]\s", primeira))
        )

    indice_min = len(texto)
    achou = False
    for padrao in padroes_cabecalho:
        for m in re.finditer(padrao, texto):
            if not _eh_bloco_real(m.start(), m.end()):
                continue
            if m.start() < indice_min:
                indice_min = m.start()
                achou = True

    if not achou:
        return texto.rstrip()

    recortado = texto[:indice_min]
    # Engole separadores '---' e linhas em branco logo antes do bloco.
    recortado = re.sub(r"(?:\s*\n\s*-{3,}\s*\n)+\s*$", "\n", recortado)
    return recortado.rstrip()


def _formatar_historico(historico: list, orcamento: dict) -> str:
    if not historico:
        return ""

    linhas = ["\nHISTORICO RECENTE DA CONVERSA:"]
    turnos = historico[-orcamento["historico_turnos"]:]
    for turno in turnos:
        role = "Rodolfo" if turno.get("role") == "user" else "Al IAdo PV"
        content = _limitar_texto(
            str(turno.get("content", "")),
            orcamento["historico_chars"],
        )
        linhas.append(f"\n{role}:\n{content}")
    return "\n".join(linhas)


def _contexto_temporal() -> str:
    """Gera bloco com data, hora e dia da semana atuais."""
    dias = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
            "sexta-feira", "sábado", "domingo"]
    meses = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
             "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
    agora = agora_local()
    saudacao = _saudacao_pelo_horario()
    return (
        f"DATA E HORA ATUAL: {dias[agora.weekday()]}, {agora.day} de "
        f"{meses[agora.month - 1]} de {agora.year}, às {agora.strftime('%H:%M')} "
        f"(fuso {agora.tzname() or FUSO_PADRAO}). "
        f"Período do dia: {saudacao.lower()}."
    )


def _bloco_anexos(anexos_texto: str, orcamento: dict) -> str:
    """
    Monta o bloco de ARQUIVOS ANEXADOS para o prompt. Vazio quando nao ha
    anexos de texto. O conteudo ja vem consolidado por
    `montar_bloco_texto_anexos`; aqui so aplicamos o cap de chars do provedor
    e envolvemos com cabecalho + instrucao de uso.
    """
    if not anexos_texto or not anexos_texto.strip():
        return ""
    corpo = _limitar_texto(anexos_texto, orcamento.get("anexos_chars", 9_000))
    return (
        "ARQUIVOS ANEXADOS PELO PESQUISADOR (leia e use quando pertinente):\n"
        f"{corpo}\n"
        "Os arquivos acima foram enviados agora pelo Rodolfo nesta mensagem. "
        "Leia, interprete e use o conteudo quando for pertinente a pergunta. "
        "Trate-os como fonte prioritaria desta resposta; nao invente nada alem "
        "do que o anexo traz. Se a pergunta for sobre o anexo, responda a partir dele."
    )
