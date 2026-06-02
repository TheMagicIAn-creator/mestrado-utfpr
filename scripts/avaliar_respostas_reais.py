"""
avaliar_respostas_reais.py — Al IAdo PV

Harness de "amostra real": gera a resposta de verdade do agente (RAG real +
Groq) para um conjunto curado de perguntas e a avalia programaticamente
(barato, determinístico) com correção/retry. Complementa a bateria
determinística `avaliar_agente_100.py` (que valida políticas/funções puras sem
gastar cota de API).

Para cada pergunta:
  1. `preparar_prompt(...)` monta o prompt com RAG real (ChromaDB + embeddings).
  2. `llm.invoke(...)` no Groq, com backoff para o limite de 12k tokens/min.
  3. Avaliação programática da resposta:
       - não escreve um bloco final de Referências/Bibliografia/📚 Fontes
         (`remover_bloco_fontes_llm` deve ser no-op — o rodapé é do sistema);
       - quando a pergunta pede literatura, cita autor/ano;
       - contém os termos-chave esperados (ex.: Weibull → MTTF/B10);
       - responde em português, com tamanho são;
       - nunca afirma que algo "não está na base".
  4. Correção: se falhar, 1 retry com instrução reforçada e reavaliação.
  5. (Opcional, flag --juiz) LLM-juiz: nota 0–5 + justificativa (1 chamada Groq).

Gera relatório em notas/sessoes/ e grava 1 memória por pergunta no ChromaDB
(coleção de sessões), como a bateria determinística.

Uso:
  python scripts/avaliar_respostas_reais.py                 # todas, com retry
  python scripts/avaliar_respostas_reais.py --limite 6      # smoke rápido
  python scripts/avaliar_respostas_reais.py --juiz          # + LLM-juiz
  python scripts/avaliar_respostas_reais.py --sem-retry     # sem correção
  python scripts/avaliar_respostas_reais.py --sem-memoria   # não grava ChromaDB

Sem GROQ_API_KEY no .env, o script encerra com aviso (não falha o CI).

Autor: Rodolfo Torres (UTFPR)
"""
from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
import sys
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage

from src.conhecimento.agente import (
    carregar_perfil,
    deve_consultar_literatura,
    formatar_referencias_markdown,
    preparar_prompt,
    remover_bloco_fontes_llm,
)
from src.conhecimento.indexador import upsert_em_lotes
from src.conhecimento.provedores import inicializar_provedor
from src.core.config import (
    MODELO_EMBEDDINGS,
    NOME_COLECAO,
    NOME_COLECAO_AVALIACOES,
    NOME_COLECAO_SESSOES,
    PASTA_CHROMADB,
    PASTA_NOTAS,
)

# Provedor real: Groq (texto puro, 12k tokens/min). Chave "2" em provedores.py.
ESCOLHA_GROQ = "2"
NOME_PROVEDOR_GROQ = "Groq (LLaMA 3.3)"

# Limites de sanidade da resposta.
MIN_CHARS_RESPOSTA = 80
MAX_CHARS_RESPOSTA = 14_000


class QuotaExcedida(RuntimeError):
    """Cota diaria/por-minuto do Groq esgotada (HTTP 429). NAO e falha de
    qualidade da resposta — e limite externo de API. Tratada a parte para
    nao contaminar o veredito da bateria."""

# ── RAG carregado uma única vez (um SentenceTransformer; dois modelos torch
#    vivos ao mesmo tempo provocam access violation no Windows) ──────────────
_RAG: dict = {}


def _carregar_rag() -> tuple:
    if "modelo" not in _RAG:
        _RAG["modelo"] = SentenceTransformer(MODELO_EMBEDDINGS)
        client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        _RAG["colecao"] = client.get_or_create_collection(name=NOME_COLECAO)
        _RAG["colecao_sessoes"] = client.get_or_create_collection(
            name=NOME_COLECAO_SESSOES
        )
    return _RAG["modelo"], _RAG["colecao"], _RAG["colecao_sessoes"]


# ============================================================
# CONJUNTO CURADO DE PERGUNTAS
# ============================================================
# Cada item: (nome, pergunta, termos). `termos` é uma lista de grupos; cada
# grupo é uma tupla de sinônimos — a resposta passa o grupo se contiver QUALQUER
# sinônimo (comparação sem acento/caixa). A política de literatura (citar ou
# não) é derivada em tempo de execução de `deve_consultar_literatura`.

PERGUNTAS: list[tuple[str, str, list[tuple[str, ...]]]] = [
    # ── Técnicas sem pedido de literatura ──────────────────────────────────
    ("ae_anomalia",
     "Explique como o autoencoder detecta anomalias no lado CA do inversor.",
     [("reconstru",), ("erro", "limiar", "anomalia")]),
    ("weibull_rul",
     "Como estimo a vida util remanescente com analise de Weibull?",
     [("weibull",), ("mttf", "b10", "vida util", "rul")]),
    ("npr_fmea",
     "O que significa o NPR no FMEA do nosso projeto?",
     [("npr",), ("severidade", "ocorrencia", "deteccao")]),
    ("limiar_p99",
     "Por que escolhemos o limiar p99 para o erro de reconstrucao?",
     [("p99", "percentil", "3 sigma", "3σ", "sigma"), ("limiar",)]),
    ("injecao_falhas",
     "Como funciona a injecao de falhas sinteticas baseada no FMEA?",
     [("sintetic",), ("fmea", "falha")]),
    ("roc_auc",
     "Como interpreto a curva ROC e o AUC do detector de anomalias?",
     [("auc",), ("roc", "verdadeiro positivo", "taxa")]),
    ("paderborn_uso",
     "Para que serve o dataset de Paderborn no projeto?",
     [("paderborn", "inversor"), ("saudavel", "normal", "normalidade")]),
    ("isolation_forest",
     "Qual a intuicao do Isolation Forest para deteccao de anomalias?",
     [("isolation", "isolamento", "arvore", "isola")]),
    ("thd_ca",
     "O que e THD e por que ela importa no lado CA do inversor?",
     [("thd", "distorcao harmonica", "harmonic")]),
    ("rcm_metodo",
     "Como a metodologia RCM orienta a dissertacao?",
     [("confiabilidade", "rcm", "manutencao centrada")]),
    ("baseline_saudavel",
     "Por que modelar o comportamento saudavel em vez de aprender as falhas?",
     [("saudavel", "normal"), ("desvio", "anomalia", "raro")]),
    ("features_ca",
     "Quais features extraimos dos sinais CA do inversor?",
     [("fft", "rms", "thd", "espectr", "frequencia", "estatistic")]),
    ("smd",
     "O que e a severidade minima detectavel (SMD) no nosso pipeline?",
     [("severidade",), ("detect",)]),
    ("matriz_confusao",
     "Como leio a matriz de confusao do classificador de falhas?",
     [("confusao",), ("recall", "precis", "verdadeiro", "falso positivo")]),
    ("desbalanceamento",
     "Como tratamos o desbalanceamento de classes na deteccao?",
     [("desbalance", "classe", "minoritar", "raro")]),
    ("proximo_passo",
     "Qual o proximo passo do pipeline de ML da dissertacao?",
     [("rul", "weibull", "vida util", "integr")]),

    # ── Pedem literatura explicitamente (esperam citacao autor/ano) ────────
    ("lit_anomalia",
     "Cite artigos sobre deteccao de anomalias em inversores fotovoltaicos.",
     [("anomalia", "inversor")]),
    ("lit_falhas_ca",
     "Segundo a literatura, o que se sabe sobre falhas no lado CA do inversor?",
     [("ca", "falha", "inversor")]),
    ("lit_manut_preditiva",
     "Quais autores tratam de manutencao preditiva em sistemas fotovoltaicos?",
     [("manutencao", "preditiv")]),
    ("lit_weibull",
     "Com base na literatura, descreva o uso de Weibull em confiabilidade.",
     [("weibull",), ("confiabilidade", "vida", "falha")]),
    ("lit_autoencoder",
     "Liste referencias sobre autoencoders para deteccao de anomalias.",
     [("autoencoder", "reconstru", "anomalia")]),
    ("lit_fmea_pv",
     "O que a bibliografia diz sobre FMEA em sistemas fotovoltaicos?",
     [("fmea",), ("fotovoltaic", "pv", "sistema")]),
    ("lit_rul_eletronica",
     "Faca uma revisao bibliografica sobre RUL em eletronica de potencia.",
     [("rul", "vida util"), ("eletronica de potencia", "inversor", "igbt", "potencia")]),
    ("lit_tcc",
     "Cite o TCC do Rodolfo e o que ele concluiu sobre o inversor.",
     [("inversor",), ("npr", "210", "critico", "ceamazon")]),
    ("lit_injecao",
     "Quais referencias embasam a injecao de falhas sinteticas?",
     [("falha", "sintetic", "fmea")]),
    ("lit_estado_arte",
     "Levante o estado da arte de Machine Learning para falhas em inversores.",
     [("machine learning", "aprendizado", "ml", "modelo"), ("inversor", "falha")]),

    # ── Proveniência (autor/fonte específicos) ─────────────────────────────
    ("prov_stender",
     "O que o Stender diz sobre o dataset de Paderborn?",
     [("paderborn", "inversor", "saudavel", "dados")]),
    ("prov_torres",
     "Resuma as conclusoes do TCC de Torres (2024) sobre o sistema do CEAMAZON.",
     [("npr", "210", "inversor", "ceamazon", "critico")]),
    ("prov_nasa",
     "O que a literatura da NASA documenta sobre prognostico e RUL?",
     [("prognostic", "rul", "vida util", "degrad")]),
    ("prov_golnas",
     "Segundo Golnas, qual a contribuicao do inversor para as falhas em SFVs?",
     [("inversor",), ("43", "36", "falha", "perda")]),

    # ── Panorama / borda ───────────────────────────────────────────────────
    ("resumo_projeto",
     "Resuma o projeto da dissertacao em um paragrafo.",
     [("inversor", "fotovoltaic"), ("anomalia", "preditiv", "falha")]),
    ("datasets_projeto",
     "Quais sao os datasets do projeto e para que cada um serve?",
     [("paderborn",), ("pv farms", "train_data", "ghoneim", "classificacao", "cc")]),
    ("fmea_fmeca",
     "Explique a diferenca entre FMEA e FMECA.",
     [("fmea",), ("fmeca", "criticidade", "critic")]),
    ("limiar_mu3sigma",
     "Como o limiar mu+3sigma se relaciona com o p99 do erro de reconstrucao?",
     [("limiar", "sigma", "3σ", "p99", "percentil"), ("reconstru", "erro")]),
    ("inversor_critico",
     "Por que o inversor e o componente mais critico do sistema?",
     [("npr", "210", "critic"), ("inversor",)]),
    ("rul_decisao",
     "O que e RUL e como ele apoia a decisao de manutencao?",
     [("rul", "vida util"), ("manutencao", "decis")]),
]


# ============================================================
# UTILIDADES DE AVALIAÇÃO
# ============================================================

def _norm(texto: str) -> str:
    """Minúsculas + remoção de acentos, para comparação robusta."""
    t = unicodedata.normalize("NFKD", texto or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower()


_PT_MARCADORES = re.compile(
    r"\b(de|que|para|com|nao|uma|dos|das|pelo|pela|sao|esta|isso|"
    r"porque|tambem|sobre|entre|quando|onde|como)\b"
)

_NEGA_BASE = [
    "nao esta na base", "nao consta na base", "nao encontrei na base",
    "nao ha na base", "nao existe na base", "nao foi encontrado na base",
    "nao tenho acesso a base", "not in the database", "nao esta presente na base",
    "nao ha informacoes na base", "nao consta na literatura indexada",
    "nao ha registros na base", "nao foi possivel encontrar na base",
    "nao localizei na base", "base de conhecimento nao contem",
]

_CITACAO = re.compile(r"\b(19|20)\d{2}\b|\bet al\b|\bapud\b", re.IGNORECASE)


def _tem_termos(resp_norm: str, termos: list[tuple[str, ...]]) -> tuple[bool, str]:
    faltando = []
    for grupo in termos:
        if not any(_norm(s) in resp_norm for s in grupo):
            faltando.append("/".join(grupo))
    if faltando:
        return False, "faltam termos: " + "; ".join(faltando)
    return True, ""


def avaliar(resposta: str, termos: list[tuple[str, ...]],
            consultar: bool) -> list[tuple[str, bool, str]]:
    """Roda os checks programáticos. Retorna lista (nome, ok, detalhe)."""
    resp = resposta or ""
    resp_norm = _norm(resp)
    checks: list[tuple[str, bool, str]] = []

    # 1. tamanho são
    n = len(resp.strip())
    checks.append((
        "tamanho",
        MIN_CHARS_RESPOSTA <= n <= MAX_CHARS_RESPOSTA,
        f"{n} chars",
    ))

    # 2. português (heurística por marcadores)
    marcadores = len(set(m.group(0) for m in _PT_MARCADORES.finditer(resp_norm)))
    checks.append((
        "pt_br",
        marcadores >= 3,
        f"{marcadores} marcadores PT",
    ))

    # 3. sem bloco final de fontes do próprio LLM (rodapé é do sistema)
    sem_bloco = remover_bloco_fontes_llm(resp) == resp
    checks.append((
        "sem_bloco_fontes_llm",
        sem_bloco,
        "ok" if sem_bloco else "LLM escreveu um bloco final de Referencias/Fontes",
    ))

    # 4. nunca nega a base
    nega = next((p for p in _NEGA_BASE if p in resp_norm), "")
    checks.append((
        "nao_nega_base",
        not nega,
        "ok" if not nega else f"negou a base: '{nega}'",
    ))

    # 5. termos-chave do tópico
    ok_termos, det_termos = _tem_termos(resp_norm, termos)
    checks.append(("termos_chave", ok_termos, det_termos or "ok"))

    # 6. citação quando a pergunta pede literatura
    if consultar:
        tem_cit = bool(_CITACAO.search(resp))
        checks.append((
            "cita_autor_ano",
            tem_cit,
            "ok" if tem_cit else "pediu literatura mas nao citou autor/ano",
        ))

    return checks


def _instrucao_reforco(checks: list[tuple[str, bool, str]], pergunta: str) -> str:
    """Monta um adendo de correção a partir dos checks que falharam."""
    falhou = {nome for nome, ok, _ in checks if not ok}
    linhas = ["", "", "INSTRUCOES DE CORRECAO (siga rigorosamente):"]
    if "sem_bloco_fontes_llm" in falhou:
        linhas.append(
            "- NAO escreva uma secao final de Referencias/Bibliografia/Fontes. "
            "As fontes sao anexadas automaticamente pelo sistema."
        )
    if "nao_nega_base" in falhou:
        linhas.append(
            "- NAO afirme que algo nao esta na base; use o contexto fornecido e "
            "responda com o conhecimento do projeto."
        )
    if "cita_autor_ano" in falhou:
        linhas.append(
            "- Cite explicitamente autor e ano das fontes fornecidas no contexto."
        )
    if "termos_chave" in falhou or "tamanho" in falhou or "pt_br" in falhou:
        linhas.append(
            f"- Responda em portugues, de forma tecnica e completa, abordando "
            f"diretamente a pergunta: {pergunta}"
        )
    return "\n".join(linhas)


# ============================================================
# CHAMADA AO LLM (com backoff de rate limit)
# ============================================================

def invocar_groq(llm, prompt: str, max_tentativas: int = 4) -> str:
    mensagens = [HumanMessage(content=prompt)]
    for tentativa in range(1, max_tentativas + 1):
        try:
            resposta = llm.invoke(mensagens)
            return resposta.content or ""
        except Exception as exc:  # noqa: BLE001
            erro = str(exc)
            if "429" in erro and tentativa < max_tentativas:
                m = re.search(r"retry in (\d+)", erro)
                espera = (int(m.group(1)) + 5) if m else 30
                print(f"    ⏳ 429 — aguardando {espera}s "
                      f"({tentativa}/{max_tentativas - 1})...")
                time.sleep(espera)
            elif ("413" in erro or "Request too large" in erro):
                raise RuntimeError(f"prompt grande demais: {erro}") from exc
            elif tentativa < max_tentativas:
                print(f"    ⚠️ erro transitorio ({erro[:80]}); retry...")
                time.sleep(8)
            elif "429" in erro:
                # Esgotou as tentativas ainda em 429: cota da API estourou.
                raise QuotaExcedida(erro) from exc
            else:
                raise
    return ""


def julgar(llm, pergunta: str, resposta: str) -> tuple[int | None, str]:
    """LLM-juiz opcional: nota 0–5 + justificativa (1 chamada Groq)."""
    rubrica = (
        "Voce e um avaliador rigoroso de um assistente de pesquisa de mestrado "
        "em engenharia eletrica (deteccao preditiva de falhas em inversores "
        "fotovoltaicos). Avalie a RESPOSTA a PERGUNTA quanto a: correcao tecnica, "
        "aderencia ao tema, clareza, e portugues. Responda em UMA linha no formato "
        "exato 'NOTA: X | <justificativa curta>' com X inteiro de 0 a 5.\n\n"
        f"PERGUNTA: {pergunta}\n\nRESPOSTA:\n{resposta[:4000]}"
    )
    try:
        saida = invocar_groq(llm, rubrica, max_tentativas=3)
    except Exception as exc:  # noqa: BLE001
        return None, f"juiz indisponivel: {exc}"
    m = re.search(r"nota[:\s]*([0-5])", saida, re.IGNORECASE)
    nota = int(m.group(1)) if m else None
    just = saida.strip().replace("\n", " ")[:300]
    return nota, just


# ============================================================
# PERSISTÊNCIA (memória + relatório) — espelha avaliar_agente_100.py
# ============================================================

def gravar_memoria(resultados: list[dict], timestamp: str) -> int:
    modelo, _, _ = _carregar_rag()
    client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
    # Memória de AVALIAÇÃO, separada da memória de produção (sessoes_pv).
    colecao = client.get_or_create_collection(name=NOME_COLECAO_AVALIACOES)

    ids, documentos, metadados = [], [], []
    for item in resultados:
        status = "PASSOU" if item["ok"] else "FALHOU"
        falhas = ", ".join(item["falhas"]) or "nenhuma"
        doc = (
            f"# Avaliacao de resposta real - {item['nome']}\n\n"
            f"- Data: {timestamp}\n"
            f"- Pergunta: {item['pergunta']}\n"
            f"- Pediu literatura: {item['consultar']}\n"
            f"- Resultado: {status}\n"
            f"- Corrigido no retry: {item['corrigido']}\n"
            f"- Checks que falharam: {falhas}\n"
            f"- Nota do juiz: {item.get('nota')}\n\n"
            f"Trecho da resposta:\n{item['resposta'][:600]}\n\n"
            "Memoria operacional: este teste valida a RESPOSTA real do agente "
            "(RAG + Groq) — politica de citacao, ausencia de bloco de fontes "
            "duplicado, termos-chave do dominio e proibicao de negar a base."
        )
        ids.append(f"avaliacao_respostas_reais_{timestamp}_{item['indice']:03d}")
        documentos.append(doc)
        metadados.append({
            "tipo": "avaliacao_respostas_reais",
            "data": timestamp,
            "indice": str(item["indice"]),
            "nome": item["nome"],
            "ok": str(item["ok"]),
            "origem": "scripts/avaliar_respostas_reais.py",
        })

    embeddings = modelo.encode(documentos, show_progress_bar=True).tolist()
    upsert_em_lotes(colecao, ids, embeddings, documentos, metadados, tamanho_lote=100)
    verificados = colecao.get(ids=ids)
    return len(verificados.get("ids", []))


def gravar_relatorio(resultados: list[dict], memorias: int, timestamp: str,
                     usou_juiz: bool) -> Path:
    pasta = PASTA_NOTAS / "sessoes"
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{timestamp}_avaliacao_respostas_reais.md"

    total = len(resultados)
    falhas = [r for r in resultados if not r["ok"]]
    corrigidos = [r for r in resultados if r["corrigido"]]

    linhas = [
        f"# Avaliacao de respostas reais (Groq) - {total} perguntas",
        "",
        f"- Data: {timestamp}",
        f"- Provedor: {NOME_PROVEDOR_GROQ}",
        f"- Total: {total}",
        f"- Passaram: {total - len(falhas)}",
        f"- Falharam: {len(falhas)}",
        f"- Corrigidos no retry: {len(corrigidos)}",
        f"- Memorias gravadas: {memorias}",
    ]
    if usou_juiz:
        notas = [r["nota"] for r in resultados if r.get("nota") is not None]
        media = round(sum(notas) / len(notas), 2) if notas else "—"
        linhas.append(f"- Nota media do juiz (0-5): {media} (n={len(notas)})")
    linhas.extend(["", "## Casos", ""])

    for item in resultados:
        status = "PASS" if item["ok"] else "FAIL"
        marca_corr = " (corrigido)" if item["corrigido"] else ""
        linhas.append(f"### {item['indice']:02d}. {item['nome']} — {status}{marca_corr}")
        linhas.append("")
        linhas.append(f"- Pergunta: {item['pergunta']}")
        linhas.append(f"- Pediu literatura: {item['consultar']}")
        if item["falhas"]:
            linhas.append(f"- Checks que falharam: {', '.join(item['falhas'])}")
        if item.get("nota") is not None:
            linhas.append(f"- Juiz: NOTA {item['nota']} — {item.get('juiz_just', '')}")
        detalhes = "; ".join(f"{n}={'ok' if ok else 'FALHOU'}"
                             for n, ok, _ in item["checks"])
        linhas.append(f"- Checks: {detalhes}")
        trecho = item["resposta"][:500].replace("\n", " ")
        linhas.append(f"- Trecho: {trecho}")
        linhas.append("")

    if falhas:
        linhas.extend(["## Falhas a investigar", ""])
        for item in falhas:
            linhas.append(
                f"- {item['indice']:02d} {item['nome']}: {', '.join(item['falhas'])}"
            )
    else:
        linhas.extend([
            "## Parecer", "",
            "Todas as respostas reais passaram nos criterios programaticos. "
            "O agente cita literatura quando solicitado, nao duplica o bloco de "
            "fontes, mantem os termos-chave do dominio e nunca nega a base.",
        ])

    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliacao de respostas reais (Groq).")
    parser.add_argument("--limite", type=int, default=0,
                        help="avalia apenas as N primeiras perguntas (0 = todas).")
    parser.add_argument("--pausa", type=float, default=4.0,
                        help="pausa (s) entre perguntas para respeitar o rate limit.")
    parser.add_argument("--juiz", action="store_true",
                        help="ativa o LLM-juiz (1 chamada Groq extra por pergunta).")
    parser.add_argument("--sem-retry", action="store_true",
                        help="desativa a correcao/retry em caso de falha.")
    parser.add_argument("--com-memoria", action="store_true",
                        help="grava memorias no ChromaDB (NAO e o padrao em avaliacao).")
    parser.add_argument("--sem-memoria", action="store_true",
                        help="(compat.) nao grava memorias — ja e o padrao.")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  GROQ_API_KEY ausente no .env — harness de respostas reais "
              "ignorado (sem custo de API). Configure a chave para rodar.")
        return 0

    print(f"\n🟢 Inicializando {NOME_PROVEDOR_GROQ}...")
    llm, nome_provedor = inicializar_provedor(ESCOLHA_GROQ)

    print("📚 Carregando RAG (embeddings + ChromaDB)...")
    modelo, colecao, colecao_sessoes = _carregar_rag()
    perfil = carregar_perfil()

    perguntas = PERGUNTAS[: args.limite] if args.limite > 0 else PERGUNTAS
    print(f"🧪 Avaliando {len(perguntas)} perguntas reais "
          f"(retry={'off' if args.sem_retry else 'on'}, "
          f"juiz={'on' if args.juiz else 'off'})\n")

    resultados: list[dict] = []
    for indice, (nome, pergunta, termos) in enumerate(perguntas, 1):
        print(f"[{indice:02d}/{len(perguntas)}] {nome}: {pergunta}")
        consultar = bool(deve_consultar_literatura(pergunta, colecao))

        try:
            prompt, citacoes = preparar_prompt(
                pergunta=pergunta,
                perfil=perfil,
                modelo_embeddings=modelo,
                colecao=colecao,
                historico=[],
                colecao_sessoes=colecao_sessoes,
                nome_provedor=nome_provedor,
            )
            resposta = invocar_groq(llm, prompt)
            checks = avaliar(resposta, termos, consultar)
            corrigido = False

            if not all(ok for _, ok, _ in checks) and not args.sem_retry:
                print("    ↻ retry com instrucao reforcada...")
                reforco = _instrucao_reforco(checks, pergunta)
                resposta2 = invocar_groq(llm, prompt + reforco)
                checks2 = avaliar(resposta2, termos, consultar)
                # adota o retry se nao piorou
                if sum(ok for _, ok, _ in checks2) >= sum(ok for _, ok, _ in checks):
                    resposta, checks, corrigido = resposta2, checks2, True

            ok = all(o for _, o, _ in checks)
            falhas = [n for n, o, _ in checks if not o]

            nota, juiz_just = (None, "")
            if args.juiz:
                nota, juiz_just = julgar(llm, pergunta, resposta)

            status = "PASS" if ok else "FAIL"
            extra = f" nota={nota}" if nota is not None else ""
            print(f"    → {status}{extra}"
                  + (f" | falhou: {', '.join(falhas)}" if falhas else ""))
            quota = False

        except QuotaExcedida as exc:
            resposta = f"[quota excedida: {exc}]"
            checks = [("quota", False, "cota diaria/por-minuto do Groq esgotada")]
            ok, falhas, corrigido = False, ["quota"], False
            nota, juiz_just, quota = (None, "", True)
            print("    → SKIP (cota da API esgotada — nao e falha de qualidade)")

        except Exception as exc:  # noqa: BLE001
            resposta = f"[erro: {exc}]"
            checks = [("execucao", False, str(exc))]
            ok, falhas, corrigido = False, ["execucao"], False
            nota, juiz_just, quota = (None, "", False)
            print(f"    → ERRO: {exc}")

        resultados.append({
            "indice": indice, "nome": nome, "pergunta": pergunta,
            "consultar": consultar, "resposta": resposta, "checks": checks,
            "ok": ok, "falhas": falhas, "corrigido": corrigido,
            "nota": nota, "juiz_just": juiz_just, "quota": quota,
        })

        if indice < len(perguntas) and args.pausa > 0:
            time.sleep(args.pausa)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    memorias = 0
    if args.com_memoria:  # padrão: NÃO grava (avaliação não contamina produção)
        print("\n💾 Gravando memorias no ChromaDB...")
        memorias = gravar_memoria(resultados, timestamp)
    relatorio = gravar_relatorio(resultados, memorias, timestamp, args.juiz)

    quota_skips = [r for r in resultados if r.get("quota")]
    falhas_qual = [r for r in resultados if not r["ok"] and not r.get("quota")]
    avaliadas = len(resultados) - len(quota_skips)
    passaram = sum(1 for r in resultados if r["ok"])
    print(f"\n{'='*60}")
    print(f"Perguntas             : {len(resultados)}")
    print(f"Respondidas/avaliadas : {avaliadas}")
    print(f"Passaram              : {passaram}")
    print(f"Falhas de qualidade   : {len(falhas_qual)}")
    print(f"Puladas (cota da API) : {len(quota_skips)}")
    print(f"Corrigidos no retry   : {sum(1 for r in resultados if r['corrigido'])}")
    print(f"Memorias gravadas     : {memorias}")
    print(f"Relatorio             : {relatorio}")
    if quota_skips:
        print("\n⚠️  Algumas perguntas nao foram avaliadas por esgotamento da cota "
              "diaria do Groq (100k tokens/dia no tier gratuito). Rode novamente "
              "amanha, ou use --limite/--pausa para caber no orcamento. Isso NAO "
              "conta como falha de qualidade.")

    # Exit 1 apenas para falhas REAIS de qualidade; cota esgotada nao reprova.
    return 1 if falhas_qual else 0


if __name__ == "__main__":
    raise SystemExit(main())
