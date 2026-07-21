# Arquitetura do `src/` — Al IAdo PV

Mapa rápido para não se perder. O pacote tem **4 camadas** com dependência só
para baixo (nada em `core/` importa coisas de cima):

```
core/  ──►  conhecimento/   (RAG / agente)
       ──►  ml/             (pipeline e experimentos)
                 │
                 ▼
       interface/ + orquestrador.py   (UI Streamlit e init do backend)
```

Regra de ouro: **`core/` é a fundação** (todos importam dela; ela não importa
ninguém). `conhecimento/` e `ml/` são irmãos e quase não se cruzam. A
`interface/` e o `orquestrador.py` ficam no topo e amarram tudo.

---

## `core/` — fundação (4 arquivos)
| Arquivo | O que faz |
|---|---|
| `config.py` | Ponto único de verdade: caminhos, constantes RAG/ML, env. |
| `logs.py` | Logging rotativo sem emoji (terminal legível no Windows). |
| `seguranca.py` | Cibersegurança stdlib-only: máscara de segredos, anti path-traversal, pickle verificado por SHA-256, guarda anti-injeção. |
| `utils.py` | Caminhos projeto-relativos, UTF-8 no Windows, parse de `autor_titulo_ano.pdf`. |

## `conhecimento/` — o cérebro do agente / RAG
| Arquivo | O que faz |
|---|---|
| `agente.py` | **Maior arquivo.** Expansão, recuperação híbrida, fusão RRF, reranking e montagem do prompt. |
| `ferramentas.py` | **Tool calling.** Specs, roteador determinístico (linguagem→ferramenta) e a implementação de cada ferramenta do chat. |
| `indexador.py` | Indexa PDFs e sessões no ChromaDB (chunking por página, dedupe SHA-256). |
| `indice_lexical.py` | Índice lexical BM25 em SQLite FTS5, derivado da literatura. |
| `indice_portatil.py` | Exporta e restaura um snapshot gzip versionável do índice literário. |
| `multiagente.py` | Contratos da equipe: Gemini Pro conversa/sintetiza; Gemini Flash audita evidências e memória. |
| `memoria_persistente.py` | Memória JSON validada, atômica, deduplicada e recuperada por relevância. |
| `obsidian.py` | Indexação do vault completo, busca híbrida histórica e espelho Markdown da memória validada. |
| `processador_pdf.py` | Ingestão de PDF novo: metadados, nome padrão, tema, cópia, indexa, nota Obsidian. |
| `consolidar_memoria.py` | Consolida sessões `.md` em memória via LLM, reindexa e arquiva. |
| `leitor_anexos.py` | Leitura **efêmera** de anexos da conversa (PDF/CSV/XLSX/DOCX/imagem). |
| `web_search.py` | Busca web sem API, com classificação de confiança A–D da fonte. |
| `provedores.py` | Adaptador leve do SDK Gemini e papéis fixos por nível (Pro/Flash/Flash-Lite). |
| `retrieval_metrics.py` | Métricas puras de **recuperação** RAG (Recall@k, MRR, nDCG). |
| `index_lock.py` | Lock in-process que serializa escritas concorrentes no ChromaDB. |

## `ml/` — pipeline e experimentos
**Pipeline CA principal** (em ordem; cada etapa alimenta a seguinte):
`features_ca` → `autoencoder` → `injecao_falhas` → `validacao` → `rul_weibull`,
coordenadas por `pipeline.py` e rastreadas por `proveniencia.py`.

| Arquivo | O que faz |
|---|---|
| `features_ca.py` | Etapa 1: extrai features de tempo/frequência/inter-fase do Paderborn. |
| `autoencoder.py` | Etapa 2: Autoencoder de normalidade + limiar operacional (p99). |
| `injecao_falhas.py` | Etapa 3: falhas sintéticas orientadas pelo FMEA + SMD. |
| `validacao.py` | Etapa 4: métricas no limiar congelado (ROC/PR/F1/AUC). |
| `rul_weibull.py` | Etapa 5: TTF, Weibull 2P, RUL condicional. |
| `pipeline.py` | Coordena estado/execução das etapas e grava manifestos. |
| `proveniencia.py` | Manifestos de proveniência e estado ready/stale/pending. |
| `split_temporal.py` | Split temporal com purga (anti-vazamento), compartilhado. |
| `dados_avaliacao.py` | Constrói o banco E1 comum usado nas comparações entre detectores. |
| `estatistica.py` | ICs, bootstrap e métricas estatísticas compartilhadas. |
| `exec_etapa_isolada.py` | Executa uma etapa pesada do pipeline em subprocesso. |
| `eda.py` | Análise exploratória do Paderborn (Plotly). |
| **Experimentos por artigo** | |
| `experimentos_artigos.py` | Registry de experimentos, métricas, artefatos, runners de classificação e anomalia. |
| `protocolos_artigos.py` | Protocolo de decisão **por artigo** (Francisti/Ibrahim) + injeção FMEA no espaço de features. |
| `modelos_anomalia.py` | **Módulo folha**: scorer de anomalia não-supervisionado (AE-LSTM). Existe para quebrar o ciclo `experimentos`↔`protocolos`. |
| `exec_experimento_isolado.py` | Roda experimento em subprocesso isolado (crash de lib pesada não derruba o app). |
| **Classificação PV (CC)** | |
| `classificador_pv.py` | Pipeline CLI de classificação PV Farms (Ghoneim) + carregamento de dados. |
| `classificador_pv_infer.py` | Persistência/inferência do classificador (pickle verificado SHA-256). |
| `resultados.py` | Lê/resume artefatos JSON/CSV/PNG do pipeline e experimentos para o chat. |

## `interface/` + raiz do pacote
| Arquivo | O que faz |
|---|---|
| `interface/streamlit_app.py` | UI Streamlit completa: estado, sidebar, chat, streaming, render de imagens. |
| `orquestrador.py` | Coordenação leve do backend na init (reprocessamento por sinal + indexação de PDFs novos). |

---

## Dois fluxos para entender o todo

**1. Pergunta no chat** (`streamlit_app` → `agente`/`ferramentas`):
`ferramentas.decidir_acao` decide se é caso de **ferramenta** (rodar/consultar ML)
ou de **RAG**. Se RAG: `agente` expande a query → combina ChromaDB semântico e
BM25 por RRF → reranking → Gemini Flash audita a cobertura → Gemini Pro responde com
citações por página, memória classificada do Obsidian e memória validada pertinente.

**2. Experimento de ML** (`ferramentas` → `experimentos_artigos`):
roda em subprocesso isolado; cada artigo usa seu **protocolo próprio**
(`protocolos_artigos`) com split temporal e injeção FMEA; os scorers vêm de
`modelos_anomalia`. Resultados em `resultados/experimentos/<key>/`.

> Os entrypoints ficam na **raiz do repo** (não em `src/`): `app.py` (Streamlit),
> `main.py` (chat no terminal) e `watcher.py` (monitora `novos_pdfs/`).
