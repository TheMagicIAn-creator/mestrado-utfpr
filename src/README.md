# Arquitetura do `src/` — Al IAdo PV

Mapa rápido para não se perder. O pacote tem **4 áreas**. `core/` é a base;
os módulos `conhecimento/ferramentas*` funcionam como adaptadores e podem acionar `ml/`;
`interface/` e `orquestrador.py` compõem os fluxos no topo:

```
                 interface/ + orquestrador.py
                         │
                 conhecimento/ (RAG / agente)
                         │ ferramentas
                         ▼
core/  ◄──────────────── ml/ (pipeline e experimentos)
```

Regra de ouro: **`core/` é a fundação** (todos importam dela; ela não importa
ninguém). `ml/` não depende do agente. A integração RAG→ML fica concentrada em
na família `conhecimento/ferramentas*`; a interface não implementa regra científica.

---

## `core/` — fundação (9 módulos)
| Arquivo | O que faz |
|---|---|
| `citacao_guarda.py` | Detecta citações sem lastro e monta restrições de fontes recuperadas. |
| `config.py` | Ponto único de verdade: caminhos, constantes RAG/ML, env. |
| `conversa_export.py` | Serializa o histórico do chat para exportação e consolidação. |
| `formatacao.py` | Formatação compartilhada de números, métricas e tabelas Markdown. |
| `logs.py` | Logging rotativo sem emoji e adaptador `print` para os scripts de ML. |
| `seguranca.py` | Cibersegurança stdlib-only: máscara de segredos, anti path-traversal, pickle verificado por SHA-256, guarda anti-injeção. |
| `tempo.py` | Relógio com fuso configurável e padrão `America/Sao_Paulo`. |
| `texto.py` | Contratos compartilhados de normalização textual e busca lexical. |
| `utils.py` | Caminhos projeto-relativos, UTF-8 no Windows, parse de `autor_titulo_ano.pdf`. |

## `conhecimento/` — o cérebro do agente / RAG
| Arquivo | O que faz |
|---|---|
| `agente.py` | Fachada compatível e coordenação final das respostas do agente. |
| `agente_interacao.py` | Interação leve, intenção, citações e utilitários de consulta. |
| `agente_recuperacao.py` | Expansão, recuperação híbrida, fusão RRF, diversificação e reranking. |
| `agente_contexto.py` | Busca coordenada, catálogo e preparação final do prompt. |
| `atalhos.py` | Registro único de respostas determinísticas anteriores ao RAG. |
| `embeddings.py` | Seleciona backend de embeddings local ou portátil. |
| `ferramentas.py` | Fachada de tool calling, specs, despacho e execução do pipeline. |
| `intencoes_ferramentas.py` | Detectores determinísticos de pedidos relacionados a ferramentas. |
| `ferramentas_academicas.py` | Adaptadores de literatura, experimentos, datasets e classificação. |
| `roteamento_ferramentas.py` | Decisão, guardas críticas e comentário dos resultados das ferramentas. |
| `indexador.py` | Indexa PDFs e sessões no ChromaDB (chunking por página, dedupe SHA-256). |
| `indice_lexical.py` | Índice lexical BM25 em SQLite FTS5, derivado da literatura. |
| `indice_portatil.py` | Exporta e restaura um snapshot gzip versionável do índice literário. |
| `multiagente.py` | Contratos da equipe: Gemini Pro conversa/sintetiza; Gemini Flash audita evidências e memória. |
| `memoria_persistente.py` | Memória JSON validada, atômica, deduplicada e recuperada por relevância. |
| `nota_cerebro.py` | Valida e grava notas curadas no vault. |
| `obsidian.py` | Leitura, indexação e espelho Markdown da memória validada no vault. |
| `consultas_obsidian.py` | Consultas cronológicas, inventário e busca híbrida no vault. |
| `persistencia_nuvem.py` | Persiste arquivos permitidos no GitHub quando o app roda na nuvem. |
| `processador_pdf.py` | Ingestão de PDF novo: metadados, nome padrão, tema, cópia, indexa, nota Obsidian. |
| `consolidar_memoria.py` | Consolida sessões `.md` em memória via LLM, reindexa e arquiva. |
| `leitor_anexos.py` | Leitura **efêmera** de anexos da conversa (PDF/CSV/XLSX/DOCX/imagem). |
| `web_search.py` | Busca web sem API, com classificação de confiança A–D da fonte. |
| `provedores.py` | Adaptador leve do SDK Gemini e papéis fixos por nível (Pro/Flash/Flash-Lite). |
| `retrieval_metrics.py` | Métricas puras de **recuperação** RAG (Recall@k, MRR, nDCG). |
| `resultados_ml.py` | Adaptador que publica o resumo científico na memória do agente. |
| `snippets.py` | Cofre literal e deduplicado de blocos de código. |
| `vault_links.py` | Cria relações entre sessões e memórias validadas no Obsidian. |
| `index_lock.py` | Lock entre threads e processos para escritas no ChromaDB. |

## `ml/` — pipeline e experimentos
**Pipeline CA principal** (em ordem; cada etapa alimenta a seguinte):
`features_ca` → `autoencoder` → `injecao_falhas` → `validacao` → `rul_weibull`,
coordenadas por `pipeline.py` e rastreadas por `proveniencia.py`.

| Arquivo | O que faz |
|---|---|
| `features_ca.py` | Etapa 1: extrai features de tempo/frequência/inter-fase do Paderborn. |
| `autoencoder.py` | Etapa 2: Autoencoder de normalidade + limiar operacional (p99). |
| `escore_anomalia.py` | Fonte única do MSE e do escore localizado operacional. |
| `graficos_autoencoder.py` | Figuras e resumo de calibração sem importar PyTorch. |
| `diagnostico_escore.py` | Comparação auditável entre MSE histórico e escore localizado. |
| `injecao_falhas.py` | Etapa 3: falhas sintéticas orientadas pela FMECA + SMD. |
| `validacao.py` | Etapa 4: métricas no limiar congelado (ROC/PR/F1/AUC). |
| `rul_weibull.py` | Etapa 5: varredura de magnitude, `a_det`, Weibull 2P censurada, margem condicional. O eixo é fração da assinatura nominal, **não tempo**. |
| `confiabilidade.py` | Funções fechadas da Weibull como DADO: R(t), F(t), f(t), h(t), H(t), B1/B10/mediana, leitura de β com ressalva de IC. |
| `pod_curva.py` | Arcabouço POD (MIL-HDBK-1823A / LS-POD NASA): limites de tolerância, critério de viabilidade do ensaio, gatilhos de deriva de campo e checagem das hipóteses. |
| `relatorio_weibull.py` | Montagem de `weibull_results.json` e `weibull_tabela.csv`. Recebe tudo por parâmetro — não importa `rul_weibull`, para não fechar ciclo. |
| `graficos_rul.py` | Figuras acadêmicas de `a_det`, confiabilidade e margem de magnitude. |
| `pipeline.py` | Coordena estado/execução das etapas e grava manifestos. |
| `proveniencia.py` | Manifestos de proveniência e estado ready/stale/pending. |
| `split_temporal.py` | Split em blocos intercalados com purga: anti-vazamento **e** cobertura de regime. Fonte única do split do pipeline. |
| `dados_avaliacao.py` | Constrói o banco E1 comum usado nas comparações entre detectores. |
| `estatistica.py` | ICs, bootstrap e métricas estatísticas compartilhadas. |
| `estilo_graficos.py` | Estilo acadêmico compartilhado por todas as figuras quantitativas. |
| `exec_etapa_isolada.py` | Executa uma etapa pesada do pipeline em subprocesso. |
| `eda.py` | Análise exploratória do Paderborn (Plotly). |
| `retroalimentacao_fmeca.py` | Consolida detectabilidade e ponto operacional para a FMECA. |
| **Comparação acadêmica vigente** | |
| `macro_comum.py` | Contratos, métricas e saídas compartilhadas do comparativo. |
| `macro_proposto.py` | Avalia o método proposto no protocolo comparável. |
| `macro_ibrahim.py` | Avalia o AE-LSTM temporal inspirado em Ibrahim (2022). |
| `macro_comparar.py` | Fonte única Proposto × Ibrahim para AUC e SMD. |
| **Harness histórico por artigo** | |
| `experimentos_artigos.py` | Registry do comparativo ativo Ibrahim/AE-LSTM, métricas e runner de anomalia. |
| `graficos_experimentos.py` | Figuras e comparações visuais do harness por artigo. |
| `protocolos_artigos.py` | Protocolo de decisão do Ibrahim/AE-LSTM + injeção FMECA no espaço de features. |
| `modelos_anomalia.py` | **Módulo folha**: scorer de anomalia não-supervisionado (AE-LSTM). Existe para quebrar o ciclo `experimentos`↔`protocolos`. |
| `exec_experimento_isolado.py` | Roda experimento em subprocesso isolado (crash de lib pesada não derruba o app). |
| **Classificação PV (CC)** | |
| `classificador_pv.py` | Pipeline CLI de classificação PV Farms (Ghoneim) + carregamento de dados. |
| `classificador_pv_infer.py` | Persistência/inferência do classificador (pickle verificado SHA-256). |
| `resultados.py` | Lê e resume artefatos JSON/CSV/PNG sem depender do agente. |

## `interface/` + raiz do pacote
| Arquivo | O que faz |
|---|---|
| `interface/streamlit_app.py` | Ponto de entrada da UI: estado, base, boas-vindas, streaming e chat. |
| `interface/sidebar.py` | Status, diagnóstico e controles laterais. |
| `interface/renderizacao_imagens.py` | Agrupamento, antevisão e download de figuras. |
| `interface/ciclo_chat.py` | Persistência da sessão e caminhos de resposta por ferramenta/RAG. |
| `interface/streamlit_proxy.py` | Referência tardia ao Streamlit atual para testes e hot-reload. |
| `orquestrador.py` | Coordenação leve do backend na init (reprocessamento por sinal + indexação de PDFs novos). |

---

## Dois fluxos para entender o todo

**1. Pergunta no chat** (`streamlit_app` → `agente`/`ferramentas`):
`roteamento_ferramentas.decidir_acao`, reexportado pela fachada, decide se é
caso de **ferramenta** (rodar/consultar ML) ou de **RAG**. Se RAG: o agente
expande a query → combina ChromaDB semântico e
BM25 por RRF → reranking → Gemini Flash audita a cobertura → Gemini Pro responde com
citações por página, memória classificada do Obsidian e memória validada pertinente.

**2. Experimento de ML** (`ferramentas` → `experimentos_artigos`):
roda em subprocesso isolado; o comparativo ativo usa o **protocolo próprio**
do Ibrahim/AE-LSTM (`protocolos_artigos`) com split temporal e injeção FMECA;
os scorers vêm de `modelos_anomalia`. Resultados em
`resultados/experimentos/<key>/`.

> Os entrypoints ficam na **raiz do repo** (não em `src/`): `app.py` (Streamlit),
> `main.py` (chat no terminal) e `watcher.py` (monitora `novos_pdfs/`).
