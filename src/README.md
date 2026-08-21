# Arquitetura do `src/` — Al IAdo PV

Mapa rápido para não se perder. O pacote tem **4 áreas**. `core/` é a base;
os módulos `conhecimento/ferramentas*` funcionam como adaptadores e podem acionar `ml/`;
`webapp_v2/` e `orquestrador.py` compõem os fluxos no topo:

```
                 webapp_v2/ + orquestrador.py
                         │
                 conhecimento/ (RAG / agente)
                         │ ferramentas
                         ▼
core/  ◄──────────────── ml/ (pipeline e experimentos)
```

Regra de ouro: **`core/` é a fundação** (todos importam dela; ela não importa
ninguém). `ml/` não depende do agente. A integração RAG→ML fica concentrada em
na família `conhecimento/ferramentas*`; a interface não implementa regra científica.
A V1 Streamlit (`src/interface/`) foi **removida** em 15/08/2026 — ver
`docs/aplicacao_web_v2.md` para as duas capacidades que a V2 ainda não portou.

---

## `core/` — fundação (10 módulos)
| Arquivo | O que faz |
|---|---|
| `citacao_guarda.py` | Detecta citações sem lastro e monta restrições de fontes recuperadas. |
| `config.py` | Ponto único de verdade: caminhos, constantes RAG/ML, env. |
| `conversa_export.py` | Serializa o histórico do chat para exportação e consolidação. |
| `formatacao.py` | Formatação compartilhada de números, métricas e tabelas Markdown. |
| `importacao.py` | Reexportações tardias que preservam fachadas sem ciclos entre módulos extraídos. |
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
**Pipeline GPVS principal** (em ordem; cada etapa alimenta a seguinte):
`features_gpvs` → `autoencoder` → `injecao_falhas` → `validacao` → `rul_weibull`,
coordenadas por `pipeline.py` e rastreadas por `proveniencia.py`.

| Arquivo | O que faz |
|---|---|
| `dados_gpvs.py` | Contrato único dos 16 ensaios, 24 features, split temporal, normalização e holdout GPVS-Faults; não publica resultado autônomo. |
| `modelos_autoencoder.py` | Implementações PyTorch do Denso 24-16-8-16-24 e do AE-LSTM temporal L=8, hidden=32 e latent=8. |
| `treino_comparacao.py` | Treina ambos sob o mesmo orçamento, calibra p99 próprio e persiste pesos/scalers locais ignorados. |
| `estatistica_comparacao.py` | Métricas binárias, IC Wilson e bootstrap no nível do ensaio para o comparativo. |
| `assinaturas_fmeca.py` | Contratos e injeções E2 compartilhadas de Contator AC, IGBT e Fusível AC no eixo adimensional `a_det`. |
| `detectabilidade.py` | Primeiro cruzamento, funções empíricas e Weibull 2P apenas diagnóstico, com censura e teste formal na grade de magnitude. |
| `avaliacao_comparativa.py` | Executa E3 nos 14 ensaios reais e E2 no holdout F0 com janelas, sementes e magnitudes pareadas. |
| `graficos_comparacao.py` | Figuras acadêmicas da comparação E2/E3 em PNG 300 dpi e PDF vetorial. |
| `publicacao_comparacao.py` | Grava dados-fonte, relatório, contrato JSON, figuras e manifesto v2 com hashes. |
| `comparacao_autoencoders.py` | Entrada canônica e orquestrador enxuto da comparação Denso versus AE-LSTM. |
| `gpvs_principal.py` | Etapa 1: contrato canônico de 24 features, split F0 e normalização de baseline GPVS. |
| `features_ca.py` | Extrator Stender preservado para experimentos históricos; não integra o pipeline canônico GPVS. |
| `autoencoder.py` | Etapa 2: Autoencoder de normalidade + limiar operacional (p99). |
| `escore_anomalia.py` | Fonte única do MSE operacional e da ablação localizada. |
| `graficos_autoencoder.py` | Figuras e resumo de calibração sem importar PyTorch. |
| `diagnostico_escore.py` | Comparação auditável entre MSE e escore localizado. |
| `injecao_falhas.py` | Etapa 3: falhas sintéticas orientadas pela FMECA + SMD. |
| `validacao.py` | Etapa 4: métricas no limiar congelado (ROC/PR/F1/AUC). |
| `rul_weibull.py` | Etapa 5: varredura de magnitude, primeiro cruzamento `a_det`, Weibull 2P exploratória e margem residual. É detectabilidade E2, **não RUL**. |
| `rul_weibull_execucao.py` | Orquestra a execução pesada e a regeneração tabular/gráfica da etapa 5. |
| `varredura_a_det.py` | A varredura de magnitude: janela saudável + assinatura crescente → `a_det`. Separada do ajuste porque tem dois consumidores (o AE denso do pipeline e qualquer detector via `scorer`). `rul_weibull` reexporta. |
| `weibull_por_modelo.py` | Detectabilidade E2 (`a_det` → Weibull) para **qualquer** detector, via `scorer`. É o que permite comparar AE denso × AE-LSTM nas curvas, e não só em AUC/SMD. Não reimplementa fórmula: orquestra `rul_weibull` e `confiabilidade`. |
| `confiabilidade.py` | Funções matemáticas da Weibull, posições censura-aware e diagnóstico do papel. O chamador distingue tempo de magnitude. |
| `confiabilidade_fisica_v2.py` | Cenários bibliográficos de taxa constante, conversões dimensionais e funções exponenciais físicas; não estima vida pelo GPVS. |
| `graficos_confiabilidade_fisica_v2.py` | Figuras acadêmicas de confiabilidade, probabilidade, densidade, taxa de falha e marcos B1/B10 dos cenários bibliográficos. |
| `confiabilidade_componentes.py` | Contrato canônico de confiabilidade física: cenários exponenciais rastreáveis, unidades explícitas e separação entre taxas diretas e derivadas. |
| `graficos_confiabilidade.py` | Publica R(t), F(t), f(t), h(t) e as taxas por componente em PNG 300 dpi e PDF vetorial. |
| `pod_curva.py` | Arcabouço POD (MIL-HDBK-1823A / LS-POD NASA): limites de tolerância, critério de viabilidade do ensaio, gatilhos de deriva de campo e checagem das hipóteses. |
| `relatorio_weibull.py` | Montagem de `weibull_results.json` e `weibull_tabela.csv`. Recebe tudo por parâmetro — não importa `rul_weibull`, para não fechar ciclo. |
| `graficos_rul.py` | Figuras acadêmicas de primeiro cruzamento, não detecção, diagnóstico Weibull e margem de magnitude. |
| `pipeline.py` | Coordena estado/execução das etapas e grava manifestos. |
| `proveniencia.py` | Manifestos de proveniência e estado ready/stale/pending. |
| `split_temporal.py` | Split em blocos intercalados com purga: anti-vazamento **e** cobertura de regime. Fonte única do split do pipeline. |
| `dados_avaliacao.py` | Constrói o banco E1 comum usado nas comparações entre detectores. |
| `estatistica.py` | ICs, bootstrap e métricas estatísticas compartilhadas. |
| `estilo_graficos.py` | Estilo acadêmico compartilhado por todas as figuras quantitativas. |
| `exec_etapa_isolada.py` | Executa uma etapa pesada do pipeline em subprocesso. |
| `eda.py` | Análise exploratória do conjunto Stender (Paderborn University; Plotly). |
| `retroalimentacao_fmeca.py` | Consolida detectabilidade e ponto operacional para a FMECA. |
| `gpvs.py` | Adaptador e metadados de referência do GPVS-Faults. |
| `validacao_gpvs_principal.py` | Etapa 4 composta: E2 FMECA e E3 real com um detector canônico e bootstrap por ensaio. |
| **Comparação acadêmica vigente** | |
| `macro_comum.py` | Contratos, métricas e saídas compartilhadas do comparativo. |
| `macro_proposto.py` | Avalia o método proposto no protocolo comparável. |
| `macro_ibrahim.py` | Avalia o AE-LSTM temporal inspirado em Ibrahim (2022). |
| `macro_comparar.py` | Fonte única Proposto × Ibrahim para AUC e SMD. |
| `bracos_modelo.py` | Registro único dos DOIS braços do cerne (AE denso × AE-LSTM): id, rótulo, cor, pasta de resultados e construção do detector. Fonte única da identidade — antes ela vivia em quatro cópias. Cada braço grava em `resultados/modelos/<id>/`; cruzamento vai para `resultados/comparacao/`. |
| `macro_weibull.py` | As quatro curvas (papel de Weibull, `S_D`, `f_D`/`F_D`, `h_D`) **por modelo**, sobre o GPVS. Ponto de entrada separado porque a varredura de magnitude é cara; limiar calibrado por modelo, trajetórias e ruído compartilhados. |
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
| `resultados_weibull.py` | Produz a síntese textual da detectabilidade E2 a partir dos artefatos Weibull versionados. |
| `resultados_gpvs.py` | Formata o resumo E3 do GPVS-Faults sem ampliar a fachada geral de resultados. |

## `webapp_v2/`, `interface/` legada + raiz do pacote
| Arquivo | O que faz |
|---|---|
| `webapp_v2/app.py` | Aplicação ASGI V2, API, arquivos estáticos e cabeçalhos de segurança. |
| `webapp_v2/contracts.py` | Valida apenas resultados V2 e fornece contratos somente leitura. |
| `webapp_v2/agent_adapter.py` | Liga o chat ao mesmo RAG, Gemini e ferramentas, com aquecimento explícito. |
| `webapp_v2/launcher.py` | Entrada canônica e bloqueio da execução acidental via Streamlit. |
| `webapp_v2/rendering.py` | Converte Markdown acadêmico em HTML sem aceitar HTML bruto do modelo. |
| `webapp_v2/scientific_context.py` | Reconcilia respostas do agente com os contratos que alimentam as figuras. |
| `webapp_v2/session_journal.py` | Grava e reindexa sessões V2 sem estado do Streamlit. |
| `base_runtime.py` | Restaura índices, escolhe embeddings e prepara BM25 sem depender da UI. |
| `orquestrador.py` | Coordenação leve do backend na init (reprocessamento por sinal + indexação de PDFs novos). |

---

## Dois fluxos para entender o todo

**1. Pergunta no chat** (`webapp_v2/agent_adapter` → `agente`/`ferramentas`):
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

> Os entrypoints ficam na **raiz do repo** (não em `src/`): `app.py` (ASGI),
> `main.py` (chat no terminal) e `watcher.py` (monitora `novos_pdfs/`).
