# Evidence RAG — Refinamento híbrido R4

> Estado: ranking híbrido candidato medido e não promovido; gold set provisório e pendente de revisão do pesquisador em R6.

## Auditoria R4

- Corpus: 44 PDFs e 12556 chunks.
- Snapshot portátil: schema v2; `raw_text` preservado; o índice contextual R3 permanece paralelo e recebe filtros consultivos e expansão de vizinhança no ranking.
- Hash SHA-256 do corpus: `0ef91e96379c546c7bee42434a935e86d3711a3a31500476245274518c6612b0`.
- Hash de texto/embedding do snapshot: `e06190651987aa42966b81dcbb4e4bbb27623ee5d8b52e5baa18559dd4bddf59`.
- Índice semântico: ChromaDB com `paraphrase-multilingual-MiniLM-L12-v2` na revisão `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
- Backend de indexação: sentence-transformers; backend de consulta equivalente: onnxruntime.
- Índice lexical: SQLite FTS5 BM25 (disponível: true).
- Fusão: Reciprocal Rank Fusion com constante 60; reranker `deterministic_local_v2` e diversificação local.
- IDs: SHA-256 documental mais índice ordinal do chunk; páginas preservadas nos metadados.
- Filtros de metadados são aplicados somente quando a pergunta nomeia explicitamente um autor ou fonte; a busca global permanece ativa.
- O caminho atual não possui Evidence Package nem Evidence Guard determinístico.
- R4 usa filtros consultivos por autor explícito e vizinhança de raio 1; nenhum filtro elimina a busca global.

## Gold set R1

- Perguntas: 40 (39 recuperáveis e 1 reservada à futura avaliação de abstenção).
- Categorias: autoencoders=3, comparacao_autores=2, componente=3, conceito=3, confiabilidade=4, fmeca=3, gpvs_faults=2, localizacao_direta=4, metodo=3, multi_hop=3, multilingue=3, rcm=3, revisao_ampla=2, sinonimo=2.
- Todas as evidências provisórias foram validadas contra arquivo, hash, página e chunk do snapshot.
- O conjunto não é verdade final: a promoção R6 permanece bloqueada até revisão humana do pesquisador.

## Métricas de retrieval

| Métrica | k=5 | k=8 |
|---|---:|---:|
| Recall por página@k | 0.3077 | 0.3590 |
| Recall por chunk exato@k | 0.2051 | 0.2564 |
| Recall por documento@k | 0.7265 | 0.7863 |
| Precision@k | 0.0667 | 0.0481 |
| Hit Rate@k | 0.3333 | 0.3846 |
| MRR@k | 0.2423 | 0.2487 |
| nDCG@k | 0.2419 | 0.2581 |

- Latência aquecida média: 801.8 ms; p50=812.1 ms; p95=890.4 ms.
- Contexto médio no maior k: 14182 caracteres.
- Consultas recuperáveis sem acerto no top-5: 26.

A diferença entre Recall documental e Recall por página mostra que o baseline frequentemente localiza a fonte correta, mas não a passagem citável correta. O Recall por chunk exato permanece como controle estrito das fronteiras de segmentação.

## Comparação baseline x candidato

| Métrica | R3 | R4 | Delta |
|---|---:|---:|---:|
| recall@5 | 0.294872 | 0.307692 | +0.012820 |
| recall@8 | 0.346154 | 0.358974 | +0.012820 |
| precision@5 | 0.061538 | 0.066667 | +0.005129 |
| precision@8 | 0.044872 | 0.048077 | +0.003205 |
| hit_rate@5 | 0.307692 | 0.333333 | +0.025641 |
| hit_rate@8 | 0.358974 | 0.384615 | +0.025641 |
| mrr@5 | 0.242308 | 0.242308 | +0.000000 |
| mrr@8 | 0.248718 | 0.248718 | +0.000000 |
| ndcg@5 | 0.241455 | 0.241911 | +0.000456 |
| ndcg@8 | 0.257633 | 0.258089 | +0.000456 |
| strict_chunk_recall@5 | 0.179487 | 0.205128 | +0.025641 |
| strict_chunk_recall@8 | 0.230769 | 0.256410 | +0.025641 |
| document_recall@5 | 0.726496 | 0.726496 | +0.000000 |
| document_recall@8 | 0.786325 | 0.786325 | +0.000000 |
| context_chars_mean_at_max_k | 14199.000000 | 14182.000000 | -17.000000 |

- Identidade do corpus preservada: true.
- Contrato de ranking preservado: false.
- Métricas científicas idênticas: false.
- Ganho de qualidade observado: true.
- Consultas com regressão de Recall@5: nenhuma.
- Regressões críticas em perguntas simples: nenhuma.
- Decisão: candidato não promovido; avaliação continua em R5–R6.
- Latência é informativa e não participa do gate científico de qualidade.

## Diagnóstico por categoria

| Categoria | Perguntas | Recall página@5 | Recall documento@5 | Latência média (ms) |
|---|---:|---:|---:|---:|
| autoencoders | 3 | 0.3333 | 1.0000 | 754.0 |
| comparacao_autores | 2 | 0.5000 | 1.0000 | 933.4 |
| componente | 3 | 0.3333 | 0.6667 | 811.7 |
| conceito | 3 | 0.0000 | 0.0000 | 782.3 |
| confiabilidade | 4 | 0.2500 | 0.5000 | 767.3 |
| fmeca | 3 | 0.3333 | 1.0000 | 798.3 |
| gpvs_faults | 2 | 1.0000 | 1.0000 | 757.6 |
| localizacao_direta | 4 | 0.7500 | 1.0000 | 858.1 |
| metodo | 3 | 0.0000 | 0.3333 | 808.7 |
| multi_hop | 3 | 0.0000 | 0.4444 | 777.7 |
| multilingue | 3 | 0.3333 | 1.0000 | 748.3 |
| rcm | 3 | 0.3333 | 1.0000 | 819.0 |
| revisao_ampla | 2 | 0.0000 | 0.5000 | 865.2 |
| sinonimo | 2 | 0.5000 | 1.0000 | 779.1 |

## Limitações e próximo gate

- Métricas de retrieval não medem fidelidade da resposta gerada.
- O schema v2 registra estratégia e hash de conteúdo, mas tamanho e overlap do snapshot legado permanecem desconhecidos; nenhum valor foi inferido.
- A expansão vigente possui regras temáticas manuais e ainda associa Paderborn a Stender; não existe regra ou fonte bibliográfica direta para os 16 ensaios GPVS-Faults.
- A recuperação não aplica filtro de página e ainda não valida citações após a geração.
- Perguntas de abstenção serão pontuadas apenas após o Evidence Guard (R5).
- O baseline deve permanecer disponível para comparação e rollback durante R2–R6.
- Contextual Retrieval só poderá ser promovido após ganho mensurável e ausência de regressão crítica.

### Consultas sem acerto no top-5

- `direct-tcc-rates-002`: Mostre a tabela do TCC que reúne taxas de falha de componentes fotovoltaicos, incluindo inversor e fusível.
- `concept-reliability-005`: Qual é o conceito de confiabilidade de um sistema fotovoltaico sob condições e período definidos?
- `concept-failure-mode-effect-006`: Qual é a diferença entre modo de falha, falha e efeito da falha?
- `concept-pod-007`: O que significa probabilidade de detecção em ensaios não destrutivos e que cautela metodológica é indicada?
- `method-weibull-estimation-010`: Quais métodos são comparados para estimar os parâmetros de forma e escala da Weibull 2P?
- `method-censoring-weibull-011`: Por que dados suspensos ou censurados devem ser preservados numa análise Weibull?
- `method-maintenance-interval-012`: Como o intervalo ótimo de manutenção é relacionado à RCM e aos parâmetros de confiabilidade?
- `component-inverter-alerts-014`: Quais falhas internas e alertas são observados em inversores fotovoltaicos segundo Monteiro?
- `component-ac-tickets-015`: Que participações de tickets são atribuídas ao Contator AC, IGBT e Fusíveis AC?
- `fmeca-definition-016`: Como o TCC distingue FMEA de FMECA e situa a análise de criticidade?
- `fmeca-sod-npr-017`: Como severidade, ocorrência e detecção compõem o NPR na FMECA?
- `rcm-strategies-019`: Quais estratégias de manutenção são integradas pela abordagem RCM da NASA?
- `rcm-rigorous-fmea-020`: Em que a RCM rigorosa se apoia e que tipo de cálculo ela inclui?
- `reliability-system-composition-023`: Como a confiabilidade de componentes se combina em sistemas série e paralelo?
- `reliability-inverter-mil-024`: Como Shuttleworth calcula taxas de falha de inversores fotovoltaicos com MIL-HDBK-217?
- `reliability-risk-definition-025`: Como uma revisão recente diferencia falha de confiabilidade e risco em sistemas fotovoltaicos?
- `autoencoder-reconstruction-026`: Como o erro de reconstrução é usado por autoencoders na detecção de anomalias?
- `autoencoder-evaluation-028`: Quais métricas e resultados são usados por Ibrahim para avaliar a detecção do AE-LSTM?
- `multilingual-weibull-en-032`: How are the shape and scale parameters of a two-parameter Weibull distribution estimated?
- `multilingual-fmeca-es-033`: ¿Cuál es la diferencia entre FMEA y FMECA en el análisis de criticidad?
- `synonym-unreliability-034`: Como a não confiabilidade, também chamada probabilidade acumulada de falha, se relaciona com R(t)?
- `review-pv-reliability-036`: Faça um panorama da literatura sobre risco e confiabilidade de sistemas e inversores fotovoltaicos.
- `review-anomaly-pv-037`: Revise os trabalhos da base sobre detecção de anomalias e manutenção inteligente em sistemas fotovoltaicos.
- `multihop-fmeca-rcm-038`: Relacione a priorização S/O/D da FMECA com a seleção de tarefas de manutenção na RCM.
- `multihop-weibull-maintenance-039`: Como estimativas Weibull de forma e escala podem apoiar a definição de intervalos de manutenção?
- `multihop-failures-maintenance-040`: Conecte falhas observadas em inversores fotovoltaicos à análise de risco e às estratégias de manutenção inteligente.
