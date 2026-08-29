# Evidence RAG — baseline R0–R1

> Estado: baseline vigente medido; gold set provisório e pendente de revisão do pesquisador em R6.

## Auditoria R0

- Corpus: 44 PDFs e 12556 chunks.
- Snapshot portátil: schema v1; texto bruto no campo `documento` e embeddings armazenados por chunk.
- Hash SHA-256 do corpus: `0ef91e96379c546c7bee42434a935e86d3711a3a31500476245274518c6612b0`.
- Índice semântico: ChromaDB com `paraphrase-multilingual-MiniLM-L12-v2` na revisão `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
- Backend de indexação: sentence-transformers; backend de consulta equivalente: onnxruntime.
- Índice lexical: SQLite FTS5 BM25 (disponível: true).
- Fusão: Reciprocal Rank Fusion com constante 60; reranking e diversificação locais vigentes.
- IDs: SHA-256 documental mais índice ordinal do chunk; páginas preservadas nos metadados.
- O schema v1 ainda não separa `raw_text` de `retrieval_text` e não registra contexto do chunk.
- O caminho atual não possui Evidence Package nem Evidence Guard determinístico.
- Nenhum parâmetro de ranking foi modificado nesta etapa.

## Gold set R1

- Perguntas: 40 (39 recuperáveis e 1 reservada à futura avaliação de abstenção).
- Categorias: autoencoders=3, comparacao_autores=2, componente=3, conceito=3, confiabilidade=4, fmeca=3, gpvs_faults=2, localizacao_direta=4, metodo=3, multi_hop=3, multilingue=3, rcm=3, revisao_ampla=2, sinonimo=2.
- Todas as evidências provisórias foram validadas contra arquivo, hash, página e chunk do snapshot.
- O conjunto não é verdade final: a promoção R6 permanece bloqueada até revisão humana do pesquisador.

## Métricas baseline

| Métrica | k=5 | k=8 |
|---|---:|---:|
| Recall por página@k | 0.2692 | 0.2692 |
| Recall por chunk exato@k | 0.1538 | 0.1795 |
| Recall por documento@k | 0.6752 | 0.7350 |
| Precision@k | 0.0564 | 0.0353 |
| Hit Rate@k | 0.2821 | 0.2821 |
| MRR@k | 0.2295 | 0.2295 |
| nDCG@k | 0.2253 | 0.2253 |

- Latência aquecida média: 989.7 ms; p50=978.7 ms; p95=1193.7 ms.
- Contexto médio no maior k: 12794 caracteres.
- Consultas recuperáveis sem acerto no top-5: 28.

A diferença entre Recall documental e Recall por página mostra que o baseline frequentemente localiza a fonte correta, mas não a passagem citável correta. O Recall por chunk exato permanece como controle estrito das fronteiras de segmentação.

## Diagnóstico por categoria

| Categoria | Perguntas | Recall página@5 | Recall documento@5 | Latência média (ms) |
|---|---:|---:|---:|---:|
| autoencoders | 3 | 0.3333 | 1.0000 | 1054.1 |
| comparacao_autores | 2 | 0.2500 | 1.0000 | 1166.9 |
| componente | 3 | 0.0000 | 0.6667 | 946.3 |
| conceito | 3 | 0.0000 | 0.0000 | 1002.7 |
| confiabilidade | 4 | 0.2500 | 0.5000 | 977.8 |
| fmeca | 3 | 0.3333 | 1.0000 | 1041.6 |
| gpvs_faults | 2 | 0.0000 | 0.0000 | 948.3 |
| localizacao_direta | 4 | 0.5000 | 1.0000 | 1045.0 |
| metodo | 3 | 0.0000 | 0.0000 | 926.0 |
| multi_hop | 3 | 0.0000 | 0.4444 | 992.7 |
| multilingue | 3 | 0.3333 | 1.0000 | 873.6 |
| rcm | 3 | 1.0000 | 1.0000 | 826.5 |
| revisao_ampla | 2 | 0.0000 | 0.5000 | 1121.1 |
| sinonimo | 2 | 0.5000 | 1.0000 | 1017.7 |

## Limitações e próximo gate

- Métricas de retrieval não medem fidelidade da resposta gerada.
- O snapshot v1 não registra tamanho/overlap de chunk nem proveniência do texto contextual.
- A expansão vigente possui regras temáticas manuais e ainda associa Paderborn a Stender; não existe regra ou fonte bibliográfica direta para os 16 ensaios GPVS-Faults.
- A recuperação não aplica filtro de página e ainda não valida citações após a geração.
- Perguntas de abstenção serão pontuadas apenas após o Evidence Guard (R5).
- O baseline deve permanecer disponível para comparação e rollback durante R2–R6.
- Contextual Retrieval só poderá ser promovido após ganho mensurável e ausência de regressão crítica.

### Consultas sem acerto no top-5

- `direct-tcc-rates-002`: Mostre a tabela do TCC que reúne taxas de falha de componentes fotovoltaicos, incluindo inversor e fusível.
- `direct-stender-signals-003`: Em qual página Stender lista os sinais disponíveis no dataset do inversor IGBT trifásico?
- `concept-reliability-005`: Qual é o conceito de confiabilidade de um sistema fotovoltaico sob condições e período definidos?
- `concept-failure-mode-effect-006`: Qual é a diferença entre modo de falha, falha e efeito da falha?
- `concept-pod-007`: O que significa probabilidade de detecção em ensaios não destrutivos e que cautela metodológica é indicada?
- `authors-anomaly-009`: Como Ahirwar e Ibrahim descrevem o uso de AE-LSTM em anomalias de geração solar?
- `method-weibull-estimation-010`: Quais métodos são comparados para estimar os parâmetros de forma e escala da Weibull 2P?
- `method-censoring-weibull-011`: Por que dados suspensos ou censurados devem ser preservados numa análise Weibull?
- `method-maintenance-interval-012`: Como o intervalo ótimo de manutenção é relacionado à RCM e aos parâmetros de confiabilidade?
- `component-igbt-dataset-013`: Quais características e sinais elétricos são informados para o inversor IGBT de dois níveis?
- `component-inverter-alerts-014`: Quais falhas internas e alertas são observados em inversores fotovoltaicos segundo Monteiro?
- `component-ac-tickets-015`: Que participações de tickets são atribuídas ao Contator AC, IGBT e Fusíveis AC?
- `fmeca-definition-016`: Como o TCC distingue FMEA de FMECA e situa a análise de criticidade?
- `fmeca-sod-npr-017`: Como severidade, ocorrência e detecção compõem o NPR na FMECA?
- `reliability-system-composition-023`: Como a confiabilidade de componentes se combina em sistemas série e paralelo?
- `reliability-inverter-mil-024`: Como Shuttleworth calcula taxas de falha de inversores fotovoltaicos com MIL-HDBK-217?
- `reliability-risk-definition-025`: Como uma revisão recente diferencia falha de confiabilidade e risco em sistemas fotovoltaicos?
- `autoencoder-reconstruction-026`: Como o erro de reconstrução é usado por autoencoders na detecção de anomalias?
- `autoencoder-evaluation-028`: Quais métricas e resultados são usados por Ibrahim para avaliar a detecção do AE-LSTM?
- `gpvs-domain-contrast-030`: Qual documento descreve um dataset de inversor IGBT trifásico que não deve ser confundido com o GPVS-Faults?
- `multilingual-weibull-en-032`: How are the shape and scale parameters of a two-parameter Weibull distribution estimated?
- `multilingual-fmeca-es-033`: ¿Cuál es la diferencia entre FMEA y FMECA en el análisis de criticidad?
- `synonym-unreliability-034`: Como a não confiabilidade, também chamada probabilidade acumulada de falha, se relaciona com R(t)?
- `review-pv-reliability-036`: Faça um panorama da literatura sobre risco e confiabilidade de sistemas e inversores fotovoltaicos.
- `review-anomaly-pv-037`: Revise os trabalhos da base sobre detecção de anomalias e manutenção inteligente em sistemas fotovoltaicos.
- `multihop-fmeca-rcm-038`: Relacione a priorização S/O/D da FMECA com a seleção de tarefas de manutenção na RCM.
- `multihop-weibull-maintenance-039`: Como estimativas Weibull de forma e escala podem apoiar a definição de intervalos de manutenção?
- `multihop-failures-maintenance-040`: Conecte falhas observadas em inversores fotovoltaicos à análise de risco e às estratégias de manutenção inteligente.
