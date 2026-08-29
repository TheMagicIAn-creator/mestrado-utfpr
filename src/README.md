# Arquitetura de `src/`

O projeto tem quatro camadas. `core/` fornece contratos comuns; `ml/` produz as
publicações científicas; `conhecimento/` implementa o agente e o RAG; `webapp/`
expõe a experiência ASGI. A interface não recalcula nem redefine métricas.

```text
webapp/ -> conhecimento/ -> ml/
              |             |
              +---- core/ <--+
```

## `core/`

| Arquivo | Responsabilidade |
|---|---|
| `citacao_guarda.py` | Validação de lastro das citações. |
| `config.py` | Caminhos e configuração por ambiente. |
| `conversa_export.py` | Exportação do histórico. |
| `formatacao.py` | Formatação acadêmica compartilhada. |
| `identidade.py` | Nome configurável do pesquisador na interface. |
| `importacao.py` | Importações tardias sem ciclos. |
| `logs.py` | Logging do projeto. |
| `seguranca.py` | Máscara de segredos e validações de entrada. |
| `tempo.py` | Relógio com fuso explícito. |
| `texto.py` | Normalização lexical. |
| `utils.py` | Caminhos portáveis e utilitários. |

## `conhecimento/`

| Arquivo | Responsabilidade |
|---|---|
| `agente.py` | Fachada e perfil acadêmico do ALIAdo. |
| `agente_contexto.py` | Montagem do contexto final. |
| `agente_interacao.py` | Interação e utilitários leves. |
| `agente_recuperacao.py` | Recuperação híbrida e reranking. |
| `atalhos.py` | Respostas determinísticas anteriores ao RAG. |
| `base_runtime.py` | Inicialização progressiva do conhecimento. |
| `benchmark_retrieval.py` | Gold set, validação de evidências e benchmark versionado do RAG. |
| `catalogo_bibliografico.py` | Catálogo versionado, identidade SHA-256 e metadados das fontes. |
| `cliente_llm.py` | Fachada compatível e neutra que encaminha `invoke`, JSON e streaming ao Router. |
| `consolidar_memoria.py` | Consolidação de sessões. |
| `contratos_llm.py` | Contratos neutros de pedido, resultado, uso e streaming de LLM. |
| `roteador_llm.py` | Política explícita de seleção, retry, fallback, escalonamento e validação cruzada. |
| `consultas_obsidian.py` | Consultas ao vault. |
| `embeddings.py` | Backend de embeddings. |
| `ferramentas.py` | Especificação e despacho das ferramentas. |
| `ferramentas_academicas.py` | Literatura, dataset e resultados canônicos. |
| `index_lock.py` | Lock de escrita do índice. |
| `indexador.py` | Indexação de PDFs e sessões. |
| `indice_lexical.py` | BM25 em SQLite FTS5. |
| `indice_portatil.py` | Snapshot portável do índice. |
| `intencoes_ferramentas.py` | Predicados lexicais do roteamento. |
| `leitor_anexos.py` | Leitura efêmera de anexos. |
| `memoria_persistente.py` | Memória validada e atômica. |
| `multiagente.py` | Auditoria e síntese multiagente. |
| `nota_cerebro.py` | Escrita de notas curadas. |
| `obsidian.py` | Integração com o vault. |
| `persistencia_nuvem.py` | Persistência permitida em ambiente remoto. |
| `processador_pdf.py` | Ingestão bibliográfica. |
| `provedores/` | Gateway, registry e adapters OpenAI/Gemini com fachada compatível. |
| `resultados_ml.py` | Indexação do resumo científico. |
| `retrieval_metrics.py` | Métricas do RAG. |
| `roteamento_ferramentas.py` | Decisão e comentário de ferramentas. |
| `snippets.py` | Cofre de trechos de código. |
| `vault_links.py` | Relações entre notas. |
| `web_search.py` | Busca externa com nível de confiança. |

## `ml/`

O GPVS-Faults é o único dataset ativo. A publicação experimental compara
Autoencoder Denso e AE-LSTM sob o mesmo protocolo. A publicação física é um
cenário bibliográfico separado e não deriva taxas de falha do GPVS.

| Arquivo | Responsabilidade |
|---|---|
| `dados_gpvs.py` | 16 ensaios, 24 features, split, normalização e holdout. |
| `modelos_autoencoder.py` | Denso 24-16-8-16-24 e AE-LSTM temporal. |
| `treino_comparacao.py` | Treino pareado, cinco sementes, top-k e limiar p99,9 rastreável. |
| `estatistica_comparacao.py` | Métricas, Wilson e bootstrap por ensaio. |
| `avaliacao_comparativa.py` | Avaliação E3 dos modelos congelados. |
| `graficos_comparacao.py` | Figuras comparativas E3 em PNG e PDF. |
| `publicacao_comparacao.py` | Tabelas, relatório, contrato e manifesto v2. |
| `comparacao_autoencoders.py` | Entrada da campanha Denso versus AE-LSTM. |
| `confiabilidade_componentes.py` | Cenários exponenciais diretos e derivados. |
| `graficos_confiabilidade.py` | Figuras de R(t), F(t), f(t), h(t) e taxas. |
| `publicacao_confiabilidade.py` | Publicação física e manifesto v2. |
| `pipeline.py` | Orquestra as duas publicações canônicas. |
| `proveniencia.py` | Hashes LF, entradas, dependências e saídas. |
| `resultados.py` | Leitura acadêmica dos contratos publicados. |
| `estilo_graficos.py` | Estilo quantitativo compartilhado. |

## `webapp/`

`python -m src.webapp` inicia a aplicação ASGI. `contracts.py` valida os
resultados sem recalcular; `chart_data.py` prepara séries visuais compactas;
`agent_adapter.py` preserva Gemini, RAG híbrido, memória e auditoria; os
painéis de comparação e confiabilidade carregam sob demanda.

## Fluxos

1. Chat: `webapp` -> `roteamento_ferramentas.py` -> ferramenta determinística ou RAG.
2. Cálculo: `pipeline.py` -> comparação E3 -> confiabilidade bibliográfica.
3. Publicação: dados-fonte + JSON metodológico + PNG/PDF + manifesto v2.
