# Arquitetura — Al IAdo PV

Pacote Python modular. Ponto de entrada: `app.py` (Streamlit), que dispara o
orquestrador no backend. Há também `main.py` (chat no terminal).

```
src/
├── core/                 infraestrutura compartilhada
│   ├── config.py         caminhos, constantes, KMP_DUPLICATE_LIB_OK
│   ├── utils.py          UTF-8 seguro, caminhos relativos
│   └── logs.py           logging estruturado (logs/al_iado_pv.log)
├── conhecimento/         cérebro do agente (RAG + ferramentas)
│   ├── agente.py         pipeline RAG 3 camadas, PERFIL_COMPACTO, prompt
│   ├── ferramentas.py    tool calling (specs + roteador + execução)
│   ├── provedores.py     multi-provedor de LLM (Gemini / Groq)
│   ├── indexador.py      indexa PDFs/tabelas no ChromaDB
│   ├── indice_portatil.py exporta/importa snapshot gzip do índice
│   ├── leitor_anexos.py  leitura de anexos (PDF/CSV/Office/imagem)
│   └── web_search.py     busca leve + níveis de confiança A-D
├── ml/                   pipeline e experimentos de ML
│   ├── pipeline.py       registry das etapas + estado ready/stale/pending
│   ├── proveniencia.py   manifesto + hash + detecção de stale
│   ├── split_temporal.py divisão temporal com purga (anti-vazamento)
│   ├── dados_avaliacao.py banco E1 comum para comparações locais
│   ├── estatistica.py    ICs, bootstrap e métricas metodológicas
│   ├── exec_etapa_isolada.py executa etapa pesada em subprocesso
│   ├── features_ca.py    features CA do Paderborn
│   ├── autoencoder.py    modelo de normalidade (limiar p99)
│   ├── injecao_falhas.py falhas sintéticas FMEA (schema E2) + SMD_95
│   ├── validacao.py      validação interna E2 (holdout, ROC+PR, ICs)
│   ├── rul_weibull.py    RUL / Weibull
│   ├── classificador_pv.py classificação supervisionada PV Farms (CC)
│   ├── experimentos_artigos.py experimentos de ML por artigo-base
│   ├── exec_experimento_isolado.py roda experimento pesado em subprocesso
│   └── resultados.py     leitura/resumo de artefatos
└── orquestrador.py       coordena init + pipeline
```

## Fluxos
- **Init local:** `app.py` → `configurar_saida_utf8` +
  `KMP_DUPLICATE_LIB_OK` → `carregar_base` (ChromaDB, embeddings, perfil) →
  watcher + orquestrador.
- **Init nuvem:** `app.py` → restauração do snapshot portátil → encoder ONNX
  sob demanda → perfil. O deploy não inicia watcher nem orquestrador.
- **RAG:** pergunta → expansão de query → busca híbrida → reranking → prompt
  (com `PERFIL_COMPACTO`) → LLM.
- **Ferramentas (chat):** `decidir_acao` roteia para pipeline ML, experimentos,
  catálogo de literatura, `consultar_datasets`, `comparar_abordagens_ml`, etc.
- **Pipeline ML:** `features_ca → autoencoder → injecao_falhas → validacao →
  rul_weibull`, cada etapa com manifesto de proveniência.

## Execução local e nuvem
- **PC:** possui `dados/brutos/`, treina os modelos, regenera os experimentos e
  publica apenas os artefatos científicos verificáveis.
- **Streamlit Cloud:** restaura `artefatos/literatura_indexada.jsonl.gz` em um
  ChromaDB efêmero e consulta os JSONs, CSVs e PNGs versionados em `resultados/`.
  Sem os datasets brutos, não tenta representar uma execução de treino como
  concluída na nuvem. Para manter a memória dentro do limite do serviço, usa a
  variante ONNX quantizada do mesmo MiniLM do índice, carrega a sessão apenas
  na primeira busca e libera o tokenizer antes da inferência. Gemini e Groq
  usam adaptadores diretos dos SDKs oficiais; as integrações LangChain, que
  carregavam PyTorch indiretamente, não entram no processo web.
- `AL_IADO_CHROMADB_DIR` permite redirecionar o ChromaDB sem alterar o código;
  `AL_IADO_INDICE_LITERATURA` permite apontar para outro snapshot portátil e
  `AL_IADO_DATASET_PADERBORN` permite simular o modo de consulta em testes.
  `AL_IADO_EMBEDDINGS_BACKEND` aceita `auto`, `onnx` ou
  `sentence-transformers`; `AL_IADO_ONNX_THREADS` limita threads do backend
  leve. Em `auto`, ausência do dataset ativa ONNX.

## Isolamento de cargas pesadas (subprocesso)
Experimentos por artigo que carregam bibliotecas pesadas (`torch`)
rodam em **subprocesso** via
`exec_experimento_isolado.executar_experimento_isolado(key)`. Um segfault,
conflito de OpenMP ou estouro de memória derruba apenas o filho — o app
Streamlit segue de pé e recebe uma mensagem de falha legível. O progresso é
lido do stdout do filho e encaminhado ao vivo; o resultado volta por um JSON
temporário. Degradação honesta: se o subprocesso não puder ser lançado, cai
para execução in-process. Variáveis: `AL_IADO_SEM_ISOLAMENTO=1` força
in-process (debug/CI); `AL_IADO_EXP_CHILD=1` é o marcador interno do filho.

## Eixos de ML
- **Paderborn (CA):** detecção de anomalia por modelagem de normalidade.
- **PV Farms (CC):** classificação supervisionada de falhas conhecidas.

Os dois **não se fundem** — ver `docs/datasets.md` e `docs/metodologia_ml.md`.

## Instalação modular
```powershell
pip install -r requirements.txt              # ambiente completo (pins exatos)
# ou por grupo:
pip install -r requirements-ui.txt -r requirements-rag.txt -r requirements-ml.txt
pip install -r requirements-dev.txt              # testes/lint
```

## Padronização visual (tabelas e gráficos)
- `src/core/formatacao.py` — fonte única de formatação numérica e de
  tabelas Markdown do chat (política de casas decimais por tipo de valor,
  p-valores em convenção acadêmica, construtor de tabela com alinhamento
  uniforme). Todo número exibido ao usuário passa pelos `fmt_*`.
- `src/ml/estilo_graficos.py` — fonte única de estilo matplotlib:
  `aplicar_estilo()` fixa DPI (150), bbox, fontes e grade via rcParams;
  `TAM` define os tamanhos canônicos (unico 12x5, quadrado 7x6,
  painel_3 15x5, painel_6 15x8) e os helpers `tam_barras_h/v` e
  `tam_matriz` cobrem gráficos que crescem com N. Nenhum módulo de plot
  pode fixar figsize numérico ou dpi próprio — o teste
  `tests/test_formatacao_estilo.py` trava a regressão.
