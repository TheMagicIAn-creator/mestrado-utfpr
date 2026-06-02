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
│   ├── leitor_anexos.py  leitura de anexos (PDF/CSV/Office/imagem)
│   └── web_search.py     busca leve + níveis de confiança A-D
├── ml/                   pipeline e experimentos de ML
│   ├── pipeline.py       registry das etapas + estado ready/stale/pending
│   ├── proveniencia.py   manifesto + hash + detecção de stale
│   ├── split_temporal.py divisão temporal com purga (anti-vazamento)
│   ├── features_ca.py    features CA do Paderborn
│   ├── autoencoder.py    modelo de normalidade (limiar p99)
│   ├── injecao_falhas.py falhas sintéticas FMEA (schema E2) + SMD_95
│   ├── validacao.py      validação formal (limiar congelado, ROC+PR, E2)
│   ├── rul_weibull.py    RUL / Weibull
│   ├── classificador_pv.py classificação supervisionada PV Farms (CC)
│   ├── experimentos_artigos.py experimentos de ML por artigo-base
│   └── resultados.py     leitura/resumo de artefatos
└── orquestrador.py       coordena init + pipeline
```

## Fluxos
- **Init:** `app.py` → `configurar_saida_utf8` + `KMP_DUPLICATE_LIB_OK` →
  `carregar_base` (embeddings, ChromaDB, perfil) → orquestrador.
- **RAG:** pergunta → expansão de query → busca híbrida → reranking → prompt
  (com `PERFIL_COMPACTO`) → LLM.
- **Ferramentas (chat):** `decidir_acao` roteia para pipeline ML, experimentos,
  catálogo de literatura, `consultar_datasets`, `comparar_abordagens_ml`, etc.
- **Pipeline ML:** `features_ca → autoencoder → injecao_falhas → validacao →
  rul_weibull`, cada etapa com manifesto de proveniência.

## Eixos de ML
- **Paderborn (CA):** detecção de anomalia por modelagem de normalidade.
- **PV Farms (CC):** classificação supervisionada de falhas conhecidas.

Os dois **não se fundem** — ver `docs/datasets.md` e `docs/metodologia_ml.md`.

## Instalação modular
```powershell
pip install -r requirements.txt              # ambiente completo (pins exatos)
# ou por grupo:
pip install -r requirements-ui.txt -r requirements-rag.txt -r requirements-ml.txt
pip install -r requirements-extras-prophet.txt   # Prophet (opcional)
pip install -r requirements-extras-rl.txt        # PPO/RL (opcional)
pip install -r requirements-extras-orange.txt    # CN2/Orange (opcional)
pip install -r requirements-dev.txt              # testes/lint
```
