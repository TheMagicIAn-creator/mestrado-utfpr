# Reprodutibilidade — Al IAdo PV

## Proveniência e estado das etapas
Cada etapa do pipeline grava um **manifesto v2**
(`resultados/manifestos/<etapa>.json`) com `code_sha256` normalizado para LF
(`code_hash_mode = text_lf_utf8`), `code_dependencies`, `parameters`, hash dos
artefatos upstream, `output_artifacts` e `git_commit`. O estado é de 3 valores
(`estado_pipeline()`):

- **ready** — artefatos presentes e manifesto compatível;
- **stale** — artefatos existem, mas o código, os parâmetros ou um artefato
  upstream mudaram;
- **pending** — artefato ausente **ou sem manifesto** (não verificado).

Um artefato **nunca** é tratado como válido só por existir. Nada é apagado
automaticamente; recalcular é sob comando explícito (com confirmação em duas
etapas para ações destrutivas).
Manifestos v1 continuam legíveis, mas aparecem como **stale** até serem
regenerados no schema v2.

## Portabilidade
- Artefatos gravam **caminhos relativos** ao projeto (`to_project_relative_path`),
  resolvidos para absoluto só na interface (`resolve_project_path`).
- `KMP_DUPLICATE_LIB_OK=TRUE` é definido cedo (config/app/main) para evitar crash
  de OpenMP duplicado no Windows.

## Memória
- Memória de **produção** (`sessoes_pv`) é separada da de **avaliação**
  (`avaliacoes_agente`). Os scripts de avaliação **não gravam memória por padrão**
  (`--com-memoria` para opt-in) — avaliação não contamina produção.

## Verificação local
```powershell
python scripts/verificar_ambiente.py    # imports, versões, chaves, datasets, ChromaDB, pipeline
python scripts/verificar_datasets.py    # SHA-256 + linhas + classes dos datasets
python -m pytest                        # suíte completa (contagem: pytest --collect-only -q)
# O CI (.github/workflows/ci.yml) roda apenas o subconjunto LEVE de testes
# (sem torch/chromadb/APIs); os demais rodam somente no ambiente local.
```

## Recalcular resultados (exige `dados/brutos/`)
```powershell
python src/ml/features_ca.py
python src/ml/autoencoder.py     # limiar.json (score operacional + referências) + manifesto
python src/ml/injecao_falhas.py  # falhas FMEA (E2) + schema
python src/ml/validacao.py       # ROC + PR + matriz, limiar congelado, report E2
python src/ml/rul_weibull.py
```
Enquanto não recalculados com o código atual, as etapas aparecem **stale/pending**
(comportamento correto).
