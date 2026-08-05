# Auditoria pos-merge dos resultados academicos

Data: 2026-08-05

Base auditada: `main` apos o merge documental do PR #95 (`edb2fa7`).

## Veredito

Os artefatos centrais do autoencoder estao coerentes para uso academico no
estado publicado. A auditoria nao encontrou divergencia numerica entre
`limiar.json`, tabelas CSV, JSONs de validacao/injecao/FMECA, figuras publicadas
e manifestos v2. Por isso, esta etapa registra a evidencia de auditoria sem
alterar resultados gerados.

## Escopo verificado

- Limiar operacional e nomenclatura canonica em
  `resultados/autoencoder/limiar.json`.
- Calibracao academica em
  `resultados/autoencoder/calibracao_autoencoder.csv` e `.md`.
- Validacao sintetica E2 em
  `resultados/autoencoder/validacao_tabela.csv`,
  `validacao_tabela.md` e `validacao_report.json`.
- Injecao de falhas e retroalimentacao FMECA em
  `injecao_falhas_report.json`, `injecao_smd_tabela.csv` e
  `retroalimentacao_fmeca.json`.
- Manifestos v2 em `resultados/manifestos/*.json`.
- Figuras publicadas em `resultados/autoencoder/*.png`.

## Evidencias

### Limiar e calibracao

- Escore operacional canonico: `localizado`.
- `score_threshold`: `7.826176`.
- Referencia MSE p99: `2.545433`.
- `threshold_fallback_percentile`: `99.0`.
- `threshold_effective_percentile`: `99.9`.
- `sigma_multiplier`: `3.0`.
- `top_k`: `5`.
- Fonte do limiar: `bloco_calibracao_temporal`.

Na calibracao, o MSE p99 do CSV coincide com `mse_p99` em `limiar.json`. A taxa
de excedencia por MSE no bloco de calibracao e `1/91 = 1.10%`. A taxa de
excedencia pelo escore operacional localizado e `4/91 = 4.40%` na calibracao e
`9/88 = 10.23%` no teste isolado. Essa diferenca deve permanecer explicita em
texto academico: os graficos de reconstrucao usam escala MSE, enquanto a decisao
operacional usa o escore localizado.

### Validacao sintetica E2

- Foram conferidas 9 linhas: 3 falhas x 3 severidades.
- Todas as linhas carregam `evidence_level = E2`.
- Todas as linhas usam `score_method = localizado`,
  `score_threshold = 7.826176` e `threshold_effective_percentile = 99.9`.
- O CSV e o JSON batem para precision, recall, F1, accuracy, AUC-ROC, AUC-PR,
  specificity, FNR, `precision_raro` e `f1_raro`.
- Faixa observada: recall de `0.125` a `1.000`, F1 de `0.200` a `0.941` e
  AUC-ROC de `0.589` a `0.984`.
- O holdout saudavel registrou FPR de `12.5%` no ponto operacional
  (`5/40` janelas saudaveis).

O arquivo `validacao_tabela.md` declara que o limiar operacional esta congelado,
mas nao repete todos os campos canonicos do ponto de operacao. Isso nao e
inconsistencia numerica, porque os campos completos estao no CSV e em
`validacao_report.json`. Em uma futura regeneracao da etapa `validacao`, vale
incluir no gerador uma linha explicita com
`localizado / percentil efetivo 99.9`.

### Injecao e FMECA

- `injecao_falhas_report.json` usa o mesmo `score_threshold`,
  `score_method`, `mse_p99` e percentil efetivo de `limiar.json`.
- `retroalimentacao_fmeca.json` usa `limiar_operacional = 7.826176`,
  `score_method = localizado`, `percentil_efetivo = 99.9` e
  `evidence_level = E2`.
- A regra FMECA publicada permanece: `D_proj = min(D_campo, D_mon); S e O
  inalterados`.
- Foram verificadas 3 linhas FMECA, uma por falha critica.

### Manifestos v2

Foram encontrados 5 manifestos v2: `features_ca`, `autoencoder`,
`injecao_falhas`, `validacao` e `rul_weibull`. Todos declaram
`code_hash_mode = text_lf_utf8`, dependencias cientificas e hashes de outputs.
Os 30 hashes de `output_artifacts` recalculados localmente coincidem com os
valores publicados.

### Figuras

Foram abertas 14 figuras PNG em `resultados/autoencoder`. Todas estao legiveis
e acima de 900 x 500 px. A menor figura em area foi `curva_treino.png`
(`1817 x 814`).

## Riscos residuais academicos

- O nivel de evidencia continua sendo E2: validacao sintetica orientada pela
  FMECA, nao prova industrial E3.
- O FPR observado no holdout saudavel (`12.5%`) e a taxa de excedencia do escore
  operacional no teste isolado (`10.23%`) devem ser apresentados como limitacao
  metodologica, nao como desempenho de campo.
- Alguns graficos auxiliares historicos coexistem com os artefatos centrais
  regenerados. Ao escrever a dissertacao, priorizar os artefatos cobertos pelos
  manifestos v2 e citar auxiliares apenas quando sua finalidade estiver clara.

## Conclusao

Nao ha necessidade de alterar resultados numericos nesta etapa. A melhoria
pertinente e registrar esta auditoria para deixar explicito o estado academico
dos artefatos, seus limites e a consistencia entre manifestos, tabelas, JSONs e
figuras.
