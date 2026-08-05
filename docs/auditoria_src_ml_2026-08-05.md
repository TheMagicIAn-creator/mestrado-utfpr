# Auditoria geral de `src/ml` — 2026-08-05

## Escopo

Auditoria da pasta `src/ml` para a metodologia do Autoencoder denso, desde o
pipeline CA até injeção FMECA, validação E2, matrizes de confusão, Weibull/RUL
e comparação acadêmica com Ibrahim/AE-LSTM.

Arquivos classificados:

| Grupo | Arquivos |
|---|---|
| Infraestrutura do pacote | `__init__.py` |
| Pipeline CA principal | `features_ca.py`, `autoencoder.py`, `injecao_falhas.py`, `validacao.py`, `rul_weibull.py`, `pipeline.py`, `proveniencia.py` |
| Métricas, splits e gráficos | `escore_anomalia.py`, `estatistica.py`, `split_temporal.py`, `dados_avaliacao.py`, `graficos_autoencoder.py`, `estilo_graficos.py`, `resultados.py` |
| Comparação acadêmica vigente | `macro_proposto.py`, `macro_ibrahim.py`, `macro_comum.py`, `macro_comparar.py`, `modelos_anomalia.py`, `experimentos_artigos.py`, `protocolos_artigos.py` |
| Auxiliares e domínio separado | `classificador_pv.py`, `classificador_pv_infer.py`, `comparacao_literatura.py`, `diagnostico_escore.py`, `eda.py`, `exec_etapa_isolada.py`, `exec_experimento_isolado.py`, `retroalimentacao_fmeca.py` |

## Conclusões auditadas

1. **Comparação quantitativa ativa**
   O núcleo executável vigente é Proposto × Ibrahim/AE-LSTM. `REGISTRO` e
   `PROTOCOLOS` mantêm apenas `ibrahim`; Francisti, Isolation Forest e Prophet
   não são linhas executáveis da comparação. O comparativo publicado fica em
   `resultados/macro/`.

2. **Dataset e frequência de 10 kHz**
   A menção a 10 kHz pertence ao Paderborn (`Inverter_Data_Set.csv`), domínio
   **CA**, usado como eixo principal de modelagem de normalidade. Ela não é uma
   evidência de uso do dataset CC. O domínio **CC** é PV Farms, isolado em
   `classificador_pv.py`/`classificador_pv_infer.py`, e suas métricas não se
   transferem ao pipeline CA.

3. **FMECA vs FMEA**
   A fonte metodológica das falhas sintéticas é FMECA (`docs/fmeca.md` e
   `src/ml/injecao_falhas.py`). Menções restantes a FMEA em `src/ml` são
   conceituais, por exemplo para explicar que FMECA = FMEA + criticidade, ou
   referências bibliográficas. O índice usado operacionalmente é NPR da FMECA;
   `D` da FMECA não é confundido com detectabilidade do Autoencoder.

4. **Parâmetros rastreados pelo pipeline**
   Os manifestos v2 rastreiam os principais parâmetros:

   | Etapa | Parâmetros principais |
   |---|---|
   | `features_ca` | `FS=10000`, `F0=60`, `JANELA=1024`, `SOBREPOSICAO=512`, `HARMONICOS` |
   | `autoencoder` | `LATENTE_DIM=16`, `EPOCHS=150`, `BATCH_SIZE=32`, `LR=0.001`, `DROPOUT=0.2`, `PACIENCIA=20`, split 60/20/20, `THRESHOLD_METHOD=p99` |
   | `injecao_falhas` | severidades `[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`, `ALVO_SMD=0.95`, `N_JANELAS_SMD=100` |
   | `validacao` | severidades `[0.30, 0.50, 1.00]`, `N_JANELAS_SAUDAVEL=40`, `N_JANELAS_FALHA=40`, `PREVALENCIA_RARA=0.05` |
   | `rul_weibull` | `N_TRAJ=100`, `N_STEPS=120`, `N_BOOTSTRAP=250`, `MIN_EVENTOS_WEIBULL=10`, `PERSISTENCIA_CRUZAMENTO=3`, `TTF_UNIDADE=passo_sintetico_de_degradacao` |

   O limiar operacional do Autoencoder permanece no artefato `limiar.json`:
   `score_method=localizado`, `top_k=5`, `score_threshold=7.826175715408156`,
   `threshold_effective_percentile=99.9`.

5. **Métricas e matrizes**
   `validacao.py` calcula matriz de confusão binária, accuracy, precision,
   recall, F1, MCC, AUC-ROC, AUC-PR, specificity, FPR/FNR e reprojeção para
   falha rara. O verificador `scripts/verificar_resultados_fmeca.py` cruza JSON,
   CSV e PNGs publicados, incluindo matrizes de confusão e 13 gráficos.

6. **Weibull/RUL e tempo físico**
   A RUL continua E2 e sintética. A unidade publicada é
   `passo_sintetico_de_degradacao`; `tempo_fisico_calibrado=false`. Sem dados
   run-to-failure ou taxa de degradação calibrada em campo/bancada, não há base
   honesta para converter TTF/RUL em horas, dias ou anos. O código explicita a
   duração física da janela de aquisição (`1024 / 10000 = 0.1024 s`) apenas como
   contexto do sinal, não como passo de degradação.
   `N_TRAJ=100` é teto computacional, mas o n efetivo não excede o holdout
   independente disponível; aumentar esse teto sem novos dados apenas repetiria
   janelas e inflaria a confiança estatística.

7. **Arquivos removidos por obsolescência**
   `resultados/comparacao/comparacao_literatura.json` e `.png` foram removidos
   porque eram o comparativo E1 legado e competiam com a fonte vigente
   `resultados/macro/`.

## Limitações mantidas de forma explícita

- O Paderborn é saudável; as falhas avaliadas são sintéticas e orientadas pela
  FMECA, não falhas reais de campo.
- A macrocomparação pesada foi tentada localmente, mas excedeu 20 minutos. Os
  artefatos versionados de `resultados/macro/` não foram alterados e já estavam
  coerentes com Proposto × Ibrahim/AE-LSTM.
- `resultados/experimentos/ibrahim/resultado.json` é opcional no verificador; a
  fonte vigente para comparação acadêmica publicada é `resultados/macro/`.

## Evidências executadas nesta campanha

- Pipeline completo regenerado de `features_ca` até `rul_weibull`.
- `estado_pipeline()` retornou `ready` nas cinco etapas.
- `python scripts/verificar_resultados_fmeca.py` aprovado.
- `pytest -m "not pesado"` aprovado.
- `ruff check --select F821,F822,F823 src tests scripts` aprovado.
- CI do PR #96 aprovado.
