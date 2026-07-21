# Metodologia de ML — Al IAdo PV

Documento de referência das decisões metodológicas e de integridade acadêmica
da dissertação. Os **números** ficam nos artefatos (`resultados/...`), não aqui:
consulte sempre o artefato vigente e informe o **nível de evidência**.

## 1. Dois eixos de ML (NÃO se fundem)

| Eixo | Dataset | Domínio | Tarefa |
|---|---|---|---|
| **Principal** | Paderborn (`Inverter_Data_Set.csv`) | **CA** | Detecção de anomalia por modelagem de normalidade |
| **Complementar** | PV Farms (`train/test_data.csv`) | **CC** | Classificação supervisionada de falhas conhecidas |

**Regra rígida:** o classificador PV Farms **nunca** diagnostica falhas CA do
inversor, e métricas de PV Farms **não** se transferem ao pipeline CA. O uso
combinado é conceitual/arquitetural — não fusão de dados.

Pipeline principal: `features_ca → autoencoder → injecao_falhas → validacao → rul_weibull`.

## 2. Limiar do Autoencoder

- **Limiar operacional = percentil 99** do erro de reconstrução saudável no
  bloco temporal de **calibração**
  (controla FP ≈ 1%, robusto a distribuições assimétricas).
- **μ + 3σ = referência comparativa** (assume normalidade) — **nunca** o limiar
  em uso.
- **p95 = referência adicional.**
- O artefato `limiar.json` registra `threshold_method = "p99"` + os três valores.

## 3. Validação sintética interna E2 (limiar congelado)

`src/ml/validacao.py` carrega o limiar de `limiar.json` (**congelado**) e calcula
as métricas nesse limiar fixo — **não** otimiza o limiar no conjunto de teste.
Gera ROC, **Precision-Recall**, matriz de confusão e `validacao_report.json`
com `evidence_level = E2` e `threshold_source = bloco_calibracao_temporal`.
O protocolo canônico é `treino 60% → calibração 20% → teste 20%`, com purga
nas fronteiras. Injeção e validação usam apenas janelas **não sobrepostas** do
bloco final. Isso remove vazamento de treino, mas não transforma E2 em
validação externa: as falhas continuam sintéticas.

Benchmarks exploratórios (ex.: `experimentos_artigos.py`) que escolhem o limiar
no próprio conjunto avaliado são rotulados `threshold_source =
exploratorio_no_conjunto_avaliado` → **E1**, não estimativa de generalização.

## 4. Divisão temporal com purga (anti-vazamento)

Janelas com 50% de sobreposição **não** podem ser divididas aleatoriamente
(janelas vizinhas quase idênticas vazariam entre treino/val/teste).
`src/ml/split_temporal.py::split_temporal_com_purga` faz blocos contíguos no
tempo com zona de **purga** na fronteira.

## 5. Níveis de evidência (E0–E3)

| Nível | Significado |
|---|---|
| **E0** | Hipótese |
| **E1** | Benchmark exploratório (perturbação genérica / dataset rotulado) |
| **E2** | Validação sintética orientada pela FMECA (injeção/validação do pipeline) |
| **E3** | Validação experimental externa (bancada / campo) |

O agente **sempre** informa o nível e **nunca** trata E1/E2 como prova de
desempenho industrial.

## 6. Falhas sintéticas (schema + calibração)

Cada falha injetada (`FALHAS` em `injecao_falhas.py`) declara: `evidence_level`
(E2), `hipotese_fisica`, `sinais`, `formula`, `severity_definition`, `source` e
`limitations`. **Contator AC:** o ruído gaussiano é um **proxy** do transiente de comutação
e exige **calibração física** — não afirmar alta sensibilidade sem essa ressalva.

**SMD probabilística:** `smd_probabilistico` calcula a taxa de detecção por
severidade em janelas não sobrepostas do holdout, o intervalo de Wilson de 95%
e a **SMD₉₅** (menor severidade cuja taxa pontual é ≥ 95%). O campo
`smd_95_conservadora` exige também limite inferior do IC ≥ 95%; quando n é
insuficiente, permanece nulo em vez de transmitir certeza artificial.

## 7. Weibull e RUL sintéticos

- Uma trajetória mantém a **mesma janela-base** enquanto a severidade cresce;
  não mistura ativos/regimes operacionais a cada passo. Realizações estocásticas
  de uma mesma trajetória também ficam congeladas ao longo da severidade.
- Janelas do holdout que já excedem o limiar saudável em `t=0` são excluídas e
  contabilizadas. Um evento exige três passos consecutivos acima do limiar,
  reduzindo cruzamentos isolados por ruído.
- Cruzamentos persistentes do limiar são eventos; trajetórias sem cruzamento permanecem
  **censuradas à direita**. Censura nunca recebe jitter nem vira falha.
- O ajuste de dois parâmetros usa máxima verossimilhança censurada e intervalos
  bootstrap por trajetória. Kaplan-Meier é exibido junto da curva paramétrica.
- A **RUL restrita por Kaplan-Meier** é calculada para todas as famílias até o
  horizonte observado. A RUL paramétrica só aparece quando há eventos suficientes;
  censura acima de 50% não apaga a curva, mas a marca como extrapolação de alta
  incerteza. Sem eventos suficientes, beta, eta, MTTF e B10 permanecem nulos.
- MTTF, B10 e ambas as formas de RUL estão em **passos sintéticos E2**. Mesmo com ajuste
  convergente, não equivalem a horas, ciclos ou vida física de campo.

## 8. Métricas

Schema único (`_metricas_classificacao`): accuracy, **balanced_accuracy**,
precision, recall, f1, **MCC**, AUC, **specificity** (= TN/(TN+FP) no binário) +
**specificity_macro_ovr** (média one-vs-rest) + `specificity_tipo`,
**FPR/FNR** (binário), matriz de confusão.

## 9. Proveniência e reprodutibilidade

- **Manifesto por etapa** (`proveniencia.py`): `code_sha256`, `parameters`,
  hash dos artefatos upstream, outputs, `git_commit`. Estados **ready / stale /
  pending** (`estado_pipeline()`), exibidos no chat e na sidebar. Um artefato
  **sem manifesto = não verificado (pending)** — nunca válido só por existir.
  Nada é apagado automaticamente; recalcular é sob comando (com confirmação).
- **Caminhos relativos** nos artefatos (`to_project_relative_path`), resolvidos
  para absoluto só na interface.
- **Datasets** validados por `scripts/verificar_datasets.py` (SHA-256, linhas);
  dados brutos não são versionados.

## 10. Protocolos de avaliação POR ARTIGO (anti "erro de simulação")

Antes, todos os experimentos de anomalia compartilhavam um harness único:
split **aleatório** de janelas sobrepostas (vazamento temporal) e limiar
escolhido **no próprio teste** (oráculo) para os modelos sem decisão nativa.
Agora cada artigo segue o **seu** protocolo (`src/ml/protocolos_artigos.py`)
e **nenhum limiar enxerga os rótulos do teste**:

| Artigo | Decisão de cada modelo | `threshold_source` |
|---|---|---|
| **Francisti (2025)** | Shewhart: alarme se alguma feature sai de ±3σ do treino (fixo a priori) | `shewhart_3sigma_a_priori` |
| **Ibrahim (2022)** | IF contaminação a priori (5%); AE-LSTM limiar = p99 do erro **no treino** (congelado) | `contaminacao_a_priori_0.05`, `p99_erro_em_calibracao_temporal` |

Cortados da curadoria (não são experimentos executáveis): Sharma (PPO+IF,
baselines supervisionados, RNN/CNN), Ahirwar (voto híbrido — derivativo do
Ibrahim), o Random Forest do Francisti e o Prophet do Ibrahim (pior detector
+ dependência instável em runtime).

Infraestrutura comum (benchmark justo):
- **Split temporal com purga** (`split_temporal.py`) — nunca aleatório;
- **Injeção orientada pela FMECA no espaço de features**: cada anomalia pertence
  a uma família da FMECA de Torres (2024) — Contator AC (NPR=315), IGBT (NPR=90),
  Fusível AC (NPR=30) — perturbando apenas as features que a física daquela
  falha afeta (fonte única: docs/fmeca.md). O resultado reporta **detecção
  por família** (`deteccao_por_falha`).
- O `resultado.json` carrega o bloco **`metodologia`** (split, injeção, decisão
  por modelo, notas de fidelidade ao artigo).

**Leitura correta:** os F1 **não** são comparáveis entre protocolos (cada um
opera no seu ponto de decisão); o **AUC** é a métrica comparável. Continua
**E1** (benchmark exploratório com ground truth sintético) — não é validação
formal nem desempenho industrial. Métricas antigas com
`exploratorio_no_conjunto_avaliado` permanecem válidas como histórico, mas o
caminho padrão atual usa decisões a priori/congeladas.
