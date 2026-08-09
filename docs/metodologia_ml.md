# Metodologia de ML — Al IAdo PV

Documento de referência das decisões metodológicas e de integridade acadêmica
da dissertação. Os **números** ficam nos artefatos (`resultados/...`), não aqui:
consulte sempre o artefato vigente e informe o **nível de evidência**.

## 1. Dois eixos de ML (NÃO se fundem)

| Eixo | Dataset | Domínio | Tarefa |
|---|---|---|---|
| **Principal** | Stender (`Inverter_Data_Set.csv`) | **CA experimental, inversor/motor** | Detecção de anomalia por modelagem de normalidade |
| **Complementar** | PV Farms (`train/test_data.csv`) | **CC simulado** | Classificação supervisionada de falhas conhecidas |

**Regra rígida:** o classificador PV Farms **nunca** diagnostica falhas CA do
inversor, e métricas de PV Farms **não** se transferem ao pipeline CA. O uso
combinado é conceitual/arquitetural — não fusão de dados.

Pipeline principal: `features_ca → autoencoder → injecao_falhas → validacao → rul_weibull`.

## 2. Arquitetura do Autoencoder — o que é citável e o que é escolha nossa

Separar as duas coisas é o que sustenta o capítulo diante da banca.

**Fundamentado em Ibrahim et al. (2022).** A modelagem de normalidade por
Autoencoder e o uso do **erro de reconstrução como escore de anomalia** seguem o
artigo (§3.1, eq. 1–3). A pergunta "por que Autoencoder?" tem resposta citável.

**Escolha deliberada nossa: AE denso sobre features, não AE-LSTM temporal.**
Ibrahim usa AE-LSTM sobre o sinal; aqui se usa um Autoencoder **denso sobre
features espectrais** derivadas da FMECA (RMS, THD, harmônicos 5/7/11/13,
desbalanceamento). Três razões, nesta ordem de peso:

1. **A recorrência é redundante para o alvo.** A dinâmica intra-janela relevante
   às falhas CA já está condensada nas features espectrais de cada janela de
   ~102 ms. A LSTM reaprenderia, no eixo do tempo, o que a FFT já resolveu.
2. **Features nomeadas preservam rastreabilidade.** Mesmo com MSE operacional,
   os resíduos continuam decomponíveis por RMS, THD e harmônicos de cada fase,
   permitindo ligar o desvio às assinaturas da FMECA. Um AE-LSTM sobre sinal
   bruto reduz essa rastreabilidade, importante para a conexão com RCM.
3. **A escolha é comparada empiricamente.** O AE-LSTM inspirado no artigo é
   mantido como concorrente em `resultados/macro/`, sob o mesmo protocolo E2.
   O escore localizado também permanece como ablação, mas deixou de ser o
   método canônico quando não reproduziu seu ganho no split auditado.

### Como os hiperparâmetros foram escolhidos

Profundidade fixada a priori; largura varrida. O precedente é o do próprio
artigo-base: Ibrahim (2022), §5.2, fixa a profundidade por simplificação
(*"the number of hidden layers was chosen to be four layers"*) e **otimiza
apenas o número de neurônios por camada** (Tabela 2). O mesmo protocolo aqui:
profundidade fixa e largura varrida.

A topologia vigente é `n→16→8→16→n` (3.860 parâmetros com 108 features). A anterior era
`n→64→32→16→32→64→n`, com 19.389 parâmetros para 274 janelas de treino — 70,8
parâmetros por amostra. O janelamento de 2048 amostras e o split 50/20/30
deixam 104 janelas de treino (37,1 parâmetros por janela); por isso a rede encolheu **no mesmo
conjunto de commits** que alargou a janela. O corte veio de onde estava o peso:
as camadas de borda (`n×64` e `64×n`) somavam 14.125 dos 19.389 parâmetros.

O gargalo passou a **não** ter ReLU: com ela o latente é não negativo por
construção e unidades podem morrer em zero permanente.

> **Estado:** a varredura de largura ainda **não foi executada** — exige rerun
> local com o dataset bruto. Até que seja, os valores vigentes (latente=8,
> épocas=150, lr=1e-3, dropout=0,2) devem ser apresentados como **defaults
> dimensionados pela razão parâmetros/amostra**, não como resultado de busca.
> Escrever o contrário seria afirmar evidência inexistente.

### Lacuna declarada

O acervo indexado **não contém nenhum artigo de Autoencoder denso sobre features
handcrafted com topologia reportada** — verificado por varredura em
`literatura/` e `notas/Literatura/`. Ibrahim ancora o *método* de escolher
hiperparâmetros, não as dimensões em si.

Por isso a fundamentação acima é redigida como **escolha justificada**, não como
"segue a referência X". Buscar uma âncora direta (família provável: AE denso
sobre features de vibração/corrente em máquinas rotativas) está registrado em
`notas/Cerebro/Literatura a revisar.md` como melhoria futura de redação — não
como bloqueio.

### ReLU no gargalo — resolvido em 08/08/2026

O encoder terminava em `ReLU` no espaço latente, o que zerava componentes
negativas e podia matar unidades em zero permanente. Havia inconsistência
interna: a saída do decoder já era linear, com comentário justificando; o
gargalo não recebia o mesmo tratamento.

A decisão registrada era não alterar isoladamente, porque trocar a ativação
invalida os mesmos números que a varredura invalidaria. Foi feito junto com o
encolhimento da rede, na rodada única (PR #107): **a última camada do encoder é
linear**, e `tests/test_torch_smoke.py` impede a regressão exigindo que exista
código latente negativo.

## 3. Limiar do Autoencoder

- **Limiar operacional = MSE p99 da calibração**, registrado em
  `score_threshold`, `mse_p99` e `limiar_p99` com `score_method = mse`.
- **Escore localizado = ablação diagnóstica**, mantida para comparação, sem
  controlar a decisão operacional.
- **μ + 3σ = referência comparativa** (assume normalidade) — **nunca** o limiar
  em uso.
- **p95 = referência adicional.**
- O artefato `limiar.json` preserva campos legados (`threshold_method`,
  `limiar`, `k`, `k_localizado`) e acrescenta nomes inequívocos:
  `score_method`, `score_threshold`, `mse_p99`, `sigma_multiplier`, `top_k`,
  `threshold_fallback_percentile` e `threshold_effective_percentile`.
- A auditoria tabular de calibração fica em
  `resultados/autoencoder/calibracao_autoencoder.{csv,md}`: ela reporta, por
  bloco temporal, mediana/IQR/p99 do MSE e excedências acima da referência MSE
  p99 e do limiar operacional, sempre com IC95% de Wilson.

## 4. Validação sintética interna E2 (limiar congelado)

`src/ml/validacao.py` carrega o limiar de `limiar.json` (**congelado**) e calcula
as métricas nesse limiar fixo — **não** otimiza o limiar no conjunto de teste.
Gera ROC, **Precision-Recall**, matriz de confusão e `validacao_report.json`
com `evidence_level = E2` e `threshold_source = bloco_calibracao_temporal`.
O protocolo canônico é `treino 50% → calibração 20% → teste 30%`, com purga
nas fronteiras. Injeção e validação usam apenas janelas **não sobrepostas** do
conjunto de teste. A configuração produz 32 trajetórias independentes, sem
pseudorrepetição, e mantém o piso estatístico de 30. Isso remove vazamento de treino, mas não transforma E2 em
validação externa: as falhas continuam sintéticas.

Benchmarks exploratórios (ex.: `experimentos_artigos.py`) que escolhem o limiar
no próprio conjunto avaliado são rotulados `threshold_source =
exploratorio_no_conjunto_avaliado` → **E1**, não estimativa de generalização.

## 5. Divisão em blocos intercalados com purga (anti-vazamento + cobertura)

Janelas com 50% de sobreposição **não** podem ser divididas aleatoriamente:
janelas vizinhas são quase idênticas e vazariam entre treino/calibração/teste.
A defesa é dividir por **blocos contíguos** com zona de **purga** na fronteira.

### Por que três blocos contíguos não bastavam

Três blocos contíguos pressupõem sinal aproximadamente **estacionário**. O
conjunto Stender não é: é bancada de acionamento que varre rotação em rampa. Fatiar a
rampa em três produz três **faixas de velocidade**, não três amostras do mesmo
processo. Medido em 09/08/2026 com 224 janelas:

| bloco | mediana de F0 | IQR | n |
|---|--:|--:|--:|
| treino | 20,45 Hz | 83,13 | 136 |
| calibração | **51,11 Hz** | **1,46** | 45 |
| teste | **100,08 Hz** | 17,84 | 43 |

O IQR da calibração é de 1,46 Hz — o bloco inteiro parado num regime só. O
limiar operacional era congelado ali e aplicado a um bloco operando ao **dobro**
da fundamental: FPR de 4,4% na calibração contra **62,8%** no teste. O treino,
com IQR de 83 Hz, viu a faixa inteira; quem extrapolava era apenas o limiar.

### A correção

`split_temporal.py::split_blocos_intercalados` divide a série em
`N_BLOCOS_PADRAO = 14` blocos contíguos e os distribui alternadamente
(`T E V T T E T V E T T V E T`), com purga em toda fronteira **onde o destino
muda**. A ordem é determinística — cada conjunto recebe posições ideais
`(k+0,5)/c` e a sequência sai da ordenação delas —, sem sorteio nem semente.

Cobertura da série, antes e depois:

| conjunto | contíguo | intercalado |
|---|--:|--:|
| treino | 59% | 99,6% |
| calibração | 19% | 69,7% |
| teste | 18% | 84,6% |

Custo: 22 de 228 janelas descartadas por purga (9,6%). Restam 104 janelas de
treino, 42 de calibração e 60 de teste; destas últimas, 32 não se sobrepõem.
O desenho anterior 60/20/20 fornecia apenas 21 trajetórias independentes e
falhava o piso de 30 do próprio verificador acadêmico.

### O que muda na afirmação da dissertação

O teste **deixa de ser "o futuro"** e passa a ser **generalização entre
regimes**. Para detecção de anomalia em bancada de velocidade variável isso é
mais adequado que previsão temporal — o inversor em campo não opera em rampa
monotônica —, mas é uma afirmação **diferente** e não pode ser apresentada como
a anterior.

A garantia anti-vazamento não mudou de natureza e é verificada diretamente por
`tests/test_split_intercalado.py`: nenhuma janela vizinha cai em conjuntos
diferentes, para nenhum tamanho de série testado.

O split contíguo continua disponível em `split_temporal_com_purga`, e é o que o
protocolo E1 por artigo (`protocolos_artigos.py`) segue usando — trocá-lo lá
mudaria resultados históricos publicados.

## 6. Níveis de evidência (E0–E3)

| Nível | Significado |
|---|---|
| **E0** | Hipótese |
| **E1** | Benchmark exploratório (perturbação genérica / dataset rotulado) |
| **E2** | Validação sintética orientada pela FMECA (injeção/validação do pipeline) |
| **E3** | Validação experimental externa (bancada / campo) |

O agente **sempre** informa o nível e **nunca** trata E1/E2 como prova de
desempenho industrial.

## 7. Falhas sintéticas (schema + calibração)

Cada falha injetada (`FALHAS` em `injecao_falhas.py`) declara: `evidence_level`
(E2), `hipotese_fisica`, `sinais`, `formula`, `severity_definition`, `source` e
`limitations`. **Contator AC:** o ruído gaussiano é um **proxy** do transiente de comutação
e exige **calibração física** — não afirmar alta sensibilidade sem essa ressalva.

**SMD probabilística:** `smd_probabilistico` calcula a taxa de detecção por
severidade em janelas não sobrepostas do holdout, o intervalo de Wilson de 95%
e a **SMD₉₅** (menor severidade cuja taxa pontual é ≥ 95%). O campo
`smd_95_conservadora` exige também limite inferior do IC ≥ 95%; quando n é
insuficiente, permanece nulo em vez de transmitir certeza artificial.
Na rodada canônica há 32 repetições independentes por severidade; os intervalos
continuam obrigatórios porque essa amostra ainda produz incerteza relevante.

**Resolução da calibração de cauda:** as 42 janelas de calibração não autorizam
subdividi-la em 80/20 para selecionar automaticamente uma meta de FP de 1%:
o subbloco teria 9 observações e resolução mínima de 11,1%. A auto-calibração
agora exige pelo menos `ceil(100 / FP_ALVO)` observações no subbloco de
validação; enquanto esse contrato não é atendido, o limiar usa o fallback p99
e registra a razão em `limiar.json`.

**Separação na ablação localizada:** a média e o desvio dos resíduos por feature
são ajustados no bloco de **treino**; somente o percentil do escore é estimado
na **calibração**. Ajustar ambos na calibração reutilizava
o mesmo bloco duas vezes e, na rodada diagnóstica 50/20/30, produziu 2,38% de
FP na calibração contra 15% no teste. A separação derrubou o FP do mesmo modelo
e split para 1,67% antes da regeneração completa. `limiar.json` registra
`score_standardization_source = bloco_treino_modelo`. Mesmo corrigida, a
ablação perdeu sensibilidade no ponto operacional e não substitui o MSE p99.

## 8. Weibull e RUL sintéticos

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

## 9. Métricas

Schema único (`_metricas_classificacao`): accuracy, **balanced_accuracy**,
precision, recall, f1, **MCC**, AUC, **specificity** (= TN/(TN+FP) no binário) +
**specificity_macro_ovr** (média one-vs-rest) + `specificity_tipo`,
**FPR/FNR** (binário), matriz de confusão.

## 10. Proveniência e reprodutibilidade

- **Manifesto v2 por etapa** (`proveniencia.py`): `code_sha256` normalizado para
  LF, `code_dependencies`, `parameters`, hash dos artefatos upstream,
  `output_artifacts` e `git_commit`. Estados **ready / stale / pending**
  (`estado_pipeline()`), exibidos no chat e na sidebar. Um artefato **sem
  manifesto = não verificado (pending)**; manifesto v1 = **stale** até
  regenerar. Nada é apagado automaticamente; recalcular é sob comando (com
  confirmação).
- **Caminhos relativos** nos artefatos (`to_project_relative_path`), resolvidos
  para absoluto só na interface.
- **Datasets** validados por `scripts/verificar_datasets.py` (SHA-256, linhas);
  dados brutos não são versionados.

## 11. Protocolos de avaliação POR ARTIGO (anti "erro de simulação")

Antes, todos os experimentos de anomalia compartilhavam um harness único:
split **aleatório** de janelas sobrepostas (vazamento temporal) e limiar
escolhido **no próprio teste** (oráculo) para os modelos sem decisão nativa.
Agora cada artigo segue o **seu** protocolo (`src/ml/protocolos_artigos.py`)
e **nenhum limiar enxerga os rótulos do teste**:

| Artigo | Decisão do modelo ativo | `threshold_source` |
|---|---|---|
| **Ibrahim (2022)** | AE-LSTM temporal com limiar p99 do erro em bloco de calibração temporal, congelado antes do teste | `p99_erro_seq_temporal_calibracao` |

Cortados da curadoria executável (não são experimentos quantitativos ativos):
Francisti/Shewhart, Isolation Forest e Prophet do Ibrahim, Sharma, Ahirwar,
Stender e modelos supervisionados de domínio CC. Esses trabalhos continuam
citáveis como literatura quando forem úteis ao texto, mas não entram na tabela
comparativa vigente do Autoencoder denso proposto.

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
