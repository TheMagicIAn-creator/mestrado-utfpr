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
2. **Features nomeadas habilitam o escore localizado.** O escore é a média dos
   top-k resíduos padronizados **por feature**, o que permite atribuir o desvio a
   "harmônico 5 da fase A" e ligá-lo ao modo de falha da FMECA. Um AE-LSTM sobre
   sinal bruto perderia essa rastreabilidade — e é ela que conecta a detecção à
   RCM, que é o eixo da dissertação.
3. **A escolha é validada empiricamente, não por conveniência.** O AE-LSTM fiel
   ao artigo é mantido como **concorrente** na comparação (`resultados/macro/`),
   sob o mesmo protocolo E2. A vantagem do método proposto no IGBT — e o
   **empate** no Fusível AC, cuja assinatura é de banda larga — é exatamente o
   que a escolha do escore prevê.

### Como os hiperparâmetros foram escolhidos

Profundidade fixada a priori; largura varrida. O precedente é o do próprio
artigo-base: Ibrahim (2022), §5.2, fixa a profundidade por simplificação
(*"the number of hidden layers was chosen to be four layers"*) e **otimiza
apenas o número de neurônios por camada** (Tabela 2). O mesmo protocolo aqui:
topologia 109→64→32→16 com profundidade fixa, e o espaço latente varrido em
{8, 16, 32} pela loss de calibração.

> **Estado:** a varredura ainda **não foi executada** — exige rerun local com o
> dataset bruto. Até que seja, os valores vigentes (latente=16, épocas=150,
> lr=1e-3, dropout=0,2) devem ser apresentados como **defaults**, não como
> resultado de busca. Escrever o contrário seria afirmar evidência inexistente.

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

### Pendência conhecida: ReLU no gargalo

O encoder termina em `ReLU` no espaço latente (`src/ml/autoencoder.py:140-141`),
o que zera componentes negativas e reduz a capacidade efetiva de representação.
Há uma inconsistência interna que reforça o ponto: a **saída do decoder é
linear**, com comentário justificando a escolha (`autoencoder.py:150-151`); o
gargalo não recebeu o mesmo tratamento.

**Decisão: não alterar isoladamente.** Trocar a ativação invalida os mesmos
números que a varredura de hiperparâmetros invalidaria, e o checkpoint não
versiona essa escolha (risco de `state_dict` incompatível ao recarregar o modelo
nas etapas seguintes). As duas mudanças **rodam juntas**, numa única rodada de
revalidação. Ver `docs/auditoria_pipeline_ml.md`, §23.

## 3. Limiar do Autoencoder

- **Limiar operacional = `score_threshold` do `score_method` vigente.** Na
  execução atual, o método operacional é o escore **localizado**; por isso o
  valor operacional não deve ser rotulado como MSE p99.
- **MSE p99 = referência do erro de reconstrução médio**, registrada em
  `mse_p99` / `limiar_p99`.
- **μ + 3σ = referência comparativa** (assume normalidade) — **nunca** o limiar
  em uso.
- **p95 = referência adicional.**
- O artefato `limiar.json` preserva campos legados (`threshold_method`,
  `limiar`, `k`, `k_localizado`) e acrescenta nomes inequívocos:
  `score_method`, `score_threshold`, `mse_p99`, `sigma_multiplier`, `top_k`,
  `threshold_fallback_percentile` e `threshold_effective_percentile`.

## 4. Validação sintética interna E2 (limiar congelado)

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

## 5. Divisão temporal com purga (anti-vazamento)

Janelas com 50% de sobreposição **não** podem ser divididas aleatoriamente
(janelas vizinhas quase idênticas vazariam entre treino/val/teste).
`src/ml/split_temporal.py::split_temporal_com_purga` faz blocos contíguos no
tempo com zona de **purga** na fronteira.

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

- **Manifesto por etapa** (`proveniencia.py`): `code_sha256`, `parameters`,
  hash dos artefatos upstream, outputs, `git_commit`. Estados **ready / stale /
  pending** (`estado_pipeline()`), exibidos no chat e na sidebar. Um artefato
  **sem manifesto = não verificado (pending)** — nunca válido só por existir.
  Nada é apagado automaticamente; recalcular é sob comando (com confirmação).
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
