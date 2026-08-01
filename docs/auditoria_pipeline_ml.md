# Auditoria do pipeline de ML — cerne da dissertação

> Documento auditável e histórico. Registra a inspeção criteriosa dos
> módulos que geram os resultados centrais da dissertação (Autoencoder de
> normalidade + injeção FMECA + validação E2 + RUL/Weibull) e dos
> experimentos de comparação com a literatura. Cada achado cita
> `arquivo:linha`. Distingue **bug** (corrigir) de **resultado honesto**
> (documentar) e de **escolha não fundamentada** (amarrar à literatura).
>
> Data da 1ª rodada: 2026-07-23 · Autor da revisão: assistente, sob
> supervisão de Rodolfo Torres (UTFPR). Testes empíricos com a base
> Paderborn rodam na máquina do pesquisador (o dataset bruto não está na
> nuvem).

## 1. Sumário executivo

O pipeline está **estruturalmente correto e honesto** nas ressalvas
(censura, E1/E2, extrapolação do Weibull), mas tem **uma decisão central
que compromete o resultado** e **algumas escolhas não fundamentadas** que
a banca pode questionar.

O achado principal: o escore de anomalia é a **média do erro de
reconstrução sobre TODAS as ~109 features**. Falhas localizadas no espectro
(IGBT: harmônicos; Fusível: perda parcial de fase) mexem em poucas features
e são **diluídas** pela média — por isso o detector só enxerga bem o
Contator (transiente de banda larga) e gera censura de 70%/98% para as
outras duas. Não é bug de plotagem nem falta de feature: as features
existem (`features_ca.py` extrai harmônicos 3/5/7/11/13, THD e
desbalanceamento). O problema é **como o erro é agregado**.

Consequências já visíveis (todas explicadas por esse único ponto):
β(IGBT)=5,71 > β(Contator)=3,33 (artefato de censura), Fusível com n=1
evento, histogramas "quadrados".

## 2. Fundamentação do Autoencoder na literatura

| Item | Situação | Ação |
|------|----------|------|
| Modelagem de normalidade (treinar no saudável, detectar desvio) | **Fundamentada** — `autoencoder.py:6-14` cita Ibrahim (2022) e Ahirwar (2025); é a justificativa correta para ausência de dados de falha | Manter e citar na dissertação |
| Arquitetura específica (109→64→32→16, ReLU, dropout 0,2) | **Não fundamentada** — `autoencoder.py:96-150`: dimensões, profundidade e dropout são escolhas padrão, sem referência. Ibrahim (2022) usa **AE-LSTM**, não um AE denso sobre features handcrafted | Ler Ibrahim e (a) justificar o AE denso como alternativa deliberada, ou (b) citar uma referência de AE-sobre-features. **Ponta solta a fechar** |
| Limiar = percentil 99 do erro saudável | **Fundamentado e honesto** — `autoencoder.py:262-296`; controla FP≈1%, robusto a assimetria; μ+3σ mantido só como referência | Manter |
| Split temporal treino/calib/teste com purga | **Boa prática** — `autoencoder.py:501-522`; teste isolado nunca toca scaler/limiar | Manter |
| Escore = MSE médio sobre todas as features | **Escolha que quebra o resultado** (ver §3.1) | Corrigir |

> Pendência de leitura: o PDF de Ibrahim (2022) ainda não foi confrontado
> linha a linha com o código nesta rodada. É a próxima tarefa de grounding.

## 3. Achados por área

### 3.1 [ALTA] Escore de anomalia diluído — causa-raiz da cegueira
- **Onde:** `autoencoder.py:257`, `validacao.py:154`,
  `injecao_falhas.py:411`, `rul_weibull.py:144`.
- **O quê:** o escore é `((x - x_rec)**2).mean()` — média do erro sobre as
  ~109 features. Uma falha que perturba um subespaço pequeno (harmônicos do
  IGBT; uma fase do Fusível) é diluída pelas ~100 features intactas.
- **Evidência:** Fig. de injeção — mediana do erro do IGBT satura em ~1,7 e
  do Fusível em ~0,9, ambos **abaixo** do limiar 2,55; só o Contator (banda
  larga) cruza. Censura 0%/70%/98%.
- **Correção (fundamentável, sem forçar amplitude):** trocar/complementar o
  escore por um sensível a anomalia **localizada** — opções:
  (a) erro por-feature padronizado (z do resíduo por feature na
  calibração) e agregação por **máximo** ou **top-k**;
  (b) **Mahalanobis** no vetor de resíduos ou no espaço latente;
  (c) MSE ponderado por subespaço de falha.
  Recomprovar limiar p99 no novo escore. **Não** alterar amplitude de
  injeção (isso reintroduz detecção artificial — ver §5).
- **Risco:** muda todos os números; exige re-treino/recalibração e teste
  local. Implementar como escore alternativo mensurável lado a lado com o
  atual antes de substituir.
- **RESULTADO EMPÍRICO (2026-07-23, `src/ml/diagnostico_escore.py`, k=5,
  rodado no PC do pesquisador, E2, FP saudável 2,3% em AMBOS os escores):**
  a hipótese confirma-se. Taxa de detecção, MSE médio → localizado:
  IGBT @sev1,0 **34% → 86%**; IGBT @sev0,5 **2,3% → 39%**; Fusível @sev1,0
  **4,5% → 89%**; Contator atinge o alvo SMD em sev0,3 (vs 0,7 do MSE).
  A cegueira ao IGBT/Fusível era do ESCORE, não do método nem dos dados —
  sem alterar amplitude de injeção. Ressalvas: (i) o FP saiu 2,3% (não 1%)
  por causa do gargalo de 44 janelas (§3.5) — igual para os dois escores,
  logo a comparação é justa; (ii) o Fusível só cruza forte em severidade
  alta (12% de perda de fase é sutil de fato); (iii) o escore localizado
  e o k=5 precisam de fundamentação própria na dissertação (não trocar uma
  escolha "a esmo" por outra).

### 3.2 [MÉDIA] Hiperparâmetros e arquitetura sem lastro citável
- **Onde:** `autoencoder.py:96-108` (LATENTE=16, EPOCHS=150, BATCH=32,
  LR=1e-3, DROPOUT=0,2) e `:132-150` (topologia).
- **O quê:** escolhas "a esmo" — nenhuma referência as justifica. Encoder
  termina em `ReLU` no gargalo (`:139`), que zera componentes latentes
  negativas e pode limitar a representação.
- **Correção:** justificar por literatura ou por busca de hiperparâmetro
  documentada (ex.: varredura de latente ∈ {8,16,32} com a loss de
  calibração); considerar remover a ReLU final do encoder.

### 3.3 [ALTA→documentar] β(IGBT) > β(Contator) é artefato de censura
- **Onde:** `rul_weibull.py` — ajuste MLE censurado; regra de mínimo de
  eventos = 10.
- **O quê:** β(IGBT)=5,71 é ajustado em 13 eventos empilhados na borda da
  censura (η=143,6 **além** do horizonte 120 = extrapolação). Não é
  propriedade física; é seleção/censura. O código já marca "RUL paramétrica
  com alta incerteza", mas estampa β com o mesmo peso do ajuste bom.
- **Correção:** com a §3.1 resolvida, a censura do IGBT cai e o β
  estabiliza. Enquanto isso, marcar β como **não confiável** quando
  censura alta/poucos eventos (evita a leitura equivocada na banca).

### 3.4 [MÉDIA] Injeção do Fusível fraca demais para cruzar
- **Onde:** `injecao_falhas.py:212` — `i_a *= (1 − sev·0,12)` (máx 12% de
  uma fase, mesmo em severidade 1,0).
- **O quê:** perturbação sutil que o AE reconstrói bem → não cruza. É
  fisicamente conservador (bom), mas com o escore diluído (§3.1) resulta em
  n=1 evento.
- **Correção:** manter a amplitude (é honesta); a §3.1 é que deve tornar a
  detecção sensível. Reavaliar se "perda parcial" deveria também alterar
  desbalanceamento de forma mais marcante conforme a assinatura FMECA.

### 3.5 [MÉDIA] Gargalo de trajetórias independentes (dataset)
- **Onde:** `dados_avaliacao.py` / `split_temporal.py` — holdout rende
  "44 janelas; 43 elegíveis" (log da execução).
- **O quê:** o dataset tem ~235k amostras mas o pipeline usa **43
  trajetórias** independentes. Limita o n do IGBT e a estabilidade do
  Weibull.
- **Correção legítima (usa mais dado, sem vazamento):** extrair mais
  janelas independentes do holdout (janela/passo menores ou holdout maior),
  respeitando a purga. Melhora o n do IGBT sem forçar detecção. Não salva o
  Fusível (isso depende da §3.1).

## 4. Aplicabilidade com o dataset

O uso do Paderborn como **referência de normalidade** é coerente com a
natureza do dataset (inversor IGBT trifásico saudável). Ressalvas:
- F0 é estimado adaptativamente (`features_ca.py:158-201`) porque a
  fundamental varia com a velocidade do motor — correto e necessário.
- O gargalo real não é volume de dado, é **janelas independentes** (§3.5).
- A normalidade modelada descreve o regime saudável do ensaio, não vida
  útil de campo — as ressalvas E2 nos rodapés estão corretas e devem
  permanecer.

## 5. Pontas soltas e riscos para a banca

1. **Escore diluído** (§3.1) — sem corrigir, "o detector não detecta 2 das
   3 falhas" é a primeira pergunta da banca.
2. **AE denso vs AE-LSTM de Ibrahim** (§2) — reconciliar a fundamentação.
3. **β degenerado exibido como parâmetro** (§3.3).
4. **NÃO reintroduzir detecção artificial** — o commit `e7729f8` reduziu
   amplitudes de propósito. Aumentar amplitude para "consertar" a detecção
   é cientificamente indefensável. A correção certa é o escore (§3.1).

## 6. Plano de correção priorizado

| # | Item | Arquivo | Esforço | O que testar localmente |
|---|------|---------|---------|--------------------------|
| 1 | Escore sensível a anomalia localizada (top-k/Mahalanobis), medido lado a lado com o MSE médio | autoencoder.py + validacao.py + injecao_falhas.py + rul_weibull.py | Alto | rodar AE→injeção→validação; comparar AUC/censura por falha vs escore atual |
| 2 | Marcar β "não confiável" sob censura alta | rul_weibull.py | Baixo | `python src/ml/rul_weibull.py`; conferir título do painel |
| 3 | Mais janelas independentes do holdout (sem vazamento) | dados_avaliacao.py / split_temporal.py | Médio | rodar validação; conferir n do IGBT e RMSE-KM |
| 4 | Grounding do AE em Ibrahim (2022) — leitura confrontada | autoencoder.py (docstring/metodologia) | Médio | — (documental) |
| 5 | Remover ReLU final do encoder + justificar latente | autoencoder.py | Baixo | re-treino; comparar loss de calibração |

## 7. Pendente da 1ª rodada — RESOLVIDO na 2ª (§8–§10)

Auditados na 2ª rodada: `protocolos_artigos.py`, `modelos_anomalia.py` e o
PDF de Ibrahim (2022). Falta apenas varrer `experimentos_artigos.py` (1149
linhas — harness/plotagem dos experimentos) e `comparacao_literatura.py`
linha a linha; o essencial (protocolos + scorers + grounding) está coberto.

---

# 2ª rodada — experimentos de comparação e grounding do AE (Ibrahim 2022)

## 8. O que o Ibrahim (2022) realmente faz (leitura do PDF)

Fonte: `literatura/ml-preditivo/ibrahim_...2022.pdf` (Energies 2022, 15,
1082), confrontada com o código.

- **AE-LSTM é temporal.** A LSTM existe para "segurar a relação de sequência
  do vetor de entrada" e o AE-LSTM "aprende a correlação entre variáveis **e
  a correlação na série temporal**" (§3.1, pág. 5). A recorrência é sobre o
  **tempo**.
- **Dados do Ibrahim:** séries de potência CA/CC, irradiância e temperatura
  de duas usinas, a **intervalos de 15 min por 34 dias** (§4). O sinal de
  decisão é a **potência CA como série temporal**; as 13 anomalias são quedas
  de potência em dias específicos. NÃO é sinal elétrico a 10 kHz.
- **Hiperparâmetros por grid search** (§5, pág. 9): Ibrahim **otimiza** os
  hiperparâmetros de cada modelo; a contribuição declarada é justamente
  compará-los "with their optimized hyperparameters".
- **Erro de reconstrução:** `L(X,X̂)=‖X̂−X‖²` (eq. 3) — o MSE, igual ao do
  pipeline. Esse ponto **está** fundamentado.

## 9. Grounding do Autoencoder — veredito

| Afirmação da dissertação | Fundamentada por Ibrahim? |
|---|---|
| Modelagem de normalidade (treinar no saudável, sem rótulo de falha) | **Sim** — AE não-supervisionado, treino no normal |
| Erro de reconstrução (MSE) como escore de anomalia | **Sim** — eq. 3 do artigo |
| **Arquitetura DENSA** (109→64→32→16) sobre features handcrafted | **Não** — Ibrahim usa **AE-LSTM temporal**, não AE denso. É uma escolha diferente, hoje **sem justificativa citável** |
| Hiperparâmetros fixos sem busca | **Não** — Ibrahim faz grid search; o pipeline usa defaults |

**Conclusão (ponta solta a fechar na dissertação):** citar Ibrahim para o
*conceito* (AE de normalidade + MSE) é correto. Mas o *método específico* (AE
denso sobre features espectrais) **diverge** do AE-LSTM temporal do artigo.
É preciso, no texto: (a) justificar explicitamente a escolha do AE denso
sobre features FMECA como alternativa deliberada — p.ex. "a dinâmica
intra-janela já está codificada nas features espectrais, dispensando
recorrência" — e/ou (b) citar uma referência de AE-sobre-features; e (c)
documentar como os hiperparâmetros foram escolhidos (nem que seja uma
varredura simples de latente), para não ficarem "a esmo".

## 10. Achados nos experimentos de comparação

### 10.1 [ALTA] AE-LSTM do experimento não é fiel ao Ibrahim (eixo errado)
- **Onde:** `modelos_anomalia.py:58` — `seq = x.unsqueeze(-1)` → forma
  `(B, F, 1)`: a LSTM percorre o **eixo das features** (109 "passos"), não o
  tempo.
- **O quê:** as features (RMS, THD, harmônicos…) **não têm ordem temporal**;
  sua ordem é arbitrária. Rodar a LSTM sobre elas é usá-la como uma camada
  densa cara — **não** modela a dependência temporal que é a razão de existir
  do AE-LSTM no Ibrahim. O bloco de metodologia (`protocolos_artigos.py:459`)
  afirma "segue o artigo", o que hoje **não** se sustenta.
- **Correção:** para ser fiel, alimentar uma **sequência de janelas**
  (evolução temporal do vetor de features) e deixar a LSTM sobre o tempo; ou
  reetiquetar honestamente como "AE recorrente sobre features (adaptação),
  não o AE-LSTM temporal de Ibrahim".

### 10.2 [ALTA] Mesma diluição do escore no AE-LSTM do experimento
- **Onde:** `modelos_anomalia.py:77` e `:81` — `((rec-X)**2).mean(dim=1)`.
- **O quê:** o mesmo MSE médio sobre todas as features do §3.1 reaparece
  aqui. A causa-raiz é transversal ao pipeline e aos experimentos.

### 10.3 [documentar] O baseline ingênuo é MAIS sensível que o AE proposto
- **Onde:** `protocolos_artigos.py:349` — Francisti usa
  `score = max(|z|)` sobre as features (Shewhart 3σ).
- **O quê:** o baseline "burro" agrega por **máximo** — exatamente o que
  detecta anomalia **localizada**. Ou seja: contra falha localizada
  (IGBT/Fusível), o Z-score máx-|z| tende a **vencer** o AE de MSE médio.
  Isso **reforça** a correção §3.1: se o Autoencoder não vence uma carta de
  controle 3σ na detecção localizada, o problema é o escore médio, não o
  método. (Alinhado ao papel do Francisti no CLAUDE.md: baseline que o AE
  precisa vencer.)

### 10.4 [documentar] Assimetria E1 (features) × E2 (sinal) — não comparar por F1
- **Onde:** `protocolos_artigos.py:68-104` (injeção E1 no espaço de features,
  em unidades de σ: IGBT soma 1,5–3,0σ aos harmônicos) vs
  `injecao_falhas.py` (injeção E2 no sinal, fisicamente calibrada e fraca).
- **O quê:** nos EXPERIMENTOS a injeção é forte (features movem vários σ) →
  os detectores funcionam. No PIPELINE principal a injeção é fraca (E2) e o
  escore é médio → cegueira. **Não são comparáveis por F1** (só por AUC), e a
  cegueira do pipeline é específica do par **E2 + MSE médio** — o código já
  ressalva isso, mas o ponto deve ficar explícito na dissertação.

### 10.5 [positivo] Protocolos de comparação são metodologicamente sólidos
- Split temporal com purga (`protocolos_artigos.py:217-221`); scaler só no
  treino; cada método com regra de decisão **a priori** que nunca vê rótulos
  do teste (Shewhart 3σ; IF contaminação a priori; AE-LSTM p99 congelado em
  fatia de calibração temporal, `:428-437`); recall reportado **por família
  de falha**; blocos de "fidelidade" honestos sobre cada adaptação. Manter.

## 11. Plano de correção priorizado (consolidado, 2 rodadas)

| # | Item | Arquivo | Esforço |
|---|------|---------|---------|
| 1 | Escore sensível a falha localizada (top-k/máx-z/Mahalanobis), medido lado a lado com o MSE médio | autoencoder.py, validacao.py, injecao_falhas.py, rul_weibull.py, modelos_anomalia.py | Alto |
| 2 | AE-LSTM fiel ao Ibrahim (LSTM sobre o TEMPO) OU reetiquetar honestamente | modelos_anomalia.py, protocolos_artigos.py | Médio |
| 3 | Justificar/citar o AE denso e documentar escolha de hiperparâmetros | autoencoder.py + dissertação | Médio |
| 4 | Marcar β "não confiável" sob censura alta | rul_weibull.py | Baixo |
| 5 | Mais janelas independentes do holdout (sem vazamento) | dados_avaliacao.py, split_temporal.py | Médio |
| 6 | Remover ReLU final do encoder + justificar latente | autoencoder.py | Baixo |

## 12. Pendente (honestidade de escopo)

Falta varrer `experimentos_artigos.py` (harness/plotagem/AUC dos
experimentos) e `comparacao_literatura.py` linha a linha. O núcleo
metodológico (autoencoder, features, injeção, validação, RUL, protocolos,
scorers e o grounding no Ibrahim) está auditado.

---

# 3ª rodada — preparação do terreno para promover o escore localizado

## 13. Fundamentação do escore localizado (sem "a esmo")

O escore localizado (`diagnostico_escore.py`) decompõe-se em três peças, cada
uma com lastro na literatura indexada:

1. **Erro de reconstrução como sinal de anomalia** — Ibrahim (2022), eq. 3:
   `L(X,X̂)=‖X̂−X‖²`. Fundamentado.
2. **Padronização por-feature do resíduo** (`z_j = (|r_j|−μ_j)/σ_j`) —
   Francisti (2025): *"Z-scores were calculated to detect statistical
   anomalies, defined here as deviations exceeding ±3 standard deviations
   from the mean"*. É controle estatístico de processo (Shewhart/Z-score),
   aqui aplicado ao **resíduo do Autoencoder** em vez do sinal bruto.
   Fundamentado.
3. **Agregação pelos top-k mais desviantes** — generalização robusta da regra
   de Shewhart (que alarma pelo feature MAIS desviante, k=1) para o
   subconjunto onde a falha localizada se concentra. O princípio "a anomalia
   vive num subconjunto de variáveis" é o das cartas multivariadas de SPC e
   da análise de contribuição por feature (Narayanan, 2023 — XAI de falha).

**Justificativa do k:** k deve refletir a cardinalidade típica da assinatura
de falha. Pelas assinaturas FMECA (`protocolos_artigos.py:68-104`), uma falha
toca ~3–9 features (ex.: IGBT = `harm_5/7/11` × 3 fases). k=5 é um piso
razoável; **recomenda-se justificar com uma varredura** (`diagnostico_escore.py`
aceita `--k`) e reportar a escolha — nunca fixar sem evidência.

**Relação com o baseline Francisti (papel no CLAUDE.md):** o escore localizado
é, em essência, a regra de Shewhart do Francisti aplicada aos resíduos do AE e
suavizada por top-k. Ou seja, o AE + escore localizado **incorpora a força do
baseline** que ele precisava vencer — o que fecha o argumento em vez de deixá-lo
como concorrente vencedor.

## 14. Reavaliação do gargalo de janelas (§3.5)

Com o split (`split_temporal.py`) e a extração (`dados_avaliacao.py`) na mão, o
gargalo é **intrínseco ao dataset**, não um bug:

- O Paderborn tem ~23 s de sinal a 10 kHz → ~229 janelas **não-sobrepostas**
  totais → ~44 no bloco de teste (20%). Não há "muitos dados" em janelas
  independentes; há muitas AMOSTRAS de poucos segundos.
- **O #1 já resolve o downstream:** com detecção de 86%/89% (IGBT/Fusível), as
  44 janelas geram **~38/39 eventos** (vs 13/1 hoje) — suficiente para o
  Weibull **deixar de degenerar**. Não é preciso mais janelas para consertar o
  β; basta o escore.
- **O FP=2,3% do diagnóstico é artefato** de calibrar o limiar nas 44 janelas
  do holdout. No pipeline real, o limiar é calibrado no **bloco de calibração**
  (20% das features, com sobreposição ≈ 91 janelas) → granularidade de FP ≈ 1%.
  Normaliza ao promover.
- **Opção documentada (não hack):** para CIs mais estreitos, dá para reduzir o
  treino (ex.: 50/20/30) e ganhar ~68 janelas de teste, ao custo de menos dado
  de treino — trade-off a decidir, não melhoria gratuita. Janela sobreposta no
  teste infla n artificialmente e fica desaconselhada.

**Conclusão:** o terreno está pronto. A fundamentação existe (§13) e o gargalo
de janelas não bloqueia a correção (§14). Para promover o escore ao pipeline
operacional falta apenas a decisão + o aval da orientadora.

---

# 4ª rodada — escore localizado PROMOVIDO ao pipeline operacional

## 15. O que mudou (implementação)

Fonte única do escore: **`src/ml/escore_anomalia.py`** (módulo folha, com teste
`tests/test_escore_anomalia.py`). O pipeline passa a usar o escore **localizado**
como operacional, com o MSE médio ainda calibrado e reportado.

- `autoencoder.py`: computa a **régua por-feature** (μ/σ do |resíduo| saudável)
  na CALIBRAÇÃO, calibra **os dois** limiares (MSE p99 e localizado p99), salva
  `estatistica_residuo.npz` e grava em `limiar.json` o método operacional
  (`metodo_escore`), `limiar_mse`, `limiar_localizado` e `k_localizado`. O
  `"limiar"` passa a ser o **operacional**.
- `injecao_falhas.py`, `validacao.py`, `rul_weibull.py`: carregam a régua + o
  método e computam **o mesmo escore** que definiu o limiar (via
  `escore_anomalia.pontuar`). O TTF do Weibull agora cruza o limiar do escore
  localizado.
- **Interruptor de segurança / reversão:** `AL_IADO_ESCORE_ANOMALIA=mse`
  reproduz EXATAMENTE o pipeline antigo; `AL_IADO_ESCORE_K` ajusta o k. Sem a
  régua (artefato antigo), tudo cai para MSE — nada quebra por artefato
  faltando.
- **Reprodução para a dissertação:** rodar o pipeline com o padrão (localizado)
  e com `AL_IADO_ESCORE_ANOMALIA=mse` produz o antes/depois auditável.

Ordem de execução local (regenera todos os artefatos e gráficos):
`python src/ml/autoencoder.py` → `injecao_falhas.py` → `validacao.py` →
`rul_weibull.py`. O `diagnostico_escore.py` segue como comparação lado a lado.

Pendências (§11) que restam: #2 (AE-LSTM fiel ao Ibrahim), #3 (texto do
grounding do AE denso), #6 (ReLU do encoder).

---

# 5ª rodada — validação empírica REAL (dataset completo, PC do pesquisador)

## 16. Resultado: a hipótese do escore localizado se confirma no dado real

Pipeline rodado com `metodo_escore=localizado, k=5` no dataset completo de
Paderborn (não mais o diagnóstico isolado). Comparação com o estado MSE
anterior:

| Falha | Métrica | MSE (antes) | Localizado (agora) |
|---|---|---|---|
| **IGBT** | censura | 70% | **13%** |
| | eventos observados | 13 | **34** |
| | β (Weibull) | 5,71 (artefato, η>horizonte) | **2,74** (converge, η=83,4 dentro do horizonte) |
| | detecção @sev1,0 | 34% | **86,4%** |
| | recall (validação) @sev1,0 | 0,35 | **0,875** |
| **Fusível AC** | censura | 98% | **0%** |
| | eventos observados | 1 | **39** |
| | β (Weibull) | não estimável | **6,33** (converge, η=92,9) |
| | detecção @sev1,0 | 4,5% | **100%** |
| | recall (validação) @sev1,0 | 0,05 | **1,00** |
| **Contator AC** | β / η | 3,33 / 64,5 | 4,91 / 34,1 (novo escore, novo limiar — não comparável 1:1) |

As três falhas agora têm Weibull **convergente, sem extrapolação além do
horizonte, e sem a marca "não confiável"**. Os números de detecção do IGBT
(86,4%) batem quase exatamente com a projeção do diagnóstico da 4ª rodada
(86%) — forte evidência de que a correção é robusta, não um artefato do
diagnóstico isolado. **A causa-raiz identificada em §3.1 está confirmada e
corrigida com dado real.**

## 17. Achado NOVO — três gráficos ficaram OBSOLETOS no processo (ação corretiva)

Verificação por hash (SHA-256) contra o commit anterior a qualquer correção
desta auditoria (`af9338e`):

| Arquivo | Estado |
|---|---|
| `weibull_ttf.png`, `weibull_confiabilidade.png` | **Hash diferente do antigo — genuinamente regenerados** com o escore localizado |
| `limiar.json`, `weibull_results.json`, `injecao_falhas_report.json`, `validacao_tabela.csv` | Regenerados; valores conferidos manualmente (tabela acima) |
| `curva_treino.png`, `distribuicao_erro.png`, `erro_temporal.png` | **Hash IDÊNTICO ao commit `af9338e`** (anterior a todas as correções desta auditoria) — **nunca foram regenerados** nesta rodada |

> **Correção (2026-07-30) — este achado estava errado para 1 dos 3, e a causa
> real era outra.**
>
> Reconferindo por `git hash-object`: `distribuicao_erro.png` **é diferente** de
> `af9338e` — foi regenerado em `1f29ebd` ("resultados: regenera graficos do
> autoencoder", 23/07). Só `curva_treino.png` e `erro_temporal.png` estavam
> idênticos.
>
> E a leitura de "ficaram para trás num merge" provavelmente está errada. Os três
> plotam **MSE**, não o escore localizado. O `limiar_mse` não mudou (2,5454 em
> `af9338e` e hoje) e o `diagnostico_autoencoder.npz` é byte-idêntico. Com
> `SEED=42`, é esperado que figuras de dados que não mudaram saiam iguais. O que
> mudou em `distribuicao_erro.png` foi o **código de plotagem** (14 bins), não o
> dado.
>
> **O que de fato estava quebrado era a ferramenta de conserto.**
> `regenerar_graficos_autoencoder()` — escrita justamente para este caso, sem
> nenhum chamador e sem teste — repassava o dicionário cru de `limiar.json` aos
> plots. Mas o campo `limiar` ali passou a ser o **operacional do escore
> localizado (7,83)**, enquanto os gráficos são de MSE (p99 ≈ 2,55). Executá-la
> desenharia a linha de limiar acima de quase todos os pontos e reportaria ~4
> alarmes em vez de 14 — **figura errada, não figura atualizada**. Quem tentasse
> fechar este item pela via óbvia teria piorado o artefato.
>
> Correções aplicadas:
> - `_info_em_escala_mse()` converte o JSON salvo para a escala de MSE antes de
>   plotar (e recalcula `fp_test_pct` na mesma escala);
> - os plots foram movidos para `src/ml/graficos_autoencoder.py`, que **não
>   importa `torch`** — antes, regenerar uma figura a partir de artefatos exigia
>   a stack de ML inteira, inclusive na nuvem em modo consulta;
> - `tests/test_graficos_autoencoder.py` (8 testes) fixa o comportamento; um
>   deles falha se o limiar operacional voltar a influenciar um gráfico de MSE.
>
> As três figuras foram regeneradas em 2026-07-30 **sem dataset e sem retreino**,
> só a partir de `diagnostico_autoencoder.npz` + `limiar.json`.

Interpretação: o `injecao_falhas.py`/`validacao.py`/`rul_weibull.py` rodaram
com o modelo e o limiar corretos (por isso os números batem), mas as três
figuras específicas do **próprio `autoencoder.py`** ficaram de uma execução
anterior — provavelmente perdidas na resolução de um merge local. Isso
**não invalida os resultados numéricos** (que vêm de outros artefatos), mas
as três imagens ainda mostram o estilo antigo (histograma em degraus, sem a
suavização da correção de plotagem).

**Ação pendente:** rodar `python src/ml/autoencoder.py` novamente (idealmente
o pipeline completo, na ordem, para garantir consistência total) e
subir os resultados de novo. Como a semente é fixa (`SEED=42`) e os dados
não mudaram, o modelo/limiar devem sair numericamente idênticos — só as três
imagens devem mudar de estilo.

## 18. Achado NOVO — ressalva honesta: taxa de falso positivo subiu no teste isolado

| | FP calibração | FP teste isolado |
|---|---|---|
| MSE (antes) | 1,1% | **1,1%** |
| Localizado (agora) | 1,1% | **6,8%** |

> **Correção (2026-07-30).** O 6,8% acima é da rodada de **23/07**, com percentil
> **fixo** em p99. Ele não é mais o valor vigente. Rastreando `limiar.json` pelo
> histórico:
>
> | Rodada | Percentil | FP no teste isolado |
> |---|---|---|
> | 21/07 (`af9338e`) | p99 fixo | 1,1% |
> | 23/07 | p99 fixo | 6,8% |
> | **24–27/07 (vigente)** | **p99,9 automático** | **10,2%** |
>
> Consequência para o plano de mitigação: **"usar um percentil mais conservador"
> já foi tentado**. A auto-calibração (`limiar_por_fp_alvo`) entrou, escolheu o
> percentil mais alto da escada (p99,9) e o FP medido *subiu*. É opção testada e
> insuficiente, não opção em aberto. Restam aumentar o bloco de calibração — que
> apenas redistribui as 457 janelas, sem criar dados — e suavizar a estimativa do
> limiar por bootstrap.
>
> Com 88 janelas de teste, cada falso alarme vale 1,14 ponto percentual: os 10,2%
> são 9 alarmes. A amostra é pequena demais para o número ser preciso, o que é
> parte do problema.

O limiar do escore localizado generaliza pior da calibração (91 janelas) para
o teste isolado (88 janelas) do que o MSE médio generalizava. Hipótese mais
provável: o top-k de resíduos padronizados é uma estatística de cauda (ordem
estatística), mais sensível a ruído amostral do que uma média sobre 109
features (que se beneficia de suavização tipo lei dos grandes números). Com
poucas janelas de calibração, o p99 dessa estatística é uma estimativa mais
instável.

**Isto é um trade-off real, não um detalhe cosmético:** o ganho de recall no
IGBT/Fusível veio acompanhado de uma taxa de falso alarme operacional maior
que o 1% de projeto. Precisa constar na dissertação como limitação conhecida.
Mitigações a avaliar (não implementadas ainda): (a) aumentar o bloco de
calibração; (b) usar percentil mais conservador (ex. p99,5) para o escore
localizado; (c) suavizar a estimativa do limiar com bootstrap.

## 19. Plano atualizado

| # | Item | Status |
|---|---|---|
| Regenerar `curva_treino.png`/`distribuicao_erro.png`/`erro_temporal.png` | ✅ **feito em 2026-07-30** — ver a correção abaixo; o "feito" anterior era falso |
| Investigar/mitigar FP no teste isolado | **Pendente** — hoje 10,2%, não 6,8% (ver §17-A) |
| #2 AE-LSTM fiel ao Ibrahim | ✅ **feito** (§20) |
| #3 grounding do AE denso (texto) | Pendente |
| #6 ReLU do encoder | Pendente |

---

# 6ª rodada — AE-LSTM do Ibrahim corrigido (concorrente temporal fiel)

## 20. Decisão de rumo e correção do AE-LSTM (#2)

**Decisão metodológica (Rodolfo, aprovada):** o método PRINCIPAL continua o
Autoencoder **denso sobre features FMECA + escore localizado** (nosso,
validado na 5ª rodada). O AE-LSTM do Ibrahim entra **corrigido** como o
concorrente forte na comparação — não substitui o cerne. Isso "alinha a
pesquisa com o Ibrahim" (traz a arquitetura temporal e o benchmark) sem
reescrever o núcleo a três anos da defesa.

**Correção da infidelidade (§10.1):** `modelos_anomalia._score_ae_lstm`
rodava a LSTM sobre o **eixo das features** (`x.unsqueeze(-1)` → (B, F, 1)),
ordem arbitrária — uma densa disfarçada. Reescrito para percorrer o **TEMPO**:
uma sequência de `SEQ_LEN` janelas consecutivas (a "correlação na série
temporal" do Ibrahim). Helpers `sequencias_deslizantes` (treino) e
`sequencias_com_contexto` (teste) em `modelos_anomalia.py`;
`protocolos_artigos.protocolo_ibrahim` monta as sequências e ajusta o AE-LSTM
uma vez, com limiar p99 congelado numa fatia de calibração temporal.

**Escolha metodológica documentada (a revisar com a orientadora):** como a
injeção do protocolo é PONTUAL (uma janela por vez, para manter o MESMO banco
de teste dos outros modelos → comparável por AUC), cada item é pontuado como
"a janela ATUAL dado o histórico normal precedente" — o escore é o erro de
reconstrução no **último passo** da sequência. Alternativa possível (injetar
um trecho temporal sustentado, mais próximo do cenário do Ibrahim) muda o
ground truth e quebra a comparabilidade com IF/Z-score; por isso ficou de
fora. `SEQ_LEN=8` por padrão (env `AL_IADO_AELSTM_SEQ_LEN`), a justificar por
varredura.

**Testes:** `tests/test_ae_lstm_temporal.py` valida a mecânica (forma das
sequências, contexto, e que anomalia no último passo eleva o escore). Os
números finais (AUC do AE-LSTM vs. método proposto) saem no rerun local do
experimento do Ibrahim, com dataset — não puderam ser medidos na nuvem.

---

# 7ª rodada — pendências fechadas + revisão de código

## 21. #3 — Fundamentação do AE denso (texto para a dissertação)

Justificativa a incorporar no capítulo de metodologia, fechando a ponta solta
do §9 (o AE denso diverge do AE-LSTM temporal do Ibrahim):

> A modelagem de normalidade por Autoencoder e o uso do erro de reconstrução
> como escore de anomalia seguem Ibrahim et al. (2022). Optou-se, porém, por um
> **Autoencoder denso sobre features espectrais** derivadas da FMECA (RMS, THD,
> harmônicos 5/7/11/13, desbalanceamento), e não pelo AE-LSTM temporal do
> artigo, por três razões: (i) a **dinâmica intra-janela** relevante às falhas
> CA já está condensada nas features espectrais de cada janela de ~102 ms,
> tornando a recorrência temporal redundante para o alvo de detecção; (ii) o
> espaço de features **nomeadas** habilita o escore localizado (top-k dos
> resíduos padronizados por feature), que é **interpretável via FMECA** — o
> desvio pode ser atribuído a "harmônico 5 da fase A", ligando a detecção ao
> modo de falha; um AE-LSTM sobre sinal bruto perderia essa rastreabilidade;
> (iii) o AE-LSTM temporal fiel ao artigo é mantido como **concorrente na
> comparação** (protocolo Ibrahim), de modo que a escolha do AE denso é
> justificada empiricamente pela comparação por AUC, não por conveniência.

Hiperparâmetros (latente=16, épocas=150, lr=1e-3, dropout=0,2): ainda são
defaults. Recomendação para não ficarem "a esmo": reportar uma **varredura
simples de latente ∈ {8,16,32}** pela loss de calibração, e citar o resultado.
Não implementado (exige rerun local); é trabalho de redação + uma execução.

## 22. #18 — Mitigação do FP=6,8% (levers implementados)

O FP alto no teste isolado é do escore localizado ser uma **estatística de
cauda** (top-k). Dois levers agora **configuráveis por env**, para o
pesquisador calibrar no dado real (não dá para escolher o valor sem o dataset):

- `AL_IADO_ESCORE_PERCENTIL` (padrão 99) — subir para 99,5/99,9 eleva o limiar
  e **baixa o FP**, ao custo de recall. Gravado em `limiar.json`
  (`percentil_limiar`) para auditoria.
- `AL_IADO_ESCORE_K` (padrão 5) — k maior = agregação mais suave (menos
  sensível a ruído de cauda) = **menor variância de FP**, ao custo de
  localização. k pequeno detecta falha mais concentrada.

**Recomendação:** varrer `k ∈ {5,10,15}` × `percentil ∈ {99; 99,5}` no dado
real e escolher o par que traz o FP do teste para ~1–2% mantendo o recall do
IGBT/Fusível. É o item a levar com números para a orientadora.

## 23. #6 — ReLU no gargalo do encoder: por que NÃO mexer agora

O encoder termina em `ReLU` no espaço latente (`autoencoder.py`), o que zera
componentes negativas — subótimo em teoria. **Decisão: não alterar agora**, por
duas razões honestas: (i) mudaria a arquitetura logo após a validação empírica
da 5ª rodada, **invalidando** os números recém-confirmados; (ii) o checkpoint
não registra essa escolha, então trocar sem versionar a flag arrisca
**incompatibilidade de `state_dict`** ao carregar o modelo nas etapas seguintes.
Recomendação: fazer junto da **varredura de hiperparâmetros (#3)**, numa rodada
deliberada de re-treino, não isoladamente. Fica registrado como melhoria
pendente consciente, não esquecida.

> **Amarração explícita (2026-08-01).** As duas mudanças — trocar a ativação do
> gargalo e varrer `latente ∈ {8,16,32}` — **invalidam exatamente os mesmos
> números**: limiar, validação E2, SMD, TTF do Weibull e a comparação com o
> Ibrahim. Rodá-las em momentos separados custaria **duas** revalidações
> completas do pipeline e produziria uma janela em que os artefatos publicados
> correspondem a uma arquitetura que já não é a do código.
>
> Portanto elas formam **uma única tarefa**, não duas pendências independentes.
> O critério de conclusão do item #6 (ReLU) é o mesmo do #3 (hiperparâmetros):
> uma rodada de re-treino que produza, de uma vez, o modelo com gargalo linear e
> o latente escolhido por evidência.
>
> **Argumento adicional a favor da troca**, que não estava registrado: a saída do
> decoder **já é linear**, com comentário justificando a escolha em
> `autoencoder.py:150-151` ("Saída linear — sem ativação, features
> normalizadas"). O mesmo raciocínio se aplica ao gargalo, que também produz um
> vetor de valores normalizados sem sinal definido — ele só não recebeu o mesmo
> tratamento. É inconsistência interna, não decisão deliberada.
>
> Enquanto a rodada não acontece, `docs/metodologia_ml.md` §2 apresenta os
> hiperparâmetros como **defaults**, não como resultado de busca.

## 24. Revisão de código das mudanças da sessão (/code-review)

Revisão de correção dos módulos alterados/criados nesta sessão
(`escore_anomalia.py`, `modelos_anomalia.py`, `protocolos_artigos.py`,
`comparacao_literatura.py`, `autoencoder.py`, `rul_weibull.py`):

- **Sem bug de correção encontrado.** Escore central com fallback seguro
  (régua ausente → MSE); guardas de `k>n_features` e `sigma=0`; AE-LSTM temporal
  com contexto correto (último passo = item; padding no início; fit único +
  score de calibração/teste juntos); reconstrução de `Xn_te/Xa_te` coerente com
  o layout `[normais | anômalas]` do protocolo.
- **Cobertura de teste:** `tests/test_escore_anomalia.py` (6) e
  `tests/test_ae_lstm_temporal.py` (5) — verdes na parte numpy; a parte torch
  roda na máquina do pesquisador.
- **Risco residual (declarado):** os números finais do AE-LSTM temporal e o FP
  mitigado **não foram medidos na nuvem** (sem dataset). Exigem rerun local.
  Nenhum resultado deste relatório afirma desempenho não medido.

---

# 8ª rodada — FP auto-calibrado + experimento Ibrahim desatualizado

## 25. FP do escore localizado agora se AUTO-CALIBRA (sem ajuste manual)

Substituição do percentil fixo (§18/§22) por **auto-calibração**, atendendo ao
pedido de "sem alterar manualmente":

- Por PADRÃO, o `autoencoder.py` divide o bloco de calibração em sub-fit (80%)
  e sub-val (20%). A régua e os candidatos a limiar vêm do sub-fit; o percentil
  escolhido (`escore_anomalia.limiar_por_fp_alvo`) é o **menor** cujo FP no
  sub-val (não visto) fica ≤ `FP_ALVO` (padrão 1%). Assim o limiar generaliza
  melhor para dado saudável novo, sem env, sem sweep.
- Fallback seguro: com < 40 janelas de calibração, usa o percentil fixo (p99).
- Override manual continua possível (`AL_IADO_ESCORE_PERCENTIL`), mas não é
  necessário. `limiar.json` grava `percentil_limiar` (o escolhido) e
  `percentil_auto` (se foi automático).
- Testado: `test_limiar_por_fp_alvo_*` (sobe o percentil quando o p99 estoura;
  respeita o alvo). O FP real no teste isolado só se confirma no rerun local.

## 26. Experimento do Ibrahim rodado está DESATUALIZADO (rerun necessário)

`resultados/experimentos/ibrahim/` foi gerado pelo **AE-LSTM ANTIGO** (infiel,
LSTM sobre o eixo das features): o `metricas.csv` traz
`threshold_source=p99_erro_em_calibracao_temporal` (string anterior ao #52),
não a nova `p99_erro_seq_temporal_calibracao`. Os números (AUC AE-LSTM=0,659;
IF=0,589) são do modelo antigo. **Ação:** `git pull` + re-rodar o experimento
do Ibrahim para obter o AE-LSTM temporal fiel, que é a comparação válida.

> **Correção (2026-07-30) — esta ação já foi executada, e depois superada.**
>
> 1. O rerun **ocorreu em 24/07**: o `metricas.csv` daquela data já traz
>    `threshold_source=p99_erro_seq_temporal_calibracao` (a string nova) e
>    **AUC AE-LSTM = 0,588**, não 0,659.
> 2. Em 27/07 (`9fe0322`) a pasta `resultados/experimentos/` inteira foi
>    **deletada**, quando os macro-códigos substituíram o framework por artigo.
>
> Portanto o **0,659 não deve ser citado em lugar nenhum** — nem como resultado
> antigo, porque o valor correto daquele caminho já era 0,588.
>
> **Não confundir os dois números.** 0,588 e 0,909 medem coisas diferentes:
>
> | | Framework por artigo | Macro-código (vigente) |
> |---|---|---|
> | Injeção | espaço de features (**E1**) | sinal bruto (**E2**) |
> | Limiar | p99 congelado | auto-calibrado, FP ≈ 1% |
> | Teste | balanceado | por severidade |
> | Métrica | F1, matriz de confusão | AUC, SMD@FPR=10% |
> | AE-LSTM | 0,588 (apagado) | **0,909** (IGBT) |
>
> Os dois caminhos chamam o **mesmo** modelo (`src/ml/modelos_anomalia.py`), com
> os mesmos hiperparâmetros. A diferença é inteiramente de protocolo — por isso
> os valores não vão na mesma tabela.
>
> Para eliminar o risco de recriar artefatos E1 conflitantes sem perceber, o
> framework por artigo foi **aposentado do roteador do agente** (os módulos
> seguem no repositório, preservando o histórico). Os macro-códigos são a fonte
> única de resultado de anomalia.

---

# 9ª rodada — macro-códigos (substituem o framework de experimentos)

## 27. Por que trocar o framework de experimentos por dois macro-códigos

Problemas concretos do framework antigo (`experimentos_artigos` +
`protocolos_artigos`), levantados pelo pesquisador e confirmados:

- `metricas.csv` com **33 colunas** — ilegível como tabela de comparação.
- **Matriz de confusão enganosa**: com limiar congelado a ~1% de prevalência e
  teste balanceado (50%), o AE-LSTM previu 2 anomalias em 364 → matriz com uma
  coluna cheia e outra vazia, recall 0,5%. É artefato do protocolo, não do
  método, mas visualmente sugere "modelo quebrado".
- Cada artigo com protocolo próprio → **F1/matrizes não comparáveis** entre si.

## 28. Nova arquitetura (decidida com o pesquisador)

Dois scripts legíveis de ponta a ponta, com **avaliação e saída idênticas**:

| | `macro_proposto.py` | `macro_ibrahim.py` |
|---|---|---|
| Modelo | AE denso + escore localizado (nosso) | AE-LSTM temporal (Ibrahim 2022) |
| Features | espectrais FMECA | **as mesmas** |
| Avaliação | E2: injeção FMECA no sinal por severidade | **a mesma** |
| Limiar | auto-calibrado ~1% FP | **o mesmo critério** |
| Saída | tabela 5 colunas + gráfico | **o mesmo módulo** |

Decisões do pesquisador: (1) avaliação nossa (E2) para os dois; (2) só o
AE-LSTM no macro do Ibrahim (Isolation Forest fora); (3) scripts **importam e
orquestram** (não duplicam lógica) — legíveis para citar trechos na dissertação.

**Contrato:** cada macro fornece um *scorer* `callable(list[DataFrame]) ->
np.ndarray`; `macro_comum.avaliar_deteccao` faz o resto. Isso garante
comparação maçã-com-maçã por construção.

`macro_comparar.py` roda os dois e emite UMA tabela + UM gráfico sobreposto
(AUC na legenda). Saídas em `resultados/macro/`.

**Honestidade declarada no cabeçalho do `macro_ibrahim.py`:** o artigo usa
séries de potência de usinas (15 min × 34 dias) com anomalias reais; aqui o
MÉTODO dele é aplicado ao NOSSO problema (sinal CA + injeção FMECA), para
comparabilidade. O que muda entre os macros é a arquitetura (LSTM temporal vs.
densa) e o escore (MSE do artigo vs. localizado top-k nosso).

**Testes:** `tests/test_macro_comum.py` (3 casos: tabela de 5 colunas,
md/csv/json/png gerados, gráfico não vazio) — verdes. Os números reais exigem
rerun local com dataset.

**Status do framework antigo:** mantido no repositório por ora (não removido),
mas os macro-códigos passam a ser o caminho recomendado para a comparação da
dissertação.
