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
grounding do AE denso), #4 (β "não confiável" — deve melhorar sozinho com a
censura menor), #6 (ReLU do encoder).
