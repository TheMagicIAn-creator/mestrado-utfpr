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

## 7. Pendente nesta rodada (honestidade de escopo)

Ainda **não** auditados linha a linha: `experimentos_artigos.py`,
`protocolos_artigos.py`, `modelos_anomalia.py`, `comparacao_literatura.py`
(experimentos de comparação com a literatura) e o PDF de Ibrahim (2022)
para o grounding do §2/§5. Próxima rodada.
