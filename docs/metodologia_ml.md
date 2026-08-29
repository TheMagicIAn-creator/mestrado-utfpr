# Metodologia canônica de Machine Learning

## 1. Pergunta experimental

Comparar, sob o mesmo protocolo, um Autoencoder Denso e um AE-LSTM para
detecção de anomalias em sinais elétricos do GPVS-Faults. O dataset é a base
experimental interna da comparação, não uma família autônoma de resultados.
A confiabilidade física bibliográfica permanece metodologicamente separada.

## 2. Dataset e features

O único dataset ativo é o GPVS-Faults, DOI `10.17632/n76t439f65.1`:

- F0L/F0M: condição saudável;
- F1L-F7M: 14 ensaios de falha em dois regimes;
- 24 features: estatísticas CC, RMS/THD trifásicos, desbalanceamentos e
  estatísticas de potência CA/CC.

A taxa contratual é 10 kHz e cada janela tem 200 amostras, um ciclo nominal de
50 Hz. A qualidade da coluna `Time` é validada e a divergência do manual fica
registrada. Janelas não se sobrepõem.

## 3. Split saudável e normalização

F0L e F0M são divididos separadamente em blocos temporais de 50% treino, 15%
validação, 15% calibração e 20% teste. Duas janelas são purgadas em cada
fronteira. Sequências AE-LSTM são formadas somente dentro de cada bloco.

O baseline usa mediana e IQR. O IQR recebe piso proporcional ao treino para não
amplificar feature quase constante. O `RobustScaler` é ajustado apenas no
treino.

Nos ensaios F1-F7, a primeira metade pré-falha fornece normalização de
comissionamento e a segunda metade pré-falha permanece para especificidade.
Pesos, scaler e limiar não são reajustados.

## 4. Modelos e treino

- Denso: `24-16-8-16-24`.
- AE-LSTM: sequência 8, oculto 32 e latente 8.

Os dois recebem o mesmo orçamento de épocas, early stopping, sementes e
pré-processamento compatível. A semente 42 é a execução de referência e cinco
sementes medem estabilidade. O escore é a média dos cinco maiores erros
quadráticos por feature; no AE-LSTM, somente o último passo temporal recebe o
top-k. Cada modelo recebe seu próprio p99,9 solicitado, calculado na calibração
saudável pelo método `higher`. A saída registra o order statistic selecionado,
o percentil empírico efetivo, o tamanho da calibração e sua resolução.

Nenhum desempenho em F1-F7 participa da seleção de arquitetura, semente ou
limiar.

## 5. E3 experimental

Os modelos congelados avaliam todos os 14 ensaios. Recall, F1 e Precision são
as métricas principais. ROC-AUC e PR-AUC são complementares; especificidade,
acurácia balanceada, MCC e falso positivo saudável permanecem auxiliares. Se
nenhum positivo for previsto, Precision é `N/A`, e não zero. A unidade
inferencial é o ensaio; IC95% macro usam bootstrap de 20.000 reamostragens dos
ensaios com valor finito para a métrica.

Resultados negativos e heterogeneidade por falha permanecem publicados.
E3 significa bancada, não validação de campo.

## 6. FMECA e manutenção

A FMECA consolidada preserva funções, modos de falha, índices S, O, D e NPR de
Contator AC, IGBT e Fusível AC. Ela serve para priorizar a discussão de
manutenção e interpretar os cenários bibliográficos; não injeta falhas no
holdout, não recalcula o NPR a partir do detector e não cria uma terceira
família de resultados.

## 7. Confiabilidade física

O GPVS não contém tempos de vida, exposição de frota nem censura por ativo. As
curvas físicas são cenários bibliográficos separados, sob modelo exponencial:

`R(t)=exp(-lambda*t)`, `F(t)=1-R(t)`, `f(t)=lambda*exp(-lambda*t)` e
`h(t)=lambda`.

Taxas derivadas de participações de chamados são identificadas como cenários;
a taxa direta do fusível permanece sobreposta e rastreada até PDF, páginas e
tabela. As quatro funções são exibidas em eixos lineares. Não se ajusta
Weibull, normal, curva de banheira ou RUL sem tempos individuais, exposição e
censura.

## 8. Publicação

Os resultados vigentes ficam apenas em:

- `resultados/comparacao/`;
- `resultados/confiabilidade/`;
- `resultados/manifestos/`.

Cada figura tem dados-fonte tabulares, JSON metodológico, PNG 300 dpi e PDF
vetorial. Manifestos v2 registram código, dependências, entradas, parâmetros,
saídas e hashes. A leitura de resultados nunca recalcula o pipeline.
