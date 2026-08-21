# Metodologia canônica de Machine Learning

## 1. Pergunta experimental

Comparar, sob o mesmo protocolo, um Autoencoder Denso e um AE-LSTM para
detecção de anomalias em sinais elétricos do GPVS-Faults. A comparação separa
evidência experimental E3, detectabilidade sintética E2 e confiabilidade física
bibliográfica.

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
sementes medem estabilidade. O erro de reconstrução define o escore; cada
modelo recebe seu próprio p99 empírico, calculado na calibração saudável pelo
método `higher`.

Nenhum desempenho em F1-F7 participa da seleção de arquitetura, semente ou
limiar.

## 5. E3 experimental

Os modelos congelados avaliam todos os 14 ensaios. AUC-PR é a métrica
principal, acompanhada por ROC-AUC, sensibilidade, especificidade, acurácia
balanceada, MCC, F1 e falso positivo saudável. A unidade inferencial é o
ensaio; IC95% macro usam bootstrap de 20.000 reamostragens dos ensaios.

Resultados negativos e heterogeneidade por falha permanecem publicados.
E3 significa bancada, não validação de campo.

## 6. E2 orientada pela FMECA

As assinaturas de Contator AC, IGBT e Fusível AC são aplicadas sobre todo o
holdout saudável com janelas, magnitudes e sementes compartilhadas entre
modelos. Para cada magnitude são reportadas detecção e IC95% de Wilson.

SMD95 é a menor magnitude cujo limite inferior do IC95% alcança 95%. Quando
isso não ocorre, o resultado é `não atingido`; não há extrapolação para forçar
um valor.

O eixo `a_det` é magnitude adimensional da perturbação. Sobrevivência empírica,
incidência acumulada e risco discreto descrevem o primeiro cruzamento do
detector nesse eixo. Weibull 2P é um diagnóstico intervalar com censura à
direita e critérios formais de aceitação. Parâmetros rejeitados não sustentam
síntese.

Nenhuma curva E2 representa tempo, vida útil, RUL ou taxa de falha física.

## 7. Confiabilidade física

O GPVS não contém tempos de vida, exposição de frota nem censura por ativo. As
curvas físicas são cenários bibliográficos separados, sob modelo exponencial:

`R(t)=exp(-lambda*t)`, `F(t)=1-R(t)`, `f(t)=lambda*exp(-lambda*t)` e
`h(t)=lambda`.

Taxas derivadas de participações de chamados são identificadas como cenários;
a taxa direta do fusível permanece sobreposta e rastreada até PDF, páginas e
tabela. Não se ajusta Weibull físico sem dados apropriados.

## 8. Publicação

Os resultados vigentes ficam apenas em:

- `resultados/comparacao/`;
- `resultados/confiabilidade/`;
- `resultados/manifestos/`.

Cada figura tem dados-fonte tabulares, JSON metodológico, PNG 300 dpi e PDF
vetorial. Manifestos v2 registram código, dependências, entradas, parâmetros,
saídas e hashes. A leitura de resultados nunca recalcula o pipeline.
