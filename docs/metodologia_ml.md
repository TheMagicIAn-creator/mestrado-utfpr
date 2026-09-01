# Metodologia canônica de Machine Learning

## 1. Pergunta experimental

Determinar quantas falhas reais cada modelo detecta sem produzir uma quantidade
operacionalmente inadequada de falsos alarmes, comparando um Autoencoder Denso
e um AE-LSTM sob o mesmo protocolo. Nenhum modelo é vencedor por definição. O
GPVS-Faults é a base experimental interna da comparação, não uma família
autônoma de resultados. A confiabilidade física bibliográfica permanece
metodologicamente separada.

## 2. Dataset e features

O único dataset ativo é o GPVS-Faults, DOI `10.17632/n76t439f65.1`:

- F0L/F0M: condição saudável;
- F1L-F7M: 14 ensaios de falha em dois regimes;
- 24 features: estatísticas CC, RMS/THD trifásicos, desbalanceamentos e
  estatísticas de potência CA/CC.

A taxa contratual é 10 kHz e cada janela tem 200 amostras, um ciclo nominal de
50 Hz. A qualidade da coluna `Time` é validada e a divergência do manual fica
registrada. Janelas não se sobrepõem.

No catálogo nativo, F1 é falha completa de um IGBT, F2 é erro de 20% no
sistema de sensor/realimentação, F6 reduz em 20% o ganho do controlador PI e F7
aumenta em 20% sua constante de tempo. F6/F7 são anomalias funcionais do
sistema/circuito de controle do inversor, não falhas físicas de PCB. Nenhuma
falha sintética é usada no núcleo experimental.

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
sementes medem estabilidade. Na referência histórica, o escore é a média dos
cinco maiores erros quadráticos por feature; no AE-LSTM, somente o último passo
temporal recebe o top-k. Cada modelo recebe seu próprio p99,9 solicitado,
calculado na calibração saudável pelo método `higher`. A saída registra o order
statistic selecionado, o percentil empírico efetivo, o tamanho da calibração e
sua resolução.

Na execução vigente, `n=210` na calibração faz p99,9 selecionar a observação de
ordem 210/210: percentil empírico efetivo p100 e resolução de 0,476 ponto
percentual. Portanto p99,9 é a configuração histórica solicitada, não uma
precisão empírica que a amostra consiga resolver literalmente nem um ótimo
universal.

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

### 5.1. Ablação temporal do AE-LSTM

A análise usa a sequência causal contínua `[W_(t-7), ..., W_t]` e decide sobre
`W_t`. Ela separa as sete primeiras janelas pós-fronteira da falha sustentada,
na qual o contexto já é integralmente pós-fronteira. Um contexto reiniciado na
fronteira permanece apenas como diagnóstico auxiliar. Modelos, scalers, escores
e limiares permanecem congelados. Na execução de referência, os IC95%
pareados de Recall, F1 e Precision na falha sustentada cruzam zero. A conclusão
pré-especificada é **inconclusiva**: o ganho observado não pode ser atribuído
inequivocamente à arquitetura temporal.

### 5.2. Sensibilidade de escore e limiar

A grade `k={5,10,20}` e percentis solicitados `{99;99,5;99,9}` contém nove
configurações por modelo e semente. Todos os limiares vêm somente da calibração
saudável. Os ensaios com falha não selecionam arquitetura, semente, scaler,
`k`, percentil, limiar ou hiperparâmetro. `k=5` com p99,9 permanece apenas como
referência histórica de reprodutibilidade. A grade evidencia o trade-off entre
Recall e falsos alarmes sem promover um ótimo pelo desempenho em F1-F7.

## 6. FMECA e manutenção

A FMECA vigente cobre IGBT, sistema de sensor/realimentação e sistema/circuito
de controle do inversor. As contrapartidas nativas são F1, F2 e F6/F7,
respectivamente. Os campos S, O, D e NPR ficam nulos, com estado
`awaiting_user_fmeca`, até que o pesquisador forneça valores e fontes
compatíveis com o novo escopo.

O recorte Contator AC, IGBT e Fusível AC permanece apenas como histórico do
TCC. Nenhum valor é herdado dele. As métricas dos detectores não são convertidas
em escalas de manutenção e não participam do cálculo de NPR.

## 7. Confiabilidade física

O GPVS não contém tempos de vida, exposição de frota nem censura por ativo. As
curvas físicas são cenários bibliográficos separados, sob modelo exponencial:

`R(t)=exp(-lambda*t)`, `F(t)=1-R(t)`, `f(t)=lambda*exp(-lambda*t)` e
`h(t)=lambda`.

Taxas derivadas de participações de chamados são identificadas como cenários;
a taxa direta do fusível permanece sobreposta e rastreada até PDF, páginas e
tabela. Essas curvas pertencem ao recorte bibliográfico histórico e não
preenchem a FMECA atual. As quatro funções são exibidas em eixos lineares.

A auditoria do corpus encontrou 22 trechos sobre IGBT e 74 sobre Weibull. A
única fonte comum é o TCC, no qual os assuntos aparecem separadamente e sem
`beta` ou `eta` para IGBT. Assim, Weibull 2P continua bloqueada. Também não se
ajusta Normal, Lognormal, curva de banheira ou RUL sem tempos individuais,
exposição e censura.

O contrato lista os parâmetros ainda necessários: `beta` e `eta` para Weibull
2P; média e desvio padrão em horas para Normal; média e desvio no domínio
logarítmico para Lognormal; e a amostra de vidas para histograma. Ter apenas uma
taxa por cenário não supre esses dados.

## 8. Publicação

Os resultados vigentes ficam apenas em:

- `resultados/comparacao/`;
- `resultados/confiabilidade/`;
- `resultados/manifestos/`.

Cada figura tem dados-fonte tabulares, JSON metodológico, PNG 300 dpi e PDF
vetorial. Manifestos v2 registram código, dependências, entradas, parâmetros,
saídas e hashes. A leitura de resultados nunca recalcula o pipeline.
