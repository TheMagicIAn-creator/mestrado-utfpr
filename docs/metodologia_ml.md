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

A etapa E2 usa o **outro** caminho, e isso é deliberado. A janela injetada nasce
saudável, do holdout F0: ela é normalizada pela baseline F0 do seu ensaio de
origem e depois pelo scaler congelado — exatamente o percurso de uma janela
saudável no treino, que é a escala em que o limiar foi calibrado. Normalizá-la
por comissionamento seria impossível sem inventar um recorte pré-falha que uma
janela saudável não tem, e poria os escores numa escala para a qual o limiar
congelado nunca foi calibrado, medindo outro detector com o mesmo número. A
consequência a carregar em qualquer leitura: **POD e recall não são a mesma
grandeza**, e a divergência entre elas mede fidelidade da injeção, não erro de
alguma das duas etapas.

## 4. Modelos e treino

- Denso: `24-16-8-16-24`.
- AE-LSTM: sequência 8, oculto 32 e latente 8.

Os dois recebem o mesmo orçamento de épocas, early stopping, sementes e
pré-processamento compatível. Desde 2026-09-03 recebem também a mesma
regularização: o `Dropout(0,2)` do Denso, nas duas travessias do gargalo, passou
a ter contrapartida explícita no AE-LSTM. `nn.LSTM` de camada única ignora
silenciosamente o argumento `dropout`, de modo que o braço temporal treinava sem
regularização alguma — uma explicação concorrente para ele não converter
capacidade em detecção. A assimetria que resta é de capacidade, inerente às duas
arquiteturas, e é publicada com arquitetura, contagem de parâmetros e dropout
lado a lado.

A semente 42 é a execução de referência e cinco sementes medem estabilidade. O
escore é a média dos cinco maiores erros quadráticos por feature; no AE-LSTM,
somente o último passo temporal recebe o top-k. Cada modelo recebe seu próprio
p99 solicitado, calculado na calibração saudável pelo método `higher`. A saída
registra o order statistic selecionado, o percentil empírico efetivo, o tamanho
da calibração e sua resolução.

O ponto canônico é p99 por decisão do pesquisador em 2026-09-03. Com `n=210` na
calibração, p99 seleciona a observação de ordem 208/210: percentil empírico
efetivo p99,05. O pedido anterior de p99,9 caía na ordem 210/210 — o limiar era
literalmente o maior escore visto na calibração, com a variância de um máximo
amostral e não a de um quantil. Um percentil só é distinguível do máximo quando
`n >= (q-2)/(q-1)`, com `q = p/100`: 101 para p99, 201 para p99,5 e 1001 para
p99,9. `minimum_n_for_percentile` publica esse número e `calibrate_threshold`
aceita `strict`, que a publicação canônica usa — um percentil degenerado
interrompe a publicação em vez de sair no artefato.

O ponto histórico p99,9 continua reproduzível com `strict_threshold=False` e
permanece na grade de sensibilidade, agora marcado como degenerado.

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
`k`, percentil, limiar ou hiperparâmetro. `k=5` com p99 é a configuração
canônica; `k=5` com p99,9 permanece apenas como referência histórica de
reprodutibilidade e sai marcado como degenerado, porque com `n=210` esse pedido
não é distinguível do máximo amostral. A grade evidencia o trade-off entre
Recall e falsos alarmes sem promover um ótimo pelo desempenho em F1-F7.

## 6. FMECA e manutenção

A FMECA vigente tem escopo de 6 itens: IGBT, sensor/realimentação, controle,
PCB, contatores CA/CC e ventiladores, todos com `NPR = S * O * D` validado. Só
IGBT (F1), sensor/realimentação (F2) e controle (F6/F7) têm contrapartida no
GPVS e entram na injeção E2; os demais são criticidade/manutenção. A proveniência
é mista: os cinco itens de survey de campo levam os escores de Cristaldi et al.
(2017), Tabela 6 — IGBT (3,3,7; NPR 63), controle (6,7,3; NPR 126), PCB (7,4,6;
NPR 168), contatores (6,5,5; NPR 150), ventiladores (4,3,4; NPR 48); o
sensor/realimentação mantém o valor definido pelo pesquisador em 2026-09-01
(5,8,7; NPR 280), a ser revisado. Os λ físicos por componente ficam em
`resultados/confiabilidade/`, rotulados por fonte; PCB e software ficam sem λ,
registrado sem estimativa.

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
- `resultados/detectabilidade/`;
- `resultados/confiabilidade/`;
- `resultados/manifestos/`.

Cada figura tem dados-fonte tabulares, JSON metodológico, PNG 300 dpi e PDF
vetorial. Manifestos v2 registram código, dependências, entradas, parâmetros,
saídas e hashes. A leitura de resultados nunca recalcula o pipeline.
