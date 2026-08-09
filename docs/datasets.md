# Datasets - ALIAdo PV

Os dados brutos nao sao versionados. Eles ficam em `dados/brutos/`, que esta
no `.gitignore`. Execute `python scripts/verificar_datasets.py` para gerar o
manifesto local com SHA-256, linhas, classes, duplicatas e validacoes minimas.

## 1. Stender inverter dataset - eixo principal de normalidade CA

- Arquivo local: `dados/brutos/Inverter_Data_Set.csv`.
- Fonte: Stender, Wallscheid e Bocker (2020), *Three-Phase IGBT Two-Level
  Inverter for Electrical Drives*.
- Natureza: ensaio experimental de um inversor IGBT trifasico, motor de
  inducao e controle, em bancada da Universidade de Paderborn.
- Conteudo: aproximadamente 235 mil amostras saudaveis a 10 kHz, sem rotulos
  reais de falha.
- Uso no projeto: treino do Autoencoder de normalidade e avaliacao E2 com
  assinaturas sinteticas orientadas pela FMECA.

**Identidade sem ambiguidade:** este nao e o *Paderborn University Bearing
Dataset*. O Bearing DataCenter contem falhas de rolamentos medidas por corrente
e vibracao. O arquivo usado no projeto e o dataset de inversor de Stender. A
abreviacao antiga "Paderborn" descrevia a instituicao e induzia a confusao.

**Limite de dominio:** apesar de medir diretamente um inversor, a carga e um
motor de velocidade variavel. Nao ha rede de 60 Hz, arranjo fotovoltaico,
controle MPPT, filtro de acoplamento ou falhas de campo. O conjunto sustenta um
benchmark experimental de transferencia para sinais CA, nao validacao externa
de um inversor fotovoltaico conectado a rede.

## 2. PV Farms - benchmark supervisionado simulado

- Arquivos locais: `dados/brutos/train_data.csv` e `test_data.csv`.
- Fonte: Ghoneim, Rashed e Elkalashy (2021).
- Natureza: simulacao de uma planta fotovoltaica de 250 kW.
- Classes: normal, falha de string, string-terra e string-string.
- Arquivo recebido: 600 linhas de treino e 100 de teste, com 30 features.
- Uso: benchmark supervisionado complementar de falhas predominantemente CC.

O treino recebido possui 126 linhas exatamente duplicadas. A validacao cruzada
deve manter copias da mesma observacao no mesmo fold; o classificador usa
`StratifiedGroupKFold` para evitar vazamento. O teste nao possui linhas iguais
as do treino. Estes dados nao demonstram desempenho de campo nem diagnosticam
as falhas CA da FMECA do inversor.

## 3. GPVS-Faults - validacao experimental E3 de bancada

- Arquivos locais: `dados/brutos/gpvs/csv/CSV_Files/F0L.csv` a `F7M.csv`.
- Fonte: Bakdi et al. (2020), DOI `10.17632/n76t439f65.1`.
- Natureza: 16 ensaios experimentais de microrede PV conectada a rede; `F0`
  saudavel e sete falhas introduzidas na metade dos ensaios, em IPPT/MPPT.
- Volume observado: 2.163.480 registros, sem NaN, infinito ou tempo nao
  monotono nos arquivos auditados.
- Uso: validacao externa por ensaio, separada do treino Stender.

O protocolo em `src/ml/gpvs.py` usa janelas nao sobrepostas de um ciclo de
50 Hz, 24 features dos sensores primarios, split temporal com purga, cinco
sementes e limiar p99. O PCA de quatro componentes e o baseline linear sob o
mesmo split. Os resultados versionados ficam em `resultados/gpvs/`.

**Resultado vigente:** o limiar transferido diretamente dos ensaios `F0` teve
especificidade macro 0,007 e foi rejeitado como ponto operacional. Com
adaptacao local usando somente o inicio saudavel de cada ensaio, o AE obteve
AUC macro 0,815 (IC95% 0,745-0,881), sensibilidade 0,445 e especificidade
0,974. F1, F2 e F5 foram os cenarios mais detectaveis; F4, F6 e parte de F3/F7
permanecem limitacoes. O PCA obteve AUC macro 0,794.

**Discrepancia de amostragem:** o ReadMe declara `9,9989 us`, mas a mediana
observada nos vetores `Time` dos 16 CSVs e aproximadamente `99,9969 us`
(~10 kHz). O adaptador infere a taxa do vetor de tempo e nao usa o valor
declarado no manual.

Este E3 e **experimental de bancada**, nao de campo. Ele nao demonstra
prevalencia industrial, nao transforma deteccao em diagnostico causal e nao
fornece tempos de vida para Weibull/RUL fisico.

## 4. Outros candidatos auditados em 9 de agosto de 2026

| Dataset | Natureza | Aderencia ao tema | Uso defensavel | Nao usar para |
|---|---|---|---|---|
| PV residencial | telemetria de um inversor, 862.438 linhas | direta para operacao PV | anomalia operacional e estudo temporal apos saneamento | diagnostico causal, vida de componentes ou generalizacao entre unidades |
| PMSM inverter faults | experimental, acionamento de motor, 10.892 linhas | direta para hardware de inversor; indireta para PV | benchmark complementar de classificacao/transferencia | split aleatorio por linha, Weibull ou RUL de campo |
| PV Farms | simulacao de planta PV, 700 linhas recebidas | direta para falhas de strings CC | benchmark supervisionado E1 | evidencia experimental ou falha CA do inversor |
| Bearing DataCenter | experimental, rolamentos | indireta | benchmark de processamento de corrente/vibracao | validacao de falha de eletronica de potencia |

## 5. O que um dataset precisa ter para Weibull fisico

Mais linhas de telemetria nao equivalem a mais eventos independentes. Para
estimar vida util ou confiabilidade de campo sao necessarios, no minimo:

1. identificador de varias unidades ou varias trajetorias independentes;
2. inicio de observacao e exposicao acumulada em tempo ou ciclos;
3. data e modo de cada falha, ou data de censura para unidades sem falha;
4. registro de reparo, substituicao e retorno a operacao;
5. condicoes de carga, ambiente e politica de manutencao;
6. numero de eventos suficiente por modo de falha.

Nenhum dos datasets atualmente disponiveis cumpre esse contrato. O Weibull
publicado no projeto permanece E2 ilustrativo sobre `a_det`, a magnitude da
assinatura necessaria para deteccao. Ele nao estima horas, dias ou anos de vida.

## 6. Regra de separacao

- Stender: normalidade CA experimental em acionamento de motor.
- GPVS-Faults: validacao externa experimental em sistema PV conectado a rede.
- PV residencial: comportamento operacional de uma unica unidade.
- PMSM: falhas de hardware em acionamento de motor.
- PV Farms: classificacao supervisionada de falhas CC simuladas.
- Weibull fisico: somente com dados independentes de falha/censura e exposicao.

Metricas nao sao transferidas entre esses protocolos. E3 de bancada nao deve
ser apresentado como desempenho de campo ou prova industrial.
