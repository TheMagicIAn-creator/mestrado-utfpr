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

## 3. Candidatos auditados em 9 de agosto de 2026

| Dataset | Natureza | Aderencia ao tema | Uso defensavel | Nao usar para |
|---|---|---|---|---|
| GPVS-Faults | experimental, microrede PV conectada a rede | direta | validacao externa de deteccao e classificacao por cenario | Weibull fisico sem tempos de vida independentes |
| PV residencial | telemetria de um inversor, 862.438 linhas | direta para operacao PV | anomalia operacional e estudo temporal apos saneamento | diagnostico causal, vida de componentes ou generalizacao entre unidades |
| PMSM inverter faults | experimental, acionamento de motor, 10.892 linhas | direta para hardware de inversor; indireta para PV | benchmark complementar de classificacao/transferencia | split aleatorio por linha, Weibull ou RUL de campo |
| PV Farms | simulacao de planta PV, 700 linhas recebidas | direta para falhas de strings CC | benchmark supervisionado E1 | evidencia experimental ou falha CA do inversor |
| Bearing DataCenter | experimental, rolamentos | indireta | benchmark de processamento de corrente/vibracao | validacao de falha de eletronica de potencia |

O candidato prioritario para a proxima validacao externa e o **GPVS-Faults**:
ele possui 16 ensaios experimentais, modos MPPT/IPPT e falhas de arranjo,
inversor, rede, sensor e controlador introduzidas no meio de cada ensaio. Deve
ser mantido como protocolo separado, sem misturar seus registros ao treino de
normalidade do conjunto Stender.

## 4. O que um dataset precisa ter para Weibull fisico

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

## 5. Regra de separacao

- Stender: normalidade CA experimental em acionamento de motor.
- GPVS-Faults: validacao externa experimental em sistema PV conectado a rede.
- PV residencial: comportamento operacional de uma unica unidade.
- PMSM: falhas de hardware em acionamento de motor.
- PV Farms: classificacao supervisionada de falhas CC simuladas.
- Weibull fisico: somente com dados independentes de falha/censura e exposicao.

Metricas nao sao transferidas entre esses protocolos. Resultados E1 ou E2 nao
devem ser apresentados como desempenho industrial E3.
