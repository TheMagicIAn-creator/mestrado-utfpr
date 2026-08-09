# Auditoria de datasets para inversores, deteccao e confiabilidade

Data: 2026-08-09

Base Git auditada: `main` em `fadf2a1`.

> **Atualizacao posterior no mesmo dia:** o GPVS-Faults foi baixado, auditado e
> executado em protocolo separado. O resultado E3 de bancada e seus limites
> estao em `resultados/gpvs/` e `docs/datasets.md`; as recomendacoes abaixo
> permanecem como registro do estado anterior a essa execucao.

## Veredito executivo

1. O parecer recebido confunde dois datasets diferentes da Universidade de
   Paderborn. O projeto usa o dataset de inversor IGBT de Stender, nao o
   Bearing DataCenter. A critica de dominio continua parcialmente valida
   porque o inversor aciona um motor, mas nao se trata de falha de rolamento.
2. O GPVS-Faults foi o melhor candidato recebido para validacao externa direta
   em sistema fotovoltaico conectado a rede e foi executado posteriormente no
   mesmo dia. Ele nao substitui dados de vida.
3. O PV residencial possui muito mais linhas, mas somente dez blocos de estado
   `Fault` em uma unica serie. Ele pode sustentar deteccao operacional depois
   de saneamento; nao sustenta diagnostico causal, Weibull fisico ou RUL.
4. O PMSM mede falhas reais de hardware de inversor, mas as nove classes estao
   concatenadas em nove blocos de uma unica execucao e as features derivadas
   publicadas contem divisoes por zero. Serve como benchmark exploratorio, com
   protocolo por bloco, nao como eixo principal da dissertacao.
5. Nenhum conjunto atualmente disponivel possui varias unidades, tempos de
   falha/censura e historico de reparo. Portanto nenhum permite converter os
   resultados E2 atuais em vida de campo.

## Correcao de identidade: Stender nao e Bearing DataCenter

O arquivo local `Inverter_Data_Set.csv` corresponde a Stender, Wallscheid e
Bocker (2020), *Data Set Description: Three-Phase IGBT Two-Level Inverter for
Electrical Drives*. A fonte institucional descreve aproximadamente 235 mil
amostras de um inversor IGBT trifasico em bancada com motor e controlador.

O Bearing DataCenter, tambem da Universidade de Paderborn, e outro projeto:
ele fornece corrente e vibracao para diagnostico de rolamentos. O nome curto
"Paderborn" usado pelo repositorio tornou a confusao previsivel e deve ser
substituido, na comunicacao, por "Stender inverter dataset (Paderborn
University)".

Fontes:

- Stender: https://ris.uni-paderborn.de/record/30034
- arquivo de dados: https://www.kaggle.com/datasets/stender/inverter-data-set
- Bearing DataCenter: https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter

## Inventario e integridade local

| Material | SHA-256 | Estrutura observada |
|---|---|---|
| Stender `Inverter_Data_Set.csv` | `95dd7e27751659228d22e04614fd5e91ac44e92df15cbe3f3bd60592e3e3953b` | 234.527 linhas, 26 colunas |
| PV Farms treino | `eb0fe7efb8de9e389af2d2db7ef1fd00a94e9a68fad21f625679a633f61ee7c0` | 600 linhas, 30 features + classe |
| PV Farms teste | `3930647400092547acd8cad41087a0013f44c9ba9dc4d44fc26cc68d02ebf827` | 100 linhas, 30 features + classe |
| `13974425.zip` | `2d191deacf13dbe7a38c1c5b97b0b92ee4327169b622225193d4887b073dce61` | PMSM, 10.892 linhas, 26 colunas |
| `archive.zip` | `307e768a3de88c5a360ed4b57382bb7ca3f1abcc9a84cffcdab5a34160ff4714` | PV residencial, 862.438 linhas, 16 colunas |

Os ZIPs permaneceram fora do repositorio. Nenhum dado bruto foi preparado
para commit.

## Stender inverter dataset

### O que sustenta

- dado experimental de um inversor IGBT trifasico saudavel;
- sinais CA de corrente e tensao por fase em ordem temporal;
- treino de um modelo de normalidade;
- teste de transferencia de assinaturas CA sinteticas em uma bancada real.

### O que nao sustenta

- desempenho de inversor fotovoltaico conectado a rede;
- falhas reais de IGBT, contator ou fusivel;
- frequencia de rede fixa, MPPT, irradiancia ou interacao com filtro/rede;
- taxa de falha, tempo de vida ou RUL de componentes.

O uso atual e defensavel somente se descrito como benchmark experimental de
normalidade e validacao interna E2. "Paderborn" nao deve ser usado como nome de
dataset sem o sobrenome Stender.

## PV Farms simulado

A fonte informa uma planta de 250 kW simulada, tres falhas de strings e estado
normal. O treino local possui as distribuicoes 100/153/149/198; o teste possui
25 exemplos de cada classe.

Achados de qualidade:

- 126 das 600 linhas de treino sao duplicatas exatas;
- nenhuma linha do teste aparece exatamente no treino;
- `I5` e `I6` sao colunas identicas no treino recebido;
- o arquivo e um benchmark de falhas CC simuladas, nao uma usina medida;
- split aleatorio comum na validacao cruzada pode separar copias da mesma
  observacao e inflar a metrica.

Acao implementada: a validacao cruzada agrupa linhas identicas por
`StratifiedGroupKFold`. O teste oficial permanece separado.

Fonte: https://www.kaggle.com/datasets/amrezzeldinrashed/fault-detection-dataset-in-photovoltaic-farms

## PMSM inverter faults

O arquivo `converted_dataset-2.csv` possui 10.892 amostras a 10 Hz, nove
classes (`F0` a `F8`) e oito sensores de corrente, tensao e temperatura. As
classes aparecem como nove blocos contiguos, exatamente um bloco por classe.
O `Timestamp` nao vem da aquisicao: o script o cria como `0,0; 0,1; ...` depois
de concatenar os dados.

Achados de qualidade:

- forte desbalanceamento: `F0=4.295`, menor classe `F4=341`;
- 20 ausencias esperadas nas primeiras diferencas e medias moveis;
- 21 infinitos em `Current_Imbalance` e 14 em cada corrente normalizada;
- corrente bruta e duas conversoes da mesma corrente sao colineares;
- a formula de desbalanceamento usa media assinada no denominador e produz
  valores negativos, extremos e infinitos;
- as correntes normalizadas dividem por `IDC` sem guarda para zero;
- `Power_AC` assume duas fases, embora o sistema seja trifasico;
- split aleatorio por linha mede reconhecimento do mesmo ensaio, nao
  generalizacao para nova aquisicao.

Uso recomendado: partir das colunas brutas, revisar unidades e formulas,
preservar blocos de aquisicao e tratar o resultado como transferencia para
acionamento de motor. Nao usar o CSV derivado como esta.

Fonte: https://zenodo.org/records/13974425

## PV residencial

Perfil observado:

- 862.438 linhas entre 2022-11-05 e 2026-02-06;
- passo mediano de 60 s;
- 842.856 linhas `Normal`, 19.106 `Wait` e 476 `Fault`;
- dez blocos de `Fault` separados por mais de uma hora, em nove datas;
- 819 repeticoes de timestamp, 9.921 lacunas acima de 90 s e 1.188 acima de
  uma hora; maior lacuna de aproximadamente 4,08 dias;
- sem valores ausentes e sem linhas completas duplicadas.

Riscos metodologicos:

- `Working Mode.1` e uma codificacao exata de `Working Mode`; usa-la como
  feature revela o alvo;
- em `Fault`, a tensao CA, a frequencia e a corrente CA sao zero em 100% das
  linhas; um classificador aprenderia o estado de desligamento, nao a causa;
- `PF` tem apenas dois valores e quase sempre vale `-0.001`;
- contadores acumulados apresentam saltos de ate 20 kWh e 9 h;
- o intervalo muda para cerca de 300 s em varios blocos de falha, criando uma
  assinatura de amostragem associada ao rotulo;
- nao ha identificador de equipamento, causa, componente trocado, reparo ou
  censura. Os contadores indicam uma unica unidade.

O dataset e promissor para um estudo separado de anomalia operacional com
split temporal, exclusao de vazamento e comparacao contra regras simples. Ele
nao aumenta o numero de falhas independentes para Weibull.

Fonte: https://www.kaggle.com/datasets/mark0ndz/dataset-pv-inverter-residential

## GPVS-Faults

O GPVS-Faults possui 16 ensaios experimentais de uma microrede PV conectada a
rede, em MPPT e IPPT. Cada ensaio introduz uma falha manualmente na metade da
captura. Ha cenarios de arranjo PV, inversor, rede, sensor e controlador, com
correntes/tensoes trifasicas e variaveis CC em alta frequencia.

E o melhor candidato para uma validacao externa porque reduz o deslocamento de
dominio sem misturar datasets. Ainda assim, os 16 ensaios sao cenarios de falha
introduzida, nao vidas independentes de componentes.

Fonte e licenca CC BY 4.0:
https://data.mendeley.com/datasets/n76t439f65/1

## Contrato minimo para Weibull fisico

Uma base apta precisa representar unidades ou ensaios independentes e conter
tempo/ciclos de exposicao, evento por modo de falha, censura, reparos e
covariaveis operacionais. Repetir amostras do mesmo evento aumenta resolucao
temporal, nao o denominador estatistico de vidas.

Os graficos atuais ja declaram `a_det` como magnitude de assinatura e marcam a
evidencia E2. Portanto o caminho correto e manter esse experimento como
demonstracao metodologica e adicionar GPVS-Faults como validacao externa. A
conversao para horas, dias, MTTF de campo ou curva de vida deve permanecer
bloqueada ate existir um dataset com o contrato acima.

## Estado de proveniencia apos os merges

O commit `fadf2a1` alterou `weibull_results.json` e tres figuras sem atualizar
`resultados/manifestos/rul_weibull.json`. O manifesto ainda aponta para o
commit `d9afe11` e hashes anteriores. No checkout local, todas as etapas do
pipeline aparecem como `stale`; isso exige regeneracao completa pelo pipeline,
nao edicao manual de hashes.

O verificador FMECA tambem foi escrito para limites de tres blocos contiguos e
interpreta incorretamente a nova lista de blocos intercalados como incompleta.
Esse verificador precisa aceitar os dois schemas sem reduzir as exigencias de
purga e nao sobreposicao.

## Decisao

1. Manter Stender como treino de normalidade E2, com nome e limite de dominio
   corrigidos.
2. Manter PV Farms apenas como benchmark simulado E1 e usar CV por grupos.
3. Nao integrar PMSM derivado nem PV residencial ao resultado principal neste
   estado.
4. Preparar GPVS-Faults como proxima validacao externa em protocolo separado.
   **Concluido:** ver `resultados/gpvs/`.
5. Nao alterar a interpretacao de Weibull para vida fisica.
6. Regenerar os artefatos atuais e seus manifestos depois das correcoes de
   codigo, preservando a marcacao E2.
