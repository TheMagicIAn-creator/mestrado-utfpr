# Auditoria de consolidação do pipeline GPVS-Faults

Data: 9 de agosto de 2026. Base Git: `a4aa42c`. Esta auditoria substitui, para
uso corrente, os resultados que comparavam transferência estrita, AE
adaptativo e PCA. Os relatórios anteriores permanecem como histórico.

## Decisão

O GPVS-Faults passou a ser o único dataset dos resultados canônicos. Stender,
PMSM, PV Farms, telemetria residencial e Bearing DataCenter não fornecem
amostras, parâmetros ou métricas ao pipeline principal.

## Implementação

- 24 features físicas extraídas em janelas não sobrepostas de um ciclo de 50 Hz.
- 1.423 janelas saudáveis F0L/F0M.
- Split por ensaio: treino 50%, validação 15%, calibração 15% e teste 20%, com purga.
- Autoencoder `24-16-8-16-24`, MSE p99 de calibração e teste saudável isolado.
- Normalização robusta com piso de IQR; baseline de comissionamento em F1-F7.
- E2 FMECA no holdout F0 e E3 nos 14 ensaios reais F1L-F7M.
- Bootstrap macro por ensaio com 20.000 reamostragens.
- Manifestos v2 para os cinco estágios e manifesto E3 específico.

## Resultados vigentes

| Evidência | Resultado |
|---|---|
| F0 saudável | 4/281 excedências; 1,42%, IC95% Wilson 0,55%-3,60% |
| E2 Contator AC | SMD95 0,70 |
| E2 IGBT | SMD95 0,50 |
| E2 Fusível AC | SMD95 0,05; conservadora 0,10 |
| E3 AUC macro | 0,773; IC95% 0,691-0,853 |
| E3 sensibilidade | 0,406; IC95% 0,211-0,615 |
| E3 especificidade | 0,974; IC95% 0,946-0,992 |
| E3 acurácia balanceada | 0,690; IC95% 0,596-0,791 |

F1, F2 e F5 são detectadas com maior sensibilidade. F4, F6 e parte de F3/F7
continuam como limitações publicadas. Não houve seleção de cenários por
desempenho.

## Weibull E2

Foram usadas 100 trajetórias por assinatura e 120 níveis de magnitude. As três
assinaturas cruzaram o limiar até `a=1`, mas apenas Contator AC passou a triagem
visual do papel de Weibull (`R²pp=0,925`). IGBT (`0,868`) e Fusível AC (`0,568`)
não têm síntese paramétrica recomendada.

O eixo é `a_det`, fração da assinatura sintética nominal. Não representa tempo,
probabilidade de falha, confiabilidade, taxa de falha, MTTF, B10 ou RUL físico.

## Verificação

- Verificador FMECA/GPVS aprovado: JSON, CSV, hashes e 17 PNGs reconciliados.
- Cinco estágios em estado `ready`.
- Dados brutos, pesos, scaler e estado local do Obsidian permanecem fora do Git.
- Validação E3: 20 entradas e 9 saídas com hash no manifesto específico.

## Limites

E3 é bancada, não campo. A normalização usa a primeira metade pré-falha de cada
ensaio como baseline de comissionamento; não há retreino do modelo nem
recalibração do limiar. Janelas do mesmo ensaio são autocorrelacionadas, por
isso a inferência macro usa o ensaio como unidade.
