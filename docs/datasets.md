# Dataset do projeto - GPVS-Faults

Os dados brutos não são versionados. O pipeline principal usa somente os 16
arquivos em `dados/brutos/gpvs/csv/CSV_Files/`, de `F0L.csv` a `F7M.csv`.

## GPVS-Faults

- Fonte: Bakdi et al. (2020), DOI `10.17632/n76t439f65.1`.
- Natureza: microrede fotovoltaica experimental conectada à rede.
- Modos: `L` para IPPT/potência limitada e `M` para MPPT/potência máxima.
- F0: operação saudável usada para treino, validação, calibração e teste.
- F1-F7: sete falhas experimentais, duas condições por falha, usadas na E3.
- Volume auditado: 16 CSVs e aproximadamente 2,16 milhões de registros.
- Integridade: tempo monotônico, ausência de NaN/infinito e hashes registrados.

O arquivo ZIP oficial tem SHA-256
`88cd20c848fee86752870cf9b198eab45568c31355685328dd75aba982bf1a63`.
Os 16 CSVs locais totalizam 493.425.214 bytes (aproximadamente 493 MB em base decimal) e permanecem ignorados pelo Git.

## Protocolo canônico

F0L/F0M alimentam o Autoencoder Denso e o AE-LSTM sob o mesmo protocolo. Cada
modelo preserva seu próprio scaler, escore top-k (`k=5`) e limiar p99,9
solicitado, todos congelados antes
de avaliar F1L-F7M na E3 de bancada. Não há publicação autônoma de métricas do
dataset: toda métrica experimental compara os dois modelos.

Com 210 observações de calibração, o p99,9 solicitado seleciona a ordem 210/210
e corresponde a p100 empírico, com resolução de 0,476 ponto percentual.

Cada ensaio de falha usa a primeira metade pré-falha para normalização de
comissionamento e reserva a segunda metade pré-falha para especificidade. Não
há retreino nem recalibração do limiar por ensaio.

Os rótulos F1-F7 não são convertidos em classes Contator AC, IGBT e Fusível AC.
A FMECA permanece uma camada de priorização de manutenção separada. Portanto,
as métricas E3 não devem ser apresentadas como POD específica desses três
componentes.

## Qualidade de amostragem

O manual declara período de 9,9989 us, mas a mediana observada na coluna `Time`
é aproximadamente 100 us, equivalente a 10 kHz. A implementação infere a taxa
dos CSVs e registra ambos os valores; não presume que o manual esteja correto.

## Resultados vigentes

Os resultados dos dois modelos nos 14 ensaios E3 estão em
`resultados/comparacao/e3_metricas_macro.csv` e
`e3_metricas_por_ensaio.csv`. Recall, F1 e Precision são as métricas principais;
ROC-AUC e PR-AUC são complementares. Os IC95% usam o ensaio como unidade de
bootstrap. Limites de desempenho por ensaio permanecem
publicados; nenhum cenário é descartado para melhorar a média.

## Fontes fora do resultado principal

Stender, PMSM, PV Farms, telemetria residencial e Bearing DataCenter permanecem
somente como referências ou auditorias históricas. Seus dados e métricas não
são fundidos ao GPVS nem possuem resultados ativos no repositório.

## Limite para Weibull físico

O GPVS contém ensaios pré/pós-falha, não tempos de vida de unidades
independentes. Portanto não sustenta confiabilidade temporal, MTTF, taxa de
falha, distribuição de vidas ou RUL de campo. As curvas físicas vêm somente de
taxas bibliográficas rastreadas e não são ajustadas ao dataset experimental.
