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
Os CSVs locais totalizam cerca de 620 MB e permanecem ignorados pelo Git.

## Protocolo canônico

F0L/F0M alimentam um único Autoencoder. O teste saudável F0 também recebe
injeções sintéticas orientadas pela FMECA, produzindo evidência E2. O mesmo
modelo, scaler, escore e limiar avaliam F1L-F7M, produzindo evidência E3 de
bancada.

Cada ensaio de falha usa a primeira metade pré-falha para normalização de
comissionamento e reserva a segunda metade pré-falha para especificidade. Não
há retreino nem recalibração do limiar por ensaio.

## Qualidade de amostragem

O manual declara período de 9,9989 us, mas a mediana observada na coluna `Time`
é aproximadamente 100 us, equivalente a 10 kHz. A implementação infere a taxa
dos CSVs e registra ambos os valores; não presume que o manual esteja correto.

## Resultados vigentes

O detector canônico obteve, nos 14 ensaios E3:

| Métrica macro por ensaio | Estimativa | IC95% bootstrap |
|---|---:|---:|
| AUC | 0,773 | 0,691-0,853 |
| Sensibilidade | 0,406 | 0,211-0,615 |
| Especificidade | 0,974 | 0,946-0,992 |
| Acurácia balanceada | 0,690 | 0,596-0,791 |

F1, F2 e F5 são mais detectáveis. F4, F6 e parte de F3/F7 permanecem limites
publicados, não resultados descartados. O teste saudável F0 apresentou 1,42%
de excedência no limiar nominal p99.

## Fontes fora do resultado principal

Stender, PMSM, PV Farms, telemetria residencial e Bearing DataCenter permanecem
como referências, auditorias ou experimentos legados. Seus dados e métricas não
são fundidos ao GPVS. Relatórios datados anteriores podem descrevê-los como
candidatos ou eixos; isso é histórico, não o protocolo canônico vigente.

## Limite para Weibull físico

O GPVS contém ensaios pré/pós-falha, não tempos de vida de unidades
independentes. Portanto não sustenta confiabilidade temporal, MTTF, taxa de
falha ou RUL de campo. A Weibull publicada descreve magnitude de
detectabilidade E2 (`a_det`), não tempo.
