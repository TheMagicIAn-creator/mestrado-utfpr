# Validacao experimental GPVS-Faults (E3 de bancada)

Gerado em `2026-08-09T12:02:28.715352-03:00`. Dataset: DOI 10.17632/n76t439f65.1.

## Resultado principal

A transferencia estrita do limiar F0 nao e operacional: a especificidade macro foi 0.007 (IC95% 0.000-0.019) por causa do deslocamento entre ensaios. A taxa de deteccao isolada desse protocolo nao deve ser citada.

Com adaptacao local usando somente o inicio saudavel, o AE obteve AUC macro 0.815 (IC95% 0.745-0.881), sensibilidade 0.445 e especificidade 0.974. Houve deteccao sustentada em 10/14 ensaios.

O baseline PCA, sob o mesmo split, obteve AUC macro 0.794, sensibilidade 0.431 e especificidade 0.972.

## Leitura por modo de falha

- F1, F2 e F5 sao os cenarios mais detectaveis no limiar p99.
- F3 e intermitente; AUC e sensibilidade devem ser lidas separadamente.
- F4, F6 e F7 permanecem limitacoes do detector no limiar operacional.
- Resultado nulo foi preservado; nao houve selecao de cenarios por desempenho.

## Protocolo

- 24 features de sensores primarios em janelas nao sobrepostas de um ciclo de 50 Hz.
- Taxa inferida de cada vetor `Time` (aprox. 10 kHz); o manual declara 9,9989 us,
  mas os CSVs observados apresentam aproximadamente 99,9969 us.
- Gargalo linear 4D, cinco sementes, indice final pela mediana dos scores
  normalizados por seus limiares p99.
- Split temporal com purga; nenhuma janela pos-falha entra em scaler, treino,
  early stopping ou calibracao.
- IC95% por bootstrap de ensaios, nao de janelas.

## Limites de evidencia

E3 aqui significa validacao experimental externa em bancada. Nao e validacao de
campo, nao estima prevalencia industrial, nao identifica causa automaticamente e
nao fornece tempos de vida para Weibull/RUL fisico.

## Fontes

- GPVS-Faults: https://doi.org/10.17632/n76t439f65.1
- Bakdi et al. (2021): https://doi.org/10.1016/j.ijepes.2020.106457
