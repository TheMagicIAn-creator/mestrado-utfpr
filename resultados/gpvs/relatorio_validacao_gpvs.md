# Validação experimental GPVS-Faults (E3 de bancada)

Gerado em `2026-08-12T07:35:01.707312-03:00`. Dataset: DOI 10.17632/n76t439f65.1.

## Resultado principal

O Autoencoder canônico, treinado e calibrado somente nos ensaios saudáveis F0L/F0M, obteve AUC macro 0.773 (IC95% 0.691-0.853) nos 14 ensaios de falha. A sensibilidade macro no limiar congelado foi 0.406 e a especificidade pré-falha, 0.974.

Houve detecção sustentada em 9/14 ensaios. O maior AUC ocorreu em F5L (1.000) e o menor em F4M (0.534). Esses extremos são descritivos; não houve seleção de cenários por desempenho.

## Protocolo

- Uma única fonte de dados: GPVS-Faults.
- F0L/F0M: scaler, treino, early stopping, calibração do limiar e teste saudável.
- F1L-F7M: validação E3 com pesos e limiar congelados; a primeira metade pré-falha define o baseline de comissionamento e a segunda metade pré-falha mede a especificidade.
- 24 features de sensores primários em janelas não sobrepostas de um ciclo de 50 Hz.
- A taxa usada é inferida do vetor `Time` (aproximadamente 10 kHz); o período de 9,9989 µs informado no manual diverge dos CSVs e é mantido apenas como ressalva documental.
- A unidade do IC95% macro é o ensaio (bootstrap de 14 ensaios), não a janela.
- IC95% por cenário é Wilson por janela e deve ser lido com cautela devido à autocorrelação.

## Limites de evidência

E3 significa validação experimental em bancada, não validação de campo. O detector sinaliza desvio do padrão saudável; não demonstra causalidade do componente e não transforma os ensaios em tempos de vida para Weibull físico.

## Fonte

- GPVS-Faults: https://doi.org/10.17632/n76t439f65.1
