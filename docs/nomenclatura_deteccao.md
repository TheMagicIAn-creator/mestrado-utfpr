# Nomenclatura de detecção e confiabilidade

## Campo e monitoramento

`D_campo` é o índice da FMECA e cresce com a dificuldade de detectar a falha no
processo de manutenção. A saída dos Autoencoders é descrita por erro de
reconstrução, limiar, decisão binária e métricas com intervalo de confiança.
O projeto não converte esse desempenho em um novo NPR.

## Comparação experimental

Use “comparação Denso versus AE-LSTM” para os resultados E3. O nome do dataset
deve aparecer apenas em método, fonte e proveniência. Acurácia, AUC-PR, ROC-AUC,
MCC, F1, sensibilidade e especificidade são métricas dos modelos no protocolo
avaliado, não propriedades autônomas do dataset.

## Tempo físico

Somente a publicação bibliográfica usa `t` em horas ou anos. `R(t)`, `F(t)`,
`f(t)` e `h(t)` não são calculadas a partir dos escores dos Autoencoders. Termos
como vida útil, RUL, MTTF, Weibull e distribuição normal só podem ser usados se
houver dados de vida compatíveis com a estimativa pretendida.
