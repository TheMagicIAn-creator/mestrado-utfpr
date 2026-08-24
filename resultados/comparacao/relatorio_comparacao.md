# Comparação canônica: Autoencoder Denso versus AE-LSTM

## Delineamento

O GPVS-Faults é a única fonte experimental. F0L/F0M fornecem treino,
validação, calibração e teste saudável em blocos temporais disjuntos.
F1L-F7M permanecem fora do ajuste e formam a evidência E3 de bancada.

## Resultado experimental E3

- **Autoencoder Denso:** AUC-PR macro 0.861 (IC95% 0.804-0.915).
- **AE-LSTM:** AUC-PR macro 0.841 (IC95% 0.780-0.903).

A unidade inferencial do intervalo é o ensaio, não a janela. A
semente 42 é pré-fixada; cinco sementes descrevem estabilidade sem
selecionar o modelo pelo desempenho nas falhas.

A fronteira de falha é nominalmente 50% do registro porque os CSVs
não contêm canal instrumentado de disparo.
