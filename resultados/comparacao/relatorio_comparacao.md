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

## Detectabilidade E2

Contator AC, IGBT e Fusível AC usam as mesmas janelas, magnitudes e 
realizações nos dois detectores. SMD95 exige limite inferior Wilson 
de 95%; quando a condição não ocorre até a_det=1, registra-se 
`não atingido`.

Sobrevivência empírica, incidência acumulada e risco discreto vivem 
no eixo de magnitude. O Weibull 2P é apenas diagnóstico formal e 
nunca produz RUL, MTTF ou confiabilidade física.

A não aceitação de um ajuste Weibull rejeita apenas a síntese 
paramétrica correspondente; não reprova nenhum dos detectores.

Os IC95% Wilson de E2 tratam cada janela-trajetória como unidade 
Bernoulli e são apresentados como descritivos, pois janelas do 
mesmo ensaio podem permanecer autocorrelacionadas.
