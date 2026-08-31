# Comparação canônica: Autoencoder Denso versus AE-LSTM

## Delineamento

O GPVS-Faults é a única fonte experimental. F0L/F0M fornecem treino,
validação, calibração e teste saudável em blocos temporais disjuntos.
F1L-F7M permanecem fora do ajuste e formam a evidência E3 de bancada.

## Resultado experimental E3

| Modelo | Recall | F1 | Precision | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Autoencoder Denso | 0.384 (IC95% 0.180-0.596) | 0.432 (IC95% 0.215-0.652) | 0.870 (IC95% 0.667-0.997) | 0.774 (IC95% 0.691-0.853) | 0.862 (IC95% 0.806-0.916) |
| AE-LSTM | 0.387 (IC95% 0.181-0.603) | 0.433 (IC95% 0.212-0.657) | 0.942 (IC95% 0.848-0.998) | 0.749 (IC95% 0.661-0.836) | 0.841 (IC95% 0.780-0.903) |

Recall, F1 e Precision formam a camada principal. ROC-AUC e PR-AUC são medidas complementares de discriminação. Precision é N/A quando a execução não produz nenhum alarme positivo.

Precision teve valor finito em 12/14 ensaios do Autoencoder Denso e 9/14 do AE-LSTM.

## Matrizes de confusão agregadas

| Modelo | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Autoencoder Denso | 1849 | 20 | 2328 | 2835 |
| AE-LSTM | 1861 | 38 | 2310 | 2823 |

As contagens são agregadas por janela e têm uso descritivo devido à autocorrelação dentro de cada ensaio.

## Ponto operacional

| Modelo | Top-k | Limiar | Percentil solicitado | Percentil efetivo | Ordem | Resolução | FP no teste saudável |
|---|---:|---:|---:|---:|---:|---:|---:|
| Autoencoder Denso | 5 | 5.104478 | p99.9 | p100.000 | 210/210 | 0.476 p.p. | 0.712% |
| AE-LSTM | 5 | 18.583059 | p99.9 | p100.000 | 210/210 | 0.476 p.p. | 1.068% |

A unidade inferencial do intervalo é o ensaio, não a janela. A
semente 42 é pré-fixada; cinco sementes descrevem estabilidade sem
selecionar o modelo pelo desempenho nas falhas.

A fronteira de falha é nominalmente 50% do registro porque os CSVs
não contêm canal instrumentado de disparo.

O escore é a média dos cinco maiores erros quadráticos por feature. Cada modelo calibra seu próprio limiar no bloco saudável de calibração usando p99,9 solicitado; o contrato registra o order statistic e o percentil empírico efetivamente alcançável.

## Ablação temporal do AE-LSTM

A análise suplementar separa as sete primeiras janelas pós-fronteira da falha sustentada e também reinicia o contexto pós-falha do AE-LSTM. Treino, scaler, escore e limiares permanecem congelados.

| Métrica | AE-LSTM − Denso na falha sustentada | IC95% |
|---|---:|---:|
| Recall | 0.002 | -0.005 a 0.011 |
| F1 | 0.000 | -0.013 a 0.015 |
| Precision | 0.055 | -0.020 a 0.181 |

Conclusão pré-especificada: `inconclusive`. Os intervalos da falha sustentada não sustentam superioridade inequívoca.
