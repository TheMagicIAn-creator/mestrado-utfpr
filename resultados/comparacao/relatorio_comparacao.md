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

O ponto operacional reproduzido usa a média dos cinco maiores erros quadráticos por feature e p99,9 solicitado. Ele é uma referência histórica pré-fixada, não um ótimo universal. Cada modelo calibra seu próprio limiar somente no bloco saudável; o contrato registra o order statistic e o percentil empírico efetivamente alcançável.

## Ablação temporal do AE-LSTM

A análise canônica usa a sequência causal contínua [W_(t-7), ..., W_t] e decide em W_t. Ela separa as sete primeiras janelas pós-fronteira da falha sustentada, cujo contexto já é integralmente pós-fronteira. O reinício pós-falha permanece apenas como diagnóstico auxiliar. Treino, scaler, escore e limiares permanecem congelados.

| Métrica | AE-LSTM − Denso na falha sustentada | IC95% |
|---|---:|---:|
| Recall | 0.002 | -0.005 a 0.011 |
| F1 | 0.000 | -0.013 a 0.015 |
| Precision | 0.055 | -0.020 a 0.181 |

Conclusão pré-especificada: `inconclusive`. Os intervalos da falha sustentada não sustentam superioridade inequívoca.

## Sensibilidade do escore e do limiar

A grade complementar usa `k = 5, 10, 20` e percentis solicitados p99, p99,5 e p99,9, totalizando nove configurações por modelo e semente. Cada limiar é derivado exclusivamente da calibração saudável; as falhas não selecionam a configuração.

| Modelo | FP saudável mínimo-máximo | Recall E3 mínimo-máximo | F1 E3 mínimo-máximo | Precision E3 mínimo-máximo |
|---|---:|---:|---:|---:|
| Autoencoder Denso | 0.712%–1.779% | 0.384–0.413 | 0.432–0.474 | 0.870–0.934 |
| AE-LSTM | 1.068%–2.847% | 0.387–0.391 | 0.433–0.436 | 0.941–0.950 |

k=5 com p99,9 solicitado permanece somente como referência histórica de reprodutibilidade. A tabela registra também o percentil empírico efetivo, a estatística de ordem e a resolução da calibração; esta análise não promove uma configuração a partir das falhas.
