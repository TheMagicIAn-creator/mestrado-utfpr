# Comparação canônica: Autoencoder Denso versus AE-LSTM

## Delineamento

O GPVS-Faults é a única fonte experimental. F0L/F0M fornecem treino,
validação, calibração e teste saudável em blocos temporais disjuntos.
F1L-F7M permanecem fora do ajuste e formam a evidência E3 de bancada.

### Capacidade e regularização dos dois braços

| Modelo | Arquitetura | Parâmetros | Dropout |
|---|---|---:|---:|
| Autoencoder Denso | 24-16-8-16-24 | 1088 | 0.20 |
| AE-LSTM | AE-LSTM temporal: L=8, hidden=32, latent=8 | 17216 | 0.20 |

Os dois braços recebem o mesmo orçamento de treino — otimizador, taxa de aprendizado, épocas máximas, paciência e lote —, as mesmas 24 features e as mesmas sementes. Isso NÃO é a mesma capacidade: a contagem de parâmetros acima difere por cerca de uma ordem de grandeza, e a diferença é inerente às duas arquiteturas, não uma escolha de ajuste. A tabela existe para que essa assimetria seja lida junto com o resultado, e não descoberta depois dele.

Desde 2026-09-03 o dropout é igual nos dois. Antes disso o braço temporal não tinha regularização alguma — `nn.LSTM` de camada única ignora o argumento `dropout` —, o que abria uma explicação concorrente para o seu desempenho: um autoencoder grande e não regularizado tende a reconstruir bem demais, inclusive o que deveria destoar.

## Resultado experimental E3

| Modelo | Recall | F1 | Precision | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Autoencoder Denso | 0.407 (IC95% 0.212-0.611) | 0.467 (IC95% 0.266-0.668) | 0.923 (IC95% 0.870-0.968) | 0.774 (IC95% 0.691-0.853) | 0.862 (IC95% 0.806-0.916) |
| AE-LSTM | 0.387 (IC95% 0.184-0.601) | 0.436 (IC95% 0.216-0.658) | 0.947 (IC95% 0.860-1.000) | 0.759 (IC95% 0.676-0.842) | 0.847 (IC95% 0.790-0.906) |

Recall, F1 e Precision formam a camada principal. ROC-AUC e PR-AUC são medidas complementares de discriminação. Precision é N/A quando a execução não produz nenhum alarme positivo.

Precision teve valor finito em 13/14 ensaios do Autoencoder Denso e 10/14 do AE-LSTM.

## Matrizes de confusão agregadas

| Modelo | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Autoencoder Denso | 1965 | 65 | 2283 | 2719 |
| AE-LSTM | 1864 | 44 | 2304 | 2820 |

As contagens são agregadas por janela e têm uso descritivo devido à autocorrelação dentro de cada ensaio.

## Ponto operacional

| Modelo | Top-k | Limiar | Percentil solicitado | Percentil efetivo | Ordem | Resolução | FP no teste saudável | É o máximo amostral |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Autoencoder Denso | 5 | 3.461895 | p99.0 | p99.048 | 208/210 | 0.476 p.p. | 1.779% | não |
| AE-LSTM | 5 | 17.381609 | p99.0 | p99.048 | 208/210 | 0.476 p.p. | 1.423% | não |

A unidade inferencial do intervalo é o ensaio, não a janela. A
semente 42 é pré-fixada; cinco sementes descrevem estabilidade sem
selecionar o modelo pelo desempenho nas falhas.

A fronteira de falha é nominalmente 50% do registro porque os CSVs
não contêm canal instrumentado de disparo.

O ponto operacional canônico usa a média dos cinco maiores erros quadráticos por feature e o percentil saudável solicitado da tabela acima. Ele é pré-fixado, não um ótimo universal. Cada modelo calibra seu próprio limiar somente no bloco saudável; o contrato registra o order statistic, o percentil empírico efetivamente alcançável e se o limiar coincide com o máximo da calibração.

A última coluna existe porque um percentil só é representável quando a calibração é grande o bastante. Pedir p99,9 exigiria n >= 1001; a calibração saudável do GPVS tem 210 janelas, e o pedido caía na ordem 210/210 — o limiar era o MAIOR escore visto, com a variância de um máximo amostral e não a de um quantil. Por decisão do pesquisador em 2026-09-03 o ponto canônico passou a p99, que 210 observações sustentam (ordem 208/210). O ponto histórico p99,9 permanece na grade de sensibilidade, marcado como tal.

## Ablação temporal do AE-LSTM

A análise canônica usa a sequência causal contínua [W_(t-7), ..., W_t] e decide em W_t. Ela separa as sete primeiras janelas pós-fronteira da falha sustentada, cujo contexto já é integralmente pós-fronteira. O reinício pós-falha permanece apenas como diagnóstico auxiliar. Treino, scaler, escore e limiares permanecem congelados.

| Métrica | AE-LSTM − Denso na falha sustentada | IC95% |
|---|---:|---:|
| Recall | -0.020 | -0.053 a 0.001 |
| F1 | -0.032 | -0.084 a 0.004 |
| Precision | 0.011 | -0.076 a 0.098 |

Conclusão pré-especificada: `does_not_survive`. Recall ou F1 apresenta diferença pontual negativa na falha sustentada.

## Sensibilidade do escore e do limiar

A grade complementar usa `k = 5, 10, 20` e percentis solicitados p99, p99,5 e p99,9, totalizando nove configurações por modelo e semente. Cada limiar é derivado exclusivamente da calibração saudável; as falhas não selecionam a configuração.

| Modelo | FP saudável mínimo-máximo | Recall E3 mínimo-máximo | F1 E3 mínimo-máximo | Precision E3 mínimo-máximo |
|---|---:|---:|---:|---:|
| Autoencoder Denso | 0.712%–1.779% | 0.384–0.413 | 0.432–0.474 | 0.870–0.934 |
| AE-LSTM | 1.068%–1.423% | 0.385–0.388 | 0.433–0.437 | 0.943–0.950 |

k=5 com p99,9 solicitado permanece somente como referência histórica de reprodutibilidade. A tabela registra também o percentil empírico efetivo, a estatística de ordem e a resolução da calibração; esta análise não promove uma configuração a partir das falhas.
