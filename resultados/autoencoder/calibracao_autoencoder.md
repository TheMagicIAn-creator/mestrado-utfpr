# Calibração acadêmica do Autoencoder

> Split temporal com purga. O bloco de teste isolado não participa do scaler, do early stopping nem da escolha de limiar.

- Escore operacional: `localizado / percentil efetivo 99.0`.
- Limiar operacional (`score_threshold`): `5.572589`.
- Referência MSE p99 para gráficos de reconstrução: `44.808807`.
- Intervalos entre colchetes são IC95% de Wilson para a proporção de excedências.

| Bloco | n | MSE mediana | MSE IQR | MSE p99 | > ref. MSE p99 | Escore mediana | > limiar operacional |
|---|---:|---:|---:|---:|---:|---:|---:|
| treino | 126 | 0.2004 | 0.2351 | 2.3534 | 0/126 = 0.00% [0.00; 2.96] | 1.7627 | 9/126 = 7.14% [3.80; 13.02] |
| calibracao | 39 | 0.3428 | 0.2743 | 44.8088 | 1/39 = 2.56% [0.45; 13.18] | 1.9279 | 1/39 = 2.56% [0.45; 13.18] |
| teste_isolado | 39 | 0.2357 | 0.2819 | 1.4759 | 0/39 = 0.00% [0.00; 8.97] | 2.1233 | 1/39 = 2.56% [0.45; 13.18] |

Observação metodológica: os gráficos `distribuicao_erro.png` e `erro_temporal.png` estão na escala MSE; a decisão operacional do pipeline usa o escore canônico registrado em `limiar.json`.
