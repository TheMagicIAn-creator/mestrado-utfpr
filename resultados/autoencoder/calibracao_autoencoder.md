# Calibração acadêmica do Autoencoder

> Split temporal com purga. O bloco de teste isolado não participa do scaler, do early stopping nem da escolha de limiar.

- Escore operacional: `localizado / percentil efetivo 99.9`.
- Limiar operacional (`score_threshold`): `7.826176`.
- Referência MSE p99 para gráficos de reconstrução: `2.545433`.
- Intervalos entre colchetes são IC95% de Wilson para a proporção de excedências.

| Bloco | n | MSE mediana | MSE IQR | MSE p99 | > ref. MSE p99 | Escore mediana | > limiar operacional |
|---|---:|---:|---:|---:|---:|---:|---:|
| treino | 274 | 0.1619 | 0.1643 | 9.2128 | 12/274 = 4.38% [2.52; 7.50] | 4.1611 | 62/274 = 22.63% [18.07; 27.94] |
| calibracao | 91 | 0.1772 | 0.1598 | 2.5454 | 1/91 = 1.10% [0.19; 5.96] | 1.3842 | 4/91 = 4.40% [1.72; 10.76] |
| teste_isolado | 88 | 0.2319 | 0.0648 | 2.8086 | 1/88 = 1.14% [0.20; 6.16] | 4.2664 | 9/88 = 10.23% [5.47; 18.31] |

Observação metodológica: os gráficos `distribuicao_erro.png` e `erro_temporal.png` estão na escala MSE; a decisão operacional do pipeline usa o escore canônico registrado em `limiar.json`.
