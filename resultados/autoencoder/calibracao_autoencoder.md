# Calibração acadêmica do Autoencoder

> Split temporal com purga. O bloco de teste isolado não participa do scaler, do early stopping nem da escolha de limiar.

- Escore operacional: `mse / percentil efetivo 99.0`.
- Limiar operacional (`score_threshold`): `2.582821`.
- Referência MSE p99 para gráficos de reconstrução: `2.582821`.
- Intervalos entre colchetes são IC95% de Wilson para a proporção de excedências.

| Bloco | n | MSE mediana | MSE IQR | MSE p99 | > ref. MSE p99 | Escore mediana | > limiar operacional |
|---|---:|---:|---:|---:|---:|---:|---:|
| treino | 104 | 0.2159 | 0.2275 | 2.7213 | 2/104 = 1.92% [0.53; 6.74] | 0.2159 | 2/104 = 1.92% [0.53; 6.74] |
| calibracao | 42 | 0.3683 | 0.3319 | 2.5828 | 1/42 = 2.38% [0.42; 12.32] | 0.3683 | 1/42 = 2.38% [0.42; 12.32] |
| teste_isolado | 60 | 0.2472 | 0.3200 | 3.5711 | 1/60 = 1.67% [0.29; 8.86] | 0.2472 | 1/60 = 1.67% [0.29; 8.86] |

Observação metodológica: os gráficos `distribuicao_erro.png` e `erro_temporal.png` estão na escala MSE; a decisão operacional do pipeline usa o escore canônico registrado em `limiar.json`.
