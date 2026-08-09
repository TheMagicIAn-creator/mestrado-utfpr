# Calibração acadêmica do Autoencoder

> Split temporal com purga. O bloco de teste isolado não participa do scaler, do early stopping nem da escolha de limiar.

- Escore operacional: `localizado / percentil efetivo 99.9`.
- Limiar operacional (`score_threshold`): `5.298013`.
- Referência MSE p99 para gráficos de reconstrução: `1.702435`.
- Intervalos entre colchetes são IC95% de Wilson para a proporção de excedências.

| Bloco | n | MSE mediana | MSE IQR | MSE p99 | > ref. MSE p99 | Escore mediana | > limiar operacional |
|---|---:|---:|---:|---:|---:|---:|---:|
| treino | 136 | 0.1999 | 0.2275 | 2.7539 | 5/136 = 3.68% [1.58; 8.32] | 6.1381 | 77/136 = 56.62% [48.22; 64.65] |
| calibracao | 45 | 0.2506 | 0.0935 | 1.7024 | 1/45 = 2.22% [0.39; 11.57] | 1.5484 | 2/45 = 4.44% [1.23; 14.83] |
| teste_isolado | 43 | 0.2729 | 0.0967 | 3.5614 | 2/43 = 4.65% [1.28; 15.46] | 6.5971 | 27/43 = 62.79% [47.86; 75.62] |

Observação metodológica: os gráficos `distribuicao_erro.png` e `erro_temporal.png` estão na escala MSE; a decisão operacional do pipeline usa o escore canônico registrado em `limiar.json`.
