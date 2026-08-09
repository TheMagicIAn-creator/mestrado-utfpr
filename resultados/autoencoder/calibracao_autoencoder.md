# Calibração acadêmica do Autoencoder

> Split temporal com purga. O bloco de teste isolado não participa do scaler, do early stopping nem da escolha de limiar.

- Escore operacional: `mse / percentil efetivo 99.0`.
- Limiar operacional (`score_threshold`): `2.582821`.
- Referência MSE p99 para gráficos de reconstrução: `2.582821`.
- O p99 é um quantil nominal interpolado (`method=linear`); com 42 janelas, a resolução empírica da cauda é 1/42 = 2.38%, não 1%.
- Intervalos entre colchetes são IC95% de Wilson, usados como referências binomiais por janela. A sobreposição e a dependência serial limitam a interpretação inferencial.

| Bloco | n janelas | n sem compartilhamento | MSE mediana | MSE IQR | MSE p99 | > ref. MSE p99 por janela | > ref. sem compartilhamento |
|---|---:|---:|---:|---:|---:|---:|---:|
| treino | 104 | 52 | 0.2159 | 0.2275 | 2.7213 | 2/104 = 1.92% [0.53; 6.74] | 0/52 = 0.00% [0.00; 6.88] |
| calibracao | 42 | 21 | 0.3683 | 0.3319 | 2.5828 | 1/42 = 2.38% [0.42; 12.32] | 0/21 = 0.00% [0.00; 15.46] |
| teste_isolado | 60 | 32 | 0.2472 | 0.3200 | 3.5711 | 1/60 = 1.67% [0.29; 8.86] | 1/32 = 3.12% [0.55; 15.74] |

Observação metodológica: 'sem compartilhamento' retém uma janela a cada duas dentro de cada bloco, coerente com 50% de sobreposição. Isso remove amostras brutas compartilhadas, mas não garante independência estatística ou temporal. Os gráficos estão na escala MSE; a decisão operacional usa o escore registrado em `limiar.json`.
