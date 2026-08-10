# Calibração acadêmica do Autoencoder

> Split temporal com purga em quatro papéis disjuntos. O bloco de validação orienta somente o early stopping; o de calibração define somente o limiar; o teste isolado estima o falso positivo final.

- Escore operacional: `mse / percentil efetivo 99.0`.
- Limiar operacional (`score_threshold`): `0.857702`.
- Referência MSE p99 para gráficos de reconstrução: `0.857702`.
- O p99 é um quantil nominal interpolado (`method=linear`); com 210 janelas, a resolução empírica da cauda é 1/210 = 0.48%, não 1%.
- Intervalos entre colchetes são IC95% de Wilson, usados como referências binomiais por janela. A sobreposição e a dependência serial limitam a interpretação inferencial.

| Bloco | n janelas | n sem compartilhamento | MSE mediana | MSE IQR | MSE p99 | > ref. MSE p99 por janela | > ref. sem compartilhamento |
|---|---:|---:|---:|---:|---:|---:|---:|
| treino | 711 | 711 | 0.1735 | 0.1164 | 0.8350 | 7/711 = 0.98% [0.48; 2.02] | 7/711 = 0.98% [0.48; 2.02] |
| validacao_early_stopping | 209 | 209 | 0.1976 | 0.1421 | 0.8172 | 2/209 = 0.96% [0.26; 3.42] | 2/209 = 0.96% [0.26; 3.42] |
| calibracao_limiar | 210 | 210 | 0.2154 | 0.1094 | 0.8577 | 3/210 = 1.43% [0.49; 4.12] | 3/210 = 1.43% [0.49; 4.12] |
| teste_isolado | 281 | 281 | 0.2339 | 0.1315 | 0.9880 | 4/281 = 1.42% [0.55; 3.60] | 4/281 = 1.42% [0.55; 3.60] |

Observação metodológica: 'sem compartilhamento' usa a distância registrada no protocolo. No GPVS, as janelas são contíguas e sem sobreposição, portanto todas são retidas. Isso não garante independência estatística ou temporal. Os gráficos estão na escala MSE; a decisão operacional usa o escore registrado em `limiar.json`.
