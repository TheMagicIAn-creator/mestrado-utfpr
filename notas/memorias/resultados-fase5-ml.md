# Resultados da Fase 5 — Pipeline de ML
> Gerado em 24/05/2026 17:51

## Autoencoder
O Autoencoder de detecção de anomalias foi treinado com limiar operacional (percentil 99) de 2.9103. O erro de reconstrução baseline do inversor saudável tem média 0.3214 e desvio 0.5017. Foram treinadas 84 épocas.

## Injeção de Falhas Sintéticas
- Degradação Filtro LCL (NPR=210): severidade mínima detectável = 1.0.
- Desbalanceamento de Fase (NPR=150): severidade mínima detectável = 0.3.
- Falha de Sensor CA (NPR=None): severidade mínima detectável = 0.1.

## Validação Formal
- lcl_sev0.3: AUC=0.678, F1=0.109, Recall=0.060.
- lcl_sev0.5: AUC=0.795, F1=0.323, Recall=0.200.
- lcl_sev1.0: AUC=0.935, F1=0.632, Recall=0.480.
- desbalanceamento_sev0.3: AUC=0.982, F1=0.818, Recall=0.720.
- desbalanceamento_sev0.5: AUC=1.000, F1=0.980, Recall=1.000.
- desbalanceamento_sev1.0: AUC=1.000, F1=0.980, Recall=1.000.
- sensor_sev0.3: AUC=1.000, F1=0.980, Recall=1.000.
- sensor_sev0.5: AUC=1.000, F1=0.980, Recall=1.000.
- sensor_sev1.0: AUC=1.000, F1=0.980, Recall=1.000.

## RUL — Análise de Weibull
- Degradação Filtro LCL: β=2.25, η=46.0, MTTF=40.7, B10=16.9. Taxa de falha crescente.
- Desbalanceamento de Fase: β=3.32, η=29.4, MTTF=26.4, B10=14.9. Taxa de falha crescente.
- Falha de Sensor CA: β=4.23, η=5.2, MTTF=4.8, B10=3.1. Taxa de falha crescente.
