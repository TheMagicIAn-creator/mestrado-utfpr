---
al_iado: true
titulo: "Detectabilidade E2 e ajuste Weibull auditado"
tipo: resultado
status: superado
superado_em: 2026-09-03
confianca: media
nivel_evidencia: E2
registrado_em: 2026-08-09
tags: [cerebro, detectabilidade, weibull, evidencia-e2, autoencoder, resultado]
---

> [!warning] Nota superada em 2026-09-03
> Os números desta nota vieram do pipeline de macro-códigos e do recorte de
> componentes **Contator AC / IGBT / Fusível AC**, que não é a FMECA vigente.
> A FMECA canônica hoje é **IGBT / sensor-realimentação / controle**
> (`docs/fmeca.md`), e a detectabilidade `a_det` está sendo reimplementada
> sobre o pipeline novo. Use esta nota como registro histórico, nunca como
> resultado corrente.

# Detectabilidade E2 e ajuste Weibull auditado

Esta etapa modela a magnitude do primeiro cruzamento confirmado do detector em
falhas sintéticas. O eixo `a_det` é fração da assinatura nominal, não tempo.
Consequentemente, os resultados não são confiabilidade, taxa de falha, desgaste
ou RUL dos componentes.

| Modo | Detectadas/total | beta (IC95%) | eta (IC95%) | R²pp | Decisão |
|---|---:|---:|---:|---:|---|
| Contator AC | 31/31 | 3,148 [2,658; 3,958] | 0,475 [0,418; 0,529] | 0,955 | síntese exploratória E2 |
| IGBT | 12/31 | 6,578 [4,396; 10,967] | 1,112 [1,020; 1,198] | 0,947 | não reportar: 61% indetectável |
| Fusível AC | 30/31 | 8,599 [4,791; 15,772] | 0,896 [0,856; 0,927] | -0,843 | não reportar: desvio da Weibull 2P |

Os pontos empíricos usam Kaplan-Meier modificado com o tamanho total da
amostra. O bootstrap usa 1.000 réplicas de janelas sem amostras compartilhadas,
mas não presume independência temporal. No IGBT, 819 réplicas foram válidas.

Nomes canônicos: `a10`, média de `a_det`, `S_D(a)` para não detecção, `h_D(a)`
para intensidade de primeiro cruzamento e margem residual de magnitude.
`MTTF/B10/RUL` são apenas aliases legados no JSON.

Relatório técnico: `docs/auditoria_academica_weibull_confiabilidade_2026-08-09.md`.

## Conexões

- [[00 - Painel do cerebro]]

> Nota curada registrada pelo Al IAdo PV. Não é citação bibliográfica; os
> números devem ser conferidos em `resultados/autoencoder/weibull_results.json`.
