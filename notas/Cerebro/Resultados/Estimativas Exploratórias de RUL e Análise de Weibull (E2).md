---
al_iado: true
titulo: "Estimativas Exploratórias de RUL e Análise de Weibull (E2)"
tipo: resultado
status: ativo
confianca: media
nivel_evidencia: E2
registrado_em: 2026-07-29
tags: [cerebro, confiabilidade, contator-ac, evidencia-e2, fmeca, fusivel-ac, igbt, resultado, weibull-rul]
---

# Estimativas Exploratórias de RUL e Análise de Weibull

Resultados do pipeline de confiabilidade calculados a partir de passos de degradação sintética (**Evidência E2**).

## Tabela de Resultados

| Falha | NPR | Eventos/Censura | beta (IC95%) | eta (IC95%) | MTTF (IC95%) | B10 (IC95%) | RUL restrita inicial | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Contator AC | 315 | 38/0 | 4.39 [3.78; 5.54] | 39.3 [36.6; 42.4] | 35.8 [33.3; 38.7] | 23.5 [21.2; 26.6] | 35.868 | exploratório |
| IGBT | 90 | 32/6 | 2.30 [1.94; 2.75] | 85.4 [72.7; 100.1] | 75.7 [64.6; 88.7] | 32.1 [27.1; 38.7] | 72.158 | exploratório |
| Fusível AC | 30 | 37/1 | 5.42 [4.54; 6.74] | 97.1 [90.0; 102.6] | 89.6 [82.9; 95.6] | 64.1 [56.6; 72.3] | 89.368 | exploratório |

## Diretrizes Metodológicas e Ressalvas

- **Separação das Estimativas**: A **RUL restrita inicial** é exclusivamente a média residual **não paramétrica de Kaplan-Meier** truncada no horizonte observado. Nunca descrevê-la como RUL Weibull (que é a estimativa paramétrica/extrapolativa).
- **Natureza dos Dados (E2)**: Os tempos correspondem a passos de degradação sintética E2. As métricas de MTTF, B10 e RUL descrevem o experimento computacional e **não podem ser apresentadas como vida útil física ou de campo**.
- **Tratamento Estatístico**: A censura foi preservada e os intervalos de confiança (IC95%) foram obtidos via bootstrap.
- **Papel do NPR**: Prioriza o risco na FMECA, mas **não determina** o volume de eventos produzidos no experimento sintético e nem justifica causalmente a censura.

## Conexões

- [[00 - Painel do cerebro]]

> Nota curada registrada pelo Al IAdo PV. Não é citação bibliográfica;
> métricas devem ser conferidas nos artefatos de `resultados/`.
