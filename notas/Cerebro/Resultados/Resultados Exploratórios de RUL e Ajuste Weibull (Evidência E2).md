---
al_iado: true
titulo: "Resultados Exploratórios de RUL e Ajuste Weibull (Evidência E2)"
tipo: resultado
status: ativo
confianca: media
nivel_evidencia: E2
registrado_em: 2026-07-29
tags: [cerebro, confiabilidade, weibull-rul, evidencia-e2, contator-ac, igbt, fusivel-ac, fmeca, resultado]
---

# Resultados Exploratórios de RUL / Weibull

Resultados obtidos para a análise de confiabilidade e estimativa de vida útil remanescente (RUL) nos componentes avaliados.

## Tabela de Métricas

| Falha | NPR | Eventos/Censura | $\beta$ (IC95%) | $\eta$ (IC95%) | MTTF (IC95%) | B10 (IC95%) | RUL Restrita Inicial | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Contator AC | 315 | 38/0 | 4.39 [3.78; 5.54] | 39.3 [36.6; 42.4] | 35.8 [33.3; 38.7] | 23.5 [21.2; 26.6] | 35.868 | exploratório |
| IGBT | 90 | 32/6 | 2.30 [1.94; 2.75] | 85.4 [72.7; 100.1] | 75.7 [64.6; 88.7] | 32.1 [27.1; 38.7] | 72.158 | exploratório |
| Fusível AC | 30 | 37/1 | 5.42 [4.54; 6.74] | 97.1 [90.0; 102.6] | 89.6 [82.9; 95.6] | 64.1 [56.6; 72.3] | 89.368 | exploratório |

## Diretrizes Methodológicas e Separação de Estimativas

- **RUL Restrita Inicial**: Trata-se exclusivamente da média residual **não paramétrica de Kaplan-Meier**, truncada no horizonte observado. **Não** deve ser descrita como RUL Weibull.
- **RUL Weibull**: Corresponde à estimativa **paramétrica/extrapolativa**, existente apenas quando o ajuste convergiu (com ressalvas em cenários de alta censura).
- **Intervalos de Confiança**: Gerados via *bootstrap* com preservação da censura.

## Ressalvas de Evidência (E2)

> [!WARNING] Nível de Evidência: E2 (Sintético)
> - Os tempos analisados referem-se a **passos de degradação sintética (E2)**. 
> - As métricas (MTTF, B10 e RUL) descrevem estritamente o experimento computacional e **não podem ser apresentadas como vida útil física ou de campo**.
> - O NPR (Número de Prioridade de Risco) é utilizado para priorização na FMECA, **não determinando** a quantidade de eventos produzidos no experimento sintético nem explicando causalmente a censura.

## Conexões

- [[00 - Painel do cerebro]]

> Nota curada registrada pelo Al IAdo PV. Não é citação bibliográfica;
> métricas devem ser conferidas nos artefatos de `resultados/`.
