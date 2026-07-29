---
al_iado: true
titulo: "Resultados Exploratórios de Confiabilidade: Weibull e RUL (E2)"
tipo: resultado
status: ativo
confianca: media
nivel_evidencia: E2
registrado_em: 2026-07-29
tags: [cerebro, confiabilidade, weibull-rul, evidencia-e2, contator-ac, igbt, fusivel-ac, resultado]
---

# Resultados Exploratórios de Confiabilidade: Weibull e RUL (E2)

## Sumário de Métricas Sintéticas

| Falha | NPR | Eventos/Censura | $\beta$ (IC95%) | $\eta$ (IC95%) | MTTF (IC95%) | B10 (IC95%) | RUL Restrita Inicial (KM) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Contator AC** | 315 | 38/0 | 4.39 [3.78; 5.54] | 39.3 [36.6; 42.4] | 35.8 [33.3; 38.7] | 23.5 [21.2; 26.6] | 35.868 | Exploratório |
| **IGBT** | 90 | 32/6 | 2.30 [1.94; 2.75] | 85.4 [72.7; 100.1] | 75.7 [64.6; 88.7] | 32.1 [27.1; 38.7] | 72.158 | Exploratório |
| **Fusível AC** | 30 | 37/1 | 5.42 [4.54; 6.74] | 97.1 [90.0; 102.6] | 89.6 [82.9; 95.6] | 64.1 [56.6; 72.3] | 89.368 | Exploratório |

---

## Diretrizes e Separação Metodológica

1. **Diferenciação Obrigatória de RUL**:
   - **RUL Restrita Inicial**: Trata-se exclusivamente da média residual não paramétrica de **Kaplan-Meier (KM)**, truncada no horizonte observado. Jamais descrevê-la como RUL Weibull.
   - **RUL Weibull**: Curva paramétrica/extrapolativa, aplicável apenas quando o ajuste convergiu e sujeita a ressalva sob alta censura.

2. **Escopo da Evidência (E2)**:
   - Os tempos representam passos de degradação sintética **E2**.
   - Intervalos de confiança (IC95%) derivados de *bootstrap* e censura preservada.
   - As métricas (MTTF, B10 e RUL) descrevem estritamente o **experimento computacional**, não constituindo vida útil física ou de campo.

3. **Relação FMECA / NPR**:
   - O NPR prioriza o risco dentro do escopo da FMECA.
   - O NPR **não** determina causalmente a quantidade de eventos produzidos no experimento sintético nem explica a presença de censura.

## Conexões

- [[00 - Painel do cerebro]]

> Nota curada registrada pelo Al IAdo PV. Não é citação bibliográfica;
> métricas devem ser conferidas nos artefatos de `resultados/`.
