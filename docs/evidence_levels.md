# Níveis de evidência — Al IAdo PV

Toda afirmação sobre resultados **deve** informar o nível de evidência. O agente
**nunca** trata E1 ou E2 como prova de desempenho industrial.

| Nível | Significado | Onde aparece no projeto |
|---|---|---|
| **E0** | Hipótese | proposições ainda não testadas |
| **E1** | Benchmark **exploratório** — perturbação genérica ou dataset rotulado | `experimentos_artigos.py` (anomalia com perturbação genérica; classificação PV Farms). Limiar escolhido no próprio conjunto avaliado = E1 |
| **E2** | Validação **sintética orientada pelo FMEA** — ground truth de falhas injetadas | `injecao_falhas.py` (schema por falha) e `validacao.py` (limiar congelado, `__meta__.evidence_level = E2`) |
| **E3** | Validação **experimental externa** (bancada / campo) | ainda não realizada |

Regras práticas:
- O campo `evidence_level` é gravado nos artefatos (resultado de experimento,
  report de injeção/validação).
- AUC é independente de limiar (válido); F1/recall/specificity dependem do
  limiar — se ele foi escolhido no conjunto avaliado, são **exploratórios (E1)**.
- A falha de **Sensor CA** usa ruído gaussiano como **proxy** → exige calibração
  física; alta sensibilidade observada é E2, não validada em bancada.

- **Ajuste estatístico rejeitado NÃO é omitido**: quando um teste de aderência
  rejeita o modelo assumido (ex.: KS rejeita Weibull nos TTF simulados —
  campo `ajuste_weibull_adequado: false` em `weibull_results.json`), os
  parâmetros derivados (beta, eta, MTTF, B10) são **indicativos, não
  conclusivos**, e a ressalva deve acompanhar QUALQUER citação deles — no chat,
  nos gráficos e na dissertação. Reportar o p-valor junto (`p<0,05` rejeita).
- **Detecção nula também é resultado**: se uma falha injetada não cruza o
  limiar em nenhuma severidade (SMD nula — ex.: campo `smd: null` no report de
  injeção), isso é um achado de LIMITAÇÃO do detector e deve ser reportado
  como tal, nunca suprimido da análise.
