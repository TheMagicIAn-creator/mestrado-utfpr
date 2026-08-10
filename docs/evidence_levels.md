# Níveis de evidência — Al IAdo PV

Toda afirmação sobre resultados **deve** informar o nível de evidência. O agente
**nunca** trata E1 ou E2 como prova de desempenho industrial.

| Nível | Significado | Onde aparece no projeto |
|---|---|---|
| **E0** | Hipótese | proposições ainda não testadas |
| **E1** | Benchmark **exploratório** — protocolo em features ou dataset rotulado auxiliar | `experimentos_artigos.py` (Ibrahim/AE-LSTM com injeção FMECA no espaço de features; classificação PV Farms quando usada como referência auxiliar). Limiar escolhido no próprio conjunto avaliado = E1 |
| **E2** | Validação **sintética orientada pela FMECA** — ground truth de falhas injetadas | `injecao_falhas.py` (schema por falha) e `validacao.py` (limiar congelado, `__meta__.evidence_level = E2`) |
| **E3** | Validação **experimental externa** (o escopo deve dizer bancada ou campo) | `validacao_gpvs_principal.py`: realizada em 14 ensaios GPVS-Faults, em bancada; campo ainda não realizado |

Regras práticas:
- O campo `evidence_level` é gravado nos artefatos (resultado de experimento,
  report de injeção/validação).
- **E3 não é sinônimo de campo**: o resultado GPVS é E3 de bancada. Não prova
  prevalência industrial, generalização entre plantas nem vida útil.
- No GPVS, intervalos de confiança reamostram os 14 ensaios; janelas do mesmo
  ensaio não são tratadas como replicações independentes.
- AUC é independente de limiar (válido); F1/recall/specificity dependem do
  limiar — se ele foi escolhido no conjunto avaliado, são **exploratórios (E1)**.
- O **Contator AC** usa ruído gaussiano como **proxy** de transiente/chattering
  → exige calibração física; alta sensibilidade observada é E2.

- **Weibull nunca perde a ressalva E2**: a censura é preservada por MLE e os
  intervalos são obtidos por bootstrap, mas `a_det` é magnitude sintética de
  primeiro cruzamento. Não converter para horas, dias ou anos e não chamar de
  confiabilidade, MTTF, B10 ou RUL físico.
- **Detecção nula também é resultado**: se uma falha injetada não cruza o
  limiar em nenhuma severidade (SMD nula — ex.: campo `smd: null` no report de
  injeção), isso é um achado de LIMITAÇÃO do detector e deve ser reportado
  como tal, nunca suprimido da análise.
