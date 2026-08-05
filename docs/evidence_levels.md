# Níveis de evidência — Al IAdo PV

Toda afirmação sobre resultados **deve** informar o nível de evidência. O agente
**nunca** trata E1 ou E2 como prova de desempenho industrial.

| Nível | Significado | Onde aparece no projeto |
|---|---|---|
| **E0** | Hipótese | proposições ainda não testadas |
| **E1** | Benchmark **exploratório** — protocolo em features ou dataset rotulado auxiliar | `experimentos_artigos.py` (Ibrahim/AE-LSTM com injeção FMECA no espaço de features; classificação PV Farms quando usada como referência auxiliar). Limiar escolhido no próprio conjunto avaliado = E1 |
| **E2** | Validação **sintética orientada pela FMECA** — ground truth de falhas injetadas | `injecao_falhas.py` (schema por falha) e `validacao.py` (limiar congelado, `__meta__.evidence_level = E2`) |
| **E3** | Validação **experimental externa** (bancada / campo) | ainda não realizada |

Regras práticas:
- O campo `evidence_level` é gravado nos artefatos (resultado de experimento,
  report de injeção/validação).
- AUC é independente de limiar (válido); F1/recall/specificity dependem do
  limiar — se ele foi escolhido no conjunto avaliado, são **exploratórios (E1)**.
- O **Contator AC** usa ruído gaussiano como **proxy** de transiente/chattering
  → exige calibração física; alta sensibilidade observada é E2.

- **Weibull/RUL nunca perde a ressalva E2**: a censura é preservada por MLE e
  os intervalos são obtidos por bootstrap, mas os TTF continuam sintéticos.
  A unidade publicada é `passo_sintetico_de_degradacao`; sem taxa de
  degradação calibrada por campo ou bancada, não converter para horas, dias ou
  anos.
  `status_ajuste=exploratorio_descritivo` não significa vida de campo; MTTF,
  B10 e RUL devem ser acompanhados dessa ressalva no chat, gráficos e texto.
- **Detecção nula também é resultado**: se uma falha injetada não cruza o
  limiar em nenhuma severidade (SMD nula — ex.: campo `smd: null` no report de
  injeção), isso é um achado de LIMITAÇÃO do detector e deve ser reportado
  como tal, nunca suprimido da análise.
