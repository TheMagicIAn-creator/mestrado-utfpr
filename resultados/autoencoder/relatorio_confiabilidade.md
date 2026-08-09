# Relatório de confiabilidade e ponto de operação

> **Evidência E2** — validação sintética orientada pela FMECA. Não é desempenho de campo (E3). O eixo NÃO é tempo físico: é a magnitude de injeção em que a detecção se confirma.

## Confiabilidade por modo de falha

### Contator AC (NPR 315)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 4.887 | [3.77; 8.08] |
| escala η | 0.16 | [0.15; 0.18] |

| Marco | Magnitude de injeção | R nesse ponto |
|---|--:|--:|
| B1 (1% detectado) | 0.06 | 0.990 |
| B10 (10% detectado) | 0.10 | 0.900 |
| mediana | 0.15 | 0.500 |
| η (vida característica) | 0.16 | 0.368 |
| MTTF | 0.15 | 0.520 |

**Leitura de β.** taxa de falha crescente: o risco aumenta com a idade, e existe intervalo de substituição preventiva que compensa

> **B10 (0.1) < MTTF (0.1).** A distribuição é assimétrica: a média fica acima de boa parte da população, e por isso B10/B1 são melhores indicadores de decisão de manutenção que o MTTF.

> Observação vai até 0.2; além disso as curvas são extrapolação do modelo, não dado.

### IGBT

Ajuste não convergiu — sem curva de confiabilidade.

### Fusível AC (NPR 30)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 6.089 | [4.29; 10.40] |
| escala η | 0.65 | [0.59; 0.69] |

| Marco | Magnitude de injeção | R nesse ponto |
|---|--:|--:|
| B1 (1% detectado) | 0.30 | 0.990 |
| B10 (10% detectado) | 0.45 | 0.900 |
| mediana | 0.61 | 0.500 |
| η (vida característica) | 0.65 | 0.368 |
| MTTF | 0.60 | 0.529 |

**Leitura de β.** taxa de falha crescente: o risco aumenta com a idade, e existe intervalo de substituição preventiva que compensa

> **B10 (0.4) < MTTF (0.6).** A distribuição é assimétrica: a média fica acima de boa parte da população, e por isso B10/B1 são melhores indicadores de decisão de manutenção que o MTTF.

> Observação vai até 0.8; além disso as curvas são extrapolação do modelo, não dado.

---

## Ponto de operação sob o critério LS-POD

Limiar operacional adotado: **5.5726**

### A hipótese, antes do número

O método assume normalidade do lado saudável. Shapiro-Wilk no escore bruto: p = 2.07e-01; em log: p = 1.30e-03. Assimetria +0.69, curtose +0.35.

Hipótese satisfeita na escala **bruto**.

| Estimador do percentil 99 do escore saudável | Valor |
|---|--:|
| normal no escore bruto (LS-POD) | 6.2249 |
| normal em log, destransformado | 16.7034 |
| percentil 99 empírico | 5.7265 |
| **limiar adotado** | **5.5726** |

> **Os 3 estimadores ficam acima do limiar adotado.** Pelo critério LS-POD, o requisito de falso positivo de 1% **não é cumprido no bloco de teste** — e a conclusão é robusta à violação da hipótese de normalidade, porque o quantil empírico, que não assume distribuição, leva ao mesmo lugar.

### Deriva entre calibração e teste

INVALIDAÇÃO: o piso de falso positivo em campo (6.2249) ultrapassa o próprio limiar adotado (5.5726). O requisito de POF não está sendo cumprido no bloco de teste. Pela fonte, a resposta correta é investigar a causa — não reapertar o limiar, que trataria o sintoma.
