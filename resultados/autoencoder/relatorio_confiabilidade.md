# Relatório de confiabilidade e ponto de operação

> **Evidência E2** — validação sintética orientada pela FMECA. Não é desempenho de campo (E3). O eixo NÃO é tempo físico: é a magnitude de injeção em que a detecção se confirma.

## Confiabilidade por modo de falha

### Contator AC (NPR 315)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 4.386 | [3.78; 5.54] |
| escala η | 39.31 | [36.64; 42.37] |

| Marco | Magnitude de injeção | R nesse ponto |
|---|--:|--:|
| B1 (1% detectado) | 13.77 | 0.990 |
| B10 (10% detectado) | 23.53 | 0.900 |
| mediana | 36.16 | 0.500 |
| η (vida característica) | 39.31 | 0.368 |
| MTTF | 35.82 | 0.514 |

**Leitura de β.** taxa de falha crescente: o risco aumenta com a idade, e existe intervalo de substituição preventiva que compensa

> **B10 (23.5) < MTTF (35.8).** A distribuição é assimétrica: a média fica acima de boa parte da população, e por isso B10/B1 são melhores indicadores de decisão de manutenção que o MTTF.

> Observação vai até 55.0; além disso as curvas são extrapolação do modelo, não dado.

### IGBT (NPR 90)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 2.299 | [1.94; 2.75] |
| escala η | 85.43 | [72.68; 100.11] |

| Marco | Magnitude de injeção | R nesse ponto |
|---|--:|--:|
| B1 (1% detectado) | 11.55 | 0.990 |
| B10 (10% detectado) | 32.10 | 0.900 |
| mediana | 72.84 | 0.500 |
| η (vida característica) | 85.43 | 0.368 |
| MTTF | 75.68 | 0.469 |

**Leitura de β.** taxa de falha crescente: o risco aumenta com a idade, e existe intervalo de substituição preventiva que compensa

> **B10 (32.1) < MTTF (75.7).** A distribuição é assimétrica: a média fica acima de boa parte da população, e por isso B10/B1 são melhores indicadores de decisão de manutenção que o MTTF.

> Observação vai até 120.0; além disso as curvas são extrapolação do modelo, não dado.

### Fusível AC (NPR 30)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 5.416 | [4.54; 6.74] |
| escala η | 97.14 | [90.00; 102.64] |

| Marco | Magnitude de injeção | R nesse ponto |
|---|--:|--:|
| B1 (1% detectado) | 41.54 | 0.990 |
| B10 (10% detectado) | 64.11 | 0.900 |
| mediana | 90.78 | 0.500 |
| η (vida característica) | 97.14 | 0.368 |
| MTTF | 89.60 | 0.524 |

**Leitura de β.** taxa de falha crescente: o risco aumenta com a idade, e existe intervalo de substituição preventiva que compensa

> **B10 (64.1) < MTTF (89.6).** A distribuição é assimétrica: a média fica acima de boa parte da população, e por isso B10/B1 são melhores indicadores de decisão de manutenção que o MTTF.

> Observação vai até 120.0; além disso as curvas são extrapolação do modelo, não dado.

---

## Ponto de operação sob o critério LS-POD

Limiar operacional adotado: **7.8262**

### A hipótese, antes do número

O método assume normalidade do lado saudável. Shapiro-Wilk no escore bruto: p = 4.67e-09; em log: p = 6.28e-04. Assimetria +2.29, curtose +9.18.

**Hipótese violada.** Por isso o mesmo quantil é estimado por três caminhos independentes abaixo — se os três concordarem, a conclusão não depende dela.

| Estimador do percentil 99 do escore saudável | Valor |
|---|--:|
| normal no escore bruto (LS-POD) | 12.2214 |
| normal em log, destransformado | 19.2417 |
| percentil 99 empírico | 14.5838 |
| **limiar adotado** | **7.8262** |

> **Os 3 estimadores ficam acima do limiar adotado.** Pelo critério LS-POD, o requisito de falso positivo de 1% **não é cumprido no bloco de teste** — e a conclusão é robusta à violação da hipótese de normalidade, porque o quantil empírico, que não assume distribuição, leva ao mesmo lugar.

### Deriva entre calibração e teste

INVALIDAÇÃO: o piso de falso positivo em campo (12.2214) ultrapassa o próprio limiar adotado (7.8262). O requisito de POF não está sendo cumprido no bloco de teste. Pela fonte, a resposta correta é investigar a causa — não reapertar o limiar, que trataria o sintoma.
