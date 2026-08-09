# Relatório de confiabilidade e ponto de operação

> **Evidência E2** — validação sintética orientada pela FMECA. Não é desempenho de campo (E3). O eixo NÃO é tempo físico: é a magnitude de injeção em que a detecção se confirma.

## Confiabilidade por modo de falha

### Contator AC

Ajuste não convergiu — sem curva de confiabilidade.

### IGBT

Ajuste não convergiu — sem curva de confiabilidade.

### Fusível AC

Ajuste não convergiu — sem curva de confiabilidade.

---

## Ponto de operação sob o critério LS-POD

Limiar operacional adotado: **5.2980**

### A hipótese, antes do número

O método assume normalidade do lado saudável. Shapiro-Wilk no escore bruto: p = 1.88e-04; em log: p = 1.16e-05. Assimetria +1.22, curtose +4.54.

**Hipótese violada.** Por isso o mesmo quantil é estimado por três caminhos independentes abaixo — se os três concordarem, a conclusão não depende dela.

| Estimador do percentil 99 do escore saudável | Valor |
|---|--:|
| normal no escore bruto (LS-POD) | 17.0146 |
| normal em log, destransformado | 46.9553 |
| percentil 99 empírico | 16.8434 |
| **limiar adotado** | **5.2980** |

> **Os 3 estimadores ficam acima do limiar adotado.** Pelo critério LS-POD, o requisito de falso positivo de 1% **não é cumprido no bloco de teste** — e a conclusão é robusta à violação da hipótese de normalidade, porque o quantil empírico, que não assume distribuição, leva ao mesmo lugar.

### Deriva entre calibração e teste

INVALIDAÇÃO: o piso de falso positivo em campo (17.0146) ultrapassa o próprio limiar adotado (5.2980). O requisito de POF não está sendo cumprido no bloco de teste. Pela fonte, a resposta correta é investigar a causa — não reapertar o limiar, que trataria o sintoma.
