# Relatório de detectabilidade E2 e ponto de operação

> **Evidência E2** — validação sintética orientada pela FMECA. Não é desempenho de campo (E3). O eixo NÃO é tempo físico: é a magnitude de injeção em que a detecção se confirma. Portanto não há RUL, MTTF, taxa de falha ou confiabilidade física nesta seção.

## Detectabilidade sintética por modo de falha

### Contator AC (NPR 315)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 3.868 | [3.63; 4.21] |
| escala η | 0.40 | [0.38; 0.41] |

Diagnóstico visual: **R²pp = 0.933**. Aderência por bootstrap quantizado: **p = 0.004** (`desvio_detectado_bootstrap_quantizado`). Estabilidade entre grades finas: **sim**.
A curva e os parâmetros permanecem visíveis para auditoria, mas a síntese Weibull 2P é exploratória e não sustenta inferência física.

> Margem restrita KM no início: **0.36**. É descritiva no domínio observado e não é RUL.

| Marco paramétrico E2 | Magnitude de injeção | S_D nesse ponto |
|---|--:|--:|
| a01 (1% detectado) | 0.12 | 0.990 |
| a10 (10% detectado) | 0.22 | 0.900 |
| a50 (mediana) | 0.36 | 0.500 |
| η (escala característica) | 0.40 | 0.368 |
| média paramétrica de a_det | 0.36 | 0.507 |

**Leitura de β.** a intensidade parametrica do primeiro cruzamento aumenta com a magnitude injetada. Como o eixo nao e idade, isso NAO significa desgaste nem autoriza intervalo de manutencao

> β descreve somente a forma da intensidade de detecção em função da magnitude. Não implica desgaste, mortalidade infantil ou política de substituição.

> Observação vai até 0.6; qualquer marco além disso é extrapolação do modelo, não dado.

#### Estratificação por modo operacional GPVS

| Modo | n | beta | eta | p bootstrap | Grade estável | Uso 2P |
|---|---:|---:|---:|---:|---|---|
| F0L | 142 | 5.492 | 0.462 | 0.004 | sim | exploratório |
| F0M | 135 | 5.964 | 0.310 | 0.167 | sim | adotado em E2 |

> F0L (IPPT) e F0M (MPPT) são regimes do mesmo dataset GPVS. Diferenças entre eles não são eventos adicionais nem mistura de bases; são heterogeneidade operacional explícita.

### IGBT (NPR 90)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 3.379 | [3.19; 3.60] |
| escala η | 0.31 | [0.30; 0.32] |

Diagnóstico visual: **R²pp = 0.929**. Aderência por bootstrap quantizado: **p = 0.004** (`desvio_detectado_bootstrap_quantizado`). Estabilidade entre grades finas: **sim**.
A curva e os parâmetros permanecem visíveis para auditoria, mas a síntese Weibull 2P é exploratória e não sustenta inferência física.

> Margem restrita KM no início: **0.28**. É descritiva no domínio observado e não é RUL.

| Marco paramétrico E2 | Magnitude de injeção | S_D nesse ponto |
|---|--:|--:|
| a01 (1% detectado) | 0.08 | 0.990 |
| a10 (10% detectado) | 0.16 | 0.900 |
| a50 (mediana) | 0.28 | 0.500 |
| η (escala característica) | 0.31 | 0.368 |
| média paramétrica de a_det | 0.28 | 0.499 |

**Leitura de β.** a intensidade parametrica do primeiro cruzamento aumenta com a magnitude injetada. Como o eixo nao e idade, isso NAO significa desgaste nem autoriza intervalo de manutencao

> β descreve somente a forma da intensidade de detecção em função da magnitude. Não implica desgaste, mortalidade infantil ou política de substituição.

> Observação vai até 0.5; qualquer marco além disso é extrapolação do modelo, não dado.

#### Estratificação por modo operacional GPVS

| Modo | n | beta | eta | p bootstrap | Grade estável | Uso 2P |
|---|---:|---:|---:|---:|---|---|
| F0L | 142 | 4.685 | 0.369 | 0.004 | sim | exploratório |
| F0M | 135 | 6.254 | 0.237 | 0.371 | sim | adotado em E2 |

> F0L (IPPT) e F0M (MPPT) são regimes do mesmo dataset GPVS. Diferenças entre eles não são eventos adicionais nem mistura de bases; são heterogeneidade operacional explícita.

### Fusível AC (NPR 30)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 4.184 | [3.92; 4.50] |
| escala η | 0.59 | [0.57; 0.60] |

Diagnóstico visual: **R²pp = 0.886**. Aderência por bootstrap quantizado: **p = 0.004** (`desvio_detectado_bootstrap_quantizado`). Estabilidade entre grades finas: **sim**.
A curva e os parâmetros permanecem visíveis para auditoria, mas a síntese Weibull 2P é exploratória e não sustenta inferência física.

> Margem restrita KM no início: **0.53**. É descritiva no domínio observado e não é RUL.

| Marco paramétrico E2 | Magnitude de injeção | S_D nesse ponto |
|---|--:|--:|
| a01 (1% detectado) | 0.20 | 0.990 |
| a10 (10% detectado) | 0.34 | 0.900 |
| a50 (mediana) | 0.54 | 0.500 |
| η (escala característica) | 0.59 | 0.368 |
| média paramétrica de a_det | 0.53 | 0.512 |

**Leitura de β.** a intensidade parametrica do primeiro cruzamento aumenta com a magnitude injetada. Como o eixo nao e idade, isso NAO significa desgaste nem autoriza intervalo de manutencao

> β descreve somente a forma da intensidade de detecção em função da magnitude. Não implica desgaste, mortalidade infantil ou política de substituição.

> Observação vai até 0.9; qualquer marco além disso é extrapolação do modelo, não dado.

#### Estratificação por modo operacional GPVS

| Modo | n | beta | eta | p bootstrap | Grade estável | Uso 2P |
|---|---:|---:|---:|---:|---|---|
| F0L | 142 | 9.014 | 0.689 | 0.004 | sim | exploratório |
| F0M | 135 | 9.666 | 0.420 | 0.116 | sim | adotado em E2 |

> F0L (IPPT) e F0M (MPPT) são regimes do mesmo dataset GPVS. Diferenças entre eles não são eventos adicionais nem mistura de bases; são heterogeneidade operacional explícita.

---

## Ponto de operação sob o critério LS-POD

Escore operacional: **mse**; limiar adotado: **0.8577**

### A hipótese, antes do número

O método assume normalidade do lado saudável. Shapiro-Wilk no escore bruto: p = 9.58e-23; em log: p = 8.02e-07. Assimetria +3.71, curtose +20.49.

**Hipótese violada.** Por isso o mesmo quantil é estimado por três caminhos independentes abaixo — se os três concordarem, a conclusão não depende dela.

| Estimador do percentil 99 do escore saudável | Valor |
|---|--:|
| normal no escore bruto (LS-POD) | 0.7060 |
| normal em log, destransformado | 0.7608 |
| percentil 99 empírico | 0.9880 |
| **limiar adotado** | **0.8577** |

> Os estimadores divergem quanto ao cumprimento do requisito; reportar a faixa, não um veredito.

> No teste foram observadas **4/281 = 1.42%** excedências; IC95 de Wilson **[0.55%; 3.60%]**. A meta de 1% está dentro do intervalo: os dados não certificam conformidade nem violação.



### Deriva entre calibração e teste

O limite pontual LS-POD no teste foi **0.7060**, frente ao limiar **0.8577** e ao gatilho de deriva **0.6688**.

> O resultado aciona investigação como triagem. O bloco de teste não é campo e não fornece resolução para confirmar 1%; portanto não constitui invalidação industrial nem evidência de deriva física.
