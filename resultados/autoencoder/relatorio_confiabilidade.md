# Relatório de detectabilidade E2 e ponto de operação

> **Evidência E2** — validação sintética orientada pela FMECA. Não é desempenho de campo (E3). O eixo NÃO é tempo físico: é a magnitude de injeção em que a detecção se confirma. Portanto não há RUL, MTTF, taxa de falha ou confiabilidade física nesta seção.

## Detectabilidade sintética por modo de falha

### Contator AC (NPR 315)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 5.287 | [4.75; 6.12] |
| escala η | 0.46 | [0.44; 0.47] |

Triagem no papel de Weibull: **R²pp = 0.911**. Síntese paramétrica recomendada somente no escopo E2.

| Marco | Magnitude de injeção | R nesse ponto |
|---|--:|--:|
| a01 (1% detectado) | 0.19 | 0.990 |
| a10 (10% detectado) | 0.30 | 0.900 |
| a50 (mediana) | 0.42 | 0.500 |
| η (escala característica) | 0.46 | 0.368 |
| média paramétrica de a_det | 0.42 | 0.523 |

**Leitura de β.** a intensidade parametrica do primeiro cruzamento aumenta com a magnitude injetada. Como o eixo nao e idade, isso NAO significa desgaste nem autoriza intervalo de manutencao

> β descreve somente a forma da intensidade de detecção em função da magnitude. Não implica desgaste, mortalidade infantil ou política de substituição.

> Observação vai até 0.6; qualquer marco além disso é extrapolação do modelo, não dado.

### IGBT (NPR 90)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 4.316 | [3.67; 5.20] |
| escala η | 0.36 | [0.34; 0.37] |

Triagem no papel de Weibull: **R²pp = 0.790**. Síntese paramétrica não recomendada; os marcos abaixo são omitidos.

> Margem restrita KM no início: **0.33**. É descritiva no domínio observado e não é RUL.

### Fusível AC (NPR 30)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 4.115 | [3.50; 6.83] |
| escala η | 0.04 | [0.04; 0.04] |

Triagem no papel de Weibull: **R²pp = 0.455**. Síntese paramétrica não recomendada; os marcos abaixo são omitidos.

> Margem restrita KM no início: **0.04**. É descritiva no domínio observado e não é RUL.

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
