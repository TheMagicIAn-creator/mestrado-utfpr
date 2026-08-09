# Relatório de detectabilidade E2 e ponto de operação

> **Evidência E2** — validação sintética orientada pela FMECA. Não é desempenho de campo (E3). O eixo NÃO é tempo físico: é a magnitude de injeção em que a detecção se confirma. Portanto não há RUL, MTTF, taxa de falha ou confiabilidade física nesta seção.

## Detectabilidade sintética por modo de falha

### Contator AC (NPR 315)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 3.148 | [2.66; 3.96] |
| escala η | 0.47 | [0.42; 0.53] |

Triagem no papel de Weibull: **R²pp = 0.955**. Síntese paramétrica recomendada somente no escopo E2.

| Marco | Magnitude de injeção | R nesse ponto |
|---|--:|--:|
| a01 (1% detectado) | 0.11 | 0.990 |
| a10 (10% detectado) | 0.23 | 0.900 |
| a50 (mediana) | 0.42 | 0.500 |
| η (escala característica) | 0.47 | 0.368 |
| média paramétrica de a_det | 0.42 | 0.494 |

**Leitura de β.** a intensidade parametrica do primeiro cruzamento aumenta com a magnitude injetada. Como o eixo nao e idade, isso NAO significa desgaste nem autoriza intervalo de manutencao

> β descreve somente a forma da intensidade de detecção em função da magnitude. Não implica desgaste, mortalidade infantil ou política de substituição.

> Observação vai até 0.7; qualquer marco além disso é extrapolação do modelo, não dado.

### IGBT (NPR 90)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 6.578 | [4.40; 10.97] |
| escala η | 1.11 | [1.02; 1.20] |

Triagem no papel de Weibull: **R²pp = 0.947**. Síntese paramétrica não recomendada; os marcos abaixo são omitidos.

> Margem restrita KM no início: **0.94**. É descritiva no domínio observado e não é RUL.

### Fusível AC (NPR 30)

| Parâmetro | Valor | IC95 |
|---|--:|---|
| forma β | 8.599 | [4.79; 15.77] |
| escala η | 0.90 | [0.86; 0.93] |

Triagem no papel de Weibull: **R²pp = -0.843**. Síntese paramétrica não recomendada; os marcos abaixo são omitidos.

> Margem restrita KM no início: **0.85**. É descritiva no domínio observado e não é RUL.

---

## Ponto de operação sob o critério LS-POD

Escore operacional: **mse**; limiar adotado: **2.5828**

### A hipótese, antes do número

O método assume normalidade do lado saudável. Shapiro-Wilk no escore bruto: p = 3.42e-13; em log: p = 5.75e-02. Assimetria +4.71, curtose +25.88.

A normalidade **não foi rejeitada** na escala **log** ao nível de 5%. Isso não prova a hipótese, especialmente com p próximo do corte; os três estimadores continuam sendo reportados como análise de sensibilidade.

| Estimador do percentil 99 do escore saudável | Valor |
|---|--:|
| normal no escore bruto (LS-POD) | 2.5916 |
| normal em log, destransformado | 3.7612 |
| percentil 99 empírico | 3.5711 |
| **limiar adotado** | **2.5828** |

> **Os 3 estimadores pontuais ficam acima do limiar.** Isso sinaliza tensão com a meta nominal de 1%, mas não demonstra violação estatística com esta amostra.

> No teste foram observadas **1/60 = 1.67%** excedências; IC95 de Wilson **[0.29%; 8.86%]**. A meta de 1% está dentro do intervalo: os dados não certificam conformidade nem violação.

> Com n = 60, a resolução amostral é 1.67%: **o alvo de 1% está abaixo do que esta amostra consegue certificar**. Zero excedências observadas não provariam 1%. A limitação é de tamanho amostral, não uma prova de falha do limiar.

### Deriva entre calibração e teste

O limite pontual LS-POD no teste foi **2.5916**, frente ao limiar **2.5828** e ao gatilho de deriva **2.3446**.

> O resultado aciona investigação como triagem. O bloco de teste não é campo e não fornece resolução para confirmar 1%; portanto não constitui invalidação industrial nem evidência de deriva física.
