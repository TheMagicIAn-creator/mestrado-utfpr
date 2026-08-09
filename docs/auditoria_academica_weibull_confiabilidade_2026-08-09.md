# Auditoria acadêmica de Weibull, confiabilidade e RUL

Data: 09/08/2026  
Escopo: `rul_weibull.py`, funções matemáticas, gráficos, relatórios, artefatos,
proveniência e respostas do ALIAdo.

## Conclusão executiva

A etapa não mede tempo até falha. Ela varre uma assinatura sintética de falha
de `a=0` a `a=1` sobre janelas saudáveis e registra a primeira magnitude em que
o Autoencoder confirma detecção. Portanto:

- há uma distribuição de **magnitude de primeiro cruzamento do detector**;
- não há confiabilidade, taxa de falha, desgaste, vida característica ou RUL do
  componente;
- a Weibull 2P pode ser usada somente como modelo exploratório E2 da
  detectabilidade, condicionado ao experimento computacional.

Os arquivos e campos legados foram preservados para compatibilidade, mas as
interfaces acadêmicas agora usam `a10`, média de `a_det`, curva de não detecção,
intensidade de detecção e margem residual de magnitude.

## Achados corrigidos

### 1. Censura ignorada na CDF e no papel de Weibull

O gráfico calculava `(i-0,3)/(n_eventos+0,4)`. Para o IGBT, 12 detecções em 31
janelas chegavam visualmente a aproximadamente 95%, embora 19 janelas não
detectassem nem no teto. O denominador correto precisa incluir a amostra total.

A correção implementa Kaplan-Meier modificado. Para censura única no final, ele
se reduz às medianas de posto com `n_total`; o último ponto do IGBT passa a
`(12-0,3)/(31+0,4)=0,3726`. Essa é a recomendação do
[NIST/SEMATECH para probability plots censurados](https://www.itl.nist.gov/div898/handbook/apr/section2/apr221.htm)
e para a [estimativa Kaplan-Meier modificada](https://www.itl.nist.gov/div898/handbook/apr/section2/apr215.htm).

### 2. Interpretação física indevida de beta

O JSON classificava `beta > 1` como desgaste e sugeria substituição preventiva.
Essa leitura só existe quando o eixo representa idade/uso de uma população de
itens. Em `a_det`, beta descreve somente a forma da intensidade de primeiro
cruzamento conforme a magnitude injetada. A nova saída fixa
`inferencia_manutencao_autorizada=false`.

### 3. RUL, MTTF e B10 sem eixo temporal

A NASA define RUL como tempo de uso restante até o estado de falha, condicionado
ao estado atual e ao perfil futuro de uso. Esse requisito não está presente na
varredura de magnitude ([NASA, RUL Estimation in Prognosis](https://ntrs.nasa.gov/api/citations/20140010623/downloads/20140010623.pdf?attachment=true)).

Os nomes canônicos agora são:

| Nome acadêmico | Significado |
|---|---|
| `a10` | magnitude em que 10% dos primeiros cruzamentos ocorreram |
| média de `a_det` | média paramétrica da magnitude de cruzamento |
| margem restrita KM | magnitude residual média, truncada em `a=1` |
| `S_D(a)` | probabilidade de ainda não detectar até `a` |
| `h_D(a)` | intensidade paramétrica de detecção por unidade de `a` |

### 4. Adequação da Weibull não era decidida

O RMSE entre Kaplan-Meier e Weibull era pequeno mesmo quando o papel de Weibull
mostrava desvio grave. Foi acrescentado `R²pp`, calculado com posições que
preservam censura. Ele é triagem descritiva, não teste formal. O NIST recomenda
o probability plot justamente para verificar se os pontos são aproximadamente
lineares antes de interpretar a família escolhida.

### 5. Incerteza e pseudorrepetição

O bootstrap passou de 250 para 1.000 réplicas. A unidade de reamostragem é a
janela sem compartilhamento de amostras. Isso reduz duplicação direta, mas não
demonstra independência temporal; os ICs são condicionais a E2 e não devem ser
tratados como incerteza entre ativos de campo.

### 6. Relatório usava o limiar errado

`relatorio_confiabilidade.py` preferia o campo legado `limiar_localizado`
(`16,5361`) apesar de `score_method=mse` e `score_threshold=2,5828`. A seleção
agora obedece aos campos canônicos. A excedência observada no teste é
`1/60=1,67%`, IC95 de Wilson `[0,29%; 8,86%]`; a amostra não certifica
conformidade nem violação da meta nominal de 1%.

### 7. Proveniência incompleta

Injeção, validação e Weibull usam `features_paderborn.parquet` para definir os
índices exatos do holdout, mas o arquivo não estava nos hashes de entrada. Ele
foi adicionado. A etapa Weibull também passou a versionar os hashes de
`confiabilidade.py`, `relatorio_weibull.py` e `split_temporal.py`.

## Resultado após reexecução controlada

Foram mantidos dataset, modelo, scaler, limiar e 31 janelas elegíveis. Os pontos
MLE de beta/eta não mudaram; os intervalos foram recalculados com 1.000 réplicas.

| Modo | detecções/total | beta (IC95%) | eta (IC95%) | R²pp | Decisão acadêmica |
|---|---:|---:|---:|---:|---|
| Contator AC | 31/31 | 3,148 [2,658; 3,958] | 0,475 [0,418; 0,529] | 0,955 | síntese exploratória E2 permitida |
| IGBT | 12/31 | 6,578 [4,396; 10,967] | 1,112 [1,020; 1,198] | 0,947 | não reportar síntese: 61% indetectável |
| Fusível AC | 30/31 | 8,599 [4,791; 15,772] | 0,896 [0,856; 0,927] | -0,843 | não reportar síntese: desvio da Weibull 2P |

No IGBT, somente 819/1.000 réplicas bootstrap produziram o mínimo de eventos e
convergiram. No Fusível, o ponto isolado em `a≈0,218` e a concentração próxima
do teto produzem forte não linearidade no papel. Nenhum ponto foi removido.

## O que seria necessário para confiabilidade/RUL física

1. Tempos ou ciclos até falha, ou degradação longitudinal com limiar funcional,
   de múltiplos ativos identificáveis.
2. Censura operacional real, perfil de carga, temperatura, manutenção e modo de
   falha verificado.
3. Unidade temporal calibrada e separação entre ativos para validação externa.
4. Comparação de famílias e diagnóstico formal antes de escolher Weibull.
5. Avaliação de RUL em instantes sucessivos contra o tempo real restante.

Os datasets GPVS e PMSM disponíveis ajudam a validar detecção de falhas, mas não
fornecem automaticamente históricos run-to-failure de componentes. Mais linhas
de telemetria não equivalem a mais eventos de vida independentes.

## Referências metodológicas

- NIST/SEMATECH e-Handbook, Reliability Data Analysis e probability plotting.
- Nketiah et al. (2021), *Parameter estimation of the Weibull Distribution*;
  PDF fornecido `21IJAERS-09202130-Parameter..pdf`.
- Sankararaman (NASA), *Remaining Useful Life Estimation in Prognosis*.
- Virkkunen e Ylitalo (2016), *Practical Experiences in POD Determination*;
  PDF fornecido `2016-Practical_POD.pdf`.
