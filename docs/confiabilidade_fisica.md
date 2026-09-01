# Confiabilidade física bibliográfica

Esta publicação é independente dos detectores. O GPVS-Faults não contém tempos
de vida nem exposição de frota e não estima confiabilidade física.

## Modelo

Adota-se um cenário exponencial de taxa constante:

- `R(t) = exp(-lambda*t)`;
- `F(t) = 1 - R(t)`;
- `f(t) = lambda*exp(-lambda*t)`;
- `h(t) = lambda`.

O tempo primário é hora; a visualização converte por `1 ano = 8.760 horas`.
As quatro funções são publicadas em figuras separadas e eixos lineares.

## Cenários

A taxa global do inversor, `1,75e-4 falha/h`, e as participações de chamados do
TCC geram cenários de sensibilidade para Contator AC, IGBT e Fusível AC. Essa
multiplicação é uma hipótese de alocação, não uma medição de taxa por
componente.

Para o fusível, a taxa `2,17e-6 falha/h` é transcrita diretamente da Tabela 3.4
e permanece separada do cenário derivado. Contator AC e IGBT não possuem taxa
direta equivalente identificada na fonte usada.

## Rastreabilidade

`resultados/confiabilidade/metodologia.json` registra PDF, hash, página PDF,
página impressa, tabelas, valor original, conversão, fórmula, tipo direto ou
derivado e ressalva. `cenarios.csv` e `curvas.csv` são os dados-fonte das
figuras.

Não se produzem beta, eta, Weibull físico, distribuição normal, curva de
banheira ou RUL sem vidas individuais, censura e exposição adequadas por ativo.

## Contratos de modelos ainda indisponíveis

O arquivo `metodologia.json` distingue capacidade futura de resultado atual:

| Modelo | Parâmetros mínimos | Estado atual |
|---|---|---|
| Exponencial | `lambda_per_hour` rastreada | publicado como sensibilidade bibliográfica |
| Weibull 2P | `beta`, `eta_hours` | bloqueado |
| Normal | `mean_hours`, `std_hours` | bloqueado |
| Lognormal | `mu_log_hours`, `sigma_log_hours` | bloqueado |
| Histograma de vidas | tempos observados por ativo | bloqueado |

Para desbloquear os quatro últimos também são necessários tempos individuais,
exposição, censura e identificação da população. No caso Normal, deve-se ainda
avaliar a inadequação potencial do suporte negativo. Quatro taxas heterogêneas
de cenário não constituem uma amostra de vidas e não justificam uma curva em
sino.

O mesmo contrato mantém a extensão FMECA de monitoramento bloqueada: sem uma
regra validada `POD_mon -> D_mon`, `D_proj` e `NPR_proj` permanecem nulos.
