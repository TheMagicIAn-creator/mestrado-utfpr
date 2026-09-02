# Confiabilidade física por componente

## Escopo

As curvas são cenários bibliográficos históricos de sensibilidade do TCC,
independentes da base experimental usada na comparação dos detectores e do
escopo atual da FMECA. As fontes disponíveis não fornecem uma amostra de tempos
individuais de falha, exposição de frota e censura por ativo.

## Taxas

- Contator AC: 2,10e-5 falha/h, derivada de 1,75e-4 x 12%.
- IGBT: 1,05e-5 falha/h, derivada de 1,75e-4 x 6%.
- Fusível AC: 7,00e-6 falha/h, derivada de 1,75e-4 x 4%.
- Fusível: 2,17e-6 falha/h, transcrita diretamente da Tabela 3.4.

Os percentuais são participações de chamados, não frações demonstradas da taxa
de falha do inversor. Por isso, as três alocações são rotuladas como derivadas.
A ausência de taxas diretas equivalentes para Contator AC e IGBT é preservada.

## FMECA vigente

O escopo atual é formado por IGBT, sistema de sensor/realimentação e
sistema/circuito de controle do inversor. Em 2026-09-01, o pesquisador definiu
explicitamente os seguintes valores canônicos:

- IGBT: S=5, O=6, D=5 e NPR=150.
- Sistema de sensor/realimentação: S=5, O=8, D=7 e NPR=280.
- Sistema/circuito de controle do inversor: S=5, O=6, D=8 e NPR=240.

O estado é `validated`, com `NPR = S * O * D` verificado. A origem dos escores
adotados é uma decisão explícita do pesquisador. As escalas numéricas possuem
base no TCC e em outras referências; `pending_source_documentation` indica que
a localização exata dessas fontes e os critérios das escalas ainda precisam ser
catalogados. Os valores históricos de Contator AC, IGBT e Fusível AC não são
transferidos para esse novo escopo.

A validação E3 mede detecção de anomalias. Ela não produz escalas ordinais da
FMECA e não recalcula NPR.

## Modelo

Adota-se o cenário exponencial de taxa constante: R(t)=exp(-lambda*t),
F(t)=1-R(t), f(t)=lambda*exp(-lambda*t) e h(t)=lambda. A conversão usa
1 ano=8.760 horas. As figuras usam escalas lineares. A busca no corpus local
encontrou 22 trechos sobre IGBT e 74 sobre Weibull; a única fonte comum discute
os assuntos separadamente e não fornece beta ou eta para IGBT. Por isso Weibull
2P, distribuição normal, Lognormal, histograma de vidas, curva de banheira e RUL
físico permanecem bloqueados, sem parâmetros fabricados.
