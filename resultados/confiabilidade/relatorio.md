# Confiabilidade física por componente

## Escopo

As curvas são cenários bibliográficos de sensibilidade, independentes da base
experimental usada na comparação dos detectores. As fontes disponíveis não
fornecem uma amostra de tempos individuais de falha, exposição de frota e
censura por ativo.

## Taxas

- Contator AC: 2,10e-5 falha/h, derivada de 1,75e-4 x 12%.
- IGBT: 1,05e-5 falha/h, derivada de 1,75e-4 x 6%.
- Fusível AC: 7,00e-6 falha/h, derivada de 1,75e-4 x 4%.
- Fusível: 2,17e-6 falha/h, transcrita diretamente da Tabela 3.4.

Os percentuais são participações de chamados, não frações demonstradas da taxa
de falha do inversor. Por isso, as três alocações são rotuladas como derivadas.
A ausência de taxas diretas equivalentes para Contator AC e IGBT é preservada.

## Priorização FMECA

O NPR permanece independente do detector: Contator AC 315, IGBT 90 e Fusível
AC 30. Adota-se NPR=S x O x D_campo; D_campo é a dificuldade de detecção no
processo de manutenção e não uma métrica dos Autoencoders.

## Modelo

Adota-se o cenário exponencial de taxa constante: R(t)=exp(-lambda*t),
F(t)=1-R(t), f(t)=lambda*exp(-lambda*t) e h(t)=lambda. A conversão usa
1 ano=8.760 horas. As figuras usam escalas lineares. Não são estimados beta,
eta, distribuição normal, curva de banheira ou RUL físico.
