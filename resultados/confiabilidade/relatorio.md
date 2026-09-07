# Confiabilidade física por componente

## Escopo

As curvas são cenários bibliográficos históricos de sensibilidade do TCC,
independentes da base experimental usada na comparação dos detectores e do
escopo atual da FMECA. As fontes disponíveis não fornecem uma amostra de tempos
individuais de falha, exposição de frota e censura por ativo.

## Taxas

Cenários históricos do TCC (alocação proporcional por participação de chamados):

- Contator AC: 2,10e-5 falha/h, derivada de 1,75e-4 x 12%.
- IGBT: 1,05e-5 falha/h, derivada de 1,75e-4 x 6%.
- Fusível AC: 7,00e-6 falha/h, derivada de 1,75e-4 x 4%.
- Fusível: 2,17e-6 falha/h, transcrita diretamente da Tabela 3.4.

Taxas bibliográficas externas do escopo de 5 itens (datasheet/norma/campo,
faixas como par lower/upper, FIT = falha por 1e9 h):

- IGBT: 10-100 FIT (datasheet, Mitsubishi) e 685-2740 FIT (campo FIDES,
  Abunima & Teh 2020).
- Contatores CA/CC: ~100 FIT (ABB; consultado em cópia não oficial, verificar).
- Ventiladores: 1,4e-5 a 2,0e-5 falha/h (Sanyo Denki 9RA).

Nenhuma é medição de campo desta pesquisa. **PCB** e **Control Software** ficam
sem λ: nenhuma fonte decompõe a taxa por componente dentro do inversor —
registrado em `lambda_not_found`, sem zero, estimativa ou proxy.

## FMECA vigente (6 itens)

Escopo de 6 itens, todos validados por `NPR = S * O * D`. Proveniência mista:
os cinco itens de survey de campo levam os escores de Cristaldi et al. (2017),
Tabela 6; o sensor/realimentação mantém o valor definido pelo pesquisador, a ser
revisado.

- Sistema de sensor/realimentação: S=5, O=8, D=7, NPR=280 (pesquisador).
- PCB: S=7, O=4, D=6, NPR=168 (Cristaldi).
- Contatores CA/CC: S=6, O=5, D=5, NPR=150 (Cristaldi).
- Sistema/circuito de controle: S=6, O=7, D=3, NPR=126 (Cristaldi).
- IGBT: S=3, O=3, D=7, NPR=63 (Cristaldi).
- Ventiladores de refrigeração: S=4, O=3, D=4, NPR=48 (Cristaldi).

Só IGBT, sensor/realimentação e controle têm ensaio no GPVS — são os que a
injeção E2 cobre. Os demais entram como criticidade/manutenção, sem injeção.

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
