# FMECA vigente

A FMECA atual acompanha os modos nativos do GPVS-Faults que possuem ligação
direta com o inversor. Ela organiza a discussão de manutenção, mas não é
recalculada pelo desempenho dos Autoencoders.

## Escopo canônico

| Item | Função | Contrapartida experimental | S | O | D | NPR | Estado |
|---|---|---|---:|---:|---:|---:|---|
| IGBT | chaveamento da conversão CC-CA | F1L/F1M: falha completa de um IGBT | 5 | 6 | 5 | 150 | `validated` |
| Sistema de sensor/realimentação | medir e realimentar variáveis elétricas | F2L/F2M: erro de 20% | 5 | 8 | 7 | 280 | `validated` |
| Sistema/circuito de controle do inversor | regular MPPT/IPPT pelo controlador PI | F6L/F6M: ganho -20%; F7L/F7M: constante de tempo +20% | 5 | 6 | 8 | 240 | `validated` |

Os valores foram definidos explicitamente pelo pesquisador em 2026-09-01 e são
validados por `NPR = S * O * D`. Eles não foram inferidos das métricas dos
detectores nem transferidos do recorte histórico. O contrato usa
`calculation_enabled=true`, `schema_version=7` e mantém
`traceability_status=pending_source_documentation`. As escalas numéricas têm
base no TCC e em outras referências; a escolha dos valores para cada item foi
feita pelo critério técnico do pesquisador.

## Interpretação dos ensaios

- F1 representa falha física completa de um IGBT.
- F2 representa erro funcional do sistema de sensor/realimentação.
- F6 e F7 representam anomalias funcionais do sistema/circuito de controle.
- F6 e F7 não são evidência de falha física de placa de circuito impresso.
- F3, F4 e F5 continuam na avaliação E3, mas não integram o trio FMECA atual.

O núcleo experimental usa somente as condições reais de bancada F1L-F7M. Não
há injeção sintética de falhas para IGBT, sensor/realimentação ou controle.

## Limites científicos

Recall, F1, Precision, matrizes de confusão e falso positivo saudável medem o
comportamento dos detectores. Essas métricas não definem escalas ordinais de
manutenção e não recalculam NPR.

O recorte Contator AC, IGBT e Fusível AC, com valores 315, 90 e 30, permanece
apenas como registro histórico do TCC. Ele não é a FMECA canônica desta etapa e
não fornece valores para sensor/realimentação ou sistema de controle.

## Rastreabilidade documental pendente

Os valores adotados e o cálculo estão validados. Ainda faltam, para a
rastreabilidade científica completa das escalas:

- critério documentado das escalas de severidade, ocorrência e detectabilidade;
- catalogação das referências que fundamentam as escalas numéricas;
- fonte, página e tabela;
- interpretação operacional das faixas de NPR.

Essa pendência documental não transfere a autoria dos escores às referências:
os valores aplicados aos três itens são uma decisão do pesquisador.

Antes da decisão explícita de 2026-09-01, o contrato registrava esses campos
como `null` e `awaiting_user_fmeca`. Esse estado permanece apenas nos documentos
históricos datados; não descreve a FMECA vigente.
