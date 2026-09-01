# FMECA vigente

A FMECA atual acompanha os modos nativos do GPVS-Faults que possuem ligação
direta com o inversor. Ela organiza a discussão de manutenção, mas não é
recalculada pelo desempenho dos Autoencoders.

## Escopo canônico

| Item | Função | Contrapartida experimental | S | O | D | NPR | Estado |
|---|---|---|---:|---:|---:|---:|---|
| IGBT | chaveamento da conversão CC-CA | F1L/F1M: falha completa de um IGBT | `null` | `null` | `null` | `null` | `awaiting_user_fmeca` |
| Sistema de sensor/realimentação | medir e realimentar variáveis elétricas | F2L/F2M: erro de 20% | `null` | `null` | `null` | `null` | `awaiting_user_fmeca` |
| Sistema/circuito de controle do inversor | regular MPPT/IPPT pelo controlador PI | F6L/F6M: ganho -20%; F7L/F7M: constante de tempo +20% | `null` | `null` | `null` | `null` | `awaiting_user_fmeca` |

Os valores de severidade, ocorrência, detectabilidade e NPR dependem de decisão
do pesquisador e de fonte compatível com este novo escopo. Nenhum número do
recorte histórico é transferido automaticamente.

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

## Dados ainda necessários do pesquisador

Para cada um dos três itens atuais ainda faltam:

- severidade e critério da escala;
- ocorrência e base observacional ou bibliográfica;
- detectabilidade de manutenção e definição operacional;
- fonte, página e tabela;
- regra aprovada para cálculo e interpretação do NPR.

Até esses dados existirem, S, O, D e NPR permanecem nulos nos artefatos
versionados.
