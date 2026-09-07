# FMECA vigente

A FMECA atual tem escopo de **6 itens**. Ela organiza a discussão de
manutenção/criticidade, mas não é recalculada pelo desempenho dos Autoencoders.

## Escopo canônico

| Item | Contrapartida GPVS | S | O | D | NPR | Origem do escore |
|---|---|---:|---:|---:|---:|---|
| Sistema de sensor/realimentação | F2L/F2M: erro de 20% | 5 | 8 | 7 | 280 | pesquisador (a revisar) |
| PCB | — | 7 | 4 | 6 | 168 | Cristaldi (2017), Tab. 6 |
| Contatores CA/CC | — | 6 | 5 | 5 | 150 | Cristaldi (2017), Tab. 6 |
| Sistema/circuito de controle | F6L/F6M/F7L/F7M: ganho −20%; constante +20% | 6 | 7 | 3 | 126 | Cristaldi (2017), Tab. 6 |
| IGBT | F1L/F1M: falha completa de um IGBT | 3 | 3 | 7 | 63 | Cristaldi (2017), Tab. 6 |
| Ventiladores de refrigeração | — | 4 | 3 | 4 | 48 | Cristaldi (2017), Tab. 6 |

Todos são validados por `NPR = S * O * D`. A tabela mistura proveniências **de
propósito** e isso viaja no artefato (`score_provenance`): os cinco itens de
survey de campo levam os escores ordinais de Cristaldi, Khalil & Soulatiantork
(2017), Acta IMEKO, Tabela 6; o **sistema de sensor/realimentação** mantém o
valor definido pelo pesquisador em 2026-09-01, a ser revisado quando houver
fonte bibliográfica própria. O item DC-link Capacitor consta na Tabela 6
original (NPR 30) mas está **fora** do escopo de 5 itens fechado pelo
pesquisador.

## FMECA versus injeção E2

Só **IGBT, sensor/realimentação e controle** têm contrapartida de ensaio no
GPVS — são os três que a injeção E2 cobre (`injection_covered_items`). PCB,
contatores e ventiladores entram na FMECA como itens de criticidade/manutenção,
**sem** condição de injeção nativa no dataset; nenhum deles altera o escopo do
E2 nem é sintetizado.

- F1 representa falha física completa de um IGBT.
- F2 representa erro funcional do sistema de sensor/realimentação.
- F6 e F7 representam anomalias funcionais do sistema/circuito de controle e não
  são evidência de falha física de placa de circuito impresso.

## Confiabilidade física (λ)

As taxas de falha bibliográficas por componente ficam em
`resultados/confiabilidade/` (módulo `confiabilidade_componentes.py`), rotuladas
por fonte (datasheet/norma/campo) e como cenários de sensibilidade, nunca
medições desta pesquisa. λ com faixa entra como par de limites (lower/upper),
sem ponto fabricado. **PCB** e **Control Software** ficam sem λ: nenhuma fonte
decompõe a taxa por componente dentro do inversor — registrado em
`lambda_not_found`, sem usar zero, estimativa ou proxy.

## Limites científicos

Recall, F1, Precision, matrizes de confusão e falso positivo saudável medem o
comportamento dos detectores. Essas métricas não definem escalas ordinais de
manutenção e não recalculam NPR. A validação E3 mede detecção de anomalias e não
fornece valores S/O/D.

O recorte histórico do TCC (Contator AC, IGBT e Fusível AC, com valores 315, 90
e 30) permanece apenas como registro e não fornece valores ao escopo vigente.

## Norma do critério ordinal

O critério ordinal S/O/D adotado por Cristaldi (2017) segue a **IEC 60812**
(Análise dos modos de falha e efeitos — FMEA/FMECA). O material de origem
grafava "IEC-60182" por transposição de dígitos; a norma correta é a IEC 60812.
