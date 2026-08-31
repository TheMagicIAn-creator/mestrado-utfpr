# Evidence RAG — promoção R6

> Estado: implementação, restauração e gold set aprovados; promoção pronta para merge.

## Estado encontrado

- O R4 demonstrou ganho de recuperação sem regressões em página@5.
- O R5 atingiu 100% nos casos determinísticos de validade de citação,
  rejeição de claims inválidos, abstenção e separação de memória.
- O gold set permanece provisório por decisão explícita da especificação.

## Restauração independente

O índice foi reconstruído em diretório vazio usando apenas o snapshot
contextual promovido. A execução restaurada obteve:

| Métrica | R3 | R6 restaurado | Delta |
|---|---:|---:|---:|
| Recall de página@5 | 0,294872 | 0,333333 | +0,038461 |
| Recall de página@8 | 0,346154 | 0,384615 | +0,038461 |
| MRR@5 | 0,242308 | 0,255128 | +0,012820 |
| MRR@8 | 0,248718 | 0,261538 | +0,012820 |
| nDCG@5 | 0,241455 | 0,258089 | +0,016634 |
| nDCG@8 | 0,257633 | 0,274266 | +0,016633 |

- Regressões em página@5: nenhuma.
- Consultas melhoradas: `authors-anomaly-009` e `rcm-rigorous-fmea-020`.
- Latência média da execução final: 1.512 ms; p95: 1.809 ms.
- Corpus: 44 PDFs e 12.556 chunks.

## Mudança canônica

- O snapshot `artefatos/literatura_indexada.jsonl.gz` passa a conter o schema v2
  contextual validado.
- O perfil padrão passa a ser `r4_hybrid`.
- `AL_IADO_RETRIEVAL_PROFILE=baseline` oferece rollback imediato do ranking.
- O snapshot anterior continua restaurável pelo histórico Git.
- Evidence Package e Evidence Guard permanecem obrigatórios para respostas
  acadêmicas.

## Revisão humana concluída

O arquivo `literatura/gold_set_retrieval_v1.json` contém 40 perguntas: 39 de
recuperação e uma de abstenção. As evidências foram verificadas contra o
snapshot e os rótulos foram aprovados por Rodolfo Torres em 2026-08-30.

A aprovação significa confirmar que as perguntas, páginas e grupos de evidência
representam adequadamente o que o benchmark deve medir. Ela não declara que o
RAG é perfeito nem transforma inferências do modelo em fatos científicos.

O gold set recebeu estado `researcher_approved_R6`; o manifesto final registra
essa decisão, os gates R4-R5 e os hashes de promoção e rollback.
