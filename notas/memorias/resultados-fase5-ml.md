# Resultados da Fase 5 - Pipeline de ML

> Gerado em 09/08/2026 16:56 -03

Aqui está o que já existe nos artefatos do pipeline.

## Autoencoder - modelo de normalidade

| Métrica | Valor |
|---|---:|
| Escore operacional | mse / percentil efetivo 99.0 |
| Limiar operacional | 2.5828 |
| Referência MSE p99 | 2.5828 |
| Média baseline | 0.4847 |
| Desvio baseline | 0.5716 |
| Janelas de treino | 104 |
| Janelas de calibração | 42 |
| Janelas de teste | 60 |
| FP teste - escore operacional | 1/60 = 1.67% [0.29; 8.86] |
| FP teste - referência MSE p99 | 1/60 = 1.67% [0.29; 8.86] |
| FP teste - MSE sem compartilhamento de amostras | 1/32 = 3.12% [0.55; 15.74] |
| Resolução empírica da cauda na calibração | 1/42 = 2.38% |
| Épocas treinadas | 148 |

Leitura rápida: o detector usa o escore operacional registrado em `limiar.json`; os gráficos principais de reconstrução permanecem na escala MSE e são acompanhados por `calibracao_autoencoder.md`. O p99 é nominal e interpolado; a subamostra sem compartilhamento evita pseudorrepetição direta, mas não prova independência temporal.

## Injeção de falhas sintéticas
Limiar: **2.5828**. Baseline: **0.6080 ± 0.9796**.
| Falha | NPR | SMD95 | Taxa (IC95%) | n | Erro mediano |
|---|---:|---:|---:|---:|---:|
| Contator AC | 315 | 0.7 | 1.000 [0.893; 1.000] | 32 | 11.4439 |
| IGBT | 90 | ⚠️ alvo não atingido | - | - | - |
| Fusível AC | 30 | 1.0 | 1.000 [0.893; 1.000] | 32 | 3.5444 |

Leitura rápida: SMD95 é a menor severidade cuja taxa pontual de detecção atinge 95%; o intervalo de Wilson mostra a incerteza dessa estimativa.

⚠️ **Falha(s) sem SMD nesta execução** (achado relevante, não omitir na dissertação):
- **IGBT**: taxa máxima de detecção 0.406 na severidade 1.0; o alvo probabilístico de 95% não foi atingido.

## Validação sintética interna E2

| Falha | Sev. | AUC-ROC (IC95%) | Recall (IC95%) | FNR | Especificidade | n/classe |
|---|---:|---:|---:|---:|---:|---:|
| Contator AC | 0.3 | 0.912 [0.822; 0.982] | 0.344 [0.204; 0.517] | 0.656 | 0.969 | 32 |
| Contator AC | 0.5 | 0.976 [0.934; 1.000] | 0.719 [0.546; 0.844] | 0.281 | 0.969 | 32 |
| Contator AC | 1.0 | 1.000 [1.000; 1.000] | 1.000 [0.893; 1.000] | 0.000 | 0.969 | 32 |
| IGBT | 0.3 | 0.599 [0.467; 0.730] | 0.031 [0.006; 0.157] | 0.969 | 0.969 | 32 |
| IGBT | 0.5 | 0.698 [0.582; 0.827] | 0.031 [0.006; 0.157] | 0.969 | 0.969 | 32 |
| IGBT | 1.0 | 0.832 [0.736; 0.927] | 0.406 [0.255; 0.577] | 0.594 | 0.969 | 32 |
| Fusível AC | 0.3 | 0.714 [0.580; 0.845] | 0.062 [0.017; 0.201] | 0.938 | 0.969 | 32 |
| Fusível AC | 0.5 | 0.835 [0.721; 0.939] | 0.062 [0.017; 0.201] | 0.938 | 0.969 | 32 |
| Fusível AC | 1.0 | 0.971 [0.909; 1.000] | 1.000 [0.893; 1.000] | 0.000 | 0.969 | 32 |

**Leitura honesta:** a AUC mede a separação por *ranking* (independe do limiar). No PONTO DE OPERAÇÃO (limiar operacional congelado), o recall pode ser bem menor que a AUC sugere. Atenção ao baixo recall em **IGBT (sev. 0.3), IGBT (sev. 0.5), Fusível AC (sev. 0.3), Fusível AC (sev. 0.5)**: o limiar conservador perde a maior parte dessas falhas. As linhas mostram todas as severidades, sem escolher apenas a melhor AUC. O holdout usa blocos intercalados por regime, com purga; a avaliação retém janelas sem compartilhamento direto de amostras, sem presumir independência temporal. A falha continua sintética: não é desempenho industrial.

## Detectabilidade E2 / Weibull

Unidade do eixo: `a_det_fracao_da_assinatura_nominal`; tempo físico calibrado: não.

| Falha | NPR | Detectadas/total | beta (IC95%) | eta (IC95%) | média a_det (IC95%) | a10 (IC95%) | margem restrita | R2 papel | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Contator AC | 315 | 31/31 | 3.15 [2.66; 3.96] | 0.5 [0.4; 0.5] | 0.4 [0.4; 0.5] | 0.2 [0.2; 0.3] | 0.424 | 0.95 | exploratório E2 |
| IGBT | 90 | 12/31 | 6.58 [4.40; 10.97] | 1.1 [1.0; 1.2] | não reportada | não reportado | 0.941 | 0.95 | não recomendado; alta indetectabilidade |
| Fusível AC | 30 | 30/31 | 8.60 [4.79; 15.77] | 0.9 [0.9; 0.9] | não reportada | não reportado | 0.847 | -0.84 | não recomendado; desvio no papel Weibull |

**Leitura obrigatória:** esta etapa modela a distribuição da **magnitude do primeiro cruzamento confirmado do detector**. A curva S_D(a) é probabilidade de ainda não detectar; h_D(a) é intensidade de detecção por unidade de magnitude. Nenhuma delas é confiabilidade ou taxa de falha do componente. A margem restrita de Kaplan-Meier não é RUL, pois não existe eixo temporal. MTTF, B10 e RUL permanecem apenas como aliases legados no JSON.

Os pontos empíricos usam Kaplan-Meier modificado com o tamanho total da amostra. Os ICs vêm de bootstrap de janelas sem amostras compartilhadas, mas independência temporal não foi demonstrada. O NPR prioriza risco na FMECA; ele **não determina** quantos eventos o experimento sintético produzirá e não explica causalmente a indetectabilidade.

## Validação externa GPVS-Faults - E3 de bancada

| Protocolo | AUC macro (IC95%) | Sensibilidade pós-falha | Especificidade | Acurácia balanceada | n ensaios |
|---|---:|---:|---:|---:|---:|
| Transferência direta AE | 0.732 [0.638; 0.826] | 1.000 | 0.007 | 0.503 | 14 |
| AE adaptativo | 0.815 [0.745; 0.881] | 0.445 | 0.974 | 0.709 | 14 |
| PCA adaptativo | 0.794 [0.721; 0.866] | 0.431 | 0.972 | 0.701 | 14 |

**Leitura honesta:** a transferência direta do limiar do ensaio F0 é rejeitada: sua especificidade macro é 0.007. Com adaptação usando somente o início saudável de cada ensaio, o AE alcança AUC macro 0.815 e especificidade 0.974, mas sensibilidade pós-falha de 0.445. Os IC95% são bootstrap de 14 ensaios, não de janelas. E3 aqui significa bancada experimental externa; não é campo, não identifica causa automaticamente e não calibra Weibull/RUL físico.
