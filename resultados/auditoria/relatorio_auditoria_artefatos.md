# Auditoria dos artefatos acadêmicos

## Escopo e fonte dos dados

O pipeline canônico usa **GPVS-Faults** (DOI `10.17632/n76t439f65.1`). F0L/F0M formam o domínio saudável; F1L-F7M são os 14 ensaios de falha da validação externa E3. A injeção orientada pela FMECA e a análise Weibull são validações sintéticas E2 construídas sobre janelas saudáveis GPVS; não mesclam outro dataset no treinamento principal.

## Cobertura versionada

Foram inventariados **65 artefatos científicos rastreados** em `resultados/` e `dados/processados/`, excluindo os dois arquivos produzidos por esta própria auditoria. canonico: 39, legado_comparativo: 12, manifesto: 8, suplementar: 6. O CSV anexo registra tamanho, SHA-256, etapa proprietária, papel e validação estrutural de cada saída.

Hashes divergentes de manifestos: **0**. JSON/CSV estruturalmente inválidos: **0**.

Modelos (`*.pt`, `*.pkl`), dados brutos, estado local do Obsidian e figuras opcionais do benchmark Ibrahim permanecem deliberadamente ignorados; são regeneráveis ou locais e não constituem resultados canônicos publicáveis.

## Auditoria paramétrica e dos eixos

- `a_det` é magnitude de assinatura em `[0, 1]`, não tempo, vida útil nem probabilidade de falha física.
- Os cruzamentos são observados em 120 pontos (`delta_a = 1/119`). O MLE Weibull 2P usa censura por intervalo em cada célula da grade; não detecções no teto usam censura à direita sob hipótese declarada.
- O papel Weibull agrupa empates da grade. `R2pp` permanece triagem descritiva, não teste formal de aderência.
- As matrizes de confusão preservam contagens, normalizam a cor por classe real e exibem `n` mais percentual da linha.
- ECDF, densidades, ROC/PR e séries temporais mantêm grandeza e unidade explícitas; painéis com limites distintos incluem aviso para comparação pelos valores dos eixos.

## Diagnóstico Weibull 2P

| Componente | Eventos | Níveis distintos | Empates | beta | eta | R2pp | Síntese |
|---|---:|---:|---:|---:|---:|---:|---|
| Contator AC | 100/100 | 34 | 66.0% | 5.287 | 0.455 | 0.911 | recomendada |
| IGBT | 100/100 | 34 | 66.0% | 4.316 | 0.358 | 0.790 | não recomendada |
| Fusível AC | 100/100 | 6 | 94.0% | 4.115 | 0.039 | 0.455 | não recomendada |

O fusível não tem poucos eventos: há 100 cruzamentos. A limitação é resolução, pois eles se concentram em poucos níveis da grade. IGBT e fusível só permanecem não recomendados quando o desvio do modelo 2P continua após corrigir quantização e empates.

## Arquivos

- `inventario_artefatos.csv`: relação completa e verificável.
- `relatorio_auditoria_artefatos.md`: síntese metodológica desta auditoria.
