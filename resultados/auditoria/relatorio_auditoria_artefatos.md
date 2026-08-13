# Auditoria dos artefatos acadêmicos

## Escopo e fonte dos dados

O pipeline canônico usa **GPVS-Faults** (DOI `10.17632/n76t439f65.1`). F0L/F0M formam o domínio saudável; F1L-F7M são os 14 ensaios de falha da validação externa E3. A injeção orientada pela FMECA e a análise Weibull são validações sintéticas E2 construídas sobre janelas saudáveis GPVS; não mesclam outro dataset no treinamento principal.

## Cobertura versionada

Foram inventariados **73 artefatos científicos rastreados** em `resultados/` e `dados/processados/`, excluindo os arquivos produzidos por esta própria auditoria. canonico: 49, legado_comparativo: 12, manifesto: 8, suplementar: 4. O CSV anexo registra tamanho, SHA-256, etapa proprietária, papel e validação estrutural de cada saída.

Hashes divergentes de manifestos: **0**. JSON/CSV estruturalmente inválidos: **0**.

O catálogo cobre **27 figuras**: 3 com eixo temporal e 24 sem eixo temporal. Problemas automáticos de integridade visual: **0**.

Modelos (`*.pt`, `*.pkl`), dados brutos, estado local do Obsidian e figuras opcionais do benchmark Ibrahim permanecem deliberadamente ignorados; são regeneráveis ou locais e não constituem resultados canônicos publicáveis.

## Auditoria paramétrica e dos eixos

- `a_det` é magnitude de assinatura em `[0, 1]`, não tempo, vida útil nem probabilidade de falha física.
- Os cruzamentos canônicos são observados em 501 pontos (`delta_a = 0,002`), com sensibilidade em 101 e 251 pontos. A persistência ocupa `delta_a = 0,02` em todas as grades. O MLE Weibull 2P usa censura por intervalo em cada célula da grade; não detecções no teto usam censura à direita sob hipótese declarada.
- O papel Weibull agrupa empates da grade. `R2pp` permanece diagnóstico visual; a decisão usa bootstrap paramétrico com a mesma quantização e estabilidade entre as duas grades mais finas.
- Papel de probabilidade, PDF/CDF, sobrevivência e intensidade são figuras separadas. A intensidade `h_D(a)` não contém marcas empíricas sobre o eixo e é rotulada como não física.
- F0L e F0M são estratificados. Ambos pertencem ao GPVS; nenhum Paderborn ou PMSM entra nos gráficos canônicos.
- As matrizes de confusão preservam contagens, normalizam a cor por classe real e exibem `n` mais percentual da linha.
- ECDF, densidades, ROC/PR e séries temporais mantêm grandeza e unidade explícitas; painéis com limites distintos incluem aviso para comparação pelos valores dos eixos.

## Diagnóstico Weibull 2P

| Componente | Eventos | Níveis distintos | Empates | beta | eta | R2pp | p aderência | Grade estável | Síntese |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Contator AC | 277/277 | 140 | 49.5% | 3.868 | 0.397 | 0.933 | 0.004 | sim | não recomendada |
| IGBT | 277/277 | 126 | 54.5% | 3.379 | 0.312 | 0.929 | 0.004 | sim | não recomendada |
| Fusível AC | 277/277 | 143 | 48.4% | 4.184 | 0.586 | 0.886 | 0.004 | sim | não recomendada |

O número de eventos e o número de níveis distintos são grandezas diferentes: empates não removem trajetórias. A interpretação principal é a distribuição empírica global; a Weibull 2P só é adotada quando aderência, resolução e estabilidade permitem.

## Arquivos

- `inventario_artefatos.csv`: relação completa e verificável.
- `catalogo_figuras.csv`: dataset, gerador, eixos, evidência e QA visual.
- `relatorio_auditoria_artefatos.md`: síntese metodológica desta auditoria.
