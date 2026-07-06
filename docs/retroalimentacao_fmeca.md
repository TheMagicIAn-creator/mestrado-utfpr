# Retroalimentação do FMECA — proposta metodológica

Status: **PROPOSTA (E0)** — elo metodológico entre o FMECA estático do TCC
(Torres, 2024) e a detecção dinâmica da dissertação. Nada aqui foi executado;
este documento existe para que a ponte TCC → dissertação seja explícita e
defensável perante a banca.

## A ideia em uma frase

O TCC atribuiu notas de Detecção (D) por julgamento de literatura; a
dissertação MEDE a capacidade de detecção do monitoramento proposto — logo, os
resultados do detector podem substituir o D julgado por um D medido,
recalculando o NPR e fechando o ciclo do RCM.

## Mapeamento proposto

1. **D (Detecção)** — é a coluna diretamente informada pelo detector:
   - SMD baixa + recall alto no limiar operacional → D baixo (falha fácil de
     detectar COM o monitoramento proposto);
   - SMD nula (caso vigente do desbalanceamento) → D permanece alto: o
     monitoramento proposto NÃO reduz a nota de detecção dessa falha — e o
     NPR pós-monitoramento fica honesto em vez de otimista.
   - Regra de conversão sugerida (a calibrar): D_novo = f(recall no limiar
     operacional), com f decrescente e por faixas (ex.: recall ≥ 0,9 → D=2–3;
     0,5–0,9 → D=4–6; < 0,5 ou SMD nula → manter D original).
2. **O (Ocorrência)** — NÃO muda com o detector (detecção não altera a taxa
   de falha). Só mudaria com dados de campo/manutenção (E3).
3. **S (Severidade)** — invariante: a consequência funcional da falha não
   depende do monitoramento.

Resultado: NPR_pós-detector = S × O × D_novo, comparável ao "NPR
pós-manutenção" do Apêndice E do TCC (210→18, 150→10), mas com a diferença
de que o D_novo é MEDIDO (E2), não julgado.

## Salvaguardas obrigatórias

- O recálculo herda o nível de evidência do detector: **E2** (sintético
  orientado pelo FMEA). Apresentar como "NPR projetado sob validação
  sintética", nunca como NPR de campo.
- Falha com ajuste Weibull rejeitado (KS): usar apenas o recall/SMD na
  conversão de D; não usar MTTF/B10 rejeitados como argumento de O.
- A conversão recall→D deve ser definida ANTES de olhar os números (regra a
  priori), para não calibrar a régua no resultado.

## Passos para executar (quando priorizado)

1. Congelar a tabela de conversão recall→D (com a orientadora).
2. Ler recall por falha no limiar congelado (`validacao_report.json`).
3. Recalcular NPR por modo de falha e montar a tabela comparativa
   (D original TCC × D medido × NPR resultante).
4. Discutir divergências — em especial o desbalanceamento (SMD nula):
   o caso em que o ciclo RCM recomendaria revisar o método de detecção
   (outra feature, outro sensor ou outro modelo) em vez de declarar o item
   coberto.
