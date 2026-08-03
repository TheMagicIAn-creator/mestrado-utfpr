# Retroalimentação da FMECA — método

Status: **IMPLEMENTADO (E2)** — elo metodológico entre a FMECA estática do TCC
(Torres, 2024) e a detecção dinâmica da dissertação.
Código: `src/ml/retroalimentacao_fmeca.py` · Saída:
`resultados/autoencoder/retroalimentacao_fmeca.{json,md}`

> **A pré-condição que faltava foi resolvida por nomenclatura, não por escolha
> de lado.** A versão anterior deste documento assumia que detectabilidade em
> sinal e em campo eram a mesma grandeza; `docs/fmeca.md` afirmava o contrário.
>
> Elas **continuam distintas** — e é por isso que uma não substitui a outra.
> Cada uma tem nome próprio (`D_campo`, `POD_mon`, `D_mon`; ver
> `docs/nomenclatura_deteccao.md`), a FMECA oficial de `docs/fmeca.md`
> permanece intacta, e o NPR projetado é publicado **em tabela separada**, como
> análise de sensibilidade.

## A ideia em uma frase

O TCC atribuiu `D_campo` por julgamento de literatura; a dissertação **mede**
`POD_mon`, a capacidade de detecção do monitoramento proposto — e a Tab. 4.8 do
próprio TCC converte uma na outra, fechando o ciclo do RCM.

## Mapeamento

1. **D_mon** — o índice informado pelo detector, obtido pela **leitura da
   escala**, não por régua nossa: a Tab. 4.8 define D em percentual de NÃO
   detectar, e `1 − POD_mon(s_ref)` é exatamente esse percentual.
   - `POD_mon = 1` (detecção perfeita) → 0% de não detecção → `D_mon = 1`;
   - `POD_mon = 0` (falha nunca detectada) → 100% → `D_mon = 10`, e a emenda
     `min` devolve o D_campo original — o NPR fica honesto em vez de otimista,
     sem precisar de exceção no código.
   - **D_proj = min(D_campo, D_mon)**: o monitoramento é ADICIONAL ao que já
     existe em campo; acrescentar um detector não torna falha alguma mais
     difícil de detectar, logo o índice só pode melhorar ou ficar igual.
2. **O (Ocorrência)** — NÃO muda com o detector (detecção não altera a taxa
   de falha). Só mudaria com dados de campo/manutenção (E3).
3. **S (Severidade)** — invariante: a consequência funcional da falha não
   depende do monitoramento.

Resultado: `NPR_projetado = S × O × D_proj`, comparável ao "NPR pós-manutenção"
do Apêndice E do TCC (210→18, 150→10), com a diferença de que `D_proj` é
**MEDIDO** (E2), não julgado.

## Por que a circularidade não se aplica mais

A salvaguarda anterior era "congelar a régua antes de olhar os números", porque
uma régua escolhida depois dos resultados teria sido calibrada para produzir o
resultado desejado.

Ela deixou de ser necessária: **não há régua**. As faixas são as da Tab. 4.8,
publicadas em 2024, antes de qualquer medição deste projeto. Não há o que
congelar porque não há o que escolher.

> ⚠️ Pendência factual (não metodológica): `docs/fmeca.md` registra só os
> extremos da Tab. 4.8. As faixas intermediárias em
> `src/ml/retroalimentacao_fmeca.py::BORDAS_D` são reconstrução aritmética e
> devem ser conferidas no TCC. Se divergirem, muda **uma constante**.

## Salvaguardas obrigatórias

- O recálculo herda o nível de evidência do detector: **E2** (sintético
  orientado pela FMECA). Apresentar como "NPR projetado sob validação
  sintética", nunca como NPR de campo.
- Weibull/RUL sintético: usar apenas `POD_mon`/SMD na conversão; não usar
  MTTF/B10 em passos sintéticos como argumento de O, mesmo quando o MLE converge.
- A **severidade de referência** é declarada, nunca implícita (`s_ref = 1,0`), e
  a curva `POD_mon(s)` completa acompanha o escalar. Em severidade baixa o
  quadro é outro, e omitir isso seria escolher o número que convém.
- A tabela projetada é **separada** da FMECA oficial. Duas tabelas, não uma
  substituída.

## Como executar

    python -m src.ml.retroalimentacao_fmeca
    python -m src.ml.retroalimentacao_fmeca --severidade 0.5

Lê `resultados/autoencoder/validacao_report.json` (recall por falha e
severidade, no limiar congelado) e escreve `retroalimentacao_fmeca.{json,md}`
na mesma pasta. Não precisa de `torch` nem do dataset: os índices S/O/D_campo
são lidos de `src/ml/injecao_falhas.py` por AST, sem importar o módulo.

## O que discutir na dissertação

1. **A inversão da ordem de criticidade**, quando ocorrer. Ela não é mecânica:
   com detecção uniformemente perfeita a ordem se PRESERVA. Quem inverte é o
   componente que o detector trata pior, porque é o único cujo `min` não cede.
2. **O componente cujo NPR quase não se move** — sinal de que sua criticidade
   nunca veio de detectabilidade, e sim de S×O. Nenhum monitoramento resolve
   isso; a ação de manutenção tem de ser outra.
3. **Falha com `POD_mon` baixo ou SMD nula**: o ciclo RCM recomendaria revisar
   o **método de detecção** (outra feature, outro sensor, outro modelo) em vez
   de declarar o item coberto.
