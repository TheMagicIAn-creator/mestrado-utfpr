# Preflight da reconstrução executiva do ALIAdo

> **Registro histórico de baseline.** Os estados abaixo descrevem o commit de
> origem em 27/08/2026 e não devem ser lidos como estado atual. O fechamento da
> campanha está em `docs/exec-plans/ALIADO_RELATORIO_FINAL_2026-09-01.md`.

Data: 2026-08-27  
Especificação: `ALIADO_ALINHAMENTO_ARQUITETURA_METODOLOGIA.md`

## Estado Git

- Branch de origem: `main`.
- Commit de origem: `0e678aacd208535ea7333bedc007245decc4e8de`.
- `main` local alinhado ao `origin/main` no início da campanha.
- Branch de preservação: `codex/safety-master-spec-0e678aa`.
- Arquivos locais do pesquisador preservados e fora da campanha:
  - `notas/sessoes/2026-08-23_22-36-30_sessao_d_sessao_web.md`;
  - `notas/sessoes/2026-08-23_22-37-17_sessao_7_sessao_web.md`.

## Dataset e pipeline

- Dataset ativo: GPVS-Faults, DOI `10.17632/n76t439f65.1`.
- Ensaios presentes: 16 (`F0L` a `F7M`).
- Condições saudáveis: `F0L` e `F0M`.
- Ensaios com falha: 14 (`F1L` a `F7M`).
- Janelas processadas: 1.423 saudáveis e 9.391 nos ensaios de falha.
- Features canônicas: 24.
- Fronteira de falha: ponto médio nominal, sem canal instrumentado de disparo.

## Divergências C1-C6

- C1 confirmada: agente e documentação acoplados ao Gemini.
- C2 confirmada: AUC-PR publicada como métrica principal.
- C3 confirmada: limiar p99 fixo no código e nos artefatos.
- C4 já corrigida no baseline: matrizes de confusão absolutas e normalizadas existem.
- C5 confirmada: não há contrato anulável para `POD_mon`, `D_mon`, `D_proj` e `NPR_proj`.
- C6 parcialmente resolvida: o modelo exponencial está matematicamente correto, mas não há contrato extensível para distribuições parametrizadas.
- Escore localizado top-k ausente; o baseline usa MSE médio nas 24 features.

## Estado após a campanha

- C1 corrigida: Provider Gateway e Router comuns para OpenAI/Gemini.
- C2 corrigida: Recall, F1 e Precision são principais; AUCs são complementares.
- C3 corrigida no contrato: top-k e percentil configuráveis; `k=5` e p99,9 são
  configurações de trabalho, com p100 efetivo para `n=210`.
- C4 preservada: matrizes absolutas e normalizadas continuam publicadas.
- C5 corrigida estruturalmente: contrato anulável presente e publicação de NPR
  projetado bloqueada até existir mapeamento `POD_mon -> D_mon` validado.
- C6 corrigida estruturalmente: contratos de parâmetros existem; somente o
  exponencial é publicado, e Weibull/Normal/Lognormal permanecem bloqueados por
  ausência de vidas, exposição e censura.
- A ablação temporal do AE-LSTM foi inconclusiva; nenhuma superioridade
  arquitetural é declarada.

## Verificações baseline

- `scripts/verificar_projeto.py`: aprovado, sem avisos ou erros.
- `scripts/auditar_resultados.py`: aprovado, 2 manifestos e 30 artefatos.
- `scripts/avaliar_agente.py`: 15 de 15 casos aprovados.
- `pytest -m "not pesado"`: 430 aprovados, 3 ignorados e 17 excluídos.
- Testes com PyTorch real: 6 aprovados.
- Ruff `F821,F822,F823`: aprovado.
- Gate de qualidade da campanha: aprovado.
- Cobertura de código novo medida na campanha: 82,7%.
- Duplicação em código novo: 0,0%.
- Hotspots revisados: 100%.

## Restrições da campanha

- Não ler, versionar ou imprimir segredos.
- Não incluir dados brutos, pesos, índices locais ou estado do Obsidian.
- Não fabricar distribuições físicas, POD ou NPR projetado.
- Não selecionar limiar, top-k ou arquitetura usando o holdout final de falhas.
