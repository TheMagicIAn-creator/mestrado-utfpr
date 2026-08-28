# Preflight da reconstrução executiva do ALIAdo

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

## Verificações baseline

- `scripts/verificar_projeto.py`: aprovado, sem avisos ou erros.
- `scripts/auditar_resultados.py`: aprovado, 2 manifestos e 30 artefatos.
- `scripts/avaliar_agente.py`: 15 de 15 casos aprovados.
- `pytest -m "not pesado"`: 430 aprovados, 3 ignorados e 17 excluídos.
- Testes com PyTorch real: 6 aprovados.
- Ruff `F821,F822,F823`: aprovado.
- SonarQube Quality Gate: aprovado.
- Cobertura de código novo no SonarQube: 82,7%.
- Duplicação em código novo: 0,0%.
- Hotspots revisados: 100%.

## Restrições da campanha

- Não ler, versionar ou imprimir segredos.
- Não incluir dados brutos, pesos, índices locais ou estado do Obsidian.
- Não fabricar distribuições físicas, POD ou NPR projetado.
- Não selecionar limiar, top-k ou arquitetura usando o holdout final de falhas.
