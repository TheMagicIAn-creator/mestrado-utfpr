# Fechamento da auditoria geral de `src/`

**Data local:** 2026-08-05  
**Estado auditado:** `main` no merge commit `fbcc007`  
**Status:** concluída

## Método

As correções foram publicadas em PRs seriados, sempre a partir do `main`, com
CI verde antes do merge. O trabalho foi feito em worktree isolado para preservar
o checkout local do pesquisador e suas alterações não relacionadas.

## Matriz de encerramento

| # | Achado original | Resolução | Evidência |
|---:|---|---|---|
| 1 | Mapa arquitetural incompleto | Inventário integral e atualizado de `src/` | PR #96; teste `test_readme_src_inventaria_todos_os_modulos` |
| 2 | Acoplamento bidirecional `conhecimento`/`ml` | Adaptador unidirecional `conhecimento/resultados_ml.py` e guarda arquitetural | PR #97; `tests/test_dependencias_src.py` |
| 3 | Duas comparações acadêmicas concorrentes | Remoção do caminho legado e fonte única Proposto x Ibrahim | PR #96; `tests/test_comparacao_macro.py` |
| 4 | Contradição do escore operacional | Diagnóstico alinhado ao escore localizado top-k e limiar efetivo | PR #96; `tests/test_limiar.py` |
| 5 | FMEA usado onde a origem era FMECA | Descrições operacionais corrigidas e regressão textual | PR #96; `test_descricoes_operacionais_usam_fmeca` |
| 6 | Modelo/build defasados | Constante aposentada removida e identificação de build atualizada | PR #96; testes de consistência documental |
| 7 | Módulos grandes e utilitários duplicados | Texto/log centralizados e módulos decompostos por responsabilidade | PRs #99 e #102; limite global de 1.000 linhas |
| 8 | Falhas recuperáveis silenciosas | Avisos de interface e logs explícitos, sem `except Exception: pass` em `src/` | PR #98; regressão AST em `tests/test_dependencias_src.py` |
| 9 | Lock do ChromaDB restrito a threads | Lock entre processos em Windows/POSIX, validado por subprocesso real | PR #97; `tests/test_index_lock.py` |
| 10 | Módulos críticos sem teste direto | Testes diretos e matriz de responsabilidade para os alvos auditados | PR #100; `tests/test_cobertura_auditoria_src.py` |
| 11 | Macrocomparação E2E incompleta | Execução integral concluída e artefatos derivados auditados | PR #101; duração de 1.935,3 s |

## PRs incorporados

| PR | Merge commit | Conteúdo principal |
|---|---|---|
| #96 | `5103e6d` | Consistência acadêmica, nomenclatura, documentação e configuração |
| #97 | `07f2701` | Dependências unidirecionais e lock entre processos |
| #98 | `2ee3998` | Observabilidade das falhas recuperáveis |
| #99 | `ab54deb` | Utilitários compartilhados de texto e log |
| #100 | `e816750` | Cobertura direta dos módulos críticos |
| #101 | `65392c3` | Validação E2E e métricas macro inequívocas |
| #102 | `fbcc007` | Decomposição dos módulos excessivamente grandes |

## Evidência E2E

Comando executado:

```powershell
python -m src.ml.macro_comparar
```

- Duração total: **1.935,3 s** (32 min 15 s).
- Holdout: 44 janelas não sobrepostas.
- Protocolo: 17 janelas de calibração e 25 de avaliação, com purga.
- Falso positivo fora da amostra: 0,0% nos dois métodos.
- Dataset bruto, features, pesos e scaler foram usados apenas localmente e
  permaneceram ignorados pelo Git.

| Método | Contator AC AUC | IGBT AUC | Fusível AC AUC |
|---|---:|---:|---:|
| Proposto, AE denso + escore localizado | 1,000 | 0,978 | 0,927 |
| Ibrahim 2022, AE-LSTM temporal | 1,000 | 0,909 | 0,885 |

Os JSONs e PNGs versionados foram reproduzidos sem diferença. Seis tabelas
foram regeneradas apenas para distinguir explicitamente:

- `TPR @FPR=10%, sev=1.0`, usada na comparação entre métodos;
- `deteccao_limiar_sev1`, usada na leitura operacional do limiar calibrado.

O gerador também passou a aceitar o roundtrip JSON, no qual chaves numéricas de
severidade são desserializadas como strings.

## Validação final

- `518 passed, 1 skipped, 16 deselected` em `pytest -m "not pesado"`.
- `ruff F821/F822/F823`: sem achados.
- `compileall src`: concluído.
- CI dos PRs #96 a #102: verde antes de cada merge.
- Maior módulo atual de `src/`: 952 linhas.
- Limite bloqueante: 1.000 linhas por módulo Python.
- Nenhum dado bruto, modelo treinado, scaler, segredo ou estado local do
  Obsidian foi incluído nos commits.

## Conclusão

Os onze achados da auditoria foram encerrados com alteração incorporada ao
`main` e evidência automatizada ou execução integral. Não há bloqueio técnico
remanescente dentro do escopo auditado.
