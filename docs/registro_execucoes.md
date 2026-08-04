# Registro de execuções do pipeline — changelog

Propósito: impedir que números de execuções antigas circulem como vigentes
(em memórias consolidadas, sessões ou texto da dissertação). TODA execução
completa do pipeline deve ganhar uma linha aqui. Os valores citáveis vivem
nos artefatos JSON — este registro guarda apenas o CONTEXTO de cada execução
(o que mudou e por quê).

Regra de uso: ao rodar o pipeline, acrescentar uma linha com data, motivo e
o commit. Nunca editar linhas antigas — este arquivo é só-acréscimo.

| Data | Limiar operacional | Épocas AE | Motivo da execução | Commit | Observações |
|---|---:|---:|---|---|---|
| ~2026-05-24 | 2.9103 | 84 | Primeira execução completa (Fase 5) | — | Números registrados nas memórias de 24–30/05. SUBSTITUÍDA. |
| 2026-06-17 | 2.0785 | 150 | Reexecução após curadoria/refino do treino | 0ce1e77 | SUBSTITUÍDA — artefatos removidos no reset de 2026-07-07 (histórico recuperável no git). Achados da execução: SMD do desbalanceamento = null (não detectada em nenhuma severidade — limitação, ver evidence_levels.md); KS rejeitava Weibull nas 3 famílias. |
| 2026-07-07 | — | — | RESET: artefatos zerados para reexecução com semente determinística do Weibull, estilo gráfico único e comparação com a literatura | — | Próxima execução completa do pipeline + experimentos gera os artefatos vigentes. Registrar aqui ao rodar. |

## Divergências conhecidas e resolvidas

- **Weibull irreprodutível entre execuções (até 2026-07-06).** O jitter dos
  TTF censurados usava o RNG global do NumPy sem semente
  (`rul_weibull.py`), fazendo beta/eta variarem entre execuções idênticas —
  é a causa provável de a memória de 17/06 citar beta=5.87 para
  desbalanceamento enquanto o artefato vigente registra beta=2.30.
  CORRIGIDO: semente derivada do índice da falha. A partir da próxima
  execução, o Weibull é determinístico para o mesmo modelo/limiar.
- **Memórias de 24–30/05 com números da era p99=2.9103.** Anotadas com aviso
  de substituição em `notas/memorias/` (conteúdo original preservado como
  histórico).
