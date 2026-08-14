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
| 2026-08-09 | 0.8577 | melhor 125; parada 145 | Consolidação do GPVS-Faults como único dataset canônico, com quatro papéis F0, E2 FMECA e E3 real | base `a4aa42c`; manifestos v2 deste PR | VIGENTE. 1.423 janelas F0, 24 features, FP saudável no teste 1,42%. Weibull descreve `a_det`, não tempo. |
| 2026-08-13 | 0.4405 | até 250 com early stopping; 3 arquiteturas x 5 seeds | Reconstrução V2 com perda balanceada por quatro famílias físicas e seleção sem falhas | branch `codex/autoencoder-v2-experimento`; hashes em `resultados/v2/autoencoder/` | CANDIDATO V2. Arquitetura 24-16-8-16-24, seed canônica 42, FP saudável 1,07%. |

## Validações externas

| Data | Dataset | Protocolo | Evidência | Base Git | Observações |
|---|---|---|---|---|---|
| 2026-08-09 | GPVS-Faults v1 | AE estrito F0 + AE adaptativo local (5 sementes) + PCA | E3 de bancada | `b0dbe7c` + hashes no manifesto | 14 ensaios de falha; transferência F0 invalidada por deslocamento; métricas vigentes em `resultados/gpvs/validacao_gpvs_e3.json`. |
| 2026-08-09 | GPVS-Faults canônico v2 | Um AE treinado em F0; pesos/limiar congelados; baseline de comissionamento pré-falha | E3 de bancada | base `a4aa42c`; manifesto `validacao_gpvs_e3` v2 | VIGENTE. AUC 0,773 [0,691; 0,853], sensibilidade 0,406 [0,211; 0,615], especificidade 0,974 [0,946; 0,992]. A linha v1 acima está substituída. |
| 2026-08-13 | GPVS-Faults, detector denso V2 | Seleção 3 arquiteturas x 5 seeds em F0; comparação congelada com PCA nos mesmos 14 ensaios | E3 de bancada | branch `codex/autoencoder-v2-experimento` | CANDIDATO V2. AE: AUC 0,778 [0,695; 0,859], sensibilidade 0,455 [0,268; 0,645], especificidade 0,953 [0,916; 0,978]. PCA teve AUC/especificidade maiores; AE teve sensibilidade maior. |

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
