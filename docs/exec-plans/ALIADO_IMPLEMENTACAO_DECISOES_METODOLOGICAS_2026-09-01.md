# Relatório de implementação das decisões metodológicas - 2026-09-01

## Linha de base

- Especificação incorporada pelo PR #174.
- Base da implementação: `94c70fd` (`main` após o merge do PR #174).
- Branch: `codex/implementar-decisoes-metodologicas-2026-09-01`.
- Linha de base de testes: 585 aprovados, 3 ignorados e 17 pesados não
  selecionados.
- Implementação: `f96c0031f73a4a27e20a6733b9a9e64773a49f93`.
- PR final: `#175`.

## A. Implementado

1. O GPVS-Faults permanece como único dataset experimental ativo, com 16
   ensaios F0L-F7M e 24 features elétricas.
2. F0L/F0M continuam separados em treino, validação, calibração e teste
   saudável. F1L-F7M entram apenas na E3, depois do congelamento.
3. O catálogo nativo registra F1 como falha completa de IGBT, F2 como erro de
   20% no sistema de sensor/realimentação, F6 como ganho PI reduzido em 20% e
   F7 como constante de tempo PI elevada em 20%.
4. F6/F7 são explicitamente anomalias funcionais do sistema/circuito de
   controle, sem equivalência automática com falha física de PCB.
5. O núcleo experimental não usa falhas sintéticas.
6. A comparação preserva as mesmas features, partições, scaler ajustado apenas
   no treino, orçamento de treino, early stopping e sementes para os dois
   modelos.
7. O AE-LSTM usa contexto causal contínuo `[W_(t-7), ..., W_t]` e decide em
   `W_t`. A análise distingue as sete janelas de transição da falha sustentada.
8. O diagnóstico com contexto reiniciado permanece apenas auxiliar.
9. A grade de sensibilidade é exatamente `k={5,10,20}` por
   `p={99;99,5;99,9}`, totalizando nove configurações por modelo e semente.
10. Cada limiar é calibrado somente no saudável e registra estatística de
    ordem, percentil solicitado, percentil efetivo, tamanho e resolução.
11. `k=5/p99,9` permanece apenas como referência histórica reproduzível, não
    como ótimo universal.
12. Recall, F1 e Precision permanecem métricas principais; ROC-AUC e PR-AUC
    são complementares. Matrizes absolutas e normalizadas foram preservadas.
13. A FMECA vigente passa a conter IGBT, sistema de sensor/realimentação e
    sistema/circuito de controle do inversor.
14. S, O, D e NPR da nova FMECA permanecem nulos, sem herdar os valores do
    recorte histórico Contator AC/IGBT/Fusível AC.
15. A publicação científica atual não contém a antiga projeção de criticidade
    a partir de métricas do detector.
16. Os cenários exponenciais do TCC permanecem como sensibilidade
    bibliográfica histórica, separados da FMECA atual e da E3.
17. A busca no corpus local não encontrou parâmetros Weibull 2P rastreáveis
    para IGBT; a publicação mantém `beta=null` e `eta=null`.
18. Resultados, figuras e manifestos v2 foram regenerados.

## B. Arquivos alterados

### Núcleo ML

- `src/ml/dados_gpvs.py`: catálogo nativo F1-F7, vínculo com o escopo FMECA e
  declaração de ausência de falhas sintéticas.
- `src/ml/modelos_autoencoder.py`: remoção do helper que substituía somente o
  último passo sobre contexto saudável.
- `src/ml/avaliacao_comparativa.py`: contrato explícito de causalidade,
  decisão em W_t, transição, falha sustentada e ausência de cruzamento.
- `src/ml/sensibilidade_escore.py`: grade 3x3, papéis dos dados, ensaios válidos
  e referência histórica.
- `src/ml/publicacao_comparacao.py`: contrato, relatório e manifesto alinhados
  à nova grade e à análise causal.
- `src/ml/confiabilidade_componentes.py`: novo escopo FMECA anulável e auditoria
  bibliográfica de Weibull para IGBT.
- `src/ml/publicacao_confiabilidade.py` e `graficos_confiabilidade.py`:
  separação explícita dos cenários bibliográficos históricos.
- `src/ml/estilo_graficos.py`: remoção das cores mortas do trio FMECA antigo.

### Agente, aplicação e verificações

- `src/conhecimento/agente.py`: contexto autoritativo atualizado.
- `src/webapp/scientific_context.py`: contexto científico da interface alinhado
  à grade, causalidade e FMECA vigente.
- `scripts/auditar_resultados.py`: gates para a grade 3x3, causalidade, FMECA
  anulável e ausência dos campos revogados.

### Documentação

- `docs/fmeca.md`, `docs/metodologia_ml.md`, `docs/datasets.md`,
  `docs/confiabilidade_fisica.md`, `docs/mapa_de_resultados.md`,
  `docs/reproducibilidade.md` e `docs/glossario.md`: contrato científico atual.
- `docs/exec-plans/active/ALIADO_ALINHAMENTO_ARQUITETURA_METODOLOGIA.md`:
  decisões antigas substituídas pelas aprovadas em 2026-09-01.
- `README.md`, `src/README.md` e `CLAUDE.md`: escopo e operação atualizados.

### Testes e artefatos

- `tests/test_decisoes_metodologicas_20260901.py`: novas guardas metodológicas.
- Testes de comparação, sensibilidade, confiabilidade, resultados e web foram
  atualizados para o novo contrato.
- `resultados/comparacao/`, `resultados/confiabilidade/` e
  `resultados/manifestos/`: tabelas, JSON, PNG, PDF, relatórios e hashes
  regenerados.

## C. Testes

| Comando | Resultado |
|---|---|
| `python -m src.ml.comparacao_autoencoders` | concluído; 23 saídas |
| `python -m src.ml.publicacao_confiabilidade` | concluído; 14 saídas |
| `python scripts/auditar_resultados.py` | aprovado; 9 manifestos e 37 artefatos |
| `python scripts/verificar_projeto.py` | aprovado; 16 ensaios, sem legado ou avisos |
| `python -m pytest -m "not pesado"` | 590 aprovados, 3 ignorados, 17 não selecionados |
| testes Torch reais | 20 aprovados |
| testes dataset/decisões | 12 aprovados |
| testes focados de publicação/web/docs | 76 aprovados, 2 ignorados |
| `python -m ruff check --select F821,F822,F823 src tests scripts` | aprovado |
| `python -m compileall -q src tests scripts` | aprovado |

## D. Divergências corrigidas

- C1 - Provider Gateway OpenAI + Gemini: preservado, sem expansão nesta fase.
- C2 - hierarquia de métricas: Recall/F1/Precision principais; AUC
  complementar.
- C3 - limiar: referência p99,9 mantida com resolução empírica e grade
  p99/p99,5/p99,9 sem seleção pelas falhas.
- C4 - matrizes: absolutas e normalizadas preservadas e auditadas.
- C5 - FMECA: novo escopo anulável; nenhuma transformação de desempenho do
  detector em NPR.
- C6 - confiabilidade: exponencial publicada somente onde há lambda; Weibull,
  Normal e Lognormal bloqueadas sem dados adequados.

## Resultados regenerados

### Referência histórica, semente 42

| Métrica | Autoencoder Denso | AE-LSTM |
|---|---:|---:|
| Recall macro | 0,384 (IC95% 0,180-0,596) | 0,387 (0,181-0,603) |
| F1 macro | 0,432 (0,215-0,652) | 0,433 (0,212-0,657) |
| Precision macro | 0,870 (0,667-0,997; 12 ensaios válidos) | 0,942 (0,848-0,998; 9 válidos) |
| Falso positivo no teste saudável | 0,712% | 1,068% |
| Limiar k=5/p99,9 | 5,104478 | 18,583059 |

Nos dois modelos, p99,9 com `n=210` selecionou ordem 210/210, percentil
efetivo p100 e resolução de 0,476 ponto percentual.

As diferenças pareadas Denso menos AE-LSTM foram:

- Recall: -0,0026 (IC95% -0,0117 a 0,0054);
- F1: -0,0012 (IC95% -0,0179 a 0,0130);
- Precision: -0,0599 (IC95% -0,1973 a 0,0200; 9 pares válidos).

Os intervalos cruzam zero. A análise de falha sustentada também permanece
`inconclusive`; não há suporte para declarar vencedor.

## E. Pendências metodológicas

### Valores que exigem decisão do pesquisador

| Item FMECA | severity | occurrence | detectability | npr | status |
|---|---:|---:|---:|---:|---|
| IGBT | `null` | `null` | `null` | `null` | `awaiting_user_fmeca` |
| Sistema de sensor/realimentação | `null` | `null` | `null` | `null` | `awaiting_user_fmeca` |
| Sistema/circuito de controle do inversor | `null` | `null` | `null` | `null` | `awaiting_user_fmeca` |

Para preencher cada linha ainda são necessários critério da escala, valor,
fonte, página/tabela, unidade observacional e justificativa metodológica.

### Confiabilidade

- Exponencial: publicado como cenário histórico com lambda rastreada.
- Contator derivado: `2,10e-5 h^-1`.
- IGBT derivado: `1,05e-5 h^-1`.
- Fusível derivado: `7,00e-6 h^-1`.
- Fusível direto: `2,17e-6 h^-1`, Tabela 3.4 do TCC.
- Weibull 2P para IGBT: bloqueada; `beta=null`, `eta_hours=null`.
- Normal, Lognormal e histograma de vidas: bloqueados por falta de amostra de
  tempos individuais, exposição e censura.

## F. Riscos

1. A fronteira de falha dos CSVs é nominalmente 50% porque não há canal de
   disparo instrumentado.
2. As janelas de um mesmo ensaio são autocorrelacionadas; por isso o bootstrap
   usa o ensaio como unidade.
3. Precision fica indefinida em ensaios sem alarme positivo, reduzindo o número
   de pares válidos.
4. E3 é evidência de bancada e não demonstra desempenho de campo.
5. As taxas bibliográficas históricas não são medições da população estudada.

## G. Próxima ação recomendada

1. O pesquisador definir S/O/D e fontes para os três itens da FMECA vigente.
2. Revisar com a orientadora a interpretação dos resultados inconclusivos e o
   compromisso entre Recall e falsos alarmes.
3. Só habilitar Weibull físico se surgir fonte rastreável com beta/eta de IGBT
   ou uma amostra válida de tempos de falha e censura.
