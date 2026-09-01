# Relatório final da campanha de alinhamento do ALIAdo

Data de fechamento: 01/09/2026.

## A. Implementado

- Provider Gateway e Router comuns para OpenAI e Gemini, com aliases lógicos,
  retry, fallback e contratos de streaming/JSON independentes de SDK.
- Comparação E3 canônica entre Autoencoder Denso e AE-LSTM no GPVS-Faults,
  mantendo partições, sementes, orçamento, 24 features e modelos congelados.
- Recall, F1 e Precision na camada principal; ROC-AUC e PR-AUC como métricas
  complementares; matrizes absolutas e normalizadas preservadas.
- Escore top-k configurável e limiar saudável configurável. A configuração de
  trabalho é `k=5`, p99,9 solicitado; com 210 observações, o ponto efetivo é
  p100, ordem 210/210 e resolução de 0,476 ponto percentual.
- Ablação temporal do AE-LSTM com contexto corrente, transição, falha sustentada
  e reinício pós-fronteira. Resultado pré-especificado: inconclusivo.
- Sensibilidade descritiva para seis valores de k e cinco percentis, sem usar
  falhas para selecionar configuração.
- Auditoria semântica das PRs #99/#102 encerrada, com fachadas protegidas por
  testes de compatibilidade.
- Ponte GPVS-FMECA explicitada: F1L-F7M mantêm rótulos nativos e não são
  relabelados como Contator AC, IGBT ou Fusível AC.
- Contrato anulável para `POD_mon`, `D_mon`, `D_proj` e `NPR_proj`. A publicação
  bloqueia projeção e mantém os NPR base 315, 90 e 30.
- Contratos de disponibilidade para Exponencial, Weibull 2P, Normal, Lognormal
  e histograma de vidas. Somente o Exponencial é publicado com os dados atuais.
- Perfil do agente, README e documentação alinhados à arquitetura e à
  metodologia vigentes.

## B. Arquivos alterados

- `src/ml/confiabilidade_componentes.py`: contratos nulos de extensão FMECA e
  de distribuições físicas.
- `src/ml/publicacao_confiabilidade.py`: ressalvas e parâmetros faltantes no
  relatório versionado.
- `resultados/confiabilidade/metodologia.json`: schema 5, C5/C6 auditáveis.
- `resultados/confiabilidade/relatorio.md`: bloqueios metodológicos explícitos.
- `resultados/manifestos/confiabilidade_componentes.json`: hashes da publicação
  regenerada.
- `tests/test_confiabilidade_componentes.py` e
  `tests/test_resultados_canonicos.py`: ausência obrigatória de valores
  projetados/fabricados.
- `tests/test_consistencia_docs.py`: contrato documental OpenAI+Gemini.
- `CLAUDE.md`, `README.md`, `src/README.md`: arquitetura, métricas e operação.
- `docs/metodologia_ml.md`, `docs/datasets.md`, `docs/fmeca.md`,
  `docs/confiabilidade_fisica.md`, `docs/mapa_de_resultados.md`,
  `docs/reproducibilidade.md` e `docs/glossario.md`: interpretação científica
  canônica.
- `docs/arquitetura.md`, `docs/aplicacao_web.md` e
  `docs/memoria_agentes.md`: Provider Gateway/Router neutros.
- `docs/exec-plans/active/ALIADO_PREFLIGHT_2026-08-27.md`: baseline histórico
  distinguido do estado pós-campanha.

As PRs anteriores desta campanha também criaram os dados-fonte, PNG e PDF da
ablação temporal e da sensibilidade de escore/limiar em
`resultados/comparacao/`, além do relatório de auditoria das PRs #99/#102.

## C. Testes

- `pytest -m "not pesado"`: 585 aprovados, 3 ignorados e 17 pesados não
  selecionados.
- `pytest tests/test_torch_smoke.py tests/test_modelos_autoencoder_canonicos.py`:
  6 aprovados com PyTorch real.
- `pytest tests/test_dados_gpvs_canonico.py`: 4 aprovados com os 16 CSVs locais.
- suíte focada do fechamento: 56 aprovados e 2 ignorados.
- `python scripts/avaliar_agente.py`: 15/15 casos aprovados.
- `python scripts/verificar_projeto.py`: aprovado, 16 ensaios, sem aviso ou erro.
- `python scripts/auditar_resultados.py`: aprovado, 9 manifestos e 37 artefatos.
- `ruff check --select F821,F822,F823 src tests scripts`: aprovado.
- `python -m compileall -q src tests scripts`: aprovado.

## D. Divergências corrigidas

- **C1 - Gemini-only:** corrigida. OpenAI e Gemini usam Gateway/Router comuns.
- **C2 - AUC-PR principal:** corrigida. Recall, F1 e Precision são principais.
- **C3 - p99 versus p99,9:** corrigida no contrato. Percentil e top-k são
  configuráveis; o ponto solicitado e o efetivamente realizável são distintos.
- **C4 - matrizes de confusão:** corrigida e preservada em valores absolutos e
  normalizados.
- **C5 - FMECA e monitoramento:** corrigida estruturalmente. O contrato anulável
  existe e impede `NPR_proj` sem mapping validado.
- **C6 - distribuições físicas:** corrigida estruturalmente. Os contratos e
  parâmetros mínimos existem; modelos sem dados permanecem bloqueados.

## E. Pendências metodológicas

- Definir estatisticamente `POD_mon` por componente e validar na literatura um
  mapeamento para a escala ordinal `D_mon` antes de calcular `NPR_proj`.
- Obter tempos individuais até falha, exposição e censura por ativo, ou
  parâmetros bibliográficos componentizados equivalentes, antes de publicar
  Weibull 2P, Normal, Lognormal ou histogramas de vida.
- Justificar `k=5` como configuração de trabalho no texto da dissertação e
  reconhecer que p99,9 vira p100 empírico com a calibração atual.
- A fronteira pré/pós-falha do GPVS continua nominalmente no ponto médio porque
  os CSVs não contêm canal instrumentado de disparo.

## F. Riscos

- A ablação não demonstrou superioridade inequívoca do AE-LSTM; apresentar o
  modelo temporal como vencedor excederia a evidência disponível.
- Recall macro próximo de 0,39 e heterogeneidade entre ensaios limitam a
  alegação de capacidade geral de detecção, apesar da Precision elevada.
- E3 é bancada e binária no catálogo GPVS; não valida desempenho de campo nem
  POD específica para Contator AC, IGBT e Fusível AC.
- As taxas derivadas de participações de chamados são cenários de sensibilidade,
  não medições componentizadas.
- OpenAI ou Gemini podem ficar indisponíveis ou gerar custo; o Router reduz,
  mas não elimina, dependência externa.

## G. Próxima ação recomendada

1. Escrever os resultados da dissertação declarando a comparação inconclusiva
   quanto à superioridade arquitetural e mostrando o trade-off de alarmes.
2. Buscar fonte ou base de vida componentizada com censura para decidir se
   Weibull/Lognormal devem realmente entrar na dissertação.
3. Definir com a orientadora se haverá uma metodologia validável de
   `POD_mon -> D_mon`; se não houver, manter apenas o NPR base.
