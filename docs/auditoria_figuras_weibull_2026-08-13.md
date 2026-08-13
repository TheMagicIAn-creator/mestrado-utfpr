# Auditoria das figuras Weibull e dos artefatos do PR #128

- Data: 13/08/2026
- Base auditada: `main` após o merge do PR #128 (`d3df52c`)
- Dataset canônico: **GPVS-Faults**, DOI `10.17632/n76t439f65.1`

## Conclusão

Os valores publicados de detecção, SMD, validação e FMECA estão consistentes
entre JSON, CSV, relatórios e figuras. Não foi encontrada troca de dataset,
mistura com PMSM/Paderborn, redução de eventos nem erro nas matrizes, ROC ou PR.

O defeito confirmado era de **composição e nomenclatura visual**: a antiga
`weibull_distribuicao.png` reunia histograma/PDF, ECDF/CDF e papel de
probabilidade; `weibull_confiabilidade.png` reunia sobrevivência e intensidade,
com marcas de suporte no eixo da intensidade. Essas marcas eram níveis de
`a_det`, não observações de taxa, e podiam ser interpretadas como pontos de uma
taxa de falha física.

As funções foram separadas sem recalcular o autoencoder ou as trajetórias:

| Arquivo | Pergunta respondida |
|---|---|
| `weibull_distribuicao.png` | Os pontos são aproximadamente lineares no papel Weibull? |
| `weibull_funcoes_distribuicao.png` | Como PDF/histograma e CDF/ECDF se comparam? |
| `weibull_confiabilidade.png` | Qual a probabilidade empírica de ainda não detectar até `a`? |
| `weibull_intensidade_deteccao.png` | Qual a intensidade paramétrica de primeiro cruzamento por magnitude? |

## Interpretação correta

O papel de probabilidade usa
`x = ln(a_det)` e `y = ln[-ln(1-F_D)]`. Sob uma Weibull 2P adequada, os pontos
ficam aproximadamente alinhados com uma reta de inclinação beta. Portanto, os
pontos dispersos mais a reta são o diagnóstico visual que estava sendo
procurado; não são a PDF da Weibull.

A densidade `f_D(a)`, a acumulada `F_D(a)`, a sobrevivência `S_D(a)` e a
intensidade `h_D(a)` são **funções**, desenhadas como curvas. `h_D(a)` não é uma
"distribuição da taxa de falha" e não é taxa de falha do componente: o eixo é
magnitude sintética, não tempo, ciclos ou idade.

## Resultado paramétrico vigente

| Componente | Eventos | Níveis | beta | eta | R2pp | p bootstrap | Decisão 2P global |
|---|---:|---:|---:|---:|---:|---:|---|
| Contator AC | 277/277 | 140 | 3,868 | 0,397 | 0,933 | 0,004 | rejeitada; exploratória |
| IGBT | 277/277 | 126 | 3,379 | 0,312 | 0,929 | 0,004 | rejeitada; exploratória |
| Fusível AC | 277/277 | 143 | 4,184 | 0,586 | 0,886 | 0,004 | rejeitada; exploratória |

Os 277 eventos continuam presentes em cada componente. O número menor de
níveis decorre de empates na grade de 501 pontos (`delta_a=0,002`), não de
remoção de amostras. A curvatura no papel e o bootstrap quantizado rejeitam a
Weibull 2P global; por isso as retas e curvas paramétricas permanecem
tracejadas e explicitamente exploratórias.

## Cobertura dos 45 arquivos

Todos os arquivos alterados pelo commit `d3df52c` foram rastreados até seus
geradores e cruzados com os artefatos publicados.

| Grupo | Quantidade | Verificação | Resultado |
|---|---:|---|---|
| Auditoria em `resultados/auditoria` | 3 | inventário, catálogo, hashes e integridade visual | regenerados; sem divergência |
| Autoencoder/E2/Weibull | 27 | JSON/CSV, relatórios, matrizes e inspeção visual | números coerentes; composição Weibull corrigida |
| Relatórios GPVS E3 | 2 | 14 ensaios e métricas macro | coerentes com o manifesto E3 |
| Manifestos v2 | 4 | entradas, dependências e hashes de saída | nenhum hash divergente |
| Scripts de auditoria | 2 | cobertura dos novos PNGs e eixos | atualizados |
| Código científico | 6 | fórmulas, geradores, parâmetros e proveniência | nomenclatura e figuras corrigidas |
| Teste de limiar/validação | 1 | regressão do ponto operacional | preservado |

### Inventário explícito

`resultados/auditoria/`:

- `catalogo_figuras.csv`
- `inventario_artefatos.csv`
- `relatorio_auditoria_artefatos.md`

`resultados/autoencoder/`:

- `diagnostico_escore.json`, `diagnostico_escore.png`
- `injecao_falhas_comparacao.png`, `injecao_falhas_report.json`
- `injecao_falhas_resultados.png`, `injecao_smd_tabela.csv`
- `relatorio_confiabilidade.json`, `relatorio_confiabilidade.md`
- `retroalimentacao_fmeca.json`
- `validacao_matriz.png`, `validacao_matrizes_severidades.png`
- `validacao_metricas.png`, `validacao_pr.png`, `validacao_report.json`
- `validacao_roc.png`, `validacao_tabela.csv`, `validacao_tabela.md`
- `weibull_confiabilidade.png`, `weibull_distribuicao.png`
- `weibull_modos_operacao.png`, `weibull_results.json`, `weibull_rul.png`
- `weibull_sensibilidade_grade.csv`, `weibull_sensibilidade_grade.png`
- `weibull_tabela.csv`, `weibull_trajetorias_grade.csv`, `weibull_ttf.png`

Demais arquivos do commit:

- `resultados/gpvs/relatorio_validacao_gpvs.md`
- `resultados/gpvs/validacao_gpvs_e3.json`
- `resultados/manifestos/injecao_falhas.json`
- `resultados/manifestos/rul_weibull.json`
- `resultados/manifestos/validacao.json`
- `resultados/manifestos/validacao_gpvs_e3.json`
- `scripts/auditar_artefatos_resultados.py`
- `scripts/verificar_resultados_fmeca.py`
- `src/ml/diagnostico_escore.py`
- `src/ml/graficos_rul.py`
- `src/ml/injecao_falhas.py`
- `src/ml/pipeline.py`
- `src/ml/validacao.py`
- `src/ml/validacao_gpvs_principal.py`
- `tests/test_validacao_limiar.py`

## Validação executada

- 140 testes focados de confiabilidade, eixo `a_det` e auditoria: aprovados.
- Suíte completa não pesada: 945 aprovados, 3 ignorados e 16 pesados
  deliberadamente não executados.
- Auditoria FMECA/GPVS: 21 cenários E2, 14 ensaios E3 e 23 PNGs: aprovada.
- Inventário acadêmico: 73 artefatos, 27 figuras, 0 hash divergente,
  0 JSON/CSV inválido e 0 problema automático de integridade visual.
- Inspeção visual manual dos 17 PNGs do autoencoder e das quatro figuras
  Weibull redesenhadas: sem sobreposição, truncamento ou painel vazio.

Os novos PNGs foram regenerados a partir de `weibull_results.json` e
`weibull_trajetorias_grade.csv`. Nenhum dado bruto, modelo, estado local do
Obsidian ou arquivo de `.claude/` integra a alteração.
