# CODEX — Implementação das decisões metodológicas consolidadas do Projeto Mestrado

**Projeto:** ALIAdo / Mestrado UTFPR  
**Repositório:** `TheMagicIAn-creator/mestrado-utfpr`  
**Data da decisão:** 01/09/2026  
**Baseline remoto inspecionado:** `main` em `21e0a9747f738837c837175bff5eff34b78fe1dd` (merge do PR #173)

## 0. Objetivo e regra de execução

Este documento é **comando de implementação**, não pedido de nova proposta.

1. Inspecione o estado atual do `main` e da worktree antes de alterar arquivos.
2. Se a worktree estiver suja, divergente ou com conflitos:
   - não use `git reset --hard`;
   - não descarte alterações locais;
   - preserve o trabalho e use, se necessário, uma worktree/branch limpa a partir do `main` remoto atual.
3. Preserve:
   - Provider Gateway OpenAI + Gemini;
   - Router atual;
   - Evidence RAG vigente e recuperação da referência Kull do PR #173;
   - manifestos v2, proveniência e auditoria;
   - validação dos 16 arquivos GPVS.
4. Não reabra auditoria das PRs #99/#102, Sonar, Anthropic, falhas sintéticas no núcleo experimental ou `POD_mon -> D_mon -> NPR_proj`.
5. Implemente, teste, regenere resultados canônicos pertinentes e abra PR.
6. Se faltar dado que o usuário reservou para preenchimento manual, não invente valor: use `null`, `TBD_USUARIO` ou bloqueio explícito coerente com o schema.

---

# 1. Pergunta experimental e modelos

Pergunta principal:

> **Quantas falhas reais os modelos detectam sem produzir uma quantidade operacionalmente inadequada de alarmes falsos?**

Comparar:

- Autoencoder Denso;
- AE-LSTM.

Nenhum deve ser declarado vencedor por definição. Ambos permanecem na dissertação e são avaliados sob protocolo equivalente.

---

# 2. Dataset e condições experimentais

O dataset experimental permanece **GPVS-Faults**.

- `F0L/F0M`: condições saudáveis.
- `F1L-F7M`: 14 ensaios reais de falha/anomalia.

Documentar, preservando os rótulos nativos:

| GPVS | Interpretação |
|---|---|
| `F1` | falha completa de um dos IGBTs do inversor |
| `F2` | erro de 20% no sensor/sistema de realimentação |
| `F6` | alteração de -20% no ganho do controlador PI |
| `F7` | alteração de +20% na constante de tempo do controlador PI |

`F6/F7` devem ser descritos como **anomalias funcionais do sistema/circuito de controle**, não como falhas físicas de PCB.

## Consequência

Não criar falhas sintéticas no núcleo experimental para IGBT, sensor/realimentação ou sistema/circuito de controle. O GPVS já fornece condições nativas adequadas.

---

# 3. FMECA — novo recorte conceitual, sem inventar NPR

O recorte metodológico passa a ser:

1. **IGBT**;
2. **sensor/sistema de realimentação**;
3. **sistema/circuito de controle do inversor**.

Terminologia preferencial: **sistema/circuito de controle do inversor**.

PCB pode permanecer apenas como fundamento bibliográfico de criticidade quando suportado por fonte; não usar “PCB” como equivalente automático de `F6/F7`.

## 3.1. Valores reservados ao usuário

O pesquisador preencherá manualmente os índices FMECA e NPR dos novos componentes.

Portanto:

- não estimar;
- não herdar;
- não copiar;
- não transferir valores de Contator AC ou Fusível AC.

Para novas linhas, usar algo como:

```yaml
severity: null
occurrence: null
detectability: null
npr: null
status: awaiting_user_fmeca
```

O IGBT só pode preservar dados anteriores se o contrato atual demonstrar que são do mesmo recorte e fonte. Se houver dúvida de consistência, bloquear a tabela inteira até o preenchimento do usuário.

## 3.2. Retirar a antiga FMECA do estado canônico

Contator AC e Fusível AC não devem continuar como o trio canônico atual.

Não apagar evidência histórica rastreável sem necessidade; reclassificar como histórico/legado quando útil.

Atualizar no mínimo:

- `docs/fmeca.md`;
- `docs/metodologia_ml.md`;
- `docs/glossario.md`;
- `docs/mapa_de_resultados.md`;
- `docs/confiabilidade_fisica.md`, se acoplado;
- `README.md`;
- `src/README.md`;
- `CLAUDE.md`;
- testes de consistência;
- JSON/publicações que ainda afirmem o trio antigo como recorte vigente.

---

# 4. Revogar `POD_mon`, `D_mon`, `D_proj` e `NPR_proj`

Decisão definitiva:

> **essas variáveis não fazem mais parte da metodologia principal da dissertação.**

Remover do contrato científico canônico:

- `POD_mon`;
- `D_mon`;
- `D_proj`;
- `NPR_proj`;
- regra candidata `D_proj=min(D_campo,D_mon)`.

Não manter como “próxima etapa pendente”.

Se forem necessárias apenas para histórico, marcá-las como **revogadas/superseded** em documentos antigos.

Substituir testes que exigem esses campos `null` por testes que garantam que a publicação científica atual não depende deles. Se alguma API pública antiga precisar dos campos, usar compatibilidade explicitamente `deprecated`, sem influência na publicação canônica.

---

# 5. Comparação justa: Dense AE × AE-LSTM

Manter iguais:

- 24 features;
- arquivos GPVS;
- partições;
- pré-processamento compatível;
- scaler ajustado somente no treino saudável;
- orçamento de treino;
- early stopping;
- sementes;
- métricas;
- política de calibração;
- grade de sensibilidade;
- E3.

Diferença intencional:

- Dense AE: janela atual `W_t`;
- AE-LSTM: contexto temporal causal.

Não remover a memória do LSTM para “igualar” artificialmente os modelos.

---

# 6. Independência entre desenvolvimento e E3

Preservar e auditar:

```text
F0L/F0M
  -> treino
  -> validação
  -> calibração
  -> teste saudável
  -> congelamento
  -> F1L-F7M apenas avaliação E3
```

`F1-F7` não podem selecionar:

- arquitetura;
- seed;
- scaler;
- `k`;
- percentil;
- limiar;
- hiperparâmetros.

Nenhum resultado E3 retroalimenta calibração.

A grade de sensibilidade é **descritiva**, não otimização supervisionada nas falhas.

Preservar purga e construção de sequências LSTM dentro de cada split.

---

# 7. Grade final de sensibilidade top-k × percentil

Substituir a grade atual:

```python
k = {1, 3, 5, 8, 12, 24}
p = {95, 97.5, 99, 99.5, 99.9}
```

pela grade fechada:

```python
k = {5, 10, 20}
p = {99.0, 99.5, 99.9}
```

São 9 configurações por modelo.

## 7.1. Calibração

Para cada:

```text
modelo × seed × k × percentil
```

calcular o limiar somente na calibração saudável daquele modelo/seed/k.

Não reutilizar o mesmo valor numérico de threshold entre Dense e LSTM.

Registrar:

- `score_top_k`;
- percentil solicitado;
- valor do limiar;
- ordem estatística;
- percentil empírico efetivo;
- tamanho da calibração;
- resolução empírica;
- FPR no teste saudável;
- Recall macro;
- F1 macro;
- Precision macro;
- número de ensaios válidos;
- seed;
- modelo.

## 7.2. Ressalva p99.9

A calibração atual tem aproximadamente `n=210`.

Se `p99.9` selecionar `210/210`, registrar explicitamente:

- solicitado: p99.9;
- efetivo: p100 empírico;
- resolução: aproximadamente 0,476 ponto percentual.

Não ocultar isso.

## 7.3. Não escolher o “melhor k/p” pelas falhas

A grade responde:

> Como o compromisso entre detecção e falsos alarmes muda com `k` e percentil?

Não promover retrospectivamente a combinação que maximizar F1 em F1-F7.

`k=5/p99.9` pode permanecer apenas como **configuração de referência histórica/reprodutível**, não como ótimo universal.

---

# 8. Controle temporal do AE-LSTM

Usar contexto causal e contínuo.

Com `L=8`:

```text
[W_(t-7), W_(t-6), ..., W_t]
```

para decisão referente a `W_t`.

Dense AE avalia `W_t` no mesmo instante de referência.

## 8.1. Proibições

Não:

- prefixar artificialmente cada amostra anômala com sete janelas saudáveis;
- usar janelas futuras;
- misturar ensaios;
- atravessar fronteiras de split;
- reiniciar o histórico antes de cada amostra de falha para inflar contraste.

## 8.2. Diagnóstico temporal

Separar:

### A. Transição saudável -> falha
Para `L=8`, observar as primeiras `L-1 = 7` janelas após a fronteira nominal.

### B. Falha estabelecida
Avaliar região posterior, quando o contexto já é majoritariamente ou integralmente anômalo.

Objetivo:

> verificar se eventual vantagem do AE-LSTM permanece depois que o degrau inicial saudável -> falha deixa de dominar a sequência.

Essa análise não seleciona hiperparâmetro, limiar ou vencedor.

Se o diagnóstico atual de “reinício pós-fronteira” continuar correto, pode permanecer auxiliar, mas a narrativa canônica deve enfatizar **sequência causal real + transição + falha sustentada**.

---

# 9. Métricas

Métricas principais:

1. Recall;
2. F1-score;
3. Precision.

Preservar:

- matrizes de confusão absolutas;
- matrizes normalizadas;
- falsos positivos em condição saudável.

Complementares:

- ROC-AUC;
- PR-AUC.

Não adicionar métricas/gráficos sem função metodológica.

Não maximizar Recall isoladamente; interpretar sempre detecção versus falsos alarmes.

---

# 10. Confiabilidade física

Separar rigorosamente:

```text
detecção de anomalia != confiabilidade física
```

Erro de reconstrução não fornece automaticamente taxa física de falha, tempo até falha, RUL ou parâmetros Weibull.

## 10.1. Exponencial

Quando houver apenas `lambda` bibliográfico:

```text
R(t) = exp(-lambda*t)
F(t) = 1 - R(t)
f(t) = lambda*exp(-lambda*t)
h(t) = lambda
MTTF = 1/lambda
```

Explicitar hipótese de hazard constante.

## 10.2. Weibull 2P

Não inferir `beta` e `eta` de uma única taxa agregada.

Só publicar Weibull 2P com:

- tempos individuais até falha/censura; ou
- `beta/eta` bibliográficos rastreáveis; ou
- dados equivalentes que permitam estimação identificável.

### Situação atual

Há indicação de base promissora para **IGBT**.

O CODEX deve:

1. procurar no corpus/referências já presentes;
2. confirmar se a fonte fornece `beta/eta` ou dados suficientes;
3. se confirmar, implementar o cenário Weibull do IGBT com fonte, página/tabela, unidade, parâmetros e hipótese;
4. se não confirmar documentalmente, não fabricar e manter bloqueado.

Para sensor/realimentação e sistema/circuito de controle, não criar Weibull/taxa nova sem evidência bibliográfica confirmada.

Normal/Lognormal permanecem bloqueadas sem base.

---

# 11. Nova campanha experimental canônica

Após as alterações, executar nova campanha preservando:

- seed de referência `42`;
- cinco sementes de estabilidade já definidas;
- 24 features;
- Dense AE e AE-LSTM;
- partições saudáveis vigentes;
- E3 nos 14 ensaios;
- IC95% com unidade inferencial por ensaio;
- diferenças pareadas;
- matrizes de confusão;
- análise temporal;
- nova grade `3 × 3`.

Preservar/regenerar em `resultados/comparacao/`:

- `e3_metricas_macro.csv`;
- `e3_metricas_por_ensaio.csv`;
- `e3_estabilidade_sementes.csv`;
- `e3_diferencas_pareadas.csv`;
- `e3_matrizes_confusao.csv`;
- PNG/PDF correspondentes;
- `e3_ablacao_temporal.csv`;
- `e3_ablacao_temporal_por_ensaio.csv`;
- PNG/PDF da ablação;
- `e3_sensibilidade_escore_limiar.csv`;
- PNG/PDF da sensibilidade;
- JSON metodológico;
- manifesto v2.

---

# 12. Documentação canônica

Atualizar:

- `docs/metodologia_ml.md`;
- `docs/datasets.md` se necessário;
- `docs/fmeca.md`;
- `docs/confiabilidade_fisica.md`;
- `docs/mapa_de_resultados.md`;
- `docs/reproducibilidade.md`;
- `docs/glossario.md`;
- `README.md`;
- `src/README.md`;
- `CLAUDE.md`;
- testes de consistência.

Devem desaparecer do estado vigente:

- “Contator AC + IGBT + Fusível AC” como trio canônico;
- expectativa futura de `POD_mon/D_mon/D_proj/NPR_proj`;
- afirmação de que `k=5/p99.9` é ótimo;
- interpretação de `F6/F7` como falha física de PCB;
- necessidade de falha sintética no núcleo atual.

Devem aparecer:

- IGBT + sensor/realimentação + sistema/circuito de controle;
- F1/F2/F6/F7 como correspondências nativas relevantes;
- ausência de falhas sintéticas no núcleo;
- Dense e AE-LSTM sem vencedor prévio;
- grade `k={5,10,20}` × `p={99,99.5,99.9}`;
- percentil solicitado versus efetivo;
- LSTM causal e análise transição/sustentada;
- E3 sem retroalimentação;
- FMECA sem NPR projetado pelo detector.

---

# 13. Código a auditar prioritariamente

```text
src/ml/comparacao_autoencoders.py
src/ml/sensibilidade_escore.py
src/ml/avaliacao_comparativa.py
src/ml/modelos_autoencoder.py
src/ml/treino_comparacao.py
src/ml/dados_gpvs.py
src/ml/publicacao_comparacao.py
src/ml/graficos_comparacao.py
src/ml/confiabilidade_componentes.py
src/ml/publicacao_confiabilidade.py
src/ml/graficos_confiabilidade.py
```

Pesquisar globalmente por:

```text
POD_mon
D_mon
D_proj
NPR_proj
Contator AC
Fusível AC
contator_ac
fusivel_ac
SENSITIVITY_TOP_K
SENSITIVITY_PERCENTILES
99.9
contexto
trajetoria
transition
sustained
PCB
```

Corrigir o contrato de origem, não apenas substituir texto em resultados gerados.

---

# 14. Testes obrigatórios

## 14.1. Grade

Garantir:

```python
SENSITIVITY_TOP_K == (5, 10, 20)
SENSITIVITY_PERCENTILES == (99.0, 99.5, 99.9)
```

9 combinações por modelo/seed.

## 14.2. Calibração

Testar que:

- limiar vem apenas da calibração saudável;
- F1-F7 não selecionam configuração;
- order statistic, percentil efetivo e resolução são persistidos.

## 14.3. Temporalidade

Testar que:

- sequência LSTM é causal;
- último passo é `W_t`;
- não há futuro;
- não cruza split;
- não cruza experimento;
- falha sustentada não recebe prefixo saudável artificial a cada amostra.

## 14.4. FMECA

Testar que:

- novos componentes podem ficar com S/O/D/NPR `null`;
- publicação não inventa NPR;
- Contator/Fusível não são o trio canônico atual;
- `POD_mon/D_mon/D_proj/NPR_proj` não participam da publicação científica vigente.

## 14.5. Reprodutibilidade

Preservar:

- 16 arquivos GPVS;
- seeds;
- hashes;
- manifestos v2;
- auditoria de saídas.

---

# 15. Execução e validação

Executar:

```powershell
python -m src.ml.comparacao_autoencoders
python -m src.ml.publicacao_confiabilidade
python scripts/auditar_resultados.py
python scripts/verificar_projeto.py
```

Depois:

```powershell
python -m pytest -p no:cacheprovider -q -W ignore -m "not pesado"
python -m pytest -p no:cacheprovider -q -W ignore tests/test_torch_smoke.py tests/test_modelos_autoencoder_canonicos.py
python -m pytest -p no:cacheprovider -q -W ignore tests/test_dados_gpvs_canonico.py
python -m ruff check --select F821,F822,F823 src tests scripts
python -m compileall -q src tests scripts
```

Executar também testes focados das novas mudanças.

Não concluir se:

- manifesto ficar stale;
- auditoria falhar;
- `verificar_projeto.py` falhar;
- docs e código divergirem;
- resultados versionados não corresponderem ao código atual.

---

# 16. Política de resultados

Se a nova campanha continuar inconclusiva sobre superioridade do AE-LSTM, manter conclusão inconclusiva.

Não adaptar metodologia para produzir vencedor.

A grade `3 × 3` é análise de sensibilidade, não seleção de hiperparâmetro usando falhas.

---

# 17. Limite de escopo do ALIAdo

Não criar nova evolução RAG, novo provedor, novo Router ou nova camada agêntica nesta tarefa, salvo correção estritamente necessária.

Prioridade:

```text
metodologia
-> código científico
-> resultados
-> documentação
-> testes
```

---

# 18. Git / PR

Preferir branch limpa, por exemplo:

```text
codex/implementar-decisoes-metodologicas-2026-09-01
```

Antes:

```powershell
git status
git fetch origin
git log --oneline --decorate -n 10 origin/main
```

Não force-push em branch do usuário sem necessidade.

Ao terminar:

1. commits coerentes;
2. PR contra `main`;
3. descrição com decisões, arquivos, resultados, testes, limitações e itens deixados ao usuário.

---

# 19. Relatório final obrigatório

Criar:

```text
docs/exec-plans/ALIADO_IMPLEMENTACAO_DECISOES_METODOLOGICAS_2026-09-01.md
```

ou nome equivalente não conflitante.

Incluir:

## A. Baseline
- SHA inicial;
- branch;
- estado da worktree;
- conflitos e resolução.

## B. Checklist das decisões
- retirada de POD/D/NPR projetado;
- novo recorte FMECA;
- NPRs novos deixados ao usuário;
- ausência de falhas sintéticas;
- protocolo equivalente Dense/LSTM;
- `k={5,10,20}`;
- `p={99,99.5,99.9}`;
- p solicitado versus efetivo;
- LSTM causal;
- transição versus falha sustentada;
- E3 sem leakage;
- confiabilidade separada;
- Weibull somente com evidência;
- docs/resultados sincronizados.

## C. Resultados
- artefatos regenerados;
- métricas sem esconder resultados negativos;
- ablação temporal;
- grade `3 × 3`;
- FPR saudável;
- IC95%.

## D. FMECA pendente do usuário
Listar exatamente os campos faltantes para:
- sensor/sistema de realimentação;
- sistema/circuito de controle;
- IGBT, apenas se bloqueado por consistência.

Não sugerir números.

## E. Confiabilidade
Por componente, registrar:
- `lambda`;
- MTTF derivável;
- `beta`;
- `eta`;
- dados de vida/censura;
- status de publicação.

## F. Testes
Registrar comandos/resultados.

## G. Git
- SHA final;
- branch;
- PR;
- arquivos principais.

---

# 20. Definition of Done

- [ ] baseline atual sem regressão;
- [ ] trabalho local preservado;
- [ ] novo recorte FMECA refletido;
- [ ] S/O/D/NPR novos não inventados;
- [ ] `POD_mon`, `D_mon`, `D_proj`, `NPR_proj` retirados da metodologia principal;
- [ ] sem falha sintética no núcleo experimental atual;
- [ ] Dense e AE-LSTM mantidos;
- [ ] dados/features/splits/seeds/métricas equivalentes;
- [ ] sensibilidade exatamente `k={5,10,20}` e `p={99,99.5,99.9}`;
- [ ] limiar calibrado exclusivamente no saudável;
- [ ] p99.9 reporta resolução empírica real;
- [ ] AE-LSTM causal e contínuo;
- [ ] transição e falha sustentada reportadas;
- [ ] F1-F7 não selecionam hiperparâmetros;
- [ ] Recall/F1/Precision principais;
- [ ] matrizes de confusão preservadas;
- [ ] confiabilidade física não inferida do erro do AE;
- [ ] exponencial mantida quando só há `lambda`;
- [ ] Weibull só publicada com evidência suficiente;
- [ ] resultados canônicos regenerados;
- [ ] manifestos v2 atualizados;
- [ ] auditoria passou;
- [ ] verificação do projeto passou;
- [ ] testes passaram;
- [ ] documentação sincronizada;
- [ ] relatório final criado;
- [ ] PR pronto para revisão.

---

# 21. Restrições finais

**Não fazer:**

- inventar valores FMECA;
- renomear Recall como POD;
- converter Recall em detectabilidade ordinal;
- recalcular NPR com ML;
- criar falha sintética de IGBT/sensor/controle;
- usar E3 para escolher `k`/percentil;
- forçar Weibull;
- declarar AE-LSTM superior sem evidência;
- adicionar métricas/gráficos sem função;
- expandir RAG/Router;
- apagar trabalho local para resolver conflito.

**Fazer:**

- implementar;
- preservar rastreabilidade;
- documentar limitações;
- manter resultados negativos;
- deixar valores FMECA ao usuário;
- entregar código + resultados + testes + documentação coerentes.

---

## Comando final ao CODEX

> **Implemente agora este plano no repositório atual. Não produza apenas análise ou nova proposta. Faça a auditoria inicial da worktree, preserve qualquer trabalho local, aplique as mudanças científicas e documentais, regenere a campanha experimental, execute testes e auditorias, crie o relatório final e abra um PR contra `main`. Quando encontrar campos FMECA/NPR reservados ao pesquisador, deixe-os explicitamente pendentes e continue o restante da implementação sem inventar valores.**
