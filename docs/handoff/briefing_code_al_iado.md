# Briefing de implementação — Al IAdo PV
## Documento de handoff para o Claude Code

**Autor do pedido:** Rodolfo Torres (UTFPR) — mestrado em Engenharia Elétrica, defesa prevista para março de 2027
**Origem:** sessão de auditoria metodológica em 06/08/2026 (chat) + sessão anterior do agente Al IAdo PV sobre Weibull
**Repositório:** `mestrado-utfpr`, commit de referência `e5518db`
**Status do projeto:** nível de evidência E2 (sintético orientado por FMECA); E3 (campo) pendente

---

## 0. Para o Code: como este documento deve ser usado

Este é um **plano de trabalho com pontos de decisão explícitos**, não uma ordem de serviço fechada.

**Regras de interação, sem exceção:**

1. **Tudo aqui deve ser implementado**, salvo onde o próprio Rodolfo decidir em contrário após ser consultado.
2. **Onde houver `❓ DECISÃO`, PARE e PERGUNTE.** Não escolha por ele. Apresente as opções, o trade-off de cada uma e a sua recomendação — mas espere a resposta.
3. **Se você identificar qualquer parâmetro, limiar, arquitetura, número de épocas ou constante que tenha sido definido sem embasamento técnico — inclusive fora desta lista — traga para o Rodolfo com sugestão fundamentada.** Ele quer ser consultado sobre isso, explicitamente. Não silencie o problema resolvendo por conta própria.
4. **Sugestões de melhoria são bem-vindas e esperadas.** Se você tem uma proposta melhor que a daqui, apresente-a como pergunta, com a referência que a sustenta.
5. **Nunca invente fundamentação.** Se um número não tem base, diga que não tem base. É preferível declarar "escolha convencional, sem otimização" no texto da dissertação a fabricar uma justificativa.

**Regras técnicas do repositório que continuam valendo:**

- Nunca citar métrica de memória. Números saem de `resultados/manifestos/*.json`, dos artefatos vigentes em `resultados/` e do `docs/registro_execucoes.md`.
- Resultado de injeção/validação é **E2**, nunca prova de desempenho industrial.
- Uma mudança funcional por commit, medida isoladamente. Mudanças que invalidam manifesto devem ser **agrupadas na mesma janela de reexecução, mas em commits separados**, para saber qual causou qual efeito.
- Toda reexecução gera linha nova em `docs/registro_execucoes.md`.
- Limite de 1.000 linhas por módulo; dependência unidirecional `conhecimento → ml`.
- `.env` nunca vai ao GitHub.

---

## 1. Pendência imediata (fazer antes de tudo)

**`docs/registro_execucoes.md` está desatualizado.** A última linha é de 07/07 marcada como RESET, mas os artefatos vigentes em `resultados/` foram gerados por uma execução posterior que nunca foi registrada. Sem essa linha, não há como saber qual código gerou os números que estão publicados.

**Ação:** reconstruir a linha a partir dos manifestos (hash de código, data, duração) e registrá-la. Se não for possível reconstruir com confiança, marcar explicitamente como "execução de origem não registrada" em vez de inventar.

---

## 2. Bloco A — Extração de atributos (`features_ca.py`)

### A.1 🔴 CRÍTICO — Faixa de busca da frequência fundamental

**Defeito:** `estimar_f0()` busca a fundamental em `[f0_nominal − faixa_hz, f0_nominal + faixa_hz]` = **[20, 100] Hz**, com `F0 = 60` (rede brasileira) e `faixa_hz = 40.0`.

O dataset Paderborn é **acionamento de motor de velocidade variável**, não rede de 60 Hz. A própria docstring da função reconhece isso. A mediana de F0 no bloco de teste é **100,19 Hz** com o teto da busca em 100 Hz — assinatura de **estimador saturado**, não de regime operacional.

**Impacto:** F0 é a âncora de todas as features espectrais. Se o F0 verdadeiro é 130 Hz e o estimador devolve 100, o 5º harmônico é procurado 150 Hz fora do lugar, com tolerância de ±15 Hz. A THD sai errada junto. O erro cresce com a ordem do harmônico.

**Consequência interpretativa grave:** o `docs/decisao_fpr_1pct.md` atribui o FPR alto de teste a "regimes de F0 diferentes entre calibração e teste". **Se o estimador estava saturando, essa narrativa pode estar errada** — as medianas de 51,25 Hz e 100,19 Hz talvez não sejam dois regimes operacionais, mas um estimador funcionando num bloco e batendo no teto no outro. Isso muda a interpretação do resultado central da validação.

**Sequência obrigatória:**

**Passo 1 — diagnóstico sem tocar em `src/ml`.** Criar `scripts/diagnostico_f0.py`, seguindo o precedente de `scripts/diagnostico_limiar.py`. Roda o estimador com faixa ampla sobre todas as janelas e produz histograma + estatísticas descritivas de F0 por bloco (treino/calibração/teste). Como fica em `scripts/`, **nenhum manifesto vai a `stale`**.

**Passo 2 — ler a fonte primária.** Baixar e ler o relatório do dataset (Stender, Wallscheid & Böcker, 2020, DOI 10.13140/RG.2.2.23335.37280) e extrair a faixa de operação declarada. Essa citação precisa entrar no Capítulo 3 de qualquer forma.

**Passo 3 — corrigir.** O teto principiado vem de Nyquist aplicado à maior ordem harmônica de interesse:

```
13 × F0_max < 5.000 Hz  →  F0_max < 384,6 Hz
```

Faixa proposta: **[20, 384] Hz**. Refatorar a assinatura para aceitar `f_min` e `f_max` explícitos em vez de `f0_nominal ± faixa_hz`, porque a ancoragem em frequência nominal de rede é conceitualmente errada para este dataset.

❓ **DECISÃO D1 — perguntar ao Rodolfo:** confirmar a faixa [20, 384] Hz depois de ver o histograma do Passo 1, ou ajustar aos dados observados.

---

### A.2 🔴 Janela de análise — 1024 vs 2048 amostras

**Situação atual:** `JANELA = 1024` a `FS = 10.000 Hz` → **102,4 ms**, resolução espectral de **9,77 Hz por bin**.

**Fundamentação disponível:** a **IEC 61000-4-7** especifica, para medição de harmônicos, janela sincronizada de **200 ms** (12 ciclos a 60 Hz, 10 ciclos a 50 Hz), justamente para obter **resolução espectral de 5 Hz**, e define THD, grupos e subgrupos harmônicos sobre essa base.

Duplicar a janela alinha o projeto ao requisito normativo:

```
2048 / 10.000 = 204,8 ms  ≈ 12,3 ciclos a 60 Hz     (≈ janela IEC)
10.000 / 2048 = 4,88 Hz por bin                     (≈ resolução IEC de 5 Hz)
```

**Ganho adicional:** hoje a tolerância de ±15 Hz na busca de harmônicos equivale a apenas ~1,5 bin. Com 2048 passa a ~3 bins.

**Ressalva a declarar no texto:** o Paderborn é acionamento de velocidade variável, então a IEC 61000-4-7 **não se aplica por obrigação**. Ela se aplica como **critério de projeto defensável** — o que é muito mais forte que "1024 é potência de 2".

**Custo:** duplicar a janela reduz o número de janelas disponíveis, e o projeto já opera com amostra apertada (44 janelas saudáveis no holdout, 73 na calibração). Isso pode piorar a resolução do FP mensurável, que já é grosseira.

❓ **DECISÃO D2 — perguntar ao Rodolfo:** adotar 2048 (ganho normativo, menos janelas) ou manter 1024 (mais janelas, resolução espectral fora do padrão)? Apresentar a contagem de janelas resultante nos dois casos **antes** de ele decidir.

---

### A.3 🔴 Tolerância de busca de harmônicos

**Situação:** tolerância fixa de ±15 Hz, independente da ordem. Mas o desvio do alvo escala com a ordem: `desvio = ordem × erro_F0`. Com 2 Hz de erro em F0, a 13ª harmônica cai 26 Hz fora do alvo — além da tolerância.

**Duas correções possíveis:**

- **(a) Tolerância proporcional:** `tol_h = ordem × tol_F0`, ou tolerância relativa em % de `h·F0`.
- **(b) Subgrupos harmônicos da IEC 61000-4-7:** somar a energia dos bins adjacentes ao harmônico. É a solução normativa, é citável, e é robusta a flutuação de frequência por construção.

❓ **DECISÃO D3 — perguntar ao Rodolfo:** (a) é mais simples; (b) é normativa e mais defensável na banca, mas muda a definição da feature. Recomendação: (b), se o custo de reimplementação for aceitável.

---

### A.4 🟡 Nomenclatura da THD — correção de texto, sem impacto em artefato

A IEC 61000-4-7 define THD somando harmônicas até a ordem H (padrão 40 salvo especificação em contrário). O projeto calcula sobre **cinco ordens** — [3, 5, 7, 11, 13].

Isso é uma **distorção harmônica parcial**, não a THD normativa. Chamar de "THD" é atacável.

**Ação:** renomear a grandeza para algo como "distorção harmônica parcial nas ordens características" **ou** justificar explicitamente o truncamento no texto. Ajustar nome de feature, docstrings e rótulos de gráfico de forma consistente.

---

### A.5 🟢 O que NÃO precisa mudar — e precisa ser melhor explorado no texto

**As ordens harmônicas [3, 5, 7, 11, 13] são totalmente fundamentáveis** e hoje isso está subaproveitado:

- **5, 7, 11, 13** = harmônicas características de conversor trifásico de seis pulsos, dadas por `h = 6k ± 1`. É a assinatura espectral canônica de retificadores/inversores trifásicos — casa diretamente com a assinatura de falha do IGBT.
- **3ª** = sequência zero em sistema trifásico; sua presença indica desequilíbrio ou perda de fase — exatamente a assinatura atribuída ao Fusível AC.

Não é um subconjunto conveniente: é **a lista fisicamente correta** para os modos de falha da FMECA. Isso deve aparecer no Capítulo 3 como escolha justificada, não como detalhe de implementação.

**Ação para o Code:** garantir que a docstring de `features_ca.py` registre essa fundamentação, com referência a IEC 61000-4-7 e IEEE 519.

---

## 3. Bloco B — Autoencoder (`autoencoder.py`)

### B.1 🔴 Arquitetura sem fundamentação

`109 → 64 → 32 → 16`, ReLU, `DROPOUT = 0.2`, `EPOCHS = 150`, `PACIENCIA = 20`, `LR = 1e-3`, `BATCH_SIZE = 32`.

A auditoria interna (`docs/auditoria_pipeline_ml.md`) já classifica isso como **"Não fundamentada"** e marca como ponta solta.

**O que a literatura sustenta:**
- O *método* (autoencoder como detector de anomalia por erro de reconstrução) está bem ancorado em Hinton & Salakhutdinov (2006) e Sakurada & Yairi (2014).
- `LR = 1e-3` é o valor padrão do Adam (Kingma & Ba, 2015).
- `DROPOUT = 0.2` é citável via Srivastava et al. (2014), que discute taxas menores para camadas de entrada e da ordem de 0,5 para ocultas — **verificar se a taxa e a posição de aplicação no código batem com a recomendação da fonte antes de citar**.
- `PACIENCIA = 20` pode ser ancorada em Prechelt (1998).

**O que nenhuma referência vai fornecer:** que 16 é a dimensão latente certa para 109 features de sinal CA. Isso é específico do problema e só se resolve medindo.

**Ação proposta — varredura de dimensão latente.** Usar `scripts/varrer_calibracao.py` como molde e varrer `latente_dim ∈ {8, 16, 32}` no mesmo protocolo da varredura de `k` de 02/08 (detecção em severidade máxima, mesmo ponto de operação, holdout). Foi exatamente assim que o `k=5` deixou de ser "a esmo" — e o resultado dessa varredura **contradisse** a recomendação teórica anterior, o que mostra que o método funciona.

❓ **DECISÃO D4 — perguntar ao Rodolfo:** varrer a dimensão latente (transforma 🔴 em 🔵, custa uma bateria de treinos) ou declarar no texto que a arquitetura segue configuração convencional e não é a contribuição do trabalho? Recomendação: varrer.

❓ **DECISÃO D5 — perguntar ao Rodolfo:** estender a varredura a `dropout ∈ {0,1; 0,2; 0,3}` e `épocas/paciência`? Ele mencionou explicitamente épocas como item que quer decidir.

---

### B.2 🟢 Manter e destacar — a purga temporal

`PURGA_PADRAO = 2` em `split_temporal.py` é uma escolha metodologicamente forte que hoje está invisível no texto. Com janelas de 50% de sobreposição, janelas adjacentes compartilham amostras; sem purga há vazamento entre treino e calibração.

**Ação:** garantir que isso esteja documentado como decisão deliberada, não como detalhe.

---

## 4. Bloco C — Escore e limiar (`escore_anomalia.py`)

### C.1 🔵 O que já está fundamentado — apenas consolidar

Estes itens **não precisam de mudança**, precisam de visibilidade:

- **`K_LOCALIZADO = 5`** — varrido em 02/08 (`scripts/varrer_calibracao.py`, 44 janelas do holdout). `k ∈ {5,10,15} × percentil ∈ {99; 99,5; 99,9}`. Com k=15 o Fusível AC cai de 88,6% para 9,1% — o detector fica cego. **k=5, p99 é a melhor das nove configurações.** Deve virar tabela na dissertação.
- **Limiar p99** — não é constante escolhida, é definição: por construção do percentil, 1% das janelas saudáveis de calibração fica acima do p99. Limiar p99 e FP de 1% são a mesma decisão.
- **Alternativas testadas e rejeitadas** (documentadas no próprio módulo): ajuste paramétrico lognormal (erro até 5× maior quando a distribuição está errada), EVT/GPD na cauda (73 pontos dão ~18 excedências, poucas para ajuste estável), bootstrap como correção (melhora ~2%, ou seja, nada). Conclusão registrada: **o limite é o tamanho da amostra, não o estimador.**

**Ação:** consolidar isso em documento único de defesa, não deixar espalhado.

---

### C.2 ⚫ `FP_ALVO = 1%` — decisão que é do Rodolfo, não técnica

É o único parâmetro do pipeline que não é técnico: quantas inspeções desnecessárias se aceita por 100 janelas saudáveis. Sai de custo de inspeção × custo de falha não detectada. **Não há norma de FP para detecção de anomalia em inversores.**

Duas ressalvas que precisam acompanhar a decisão:
1. **O alvo não é imposto.** `AL_IADO_ESCORE_FP_ALVO` escolhe entre percentis candidatos (99,0–99,9) o menor cujo FP fique abaixo do alvo; se nenhum atinge, aceita o mais conservador assim mesmo. É o que ocorre hoje (p99,9, FPR de 10,2% no teste).
2. **O alvo não é mensurável nesta amostra.** Com 44 janelas, o menor FP não-nulo é 1/44 = 2,27%.

❓ **DECISÃO D6 — perguntar ao Rodolfo:** qual FP alvo adotar e como declará-lo. Sugestão: levar à Prof.ª Fernanda Correa como pergunta de economia de manutenção, ancorada na analogia com POD do MIL-HDBK-1823A que ele já usa na retroalimentação da FMECA.

---

## 5. Bloco D — Confiabilidade e RUL (`rul_weibull.py`)

> **Este bloco incorpora as propostas que o próprio agente Al IAdo PV já havia levantado na sessão anterior.** Elas são boas e devem ser implementadas — com as ressalvas abaixo, que são importantes.

### D.1 ⚠️ CORREÇÃO PRÉVIA OBRIGATÓRIA — unidade do TTF

Na sessão anterior o agente apresentou exemplos como `β = 2,15 [1,92; 2,41]` e `η = 420h [390h; 455h]`. **Esses números eram ilustrativos, não resultados do projeto — e a unidade está errada para este pipeline.**

O código declara explicitamente:

```python
TTF_UNIDADE = "passo_sintetico_de_degradacao"
TEMPO_FISICO_CALIBRADO = False
```

**O TTF não está em horas.** Está em passos sintéticos de degradação, e não há calibração para tempo físico. Qualquer η, MTTF, B10 ou RUL expresso em horas seria **falso**.

**Ação para o Code:** garantir que toda saída de Weibull carregue a unidade declarada, e que nenhum gráfico, tabela ou texto gerado use unidade temporal física enquanto `TEMPO_FISICO_CALIBRADO` for `False`. Adicionar teste automatizado que falhe se isso acontecer.

---

### D.2 Weibull de 3 parâmetros (t₀) — implementar com teste de aderência

**Fundamentação primária (do próprio corpus do projeto):** Lafraia (2001), *Manual de Confiabilidade, Mantenabilidade e Disponibilidade*, p. 7 — t₀ é o parâmetro de vida inicial da distribuição de Weibull e representa a distância da origem até o início da primeira falha.

Formulação em Lafraia:

```
F(t) = 1 − exp[ −((t − t₀)/η)^β ],   t ≥ t₀
R(t) = exp[ −((t − t₀)/η)^β ]
λ(t) = (β/η) · ((t − t₀)/η)^(β−1)
```

Interpretação dos casos: `t₀ = 0` reduz ao modelo 2P; `t₀ > 0` indica período livre de falhas; `t₀ < 0` indica degradação prévia ao início da operação.

**Ganho para o projeto:** dá interpretação física ao **tempo de incubação** do defeito no IGBT ou no Contator após a injeção da anomalia, e evita subestimar a confiabilidade nos instantes iniciais.

**Ressalva técnica séria:** ajustar um terceiro parâmetro com amostra pequena é instável. O projeto tem `MIN_EVENTOS_WEIBULL = 10` e `MAX_CENSURA_RUL_PCT = 50.0`. **Não adotar 3P por padrão.**

**Implementação correta:** ajustar 2P e 3P, e adotar 3P **apenas se** o teste de razão de verossimilhança indicar ganho estatisticamente significativo. Registrar o resultado do teste no artefato, não só o modelo vencedor.

❓ **DECISÃO D7 — perguntar ao Rodolfo:** nível de significância do teste de razão de verossimilhança (sugestão: α = 0,05) e comportamento quando o ganho é marginal.

---

### D.3 Intervalos de confiança por bootstrap

`rul_weibull.py` já tem `N_BOOTSTRAP = 250`. O agente sugeriu 1.000 reamostragens.

**Ganho:** apresentar β e η com intervalo de 95% em vez de estimativa pontual, o que é o padrão esperado em confiabilidade.

**Ressalva:** o `escore_anomalia.py` já documenta que bootstrap **não reduz** a variância entre amostras diferentes — ele estima a distribuição amostral a partir de uma amostra e não acrescenta informação que a amostra não tem. Serve para **medir incerteza**, não para melhorar a estimativa. O texto da dissertação precisa dizer isso, ou vira sobrevenda.

❓ **DECISÃO D8 — perguntar ao Rodolfo:** manter 250 ou subir para 1.000? (Custo computacional × estabilidade do IC.)

---

### D.4 B10/B1 em vez de MTTF como indicador de decisão

**Proposta do agente, e é boa:** o MTTF é a média da distribuição — momento em que boa parte da população já falhou, o que é inadequado para decisão de manutenção. Balizar por **B10** (confiabilidade de 90%) ou **B1** para componentes críticos como o IGBT.

```
B10 = η · [−ln(0,90)]^(1/β)
MTTF = η · Γ(1 + 1/β)
```

**Ação:** implementar B1 e B10 como saídas de primeira classe, mantendo MTTF apenas como referência comparativa — a mesma lógica que o projeto já aplica ao μ+3σ no limiar.

---

### D.5 Confiabilidade condicional e RUL dinâmico

**Proposta do agente:** apresentar `R(t + Δt | t) = R(t + Δt)/R(t)`, que responde à pergunta operacional real ("dado que operou até t sem falha, qual a probabilidade de sobreviver mais Δt?"), em vez de confiabilidade a partir de t = 0.

**Ação:** implementar. É de baixo custo e alto valor interpretativo.

---

### D.6 ❓ Weibull-PHM com η dependente do escore — DECISÃO, não implementação automática

O agente propôs tornar o parâmetro de escala função do escore do autoencoder:

```
η(s) = η₀ · exp(−α · s(t))
```

**Isto é atraente, e é também o item de maior risco desta lista.** Motivos:

1. Introduz um novo parâmetro livre (α) sem fundamentação, exatamente o problema que este documento inteiro está tentando resolver. Trocaríamos uma ponta solta por outra.
2. Cria **circularidade potencial**: o escore do autoencoder já define o TTF (é o passo em que o escore cruza o limiar). Fazer o mesmo escore modular também o η do Weibull mistura duas etapas que hoje estão separadas.
3. Aumenta o escopo do mestrado a sete meses da defesa.

❓ **DECISÃO D9 — perguntar ao Rodolfo:** implementar o PHM agora, deixar como trabalho futuro declarado na dissertação, ou implementar como ramo experimental fora do pipeline principal? **Recomendação: trabalho futuro**, com a formulação registrada no texto — dá crédito pela ideia sem assumir o risco metodológico.

---

### D.7 ❓ Parâmetros de RUL sem fundamentação

Levantados na auditoria, todos precisam de decisão do Rodolfo:

| Parâmetro | Valor | Problema |
|---|---|---|
| `N_STEPS` | 120 | Resolução da trajetória de degradação. Sem base. |
| `PERSISTENCIA_CRUZAMENTO` | 3 | Passos consecutivos acima do limiar para declarar TTF. Sem base — é lógica de debounce de alarme, defensável **se declarada como tal**. |
| `ALVO_SMD` | 0,95 | Convenção de POD (a90/95 de ensaios não destrutivos). Ancorável no MIL-HDBK-1823A — a mesma raiz já usada para POD_mon. |
| `SEVERIDADES` | 7 níveis | Grade discreta; a SMD herda a granularidade dela. |
| `PREVALENCIA_RARA` | 0,05 | Cenário de prevalência; escolha de engenharia. |

❓ **DECISÃO D10 — perguntar ao Rodolfo** sobre cada um. Para `ALVO_SMD` e `PERSISTENCIA_CRUZAMENTO`, que são os que mais mexem no resultado da Weibull, apresentar análise de sensibilidade antes de ele decidir.

---

## 6. Bloco E — Comparação com a literatura

O baseline atual (AE-LSTM de Ibrahim, 2022) foi **implementado pelo próprio pesquisador**, o que é atacável como espantalho na banca.

**Proposta:** adicionar baseline de terceiro, mantido e citável, **como linha adicional em `macro_comparar`, sem tocar no AE principal** — assim os hashes de proveniência do pipeline não se movem.

Candidatos verificados:
- **PyOD** (BSD-2-Clause, suporte a Python 3.13, paper JMLR 2019) — API `fit`/`decision_function`, aceita matriz de atributos já extraídos. Permite ignorar o limiar interno (`contamination`) e aplicar o percentil congelado do projeto.
- **USAD** (BSD-3, KDD 2020) — AE com treino adversarial, hiperparâmetros publicados. Substituto direto do Ibrahim caseiro.
- **DeepOD** (BSD, PyTorch) — variantes temporais LSTM/TCN/Transformer.

❓ **DECISÃO D11 — perguntar ao Rodolfo:** adicionar baseline externo como linha comparativa (baixo risco, alto ganho de defensabilidade) e/ou substituir o AE principal (alto risco — invalida limiar, SMD, validação, Weibull e retroalimentação de uma vez, e pode apagar a inversão da ordem de criticidade, que é o melhor resultado do trabalho). **Recomendação: só a linha comparativa.**

---

## 7. Ordem de execução proposta

| # | Ação | Invalida manifesto? |
|---|---|---|
| 1 | Registrar linha faltante em `registro_execucoes.md` | Não |
| 2 | `scripts/diagnostico_f0.py` + histograma | Não |
| 3 | Ler relatório do dataset Paderborn e citar faixa de operação | Não |
| 4 | Renomear THD → distorção parcial (texto, docstrings, rótulos) | Não |
| 5 | Consolidar documento de defesa de parâmetros | Não |
| 6 | **Decisões D1–D3 com o Rodolfo** | — |
| 7 | Corrigir faixa de F0 | **Sim** |
| 8 | Avaliar/adotar janela 2048 | **Sim** |
| 9 | Tolerância harmônica proporcional ou subgrupos IEC | **Sim** |
| 10 | Reexecução única cobrindo 7–9, commits separados | **Sim** |
| 11 | Varredura de dimensão latente | Reexecução dirigida |
| 12 | Weibull 3P com teste de razão de verossimilhança | Reexecução dirigida |
| 13 | B1/B10, confiabilidade condicional, IC bootstrap | Reexecução dirigida |
| 14 | Baseline externo em `macro_comparar` | Não (módulo separado) |

**Itens 7, 8 e 9 tocam `features_ca.py` e todos invalidam os manifestos.** Agrupar na mesma janela de reexecução, mas em commits separados e medidos individualmente — é a mesma disciplina que o `docs/decisao_fpr_1pct.md` impõe à mudança de régua μ/σ.

---

## 8. Bibliografia de apoio

> ✔ = confirmada em consulta nesta sessão · ⚠ = verificar autor, ano e página antes de citar na dissertação

**Corpus próprio do projeto (RAG)**
- Lafraia, J. R. B. (2001). *Manual de Confiabilidade, Mantenabilidade e Disponibilidade*. Rio de Janeiro: Qualitymark. — p. 7: definição de t₀ como parâmetro de vida inicial ✔ (recuperado pelo próprio agente Al IAdo PV)
- Torres, R. (2024). *Aplicação da Metodologia Reliability Centred Maintenance a Sistemas Fotovoltaicos*. TCC, UFPA — Tab. 4.8 (escala de detecção), Tab. 3.3 (Cristaldi et al., 2017)
- Ibrahim (2022) — AE-LSTM; base do escore por erro de reconstrução

**Normas**
- IEC 61000-4-7:2002+A1:2009 — janela de 200 ms, resolução de 5 Hz, definição de THD, grupos e subgrupos harmônicos ✔
- IEEE 519 — critérios de distorção harmônica ✔
- MIL-HDBK-1823A — curvas POD ⚠

**Método**
- Hinton, G. E. & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. *Science*, 313(5786), 504-507. DOI 10.1126/science.1127647 ✔
- Sakurada, M. & Yairi, T. (2014). Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction. *MLSDA'14*, 4-11. DOI 10.1145/2689746.2689747 ✔
- Chalapathy, R. & Chawla, S. (2019). Deep Learning for Anomaly Detection: A Survey. arXiv:1901.03407 ✔

**Hiperparâmetros**
- Srivastava, N. et al. (2014). Dropout. *JMLR*, 15, 1929-1958 ⚠
- Kingma, D. P. & Ba, J. (2015). Adam. *ICLR* ⚠
- Prechelt, L. (1998). Early Stopping — But When? ⚠
- Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *JMLR* ⚠

**Confiabilidade (complementares a Lafraia — verificar disponibilidade)**
- Abernethy, R. B. *The New Weibull Handbook* — advertências sobre ajuste 3P com amostra pequena ⚠
- Meeker, W. Q. & Escobar, L. A. (1998). *Statistical Methods for Reliability Data* ⚠
- Nelson, W. (1982). *Applied Life Data Analysis* ⚠

**Dataset e domínio**
- Stender, M., Wallscheid, O. & Böcker, J. (2020). Data Set Description: Three-Phase IGBT Two-Level Inverter for Electrical Drives. DOI 10.13140/RG.2.2.23335.37280 ✔ *(texto completo não acessado — abrir antes de citar a faixa de operação)*
- Harrou, F. et al. (2024). Automatic fault detection in grid-connected photovoltaic systems via variational autoencoder-based monitoring. *Energy Conversion and Management*, 314, 118665 ✔ — **sem código público**
- Bakdi, A., Guichi, A., Mekhilef, S. & Bounoua, W. (2020). GPVS-Faults. Mendeley Data, DOI 10.17632/n76t439f65.1 ✔ — protocolo E3 de bancada executado em `resultados/gpvs/`

**Bibliotecas**
- Zhao, Y., Nasrullah, Z. & Li, Z. (2019). PyOD. *JMLR*, 20(96), 1-7 ✔
- Audibert, J. et al. (2020). USAD. *KDD 2020* ✔

**Documentação interna citável**
- `docs/auditoria_pipeline_ml.md` §13, §22 + varredura de k (2026-08-02)
- `docs/decisao_fpr_1pct.md`
- `docs/auditoria_parametros.md`
- `docs/nomenclatura_deteccao.md`

---

## 9. Resumo dos pontos de decisão

| ID | Assunto | Recomendação |
|---|---|---|
| D1 | Faixa de busca de F0 | [20, 384] Hz, confirmar após histograma |
| D2 | Janela 1024 vs 2048 | 2048, se a contagem de janelas permitir |
| D3 | Tolerância harmônica | Subgrupos IEC 61000-4-7 |
| D4 | Varrer dimensão latente | Sim |
| D5 | Varrer dropout/épocas | A critério do Rodolfo |
| D6 | FP alvo | Levar à orientadora |
| D7 | Significância do teste 2P vs 3P | α = 0,05 |
| D8 | N_BOOTSTRAP 250 vs 1000 | Avaliar custo |
| D9 | Weibull-PHM η(s) | Trabalho futuro |
| D10 | N_STEPS, PERSISTENCIA, ALVO_SMD | Análise de sensibilidade primeiro |
| D11 | Baseline externo vs troca do AE | Só baseline |
| D12 | Features do lado CC no vetor E3 | Manter + publicar ablação só-CA |
| D13 | Escore localizado rebaixado a ablação | Auditar antes de aceitar |

**Nenhuma dessas decisões deve ser tomada pelo Code sem consulta.**

> **D12 e D13 foram abertos em 10/08/2026**, depois da migração para o
> GPVS-Faults. O contexto, a evidência e as opções estão em
> `docs/handoff/estado_pos_gpvs_2026-08-10.md`, que também registra o que cada
> agente mexeu desde as PRs #111–#119.
