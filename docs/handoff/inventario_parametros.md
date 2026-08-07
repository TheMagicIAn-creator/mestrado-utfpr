# Inventário de parâmetros do pipeline CA — valor, origem, fundamentação e defesa

**Projeto:** Al IAdo PV — análise preditiva de falhas em componentes CA de inversores fotovoltaicos on-grid
**Repositório auditado:** `mestrado-utfpr`, commit `e5518db` (2026-08-06)
**Propósito:** dar a Rodolfo Torres domínio explícito sobre cada número que decide um resultado, com a referência que o sustenta e a frase de defesa perante a banca.

> **Como ler.** Cada parâmetro recebe uma **classe**:
> - 🟢 **Determinado** — decorre de física, norma ou definição matemática. Não é escolha.
> - 🔵 **Medido** — foi varrido/testado no projeto; há evidência interna registrada.
> - 🟡 **Convenção** — valor padrão da literatura de ML. Citável, mas não ótimo para este problema.
> - 🔴 **Sem base** — nem norma, nem medição, nem convenção clara. **Item a fechar.**
> - ⚫ **Decisão de engenharia** — não é técnico, é escolha de custo/risco. **Deve ser sua.**

> **Aviso de verificação.** As referências marcadas com ✔ foram confirmadas em consulta a fontes nesta sessão. As marcadas com ⚠ são bibliografia clássica que você deve confirmar (autor, ano, página) antes de citar na dissertação — não cite nada daqui sem abrir a fonte.

---

## 1. `features_ca.py` — extração de atributos

| Parâmetro | Valor | Classe | Origem real | Fundamentação disponível |
|---|---|---|---|---|
| `FS` | 10.000 Hz | 🟢 | Do dataset Paderborn | É propriedade da aquisição, não sua escolha. Nyquist = 5 kHz. |
| `JANELA` | 1024 amostras | 🔴 | Potência de 2 conveniente para FFT | **Ver §1.1 — achado novo** |
| `SOBREPOSICAO` | 512 (50%) | 🟡 | Convenção de STFT | Sobreposição de 50% com janela de Hann dá reconstrução de amplitude constante (COLA). Justificativa clássica de processamento de sinais. ⚠ |
| `HARMONICOS` | [3, 5, 7, 11, 13] | 🟢 | — | **Fundamentável — ver §1.2** |
| `F0` | 60 Hz | 🔴 | Rede brasileira | **Errado para o Paderborn — ver §1.3** |
| `faixa_hz` | 40,0 | 🔴 | — | **Defeito crítico — ver §1.3** |
| Tolerância harmônica | ±15 Hz fixo | 🔴 | — | Erro escala com a ordem; ver §1.4 |
| Pesos da evidência harmônica | 0,50 e 0,25 | 🔴 | Heurística | Sem base. Sensibilidade nunca medida. |

### §1.1 A janela de 1024 amostras — achado novo

1024 amostras a 10 kHz = **102,4 ms**, que o código comenta como "~6 ciclos a 60 Hz".

A IEC 61000-4-7 especifica, para medição de harmônicos, janela sincronizada de **200 ms** — 12 ciclos a 60 Hz, 10 ciclos a 50 Hz — justamente para obter **resolução espectral de 5 Hz**. ✔

Sua janela é **metade** da janela normativa, e a resolução resultante é:

```
10.000 / 1024 = 9,77 Hz por bin
```

Isso importa muito quando combinado com a tolerância de ±15 Hz da busca de harmônicos: você tem cerca de **1,5 bin** de tolerância. É apertado.

**Duplicar a janela para 2048 amostras resolve os dois problemas de uma vez:**

```
2048 / 10.000 = 204,8 ms  ≈ 12,3 ciclos a 60 Hz   (≈ janela IEC)
10.000 / 2048 = 4,88 Hz por bin                   (≈ resolução IEC de 5 Hz)
```

**Ressalva honesta:** o Paderborn é acionamento de motor de velocidade variável, não rede de 50/60 Hz — então a IEC 61000-4-7 **não se aplica por obrigação**. Mas ela se aplica como *critério de projeto defensável*: você adota a resolução espectral que a norma de medição de harmônicos exige, e isso é infinitamente mais forte que "1024 é potência de 2".

**Frase de defesa:** "A janela foi dimensionada para obter resolução espectral de ~5 Hz, alinhada ao requisito da IEC 61000-4-7 para medição de harmônicos, garantindo separação adequada entre componentes harmônicas e inter-harmônicas."

### §1.2 As ordens harmônicas [3, 5, 7, 11, 13] — totalmente fundamentável

Não é escolha arbitrária, e você pode defendê-la com teoria consolidada:

- **5, 7, 11, 13** são as harmônicas características de conversores trifásicos de seis pulsos, dadas por `h = 6k ± 1` (k = 1, 2, ...). É a assinatura espectral canônica de retificadores/inversores trifásicos.
- **3ª** é de sequência zero em sistema trifásico equilibrado; sua presença indica desequilíbrio ou perda de fase — exatamente a assinatura que você atribui ao Fusível AC.

Ou seja: a lista não é um subconjunto conveniente, é **a lista fisicamente correta** para o modo de falha que você quer detectar. Isso é um ponto forte da sua metodologia que está subaproveitado no texto.

**Referências de apoio:** IEC 61000-4-7 (definição formal de THD, grupos e subgrupos harmônicos; ordem padrão até a 40ª salvo especificação em contrário) ✔; IEEE 519 (critérios de distorção harmônica de tensão e corrente em sistemas elétricos) ✔.

**Ponta solta:** a IEC define THD somando até a ordem H (padrão 40). Você calcula sobre 5 ordens. Isso é uma **THD parcial**, não a THD normativa. Declare isso explicitamente na dissertação, ou a banca vai apontar. Nomeie a grandeza como "distorção harmônica parcial nas ordens características" em vez de THD, ou justifique o truncamento.

### §1.3 `F0 = 60` e `faixa_hz = 40` — o defeito crítico

```python
f_min = max(5.0, 60 - 40)   # = 20 Hz
f_max = 60 + 40             # = 100 Hz
```

A fundamental é buscada em [20, 100] Hz. A mediana de F0 no bloco de teste dá **100,19 Hz**, com o teto em 100 Hz — assinatura de estimador saturado, não de regime operacional.

**Não existe referência que fundamente isso, porque está errado.** O Paderborn é acionamento de velocidade variável; a fundamental acompanha a rotação e não tem por que ficar abaixo de 100 Hz.

**O teto principiado vem de Nyquist e da sua própria lista de harmônicos.** Para que a 13ª harmônica caiba abaixo de Nyquist:

```
13 × F0_max < 5.000 Hz   →   F0_max < 384,6 Hz
```

Faixa defensável: **[20, 384] Hz**.

**O limite inferior e a distribuição real devem ser MEDIDOS, não citados.** Tentei localizar a faixa de velocidade declarada no relatório do dataset (Stender, Wallscheid & Böcker, 2020, DOI 10.13140/RG.2.2.23335.37280) e não consegui acesso ao texto completo. Duas ações:

1. Baixar o relatório do dataset e ler a faixa de operação declarada — é a fonte primária e você deve citá-la de qualquer forma no Capítulo 3.
2. Rodar o estimador com faixa ampla sobre todas as janelas e plotar o histograma de F0. **Essa medição é o seu dado**, e é mais forte que qualquer citação.

**Frase de defesa (após correção):** "A faixa de busca da fundamental foi delimitada superiormente pelo critério de Nyquist aplicado à maior ordem harmônica de interesse (13ª), e verificada contra a distribuição empírica de F0 do conjunto de dados."

### §1.4 Tolerância de ±15 Hz

Fixa, independente da ordem. Mas o desvio do alvo é `ordem × erro_F0`. Com 2 Hz de erro em F0, a 13ª harmônica é buscada 26 Hz fora — além da tolerância.

**Correção principiada:** tolerância proporcional, `tol_h = ordem × tol_F0`, ou tolerância relativa em % de `h·F0`. A IEC 61000-4-7 resolve isso de outra forma — por **subgrupos harmônicos**, somando a energia dos bins adjacentes ao harmônico ✔. Adotar subgrupos é a solução normativa e citável.

---

## 2. `autoencoder.py` — arquitetura e treino

| Parâmetro | Valor | Classe | Fundamentação disponível |
|---|---|---|---|
| Arquitetura | 109→64→32→16 | 🔴 | Sem referência. Ver §2.1 |
| `LATENTE_DIM` | 16 | 🔴 | Ver §2.1 |
| `DROPOUT` | 0,2 | 🟡 | Srivastava et al. (2014) ⚠ — ver §2.2 |
| `LR` | 1e-3 | 🟡 | É o valor padrão do Adam (Kingma & Ba, 2015) ⚠ |
| `EPOCHS` | 150 | 🟡 | Teto; o early stopping decide de fato |
| `PACIENCIA` | 20 | 🟡 | Prechelt (1998), "Early Stopping — But When?" ⚠ |
| `BATCH_SIZE` | 32 | 🟡 | Convenção; sem impacto medido |
| `SEED` | 42 | 🟢 | Reprodutibilidade. Qualquer valor serve, desde que fixo e declarado |
| `SIGMA` | 3,0 | 🟡 | Regra 3σ; **só referência comparativa**, não operacional |
| `TRAIN/CALIB/TEST` | 0,60 / 0,20 / 0,20 | 🟡 | Convenção. Ver §2.3 |
| `THRESHOLD_METHOD` | "p99" | 🟢 | Ver §3 |

### §2.1 A arquitetura — o furo real

Este é o item que a sua própria auditoria (`docs/auditoria_pipeline_ml.md`) já classifica como **"Não fundamentada"**. Reduzir 109 features para 16 em três camadas com ReLU é padrão de mercado, não resultado de análise.

**O que existe de fundamentação genérica:**

- **Hinton & Salakhutdinov (2006)**, *Reducing the Dimensionality of Data with Neural Networks*, Science 313(5786):504-507, DOI 10.1126/science.1127647 ✔ (referência canônica do autoencoder como redutor de dimensionalidade não linear).
- **Sakurada & Yairi (2014)**, *Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction*, MLSDA'14, pp. 4-11, DOI 10.1145/2689746.2689747 ✔. **Esta é a referência mais importante para você:** é o trabalho que estabelece o autoencoder para detecção de anomalia por erro de reconstrução, mostrando que ele captura correlações não lineares entre variáveis que o PCA linear perde. Justifica a *escolha do método*, não os números.
- **Bergstra & Bengio (2012)**, *Random Search for Hyper-Parameter Optimization*, JMLR ⚠ — útil para justificar *como* você escolheria, se decidir varrer.

**O que nenhuma referência vai te dar:** que 16 é a dimensão latente certa para 109 features de sinal CA. Isso é específico do seu problema.

**Duas saídas honestas, escolha uma:**

**(a) Medir.** Varredura de `latente_dim ∈ {8, 16, 32}` no mesmo protocolo da varredura de `k` que você já fez em 02/08. É barato, você já tem o script como molde (`scripts/varrer_calibracao.py`), e transforma 🔴 em 🔵. **É o caminho que eu recomendo** — foi exatamente assim que o `k=5` deixou de ser "a esmo".

**(b) Declarar.** Assumir no texto que a arquitetura segue configuração convencional e que a contribuição do trabalho não está nela. Honesto, mas entrega um flanco.

### §2.2 Dropout 0,2

Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 15:1929-1958 ⚠. O trabalho original discute taxas típicas em torno de 0,5 para camadas ocultas e valores menores (da ordem de 0,2) para camadas de entrada.

Ou seja: **0,2 é citável como convenção**, mas note que a sua aplicação é no encoder e no decoder — a auditoria de parâmetros já registra isso. Verifique se a taxa e a posição de aplicação batem com a recomendação da fonte antes de citá-la.

### §2.3 Split 60/20/20 e a purga temporal

O split em si é convenção. **A purga entre blocos, não** — e essa é uma escolha metodologicamente forte que merece destaque no Capítulo 3. Em séries temporais com sobreposição de janelas (você usa 50%), janelas adjacentes compartilham amostras; sem purga, há vazamento entre treino e calibração. `PURGA_PADRAO = 2` em `split_temporal.py` cobre exatamente a sobreposição.

**Frase de defesa:** "O split é temporal, não aleatório, com purga entre blocos dimensionada para a sobreposição das janelas, eliminando vazamento por amostras compartilhadas."

---

## 3. `escore_anomalia.py` — o escore e o limiar

| Parâmetro | Valor | Classe | Fundamentação |
|---|---|---|---|
| `METODO_ESCORE` | "localizado" | 🔵 | Medido — top-k vence MSE para falha localizada |
| `K_LOCALIZADO` | 5 | 🔵 | **Varrido em 02/08 — ver §3.1** |
| `PERCENTIL_LIMIAR` | 99,0 (auto) | 🟢🔵 | Definição + varredura — ver §3.2 |
| `FP_ALVO` | 1,0 % | ⚫ | **Decisão sua — ver §3.3** |
| `SEED_BOOTSTRAP` | 42 | 🟢 | Reprodutibilidade |
| `n_boot` | 500 | 🟡 | Convenção de bootstrap |

### §3.1 O `k = 5` — já está fundamentado, você só não sabia

Varredura executada em 2026-08-02, `scripts/varrer_calibracao.py`, 44 janelas saudáveis do holdout:

| k | percentil | Contator AC | IGBT | Fusível AC |
|---|---|---|---|---|
| **5** | **99,0** | 100% | **86,4%** | **88,6%** |
| 5 | 99,5 | 100% | 86,4% | 79,5% |
| 5 | 99,9 | 100% | 86,4% | 70,5% |
| 10 | 99,0 | 100% | 79,5% | 18,2% |
| 15 | 99,0 | 100% | 68,2% | **9,1%** |

Com k=15 o fusível cai para 9,1% — o detector fica cego. Causa física: a perda parcial de fase mexe em pouquíssimas features, e agregar mais termos no top-k dilui o sinal.

**k=5 com p99 é a melhor das nove configurações testadas.** Isto é evidência interna de primeira qualidade e deve estar na dissertação como tabela.

### §3.2 O limiar por percentil 99

**Não é uma constante escolhida, é uma definição.** Por construção do percentil, exatamente 1% das janelas saudáveis de calibração fica acima do percentil 99. Limiar p99 e FP de 1% são a mesma decisão dita de duas formas.

**Por que não μ + 3σ:** assume normalidade. A distribuição do erro de reconstrução é assimétrica, e com poucas janelas o percentil empírico é o estimador robusto. O código mantém μ+3σ registrado apenas como referência comparativa.

**Alternativas testadas e rejeitadas** (documentado em `escore_anomalia.py`):
- Ajuste paramétrico lognormal — erro −37% quando a distribuição está certa, até 5× maior quando errada (gama, Weibull, bimodal). Risco inaceitável, já que a distribuição real não é conhecida nem verificável.
- EVT / Pareto generalizada na cauda — pior que o empírico neste regime; com 73 pontos sobram ~18 excedências acima do 75º percentil, poucas para ajuste estável.
- Bootstrap como correção — melhora a dispersão em ~2%, ou seja, nada. É esperado: bootstrap estima a distribuição amostral a partir de UMA amostra, não acrescenta informação.

**Conclusão registrada: o limite é o tamanho da amostra, não o estimador.** Isso é uma resposta forte e pronta para a banca.

### §3.3 `FP_ALVO = 1%` — a decisão que é sua

Este é o único parâmetro do pipeline que **não é técnico**. É a taxa de alarme falso que você aceita pagar: 1 inspeção desnecessária a cada 100 janelas saudáveis. Sai de custo de inspeção versus custo de falha não detectada — economia de manutenção, não estatística.

**Ninguém te perguntou, e deveria ter perguntado.** Duas ressalvas que você precisa carregar junto:

1. **O alvo não é imposto.** `AL_IADO_ESCORE_FP_ALVO` escolhe entre percentis candidatos (99,0 a 99,9) o menor cujo FP fique abaixo do alvo — e, se nenhum atinge, aceita o mais conservador assim mesmo. Hoje é o que ocorre: p99,9 escolhido, FPR de 10,2% no teste.
2. **O alvo não é mensurável nesta amostra.** Com 44 janelas, o menor FP não-nulo é 1/44 = 2,27%. Distinguir 1% de 2% é impossível aqui.

**Fundamentação para escolher um valor:** não há norma de FP para detecção de anomalia em inversores. O que existe é a analogia com **POD (probability of detection) do MIL-HDBK-1823A**, que você já usa na retroalimentação da FMECA, e a literatura de manutenção baseada em condição, onde o trade-off alarme falso × detecção é tratado como decisão econômica. Vale discutir isso com a Prof.ª Fernanda — é a pergunta certa a levar.

---

## 4. `injecao_falhas.py`, `validacao.py`, `rul_weibull.py`

| Parâmetro | Valor | Classe | Observação |
|---|---|---|---|
| `SEVERIDADES` | [0,05 … 1,0] (7 níveis) | 🟡 | Grade discreta; a SMD herda a granularidade dela |
| `ALVO_SMD` | 0,95 | ⚫ | Convenção de POD (a "a90/95" de END). Ancorável no MIL-HDBK-1823A ⚠ |
| `N_JANELAS_SMD` | 100 | 🟢 | Teto; limitado pelas janelas não sobrepostas do holdout |
| `SEVS_VALIDACAO` | [0,30; 0,50; 1,00] | 🟡 | Subconjunto da grade |
| `N_JANELAS_SAUDAVEL/FALHA` | 40 / 40 | 🟡 | Teste balanceado — declare, pois afeta precisão |
| `PREVALENCIA_RARA` | 0,05 | ⚫ | Cenário de prevalência realista; escolha sua |
| `N_TRAJ` | 100 | 🟢 | Teto amostral |
| `N_STEPS` | 120 | 🔴 | Resolução da trajetória de degradação; sem base |
| `MIN_EVENTOS_WEIBULL` | 10 | 🟡 | Mínimo para ajuste de dois parâmetros ⚠ |
| `MAX_CENSURA_RUL_PCT` | 50,0 | ⚫ | Guarda de qualidade; acima disso o ajuste não é confiável |
| `PERSISTENCIA_CRUZAMENTO` | 3 | 🔴 | Passos consecutivos acima do limiar para declarar TTF. Sem base — mas é a lógica de "debounce" de alarme, defensável se declarada |
| `TTF_UNIDADE` | passo sintético | 🟢 | **Honestidade importante:** o TTF não está em unidade física. `TEMPO_FISICO_CALIBRADO = False` está corretamente declarado no código |

**Ponto a destacar:** o `ALVO_SMD = 0,95` e a `PERSISTENCIA_CRUZAMENTO = 3` são os dois que mais mexem no resultado da Weibull e são os menos fundamentados desse bloco. O 0,95 tem analogia direta com a prática de POD em ensaios não destrutivos (a90/95), que é a mesma raiz que você já cita para POD_mon — use essa ponte.

---

## 5. Plano de fechamento — em ordem de prioridade

| # | Ação | Tipo | Efeito nos artefatos |
|---|---|---|---|
| 1 | Script de diagnóstico do F0 com faixa ampla (`scripts/`) | Medição | Nenhum (fica fora de `src/ml`) |
| 2 | Ler o relatório do dataset Paderborn e citar a faixa de operação | Bibliografia | Nenhum |
| 3 | Corrigir a faixa de busca de F0 para [20, 384] Hz | Correção | **Invalida tudo — reexecutar** |
| 4 | Avaliar janela 2048 (resolução 4,88 Hz ≈ IEC) | Correção | **Invalida tudo — agrupar com o item 3** |
| 5 | Varredura de `latente_dim ∈ {8,16,32}` | Medição | Reexecução dirigida |
| 6 | Tolerância harmônica proporcional à ordem, ou subgrupos IEC | Correção | Agrupar com 3 e 4 |
| 7 | Decidir e registrar o FP alvo com a orientadora | Decisão | Nenhum imediato |
| 8 | Renomear "THD" para distorção parcial, ou justificar o truncamento | Texto | Nenhum |

**Regra de agrupamento:** os itens 3, 4 e 6 tocam `features_ca.py` e todos invalidam os manifestos. Faça-os **na mesma janela de reexecução**, mas em **commits separados e medidos individualmente** — para saber qual causou o quê. É a mesma disciplina que o `docs/decisao_fpr_1pct.md` impõe à mudança de régua μ/σ.

---

## 6. Bibliografia consolidada

**Normas**
- IEC 61000-4-7:2002+A1:2009 — medição de harmônicos e inter-harmônicos; janela de 200 ms (12 ciclos a 60 Hz), resolução de 5 Hz, definição formal de THD, grupos e subgrupos harmônicos ✔
- IEEE 519 — critérios de distorção harmônica de tensão e corrente ✔
- MIL-HDBK-1823A — curvas POD (já usado na retroalimentação da FMECA) ⚠

**Método (autoencoder e detecção de anomalia)**
- Hinton, G. E. & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. *Science*, 313(5786), 504-507. DOI 10.1126/science.1127647 ✔
- Sakurada, M. & Yairi, T. (2014). Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction. *MLSDA'14*, 4-11. DOI 10.1145/2689746.2689747 ✔
- Chalapathy, R. & Chawla, S. (2019). Deep Learning for Anomaly Detection: A Survey. arXiv:1901.03407 ✔

**Hiperparâmetros**
- Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15, 1929-1958 ⚠
- Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR* ⚠
- Prechelt, L. (1998). Early Stopping — But When? ⚠
- Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *JMLR* ⚠

**Dataset e domínio**
- Stender, M., Wallscheid, O. & Böcker, J. (2020). Data Set Description: Three-Phase IGBT Two-Level Inverter for Electrical Drives. DOI 10.13140/RG.2.2.23335.37280 ✔ *(texto completo não acessado nesta consulta — abrir antes de citar a faixa de operação)*
- Ibrahim (2022) — AE-LSTM, já no corpus, base do escore por erro de reconstrução
- Torres, R. (2024). Aplicação da Metodologia Reliability Centred Maintenance a Sistemas Fotovoltaicos. TCC, UFPA — Tab. 4.8 (escala de detecção) e Tab. 3.3 (Cristaldi et al., 2017)

**Documentação interna citável**
- `docs/auditoria_pipeline_ml.md` §13 (escore localizado), §22 (limiar e teto amostral), varredura de `k` (2026-08-02)
- `docs/decisao_fpr_1pct.md` (decisão sobre FPR e achado de F0)
- `docs/auditoria_parametros.md` (auditoria completa de parâmetros)
