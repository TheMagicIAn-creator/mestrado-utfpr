# Auditoria total de `src/` — parâmetros, recomputação, escopo e testes

**Pedido:** reavaliação pormenorizada de **todos** os parâmetros do projeto, com
correção. **Método:** 8 frentes paralelas de auditoria automatizada, cada
achado exigindo evidência em `arquivo:linha` ou número medido de artefato.

**Estado:** 9 frentes concluídas (81 achados, 51 graves); a revisão específica
das refatorações #99 e #102 foi encerrada em 2026-08-09, conforme §12.

Relatórios integrais das frentes do agente e dos PRs: `docs/auditoria_agente_e_prs.md`.

| Frente | Estado | Achados | Graves |
|---|---|---:|---:|
| Nomenclatura de "severidade" | ✅ | 14 | 10 |
| Recomputação / artefatos que não mudam | ✅ | 13 | 8 |
| Weibull e unidade de tempo | ✅ | 12 | 7 |
| Corrente contínua no escopo CA | ✅ | 10 | 8 |
| Treino (épocas, LR, dropout, latente) | ✅ | 4 | 1 |
| Necessidade dos arquivos de teste | ✅ | 3 | 0 |
| Parâmetros do agente (`core`, `conhecimento`) | ✅ | 16 | 10 |
| Premissas metodológicas nos PRs #89–#103 | ✅ | 5 | 4 |
| Refatorações #99 e #102 (semântica silenciosa) | ✅ | 4 | 3 |

---

## §1. Severidade — o mesmo erro do `D`, cometido quatro vezes

O pesquisador questionou por que a severidade da injeção vai de 0,05 a 1,0 se a
severidade da FMECA é um índice. **Ele está certo, e o problema é maior do que
supôs: há QUATRO sentidos de "severidade" no repositório.**

| # | Onde | O que é | Escala |
|---|---|---|---|
| i | `docs/fmeca.md` | Índice **S** da FMECA, julgado | ordinal **1–5** |
| ii | `injecao_falhas.py:105` | Intensidade da injeção **no sinal** | contínua 0,05–1,0 |
| iii | `protocolos_artigos.py:42` | Intensidade da injeção **em features** | múltiplos de σ (0,8–4,0) |
| iv | `rul_weibull.py:231` | Variável de **estado** que avança no "tempo" | contínua 0→1 |

### Por que é grave, e não preciosismo

- **O glossário canônico define a palavra duas vezes**, com sentidos
  incompatíveis (`docs/glossario.md:18` e `:70`) — no arquivo que declara ser o
  árbitro de conflitos entre documentos.
- **O artefato de defesa mistura os dois** a três linhas de distância:
  `retroalimentacao_fmeca.md` diz "Severidade de referência: **1.0**" na linha 3
  e traz a coluna `S` = **5** na linha 7.
- **As duas grandezas só se distinguem pela CAIXA da letra** (`S` = 5 contra
  `s` = 1,0). Em apresentação oral são a mesma palavra; em tabela LaTeX ficam a
  milímetros. É exatamente o argumento que o projeto já usou para rejeitar a
  letra `U`.
- (iii) contradiz (ii) no rótulo qualitativo: `protocolos_artigos.py:42` chama
  1,0 de "moderada"; `injecao_falhas.py:279` chama 1,00 de "severa".
- `docs/glossario.md:18` atribui escala **1–10** ao `S`, contra **1–5** na fonte
  única. Erro factual direto.

### O que o fator 0–1 realmente é

Não mede severidade de consequência. São **três normalizações heterogêneas**:

| Falha | O que o fator escala | Em 0,3 |
|---|---|---|
| Contator AC | razão σ_ruído/σ_sinal × 0,3 | SNR ≈ 20,9 dB |
| IGBT | amplitudes harmônicas relativas a σ | THD adicionada ≈ 8,0% |
| Fusível AC | fração de redução da fase A (teto 12%) | −3,6% de amplitude |

E a docstring do IGBT declara um mapeamento severidade→THD (~15%) que **a
própria fórmula não produz** (8,0%).

### Decisão

Reservar **Severidade / S** exclusivamente para a FMECA. O fator de injeção
passa a **`a_inj` — magnitude da assinatura injetada**, adotando literalmente a
analogia com o tamanho de defeito `a` da curva POD(a) do MIL-HDBK-1823A que o
projeto **já declarou** em `docs/nomenclatura_deteccao.md`. Com isso a `SMD`
vira `a_inj,95`, o análogo exato de a₉₀ — a nomenclatura fecha em si mesma.

**Migração em duas fases**, porque o custo é assimétrico:

1. **Documentos e `CLAUDE.md`** — custo zero, não entram em hash nenhum.
2. **Código (`src/ml/*.py`)** — invalida artefatos: `proveniencia.py:167` marca
   a etapa `stale` por `code_sha256`, e `:174` por mudança do dicionário
   `parameters`, cujas **chaves vêm dos nomes das constantes**. Renomear
   `SEVERIDADES` marca a cadeia inteira. Deve ser feito **na mesma rodada** da
   correção da faixa de F0 (`docs/auditoria_parametros.md` §1), que já exige
   recálculo total.

---

## §2. Por que os artefatos não mudavam — CORRIGIDO

A suspeita do pesquisador estava certa, por **dois mecanismos somados**.

### (a) "rode de novo" não forçava recálculo

`_deve_forcar` casava uma lista fechada de substrings que exigia o
**infinitivo** (`"rodar de novo"`). O pesquisador fala no **imperativo**.
Frases medidas que **NÃO** ativavam `force`:

```
rode o pipeline de novo        rode tudo de novo
retreine o autoencoder         treine o autoencoder novamente
execute o pipeline novamente   reexecute a validacao
atualize os resultados         rode a validacao mais uma vez
```

Sem `force`, a etapa READY é **pulada**.

**Corrigido** (`intencoes_ferramentas.py`): detecção por flexão verbal
(regex de verbo de recomputação + marcador de repetição) mais uma lista de
termos que já significam recálculo por si sós (`retreine`, `recalcule`,
`reprocessar`). 29 testes em `tests/test_forcar_recalculo.py` cobrem as frases
reais, e verificam que consultas de leitura **não** passam a forçar.

### (b) A mensagem de SKIP parecia execução fresca

A resposta era só `"<Etapa> ja esta pronto."`, e o chamador concatenava a
**tabela de resultados logo abaixo**. Somado ao determinismo do treino (semente
fixa ⇒ números idênticos), não havia como distinguir SKIP de recálculo real
olhando os arquivos.

**Corrigido** (`pipeline.py`): a mensagem passa a começar por

> `NAO recalculei. <Etapa> esta READY desde <created_at>; os numeros abaixo vem
> desse artefato, nao de uma execucao agora. Para forcar, peca "recalcule tudo
> do zero".`

e o retorno carrega `recalculou: False` e `artefatos_de: <data>`.

### (c) ⛔ Os manifestos foram re-emitidos sem recálculo

**O achado mais grave desta auditoria.** No commit `332bc53`, os cinco
manifestos passaram a ter `created_at` entre `01:39:59.546` e `01:40:00.094` —
**0,55 s para as cinco etapas**. A execução real registrada no commit anterior
levou **8 minutos** (01:00:06 → 01:08:16).

No diff, `code_sha256`, `input_artifacts` e **todos** os `output_artifacts`
ficaram idênticos; mudaram apenas `created_at`, `git_commit` e as novas chaves
de `parameters`. O mesmo commit acrescentou esses nomes a `parameter_names` —
mudança que faria `comparar()` disparar "parâmetros alterados" e deixar três
etapas **STALE**.

Os **números não são falsos** (vieram da execução real das 01:00). Mas o
manifesto passou a afirmar `created_at 01:39:59` para artefatos produzidos às
01:00. Se a banca cruzar `limiar.json` (`data_treino` 01:00:28) com o manifesto,
encontra **39 minutos de discrepância sem explicação** — e o manifesto é
justamente a prova de rastreabilidade.

### (d) A causa que induziu (c): a documentação

`docs/comandos.md:20` afirma que `python src/ml/autoencoder.py` grava manifesto.
**Não grava** — nenhum bloco `__main__` das cinco etapas chama
`registrar_manifesto`; a única chamada real está dentro de
`pipeline.executar_etapa`. Quem segue a documentação regenera os artefatos e
deixa o manifesto com os `output_artifacts` **antigos** → a etapa aparece STALE
logo após ser recalculada → surge a tentação de reescrever o manifesto à mão.

**O ciclo "rodo pelo terminal → fica stale → benzo o manifesto" é induzido pela
própria documentação.**

---

## §3. Weibull — o eixo não é tempo

### ⛔ O "TTF" é severidade, não tempo

`rul_weibull.py:232-240` escolhe a janela-base **uma vez**, e as 120 iterações
reaplicam severidade crescente sobre **a mesma janela congelada de 0,1024 s**.
Como as janelas têm exatamente `JANELA = 1024` linhas, `n_disp = 1024 − 1024 =
0`: o ramo aleatório da linha 239 é **código morto** e `inicio_base = 0` sempre.

Verificação numérica: `TTF/119` devolve exatamente a severidade de cruzamento —
contator `TTF ∈ [20, 55]` → `sev ∈ [0,168; 0,462]`, compatível com as taxas de
detecção do `injecao_falhas_report.json`.

**Consequência:** β, η, MTTF, B10 e as duas RUL não são estatísticas de tempo
até a falha. São **a distribuição da severidade mínima detectável por janela**,
reescalada por 119. A ressalva atual protege contra converter para horas, mas
não avisa o essencial: **não há tempo nenhum passando.**

### ⛔ A "censura" no teto é indetectabilidade estrutural

A severidade é limitada em 1,0 por construção. Quem não cruza recebe
`(n_steps, False)`, e o MLE trata isso como **sobrevivência** — assumindo que a
trajetória continuaria e falharia depois de 120. **Não existe "depois de 120"**:
o modelo de falha acaba ali.

Medido: todos os `TTF = 120` têm `evento = False` (IGBT 6, fusível 1). A taxa de
detecção do IGBT em sev 1,0 é 0,841 — **15,9% das janelas nunca são detectadas
nem na severidade máxima**, contra `censura_pct = 15,79%` no Weibull. É o mesmo
conjunto de janelas.

Chamar isso de censura **esconde o resultado mais informativo do experimento**:
16% das janelas são indetectáveis para a assinatura de IGBT no nível máximo
modelado. Isso é um limite do detector, não um dado de sobrevivência.

### Sobre converter para horas/dias/anos

**Não é defensável com o que existe no repositório.** Não há taxa de degradação
de campo, nem histórico run-to-failure, nem lei de envelhecimento calibrada. A
janela de aquisição tem duração física conhecida (0,1024 s), mas o **avanço de
severidade não tem taxa**. Converter seria inventar o número mais importante da
seção. A saída honesta é renomear o eixo, não convertê-lo.

### Demais

- `N_TRAJ = 100` é inalcançável (n efetivo 38) e o manifesto **publica 100**.
- O filtro de baseline exclui 6 de 44 janelas e é o parâmetro mais influente do
  módulo — sem registro de sensibilidade.
- `PERSISTENCIA_CRUZAMENTO = 3` introduz viés sistemático de +3 passos.

---

## §4. Corrente contínua — o CC certo foi jogado fora e o errado ficou

O pesquisador afirmou que a CC não deveria entrar. A auditoria confirma, e
encontra algo pior.

### A feature CC é inerte

Nenhuma injeção — E1 ou E2 — jamais perturba `u_dc_k`. As três funções de falha
escrevem só em colunas CA. O próprio teste do projeto **codifica** essa inércia:
`tests/test_protocolos_artigos.py:54-59` põe `tensao_dc_media` na lista
`intocaveis`.

Numa modelagem de normalidade, uma dimensão inerte não é neutra: consome
capacidade do gargalo de 16 dimensões e acrescenta variância ao escore sem
acrescentar poder de discriminação.

### E tem alavanca desproporcional

Medido em `estatistica_residuo.npz`: no canal `tensao_dc_media`, σ = 0,0883
contra mediana de 0,1998 nas 109 features — a **15ª menor σ de 109**. Como o
escore é a média dos 5 maiores `z = (|r| − μ)/σ`, a alavanca desse canal é
`1/σ = 11,33` contra 5,01 do canal mediano: **2,26× maior**.

Se ele ocupa uma das cinco vagas, parte do limiar 7,826 — que define SMD,
POD_mon, D_mon, NPR projetado e o TTF do Weibull — é fixada por uma grandeza CC
de bancada de motor.

### ⛔ A inversão de escopo

`features_ca.py:93` exclui `i_a_media, i_b_media, i_c_media, u_a_media,
u_b_media, u_c_media` com justificativa **estatística** ("média de sinal CA ≈ 0,
CV enganoso"). Essas seis features são exatamente **o componente CC dos sinais
CA** — a injeção de corrente contínua na rede, limitada por **IEC 61727** e
**IEEE 1547**, e modo de falha reconhecido do estágio de saída.

Na mesma passada, mantém `tensao_dc_media`, que é o barramento CC — do outro
lado do estágio de conversão.

**O pipeline descartou a única janela de observação de um fenômeno CC que é
indiscutivelmente do lado CA e normativamente relevante, e preservou uma
grandeza que está fora do escopo declarado.**

O problema do CV com média ≈ 0 se resolve **normalizando** o offset pelo RMS da
própria fase (`offset_relativo = |média| / rms`, adimensional e comparável ao
limite de 0,5% de I_n da IEC 61727) — não deletando a feature.

### O que foi feito, e onde a auditoria errou

`tensao_dc_media` **saiu** (08/08/2026). O vetor vai de **109 para 108**
dimensões — não "permanece com 109", como esta seção afirmava. A conta anterior
supunha que a feature de offset entraria na mesma passada; ela não entrou, e o
motivo é substantivo:

**nenhuma das três assinaturas de `docs/fmeca.md` desloca a média.** Os
harmônicos ímpares do IGBT têm média nula; a perda parcial de fase do Fusível AC
escala amplitude, e `|média|/rms` é invariante a escala; o ruído de comutação do
Contator AC é aproximadamente simétrico. Reintroduzir a feature agora criaria
**seis dimensões inertes sob injeção** — exatamente o defeito nº 2 pelo qual
`tensao_dc_media` foi removida. Trocar um canal inerte por seis não é progresso.

**❓ DECISÃO — para o pesquisador.** Adotar o offset CC exige, ANTES, estender a
assinatura do IGBT em `docs/fmeca.md` para incluir injeção de CC (chaveamento
assimétrico → offset na corrente de fase). Isso altera a FMECA, que é fonte
única e cujos modos são estipulados pelo pesquisador. A decisão está registrada
em `src/ml/features_ca.py`, no bloco sobre `FEATURES_EXCLUIR`, e travada por
`tests/test_escopo_ca.py` — que falha se a exclusão mudar sem a justificativa.

---

## §5. O que foi corrigido nesta rodada

| Item | Onde | Invalida artefato? |
|---|---|---|
| Detecção de "recalcule" por flexão verbal | `intencoes_ferramentas.py` | não |
| Mensagem de SKIP declara que não recalculou, com data | `pipeline.py` | não |
| Import circular `ferramentas` ↔ `intencoes_ferramentas` | `intencoes_ferramentas.py` | não |
| 29 testes das frases reais de recálculo | `tests/test_forcar_recalculo.py` | não |

O import circular era gratuito: `_normalizar` sempre foi um alias de
`src.core.texto.normalizar_sem_acentos`, e importá-lo de `ferramentas` fechava
o ciclo sem comprar nada.

---

## §6. O que NÃO foi corrigido, e por quê

Tudo o que segue **invalida a cadeia inteira de artefatos** e deve ser feito
numa **única rodada planejada**, com comparação antes/depois — não em edições
soltas:

1. Renomear `SEVERIDADES` → `A_INJ` no código (§1);
2. Corrigir a faixa de busca de F0 (`docs/auditoria_parametros.md` §1);
3. Remover `tensao_dc_media` (§4) — o offset CC relativo fica pendente de
   decisão sobre a FMECA, ver §4;
4. Renomear o eixo do Weibull de TTF para severidade de detecção (§3);
5. Separar indetectabilidade estrutural de censura genuína (§3).

Fazer qualquer um isolado gasta uma rodada completa de recálculo e impede saber
qual mudança causou qual efeito. **Recomendação: uma rodada só, com os cinco.**

---

## §7. Antes de qualquer rodada: refazer os manifestos

Por causa de §2(c), os cinco manifestos vigentes têm `created_at` que não
corresponde à execução que produziu os artefatos. **A primeira execução deve ser
um recálculo completo forçado**, para que os manifestos passem a ter datas
coerentes entre si e com `data_treino`.

Antes disso, corrigir §2(d): ou os blocos `__main__` das etapas passam a
registrar manifesto, ou `docs/comandos.md` deixa de afirmar que registram.

---

## §8. Para o outro agente (ChatGPT, retorno em 10/08)

Este documento é o ponto de sincronia. Decisões já tomadas, para não serem
reabertas:

- **PR #94 não é adotada** — o corte de FPR ≤ 1% derruba o recall do IGBT e do
  fusível para 0,025, colapsa a Weibull por censura e anula o NPR projetado.
  O que valia nela foi extraído (`docs/decisao_fpr_1pct.md`).
- **A causa raiz do FPR alto não é o limiar** — é a faixa de busca de F0 não
  cobrir o dataset (`docs/auditoria_parametros.md` §1).
- **Nomenclatura é fonte única**: `docs/nomenclatura_deteccao.md` para detecção,
  este documento §1 para severidade. Não introduzir sinônimos.
- **Não re-emitir manifestos fora de `executar_etapa`.**

---

## §9. Treino — as épocas NÃO são o problema

O pesquisador suspeitou que 150 épocas com paciência 20 (parada real em 75)
fosse curto demais. **Medido: não é.** O limite é outro, e aumentar épocas o
agrava.

### Os números

| Grandeza | Valor |
|---|---:|
| Parâmetros treináveis | **19.389** |
| Janelas de treino | **274** |
| Parâmetros por amostra | **70,8** |
| Passos de gradiente por época | 9 (`⌈274/32⌉`) |
| Passos até a época 75 | **675** |

Arquitetura `109→64→32→16→32→64→109`: 7.040 + 2.080 + 528 + 544 + 2.112 + 7.085.

### A curva já estagnou

Lido de `diagnostico_autoencoder.npz`:

| Épocas | Loss treino | Loss calibração |
|---|---:|---:|
| 0–5 | 5,840 | 0,751 |
| 20–25 | 1,639 | 0,437 |
| 50–55 | 1,000 | 0,368 |
| 70–75 | 0,884 | 0,357 |

Inclinação nas últimas 20 épocas: treino −0,0083/época, **calibração −0,00055/época**
(−0,15% por época). O melhor ponto de calibração foi a **época 54**, e o fim da
série está apenas **1,7% pior**. O early stopping disparou corretamente: a curva
de calibração é plana, não interrompida em queda.

**Veredito: aumentar `EPOCHS` não muda nada.** A parada não foi por teto de
épocas (150 nunca foi alcançado) — foi por ausência de melhora.

### O limite real

**70,8 parâmetros por amostra de treino.** A rede tem capacidade de memorizar o
bloco de treino inteiro várias vezes; o único regularizador é dropout 0,2. Mais
épocas aumentam a memorização, não a generalização.

O que de fato limita:

1. **Só existem 457 janelas no dataset inteiro** (234.527 linhas ÷ passo de 512).
   Destas, 274 são de treino. Aumentar a sobreposição geraria mais janelas, mas
   **correlacionadas** — n efetivo não cresce na mesma proporção.
2. **A rede é grande demais para o dado.** `109→32→16→8` teria ~4.700 parâmetros
   (4× menos), com razão de 17 por amostra. É o caminho de maior retorno se o
   objetivo for reduzir sobreajuste.
3. **ReLU no gargalo** (já registrado em `auditoria_pipeline_ml.md` §23) impede
   representação negativa e permite unidades latentes mortas — parte das 16
   dimensões pode estar em zero permanente.

### Nota de leitura: loss de treino MAIOR que a de calibração

No fim, treino 0,884 contra calibração 0,357 — razão de **2,63×**. Isso **não**
é sinal de subajuste: dropout fica ativo no treino e desligado na avaliação, e
a loss de treino é medida com ele ligado. A comparação direta entre as duas
curvas não é maçã-com-maçã, e não deve ser apresentada como se fosse.

**Recomendação:** não mexer em `EPOCHS`. Se for para mexer em arquitetura, fazer
junto da varredura de latente e da ReLU, na rodada única já planejada (§6).

---

## §10. Testes — a suspeita não se confirma

O pesquisador achou que havia arquivos demais. **Medido: 69 arquivos, 550
testes — densidade de ~8 testes por arquivo, que é saudável.**

Varredura mecânica por AST em todos os arquivos:

| Verificação | Resultado |
|---|---|
| Testes com `assert` de constante literal (tautologia) | **0** |
| Testes sem asserção alguma | **0** |
| Aparentes sem asserção, conferidos um a um | 2 — ambos legítimos |

Os dois aparentes usam formas que a varredura não reconhece como `assert`:
`np.testing.assert_allclose` (`test_embeddings.py`) e `raise AssertionError`
após `try/except` (`test_provedores_leves.py`). São asserções reais.

### Pares que parecem redundantes e não são

| Par | Veredito |
|---|---|
| `test_limiar` × `test_validacao_limiar` | Distintos: o primeiro é o limiar do AE, o segundo é "as métricas respeitam o limiar recebido" |
| `test_exec_isolado` × `test_exec_etapa_isolada` | **Módulos diferentes**: subprocesso de experimentos × processo-filho do pipeline |
| `test_resultados_imagens` × `test_resultados_ml` | Distintos: imagens × adaptador para a memória |

### Única consolidação com mérito

Três arquivos guardam invariantes **estruturais do repositório** e somam 6
testes: `test_cobertura_auditoria_src.py` (1), `test_limites_arquitetura.py` (1)
e `test_dependencias_src.py` (4). Poderiam ser um `test_arquitetura_src.py`.

**Não executado nesta rodada**: o ganho é organizacional e o custo é churn de
histórico. Fica registrado como opcional, não como dívida.

### Conclusão honesta

O volume de arquivos reflete a **superfície real** do projeto — pipeline de ML,
agente RAG, memória, interface, proveniência, segurança. Não encontrei gordura
que justifique apagar. Se algum arquivo incomoda, o critério deve ser cobertura
duplicada medida, não contagem de arquivos.

---

## §11. Frentes do agente e dos PRs — o que apareceu

Relatórios integrais em `docs/auditoria_agente_e_prs.md`. Os quatro achados que
representam **risco acadêmico direto**:

### ⛔ A guarda de citação silencia na fabricação que ela existe para pegar

`src/core/citacao_guarda.py` compara a norma citada contra o **texto dos trechos
recuperados**, não contra o identificador da fonte. Basta um artigo indexado
*mencionar* "IEC 60812" para que uma citação inventada com cláusula e página
("IEC 60812:2018, Cláusula 7.3.3, p. 27") passe **sem alerta**.

O docstring do próprio módulo dá esse caso como exemplo do que ele deveria pegar.
Artigos de FMEA/RCM citam IEC 60812 e ISO 14224 o tempo todo — o gatilho é
rotineiro, não exótico.

### ⛔ O auditor só enxerga 8 fontes; da 9ª em diante nada pode ser citado

`src/conhecimento/multiagente.py` trunca o pacote enviado ao auditor. Fontes além
da oitava não são auditadas e, portanto, não podem ser citadas — mesmo tendo sido
recuperadas e mesmo sendo as mais pertinentes.

### ⛔ Chunk de 1800 caracteres contra um encoder de 128 tokens

`src/conhecimento/embeddings.py` trunca em 128 tokens. Com chunks de ~1800
caracteres, **cerca de 70% de cada chunk nunca é embutido** — está indexado no
papel e invisível na busca semântica. O BM25 ainda o alcança; a busca vetorial
não.

### ⚠️ `stale` apaga artefato E2 versionado antes de checar se dá para recalcular

`src/ml/pipeline.py` limpa os artefatos e **só depois** verifica dependências. Numa
máquina sem `dados/brutos/`, isso destrói resultado publicado sem ter como
regenerá-lo.

### O que NÃO quebrou

A revisão das premissas é a boa notícia: **nenhuma das seis foi quebrada em
substância** pelos PRs #89–#103. O protocolo E2 segue com injeção FMECA no sinal,
o limiar continua congelado em bloco disjunto com purga de 2 janelas, o NPR
oficial (315/90/30) é idêntico ao de `docs/fmeca.md`, e as ressalvas E2≠E3 foram
até reforçadas.

O que quebrou foi a **cadeia de rastreabilidade** — o mesmo problema de §2(c).

### Falsos alarmes que corroem a confiança no alerta

O padrão de normas aceita `EN` sem exigir caixa alta, então *"en 2022"* e
*"en 2021"* (francês e espanhol, idiomas que o `CLAUDE.md` manda o agente usar)
disparam **"Normas técnicas citadas acima"**. Alerta que grita em texto legítimo
ensina a ignorar o alerta verdadeiro.

---

## §12. Frente encerrada em 2026-08-09

A revisão específica das refatorações **#99 e #102** foi concluída. A
comparação estrutural confirmou preservação das funções e fachadas, mas a
execução em processos limpos revelou ciclos de importação ocultos em onze
módulos extraídos. As fachadas agora usam reexportação tardia, e há regressões
que importam cada módulo sem depender da ordem. Também foi corrigida a
normalização de `0` e `False`, que era convertida em vazio, a exclusão prematura
de artefatos `stale` e a proveniência dos gráficos Weibull.

Relatório técnico e evidências: `docs/auditoria_prs_99_102_2026-08-09.md`.

## Referências internas

- `docs/auditoria_parametros.md` — parâmetros do pipeline de ML (rodada anterior)
- `docs/decisao_fpr_1pct.md` — por que o corte de 1% não foi adotado
- `docs/nomenclatura_deteccao.md` — `D_campo`, `POD_mon`, `D_mon`, `SMD`
- `docs/auditoria_pipeline_ml.md` — auditoria histórica, §22 e §23
