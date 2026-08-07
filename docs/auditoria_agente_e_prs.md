# Auditoria do agente e revisão dos PRs #89–#103

Apêndice de `docs/auditoria_total_src.md`, frentes 7 e 8. Relatórios integrais
dos agentes de auditoria, com a evidência de cada achado.

> **Frente incompleta:** a revisão específica dos PRs #99 e #102 (refatorações)
> não concluiu — o agente esgotou o limite de sessão duas vezes. Fica pendente,
> e **não** deve ser tratada como auditada.

---


## A. `src/core/` e `src/interface/`

## Resumo

Varri `src/core/` (9 módulos, 918 linhas) e `src/interface/` (5 módulos, 1930 linhas). O achado mais grave é na guarda de citação: ela fica **silenciosa exatamente no caso de fabricação que o próprio docstring dá como exemplo** (norma IEC citada com cláusula/página inventada), porque compara a resposta contra o texto dos *trechos* recuperados. Além disso, `formatacao.py` se declara "ponto único de verdade" de todo número exibido mas tem 5 de 7 funções sem nenhum chamador de produção; o marcador de build atesta uma versão que ficou 7 commits para trás; e o atalho de exportar conversa sequestra pedidos como "exporte o histórico de execuções do pipeline".

## Achados

### [CRITICO] A guarda de citação silencia justamente na fabricação de norma que ela existe para pegar
- **Onde:** `src/core/citacao_guarda.py:88-117` (`_texto_das_citacoes` + bloco 1 de `alerta_citacao_infundada`)
- **Evidência:** o docstring do módulo (linhas 6-8) nomeia o caso alvo: *"IEC 60812:2018, Cláusula 7.3.3, p. 27 com aspas da norma"*. Rodei exatamente esse caso:

  ```python
  fontes = {'c1': 'Carpinetti L (2016) — p. 44 — trecho: "A analise FMEA segue a
            IEC 60812 para definir severidade e ocorrencia."'}
  resp   = 'Conforme a IEC 60812:2018, Clausula 7.3.3, p. 27: "o NPR deve ser..."'
  alerta_citacao_infundada(resp, fontes)  ->  ''   # silêncio total
  ```
  Motivo: `_texto_das_citacoes` concatena o **valor inteiro** da entrada de citação, e essa entrada inclui o excerto de 280 caracteres do chunk (`_entrada_citacao` em `agente_interacao.py:742-755` monta `base — pagina — trecho: "..."`). Basta um artigo indexado *mencionar* "IEC 60812" no trecho recuperado para `_norm("IEC 60812") in fontes_norm` dar verdadeiro. A norma nunca foi indexada; a cláusula e a página seguem inverificáveis — e o pesquisador não é avisado.
- **Por que importa:** é o cenário de risco acadêmico direto — uma cláusula e uma página de norma paga entram na dissertação com aparência de verificadas. A guarda é a única defesa estrutural além do prompt, e ela falha silenciosamente no caso mais provável (artigos de FMEA/RCM citam IEC 60812 e ISO 14224 o tempo todo).
- **Correção:** comparar a norma apenas contra o **identificador da fonte** (`_rotulo_curto`, que já existe e descarta o trecho), não contra o corpo do excerto. Uma norma só está "lastreada" se o PDF dela estiver na base — mencionada dentro de outro artigo não é lastro. Alternativamente, sempre avisar quando a resposta traz cláusula/página de norma e a fonte casada não é o próprio documento normativo.
- **Invalida artefatos:** não (invalida a confiança em citações já produzidas no chat).

---

### [SERIO] `EN` no padrão de normas transforma "en 2022" (francês/espanhol) em falso alarme de citação
- **Onde:** `src/core/citacao_guarda.py:29-32` (`_PADRAO_NORMA`, alternativa `EN`, com `re.I`)
- **Evidência:** o padrão é `\b(?:IEC|ISO|IEEE|ABNT|NBR|ASTM|DIN|EN|MIL-STD|MIL-HDBK|SAE|API)\s*[-:]?\s*\d{2,6}` com `re.I`. Testado:

  | resposta | alerta emitido |
  |---|---|
  | `Ibrahim et al. proposent un autoencodeur en 2022...` | "Normas técnicas citadas acima (**en 2022**)…" |
  | `Segun Ghoneim en 2021, el modelo alcanza AUC alta.` | "Normas técnicas citadas acima (**en 2021**)…" |
  | `O modelo foi treinado com EN 1000 amostras.` | "Normas técnicas citadas acima (**EN 1000**)…" |

  O CLAUDE.md exige que o agente responda em ES/FR quando a pergunta vier nesses idiomas — ou seja, o gatilho é rotineiro, não exótico. Some-se `infundadas[:3]` (linha 114): se o modelo fabricar 5 normas, só 3 são nomeadas no aviso, sem dizer que há mais.
- **Por que importa:** aviso de fabricação que dispara em texto legítimo é o clássico "cry wolf" — o pesquisador aprende a ignorar o box `⚠️ Verificação de citações`, e aí o alerta verdadeiro (o do achado anterior, quando ele funcionar) passa batido. O truncamento em 3 esconde silenciosamente parte da evidência de fabricação.
- **Correção:** exigir caixa alta para `EN` (compilar essa alternativa sem `re.I`, ou `(?-i:EN)`), e trocar `infundadas[:3]` por "3 primeiras + (mais N)". Não há env para isso e o valor 3 não tem justificativa registrada.
- **Invalida artefatos:** não.

---

### [SERIO] `formatacao.py` se declara ponto único de verdade dos números exibidos, mas 6 das 7 funções não têm chamador de produção
- **Onde:** `src/core/formatacao.py:5-7` (docstring: *"Ponto único de verdade: TODA tabela e TODO número exibidos ao usuário passam por aqui"*)
- **Evidência:** grep em todo o repositório (`--include=*.py`, sem `__pycache__`), fora do próprio módulo:

  | símbolo | chamadores de produção | chamadores em `tests/` |
  |---|---:|---:|
  | `fmt_num` | 1 (`src/ml/resultados.py:18,35`) | 2 |
  | `fmt_metrica` | **0** | 2 |
  | `fmt_limiar` | **0** | 2 |
  | `fmt_fisico` | **0** | 2 |
  | `fmt_pct` | **0** | 2 |
  | `fmt_pvalor` | **0** | 4 |
  | `tabela_markdown` | **0** | 3 |

  As tabelas do chat são montadas à mão, com casas decimais próprias: `src/conhecimento/ferramentas_academicas.py:297` imprime AUC com `{principal:.4f}` (4 casas) enquanto a política do módulo manda 3 para métricas 0–1; `src/ml/resultados.py:445-446` usa `_fmt(..., 4)`. Ao todo há 30 formatações `:.Nf` em `validacao.py`, 15 em `ferramentas_academicas.py`, 13 em `rul_weibull.py`, etc. Armadilha adicional latente: `fmt_num` testa `isinstance(valor, (int, float))`, e `np.float32(0.909)` → `'-'`, `np.int64(120)` → `'-'` (verificado); `float('nan')` → `'nan'` e `fmt_pvalor(nan)` → `'p=nan'`.
- **Por que importa:** o mesmo AUC pode aparecer como `0,909` numa resposta e `0,9087` noutra, dependendo de qual ferramenta respondeu — e a banca vê a inconsistência antes do orientando. A política existe, está escrita, tem testes, e não governa nada do que o pesquisador realmente lê.
- **Correção:** ou fazer os montadores de tabela do chat passarem por `tabela_markdown`/`fmt_*`, ou rebaixar o docstring para "utilitário disponível" e parar de afirmar unicidade. Trocar `isinstance(..., (int,float))` por `numbers.Real` e tratar NaN/inf como `'-'`.
- **Invalida artefatos:** não (os JSONs guardam o valor cheio; o problema é a exibição).

---

### [SERIO] `MARCADOR_BUILD` é editado à mão e já atesta uma versão 7 commits atrasada
- **Onde:** `src/core/config.py:139-146`; exibido em `src/interface/sidebar.py:279` e `:299`
- **Evidência:** o comentário diz que o marcador serve para *"confirmar QUAL versão do código está no ar… se o marcador aqui não bate com o exibido, o app está rodando código antigo e precisa de Reboot"*. Valor atual: `"2026-08-05 · auditoria geral de src · comparação acadêmica unificada"`, fixado em `b1dfb91` (2026-08-05). Desde então:

  ```
  git log --oneline b1dfb91..HEAD -- src/   →  7 commits
  git diff --stat b1dfb91..HEAD -- src/     →  54 arquivos, +6594 / -5996
  ```
  entre eles `e9c6fc5 fix: torna metricas macro inequivocas`, `7f5be7a refactor: modulariza pontos criticos de src` e `a8a67c2 fix: o skip silencioso que fazia os artefatos nao mudarem`. A barra lateral de HOJE anuncia o build de 05/08 rodando código de 06/08 que mexeu em métrica.
- **Por que importa:** o único mecanismo que o pesquisador tem para saber se a nuvem está com código velho não distingue a versão que mudou métrica da anterior. Ele pode atribuir um número novo a um build antigo (ou vice-versa) ao registrar uma rodada. `AL_IADO_BUILD_LABEL` está em `.env.example`, mas comentado — o que roda é o default hardcoded.
- **Correção:** derivar o default de `git rev-parse --short HEAD` + data do commit (com fallback para a string, para o Cloud sem `.git`), e não de edição manual.
- **Invalida artefatos:** não, mas corrompe a rastreabilidade de qual código produziu qual rodada.

---

### [SERIO] O atalho de exportar conversa sequestra pedidos de exportar OUTRA coisa
- **Onde:** `src/core/conversa_export.py:27-52` (`_ALVOS` / `_ACOES_ARQUIVO`), acionado em `src/conhecimento/atalhos.py:178` — é o **primeiro** da tupla `ATALHOS`
- **Evidência:** `quer_exportar_conversa` exige apenas um termo de cada lista. `_ALVOS` inclui `"historico"`, `"chat"`, `"essa sessao"`; `_ACOES_ARQUIVO` inclui `"exporte"`, `"baixar"`, `"arquivo"`, `"salvar em"`. Testado:

  | pergunta | dispara export do chat? |
  |---|---|
  | `exporte o historico de execucoes do pipeline` | **True** |
  | `gere um arquivo com o historico de treinos` | **True** |
  | `baixar o arquivo de resultados dessa sessao` | **True** |
  | `salvar em md o historico de versoes do modelo` | **True** |
  | `qual o historico do inversor?` | False (correto) |

  Como o atalho é o primeiro em `ATALHOS` e `renderizar_chat` faz `return` logo após `_fechar_turno_simples` (`streamlit_app.py:816-824`), o pedido **nunca chega** ao roteador de ferramentas. E `docs/registro_execucoes.md` existe — "histórico de execuções" é um pedido legítimo e diferente.
- **Por que importa:** o pesquisador pede o registro de execuções e recebe um `.txt` do bate-papo, com aparência de sucesso (botão de download, sem erro). Silencioso e fácil de não notar até tentar usar o arquivo.
- **Correção:** exigir que alvo e ação estejam próximos (mesma cláusula) ou negar quando a frase contiver um substantivo de domínio concorrente (`pipeline`, `execuç*`, `resultado*`, `treino*`, `versõ*`). As duas listas não estão documentadas em lugar nenhum e são inteiramente arbitrárias.
- **Invalida artefatos:** não.

---

### [SERIO] A cadência de 6 interações é pulada quando o turno múltiplo de 6 cai num atalho
- **Onde:** `src/interface/ciclo_chat.py:78-90` (`_cadencia_atingida`), consumidores em `:106` e `:133`; caminho de atalho em `src/interface/streamlit_app.py:816-824`
- **Evidência:** `_cadencia_atingida` devolve `n` só quando `n % passo == 0` (passo=6, de `AL_IADO_CONSOLIDAR_A_CADA`). O turno de atalho chama `_fechar_turno_simples`, que **incrementa** `st.session_state.mensagens` e faz `return` — sem chamar `aprender_da_sessao_web()` nem `persistir_sessao_web()`. Portanto, se a 6ª interação for um atalho (exportar conversa, cronologia do vault, saudação — cinco atalhos, incluindo `interacao_simples` que pega qualquer "obrigado"), na 7ª interação `7 % 6 != 0` e a janela é perdida: próxima chance só em n=12.
- **Por que importa:** os dois consumidores são (a) a consolidação de memória validada e (b) o commit do transcrito da sessão no GitHub. No Streamlit Cloud o disco é efêmero: até 11 interações podem sumir num reboot porque um "obrigado" caiu no turno errado. É perda silenciosa do registro da pesquisa.
- **Correção:** trocar o teste de igualdade modular por um marco persistido (`st.session_state["ultima_consolidacao"]`) e disparar quando `n - ultima >= passo`; ou chamar as duas rotinas também em `_fechar_turno_simples`. O valor 6 está documentado em `.env.example`, mas a semântica "múltiplo exato" não está.
- **Invalida artefatos:** não.

---

### [MENOR] `TAXA_AMOSTRAGEM` e `SEMENTE_ALEATORIA` de `config.py` não são importados por ninguém
- **Onde:** `src/core/config.py:167-168`, sob o cabeçalho "CONSTANTES DO MACHINE LEARNING"; docstring do módulo em `:5-7` diz "ponto único de verdade para caminhos, constantes e parâmetros"
- **Evidência:** grep no repositório inteiro:
  - `TAXA_AMOSTRAGEM` aparece em exatamente 2 lugares: a definição em `config.py:167` e uma **redefinição local** em `src/ml/eda.py:47`. Em `src/ml/features_ca.py:76` o mesmo número vive com outro nome: `FS = 10_000  # Hz`. Nenhum `from src.core.config import TAXA_AMOSTRAGEM`.
  - `SEMENTE_ALEATORIA` aparece **só** na definição. O 42 está hardcoded em `autoencoder.py:134` (`SEED = 42`), `rul_weibull.py:658`, `macro_ibrahim.py:92` e em 6 pontos de `classificador_pv.py` / `classificador_pv_infer.py` (`random_state=42`).
- **Por que importa:** a taxa de amostragem governa o eixo de frequência da FFT, o THD e a estimativa de F0 (justamente o parâmetro do achado já registrado sobre a faixa de busca de F0). Quem for corrigir vai naturalmente editar `config.py`, que o módulo anuncia como fonte única — e não vai mudar absolutamente nada, porque `features_ca.FS` continua com o valor antigo. É um botão desconectado, e o pior tipo: o que parece conectado.
- **Correção:** ou fazer `features_ca.py`/`eda.py` importarem de `config.py`, ou deletar as duas constantes órfãs de `config.py` para não induzir edição inócua. Nenhuma das duas está em `.env.example`.
- **Invalida artefatos:** não (hoje os três valores coincidem — o risco é na próxima edição).

---

### [MENOR] Diagnóstico do log corta o erro pelo fim e ignora os arquivos rotacionados
- **Onde:** `src/interface/sidebar.py:89-99`; rotação em `src/core/logs.py:67-69`
- **Evidência:** `erros[-1][-110:]` mantém os **últimos** 110 caracteres da linha. Numa linha real de 181 chars, o que aparece é:

  ```
  | falha ao indexar literatura/stender_inverter-data-set_2020.pdf: PdfReadError arquivo corrompido na pagina 12
  ```
  — sem timestamp, sem nível, sem módulo, e sem reticências indicando corte. Além disso, `ARQUIVO_LOG.read_text()` lê só `logs/al_iado_pv.log`; o handler é `RotatingFileHandler(maxBytes=2_000_000, backupCount=3)`, então após uma rotação o painel pode exibir "Sem erros registrados no log" enquanto os erros estão em `.log.1`/`.2`/`.3`. O arquivo atual tem 33 KB, mas 2 MB são atingíveis numa reindexação completa.
- **Por que importa:** sem timestamp o pesquisador não sabe se o erro é da execução atual ou de três semanas atrás, e "sem erros" logo após uma rotação é uma afirmação falsa num painel de diagnóstico. Efeito colateral menor: a cada render do sidebar o arquivo inteiro (até 2 MB) é lido e filtrado em memória.
- **Correção:** truncar pelo início preservando o carimbo (`ln[:19] + " … " + ln[-90:]`) com marca visível de corte, e varrer também os backups (ou manter um contador de erros da sessão em memória). Os números 110, 2_000_000 e 3 não estão em `.env.example` nem têm justificativa registrada.
- **Invalida artefatos:** não.

---


## B. `src/conhecimento/` — o agente RAG

## Resumo

Auditei os parâmetros de recuperação, orçamento, memória e provedores em `src/conhecimento/`. Dois números decidem, sozinhos, o que pode ser citado: o encoder trunca em **128 tokens** enquanto os chunks têm mediana de **1622 caracteres**, e o auditor só enxerga as **8 primeiras** de até **50** citações recuperadas. Ambos são invisíveis nos docs e no `.env.example`. Há ainda o RRF neutralizado pelos pesos do rerank, 7 de 10 chaves `AL_IADO_RAG_*` indocumentadas (que o `test_consistencia_docs` estruturalmente não pega), duas políticas de retry 429 empilhadas e sem timeout, e um orçamento de contexto Obsidian inalcançável.

## Achados

### [CRITICO] O auditor só vê 8 fontes; da 9ª em diante nada pode ser citado
- **Onde:** `/home/user/mestrado-utfpr/src/conhecimento/multiagente.py:170` e `:87-99`; orçamento em `/home/user/mestrado-utfpr/src/conhecimento/agente.py:54-55`
- **Evidência:** `_pacote_fontes(citacoes, max_fontes: int = 8)` corta `list(citacoes.items())[:8]` antes de montar o pacote do auditor. `filtrar_citacoes_auditadas` mapeia os rótulos aprovados por `re.fullmatch(r"F(\d+)")` para índices da MESMA lista, então só índices 0–7 são alcançáveis. Enquanto isso `ORCAMENTOS_RAG["gemini"]` pede `n_resultados=30` e `n_resultados_revisao=50`, e `_chave_citacao` (`agente_interacao.py:731`) gera uma chave POR CHUNK (`arquivo|pag_ini|pag_fim|sha1[:16]`) — logo `citacoes` chega ao auditor com até 30 (ou 50) entradas. Em query de revisão o cap é `max_chunks_por_fonte=1` (`agente_contexto.py:79`), ou seja **50 documentos DISTINTOS recuperados e no máximo 8 citáveis**.
- **Por que importa:** é exatamente na pergunta "cite a literatura / estado da arte" que o pipeline maximiza diversidade — e é ali que 42 dos 50 papers recuperados ficam estruturalmente fora do rodapé. Pior: o contexto do prompt NÃO é filtrado (comentário em `src/interface/ciclo_chat.py:333-336` admite isso), mas `montar_restricao_fontes` manda o LLM usar "EXATAMENTE estas" — então o texto de 42 chunks está no prompt e proibido de virar citação. E se o paper on-topic cair na posição 9+, o auditor vê só ruído, devolve `insuficiente`, e `filtrar_citacoes_auditadas` retorna `{}` (linha 80-81): rodapé vazio numa pergunta que tinha 50 fontes recuperadas.
- **Correção:** casar `max_fontes` com `orcamento["n_resultados"]` (ou no mínimo com `n_resultados_revisao`), paginando a auditoria em lotes se o pacote estourar o `max_tokens=650`; e falhar ruidosamente quando `len(citacoes) > max_fontes` em vez de truncar em silêncio.
- **Invalida artefatos:** não

### [CRITICO] Chunk de 1800 chars indexado por um encoder de 128 tokens — ~70% de cada chunk nunca é embeddado
- **Onde:** `/home/user/mestrado-utfpr/src/conhecimento/embeddings.py:23` e `:127`; `/home/user/mestrado-utfpr/src/conhecimento/indexador.py:57-58`
- **Evidência:** `MAX_TOKENS = 128` e `tokenizer.enable_truncation(max_length=self.max_tokens)` — truncagem explícita no backend ONNX (o de consulta/deploy). No backend `SentenceTransformer` o `max_seq_length` do `paraphrase-multilingual-MiniLM-L12-v2` também é 128 e o código **nunca** o ajusta: `grep -rn "max_seq_length"` no repositório inteiro retorna vazio. Medido em `artefatos/literatura_indexada.jsonl.gz` (12.255 chunks): mediana **1622** chars, p90 **1790**, máx **1800**; **91,1%** dos chunks passam de 500 chars e **74,5%** passam de 1000. Com ~4 chars/token do tokenizer XLM-R em prosa acadêmica, 128 tokens ≈ 450–600 chars, isto é, o vetor de um chunk mediano representa cerca de 1/3 do texto que ele indexa.
- **Por que importa:** a busca semântica (CAMADA 2) enxerga só o começo de cada chunk. Um trecho com o dado decisivo (NPR, THD, valor de tabela) na segunda metade do chunk é irrecuperável por embedding — só o BM25 pode achá-lo. E o `.env.example` (linhas 88-90) recomenda ativamente o oposto: *"500 chars era o valor antigo; 1600-2200 e mais equilibrado para RAG academico"*, sem nenhuma menção à janela do encoder. Nenhum doc em `docs/` cita 128 ou truncagem (`grep` em `docs/*.md` e `src/README.md`: vazio).
- **Correção:** medir a taxa real de tokens/char no corpus e ou (a) baixar `TAMANHO_CHUNK_LITERATURA` para a janela útil, ou (b) indexar em dois níveis (vetor sobre janela de 128 tokens + chunk pai de 1800 para o contexto). Em qualquer caso, documentar `MAX_TOKENS=128` no `.env.example` junto do tamanho de chunk, já que as duas constantes só fazem sentido juntas. Trocar o chunk exige reindexar.
- **Invalida artefatos:** não (os artefatos de ML não dependem disso), mas invalida a premissa documentada de qualidade do índice.

### [SERIO] O rerank neutraliza o RRF: a fusão semântica+BM25 vale menos que um único termo lexical
- **Onde:** `/home/user/mestrado-utfpr/src/conhecimento/agente_recuperacao.py:333`, `:587`, `:591`, `:627`, `:487`
- **Evidência:** o RRF usa `1.0 / (60.0 + max(1, rank))` e o rerank o converte com `score += 30.0 * _rrf_score`. Por variação de query, a diferença entre o 1º e o 50º colocado vale `30*(1/61 − 1/110) = 0,22` ponto. Na mesma escala: cada termo lexical presente vale **+2,0** (`>4` chars) ou +1,0; casar autor vale **+6,0**; `PESOS_PASTA` vale até **+1,4** incondicionalmente; textbook fora de domínio leva **−12,0**. Com uma query típica de 5 variações, o spread total do RRF é ~1,1 ponto — abaixo de um único acerto lexical longo.
- **Por que importa:** CLAUDE.md descreve a fusão RRF como CAMADA 2 do pipeline, mas ela sobrevive apenas como filtro de entrada no pool; a ordenação final é decidida por heurística de sobreposição de palavras. Isso reintroduz o viés de casamento exato que o híbrido existia para corrigir, e agrava o achado anterior (o chunk cujo miolo não foi embeddado também não é resgatado pelo ranking semântico). Agrava ainda o fato de o caminho de fallback `colecao.get(where_document=...)` / busca por autor (`:396-451`) atribuir `rank` pela ordem de armazenamento, não por relevância, e mesmo assim receber crédito RRF.
- **Correção:** normalizar `_rrf_score` para [0,1] dentro do pool antes de multiplicar, e calibrar o peso contra os bônus lexicais (hoje 30 é ~15× pequeno demais). `tests/test_retrieval_metrics.py` já existe e pode ancorar a recalibração.
- **Invalida artefatos:** não

### [SERIO] 7 das 10 chaves `AL_IADO_RAG_*` não existem no `.env.example`, e o teste que deveria pegar isso é cego a `getenv` com f-string
- **Onde:** `/home/user/mestrado-utfpr/src/conhecimento/agente_interacao.py:248`; `/home/user/mestrado-utfpr/tests/test_consistencia_docs.py:33`
- **Evidência:** `_orcamento_rag` faz `env = os.getenv(f"AL_IADO_RAG_{chave.upper()}")` iterando as 10 chaves de `ORCAMENTOS_RAG`, criando as variáveis reais `AL_IADO_RAG_N_POOL`, `_N_RESULTADOS`, `_N_RESULTADOS_REVISAO`, `_MAX_CHUNKS_POR_FONTE`, `_SESSAO_CHARS`, `_HISTORICO_TURNOS`, `_HISTORICO_CHARS`, `_ANEXOS_CHARS`, `_CONTEXTO_CHARS`, `_MAX_PROMPT_CHARS`. O `.env.example` documenta **3** (linhas 62-64). O teste `test_env_example_cobre_variaveis_lidas_pelo_codigo` casa apenas `r'os\.getenv\(\s*["\']([A-Z][A-Z0-9_]*)["\']'` — literal de string. Rodei `pytest tests/test_consistencia_docs.py`: **11 passed**, com as 7 lacunas presentes. É a única construção `getenv(f"...")` do `src/` inteiro, ou seja, um ponto cego de exatamente um caso — e é o que controla top-K e diversificação.
- **Por que importa:** os knobs que decidem quantos chunks entram e quantos por fonte são ajustáveis em produção mas não descobríveis; e a garantia documental que o projeto acredita ter ("toda variável lida está no `.env.example`") é falsa justamente aqui.
- **Correção:** documentar as 10 chaves com seus defaults por provedor, e estender o teste para também extrair prefixos de f-string (`os.getenv(f"PREFIXO_{...}")` → exigir que o prefixo apareça no `.env.example`, ou proibir a construção).
- **Invalida artefatos:** não

### [SERIO] Duas camadas de retry 429 empilhadas, com políticas incompatíveis, e nenhuma chamada Gemini tem timeout
- **Onde:** `/home/user/mestrado-utfpr/src/conhecimento/provedores.py:40-41` e `:249-281`; `/home/user/mestrado-utfpr/src/conhecimento/agente.py:540-571`; `/home/user/mestrado-utfpr/src/conhecimento/agente_interacao.py:585-597`
- **Evidência:** camada interna — `_MAX_RETENTATIVAS = 2`, `_BACKOFF_BASE_S = 1.2`, espera `1.2*tentativa` (total ~3,6 s), e `_erro_transitorio` inclui explicitamente `"429"`, `"rate limit"`. Camada externa — `perguntar()` usa `max_tentativas = 3` e `_espera_retry_429`, que faz `min(120, 2**(tentativa+3)) + random.randint(0,5)`, com teto de **120 s**. Como a interna já consome os 429, a externa só vê o erro depois: pior caso são 3×3 = **9 chamadas** e ~240 s de sono, para o mesmo 429. Além disso: `grep -n "timeout" provedores.py` não retorna nada — `generate_content`/`generate_content_stream` são chamados sem `http_options`/timeout, ao contrário de `web_search.py:34` (`timeout=_TIMEOUT`) e do GitHub (`AL_IADO_GITHUB_TIMEOUT=12`).
- **Por que importa:** uma requisição pendurada trava a thread do Streamlit indefinidamente (sem timeout não há como o retry sequer disparar), e quando ela falha o usuário espera até 4 minutos por algo que o provedor já rejeitou. As duas políticas discordam sobre o que é backoff correto para o mesmo código de erro.
- **Correção:** uma política só — deixar 429/503 inteiramente em `provedores._chamar` e remover o laço de `perguntar()` (ou vice-versa), e adicionar um timeout explícito por chamada, com env documentada como as demais.
- **Invalida artefatos:** não

### [SERIO] `AL_IADO_RAG_OBSIDIAN_CHARS=18000` é inerte: um teto fixo de 6 chunks corta antes
- **Onde:** `/home/user/mestrado-utfpr/src/conhecimento/agente_contexto.py:157`; `/home/user/mestrado-utfpr/src/conhecimento/consultas_obsidian.py:332` e `:453`
- **Evidência:** `buscar_notas_obsidian` é chamado com `n_resultados=max(3, min(6, (n_resultados or 8) // 2))` — com o orçamento Gemini (`n_resultados=30`) isso dá **6**, e o laço encerra em `if incluidos >= n_resultados: break`, ANTES do teste de `max_chars`. Tamanho real dos chunks medido em `artefatos/obsidian_indexado.jsonl.gz`: `sessao_arquivada` n=4280, mediana **746**, máx 900; `memoria_consolidada` n=463, mediana **706**, máx 1200; `curada` n=87, mediana 223. Teto prático: 6 × ~900 + ~6 cabeçalhos de ~200 chars ≈ **6,5 k** — cerca de **36%** dos 18.000 configurados. O `min(6, …)` também torna o parâmetro monotônico-morto: subir `n_resultados` de 12 para 30 não muda nada no vault.
- **Por que importa:** é um dos 3 knobs de RAG que o `.env.example` documenta, e ele não faz o que anuncia. A memória do projeto (decisões, correções, notas curadas) entra no prompt com 1/3 do orçamento que o pesquisador acredita ter concedido, enquanto o Gemini paga por um budget que nunca é usado.
- **Correção:** derivar `n_resultados` do Obsidian do orçamento em vez do `min(6, ...)` fixo, ou remover `obsidian_chars` do `.env.example` e assumir a contagem de chunks como o knob real.
- **Invalida artefatos:** não

### [MENOR] Limiar de confiança da memória (0,70) é hardcoded, autorreportado pelo LLM e o próprio prompt sugere 0.0
- **Onde:** `/home/user/mestrado-utfpr/src/conhecimento/memoria_persistente.py:215`; `/home/user/mestrado-utfpr/src/conhecimento/multiagente.py:281` e `:382`
- **Evidência:** `if confianca < 0.70: raise MemoriaInvalida("Confianca insuficiente...")`. O valor vem de `float(candidato.get("confianca", 0.0))`, isto é, do JSON que o próprio auditor gera. Os dois prompts do porteiro trazem o campo no template como literal `"confianca": 0.0` e **em nenhum lugar dizem ao modelo o que a escala significa nem que abaixo de 0,70 o item é descartado**. `0.70` não aparece no `.env.example` nem em `docs/`. Os demais gatilhos de memória estão documentados (`CONSOLIDAR_LIMITE_INTERACOES=15`, `CONSOLIDAR_DIAS_ACUMULACAO=3`, `AL_IADO_CONSOLIDAR_A_CADA=6` em `src/interface/ciclo_chat.py:87`) — este é o único que decide aceitação e está fora.
- **Por que importa:** um modelo conservador que preencha 0.5 zera silenciosamente a consolidação automática (`consolidar_memoria_das_sessoes`, que roda sem gatilho explícito) e ninguém percebe: a rejeição vira apenas `rejeitadas += 1`. É o parâmetro que decide o que o agente lembra entre sessões.
- **Correção:** declarar o critério dentro do prompt ("use ≥0,70 apenas quando houver evidência literal") e expor o limiar como env documentada, com log de nível WARNING quando um candidato válido cai só por confiança.
- **Invalida artefatos:** não

### [MENOR] Docs afirmam 500/50 para sessões e memórias; o vault usa 900/100, 1200/120 e 1600/160 — e o chunk de literatura nunca sobrepõe entre páginas
- **Onde:** `/home/user/mestrado-utfpr/src/conhecimento/obsidian.py:371-377`; `/home/user/mestrado-utfpr/src/conhecimento/indexador.py:581-588`; `CLAUDE.md:445`; `docs/glossario.md:99`
- **Evidência:** `dividir_nota` escolhe `(900,100)` para `sessao_atual`/`sessao_arquivada`, `(1200,120)` para `memoria_consolidada`/`memoria_validada`, `(1600,160)` para o resto. `CLAUDE.md:445` diz "Sessões e memórias usam chunks menores (500/50)" e `docs/glossario.md:99` repete "sessões/memórias: 500/50". O par 500/50 (`src/core/config.py:158-159`) só é usado em `indexador.py:761`, o caminho legado de sessões — que `agente_contexto.py:167` só aciona como fallback quando o vault não respondeu. `test_consistencia_docs.py:61` valida apenas `TAMANHO_CHUNK_LITERATURA`, então a divergência passa. Separadamente: a literatura é chunkada **por página** (`for texto_pag, num_pag in paginas: dividir_em_chunks(texto_pag, 1800, 200)`), logo a sobreposição de 200 chars nunca cruza a quebra de página — uma frase, tabela ou equação partida entre p. 7 e p. 8 fica sem nenhum chunk que a contenha inteira.
- **Por que importa:** o doc descreve um índice que não existe, e a quebra dura de página cria pontos cegos de recuperação em exatamente o material mais citável (tabelas de NPR, equações).
- **Correção:** corrigir os dois docs para os valores reais e estender `test_chunk_de_literatura_documentado_bate_com_o_codigo` aos três pares do Obsidian; para a quebra de página, permitir carregar a sobreposição da página anterior ao iniciar a próxima.
- **Invalida artefatos:** não

---


## C. PRs #89–#103 — premissas metodológicas

## Resumo

Nenhuma das seis premissas metodológicas foi quebrada **em substância** nos PRs #89–#103: o protocolo E2 segue com injeção FMECA no sinal, o limiar continua congelado em bloco de calibração disjunto com purga de 2 janelas, o NPR oficial (315/90/30) segue idêntico a `docs/fmeca.md`, e as ressalvas E2≠E3 foram até reforçadas (unidade `passo_sintetico_de_degradacao`). O que quebrou foi a **cadeia de rastreabilidade**: refatorações cosméticas deixaram as três etapas E2 em `stale` sem re-execução, e o caminho de `stale` apaga artefatos versionados antes de checar se há como recalculá-los. Além disso, a comparação publicada que o CLAUDE.md declara "fonte única" não tem manifesto nenhum e é 7 dias mais velha que o retreino do Autoencoder.

## Achados

### [CRITICO] `stale` apaga os artefatos E2 versionados antes de verificar se há como recalculá-los
- **Onde:** `/home/user/mestrado-utfpr/src/ml/pipeline.py:567-569` (limpeza) e `:571-580` (checagem de dependências, que só ocorre **depois**); `limpar_artefatos` em `:453-460`
- **Evidência:** medi o estado real do HEAD com `estado_etapa_completo`:
  `injecao_falhas`, `validacao` e `rul_weibull` → `stale`; `features_ca` e `autoencoder` → `pending` (parquet, `.pt` e `.pkl` são gitignored, linhas 17/21 do `.gitignore`). Com `auto_deps=True` (default em `pipeline.py:526`), um pedido "rode a validação" cai em `stale` → `limpar_artefatos("validacao")` → `path.unlink()` em 13 arquivos versionados (`validacao_report.json`, `validacao_tabela.csv/md`, 5 PNGs, `weibull_results.json`, `weibull_tabela.csv`, 3 PNGs do Weibull) → só então tenta rodar `features_ca`, que falha sem `dados/brutos/`.
- **Por que importa:** na nuvem (e em qualquer clone sem o dataset) o pedido mais natural do pesquisador destrói exatamente os artefatos E2 que sustentam o capítulo de resultados, e falha em seguida. No PC recupera-se com `git checkout`; no Streamlit Cloud só no próximo redeploy. Antes desta faixa as três etapas estavam `ready` no PC, então o ramo destrutivo não era alcançável localmente — foram as refatorações de #97–#102 que o abriram.
- **Correção:** mover `limpar_artefatos` para **depois** de `dependencias_pendentes` e de confirmar que a etapa tem como executar; e nunca limpar quando alguma dependência estiver `pending`.
- **Invalida artefatos:** não (mas pode apagá-los)

### [SERIO] Os três artefatos E2 publicados não correspondem a nenhuma versão do código que está no repositório
- **Onde:** manifestos `resultados/manifestos/{injecao_falhas,validacao,rul_weibull}.json` × `src/ml/{injecao_falhas,validacao,rul_weibull}.py`
- **Evidência:** os três manifestos foram emitidos pela última vez em `332bc53` (`created_at` 2026-08-05T01:39:59–01:40:00). Depois disso, na mesma faixa, os três módulos mudaram: `0fa6638` (troca do `_log` local por `adaptar_logger_como_print`), `7f5be7a` (extração de `plotar_ttf_histogramas`/`plotar_confiabilidade`/`plotar_rul` para `src/ml/graficos_rul.py`, −251 linhas de `rul_weibull.py`) e `06ca15b` (strings FMEA→FMECA). O `comparar()` de `proveniencia.py:167` acusa hoje, para as três: `código da etapa alterado`, `dependência científica alterada`, `artefato upstream regenerado`, `artefato de saída alterado`.
- **Por que importa:** o manifesto é a prova de rastreabilidade que a banca pode cobrar. Hoje ele aponta para um `code_sha256` que não existe mais em lugar nenhum do repositório — não dá para reproduzir os números publicados a partir do código publicado, mesmo que os números estejam corretos (as mudanças são comprovadamente cosméticas). Nenhum dos documentos de auditoria da faixa registra esse efeito colateral.
- **Correção:** re-executar as três etapas pelo `pipeline.executar_etapa(..., force=True)` no PC (o único caminho que grava manifesto) e commitar artefatos + manifestos juntos, no mesmo commit. Enquanto isso não ocorrer, declarar no texto que os artefatos E2 são de 05/08 01:00 e que o código foi refatorado depois sem recálculo.
- **Invalida artefatos:** não (invalida a rastreabilidade, não os números)

### [SERIO] `output_artifacts` é hasheado em bytes crus enquanto `code_sha256` normaliza LF — o manifesto é inverificável fora do PC do pesquisador
- **Onde:** `/home/user/mestrado-utfpr/src/ml/proveniencia.py:130` (`_hashes_arquivos(outputs)` sem `texto_normalizado`) contra `:123-126` (`code_sha256` e `code_dependencies` com `texto_normalizado=True`)
- **Evidência:** conferi os 5 manifestos contra os arquivos do repositório: **todos** os hashes de saída divergem. A causa é CRLF, e é exata — para `limiar.json`, sha256 do arquivo como está no repo = `2f20f33162823725…`, sha256 do mesmo conteúdo com `\n`→`\r\n` = `5f16fe3d9f2c274a…`, que é **exatamente** o valor gravado no manifesto. Idem para `calibracao_autoencoder.csv` (`db6bc081…` LF vs `1df764cf…` CRLF = manifesto). Não há `.gitattributes` no projeto.
- **Por que importa:** qualquer auditoria externa (orientadora, banca, CI, a própria nuvem) que baixar o repositório e recalcular os hashes vai encontrar 100% de divergência e concluir "artefato adulterado", quando o problema é só fim de linha. É também parte da razão de as etapas nunca chegarem a `ready` fora da máquina do pesquisador. E, na direção oposta, esse ruído mascara adulteração real: um hash divergente hoje não distingue CRLF de edição manual.
- **Correção:** hashear saídas de texto com `sha256_arquivo_texto_normalizado` (mesma função já usada para código), mantendo bytes crus só para binários (`.pt`, `.pkl`, `.npz`, `.png`, `.parquet`); ou adicionar `.gitattributes` com `* text=auto eol=lf`. Reemitir os manifestos pela execução, nunca à mão.
- **Invalida artefatos:** não

### [SERIO] A comparação publicada — declarada "fonte única de resultado de anomalia" — não tem manifesto nem qualquer campo de proveniência, e é anterior ao retreino do Autoencoder
- **Onde:** `resultados/macro/comparacao_resultado.json`; consumida por `/home/user/mestrado-utfpr/src/conhecimento/ferramentas_academicas.py:111-112`; `src/ml/macro_comparar.py` não chama `registrar_manifesto`
- **Evidência:** as chaves de cada método no JSON publicado são apenas `['cor','falhas','fp_pct','limiar','n_aval','n_calib','nome','percentil','severidades']` — sem data, sem `git_commit`, sem hash do modelo, sem `FP_ALVO`. `git log` dá o último commit desses JSONs em **2026-07-26/29**; o `data_treino` de `limiar.json` é **2026-08-05T01:00:28** e o hash de `modelo_autoencoder.pt` mudou no manifesto (`0a1fbd0…` → `488ff80…`) na mesma faixa. Não existe `resultados/manifestos/macro*.json`. No mesmo intervalo, `b1dfb91` apagou `src/ml/comparacao_literatura.py` e `06ca15b` apagou `resultados/comparacao/comparacao_literatura.{json,png}` — o único comparativo que ainda carregava `gerado_em` e `evidence_level` no próprio artefato.
- **Por que importa:** o número que vai à banca (AUC 0,978 × 0,909; SMD 0,50 × 1,00 no IGBT) não tem como ser datado nem amarrado a um modelo, e nada no sistema detecta que ele envelheceu. A tabela `.csv`/`.md` chegou a ser reescrita em `e9c6fc5` sem re-execução (só troca de cabeçalho: `tpr_sev1`→`tpr_fpr10_sev1`); regenerei a CSV a partir do JSON publicado com o código atual e ela bate byte a byte — então **não há número falso** —, mas o PNG e o JSON não foram tocados no mesmo commit, o que confirma que não houve rodada.
- **Correção:** fazer `macro_comparar` gravar um manifesto (código, `git_commit`, hash de `modelo_autoencoder.pt`/`limiar.json`, `FP_ALVO`, `n_calib`/`n_aval`) e a ferramenta `consultar_comparacao_macro` exibir a data e avisar quando o hash do AE publicado não bater com o do artefato atual.
- **Invalida artefatos:** não, mas impede datar a comparação

### [MENOR] O relatório E2 anuncia `threshold_method: "p99"` enquanto o ponto operacional real é p99,9 com 10,2% de FP no teste
- **Onde:** `resultados/autoencoder/validacao_report.json:5` e `resultados/manifestos/autoencoder.json:22`; o valor real está em `resultados/autoencoder/limiar.json` (`threshold_effective_percentile: 99.9`, `fp_test_pct: 10.227…`)
- **Evidência:** `validacao_report.__meta__` traz `"threshold_method": "p99"`, `"score_threshold": 7.826…` e a frase "Limiar CONGELADO … NÃO otimizado no teste" — verdadeira quanto ao congelamento, mas o `limiar_p99` do MSE é `2.545`, e o 7,826 vem do percentil **99,9** do escore localizado. O FP medido no bloco de teste na mesma execução é 10,23% (9 de 88 janelas). `docs/decisao_fpr_1pct.md:89`, criado nesta faixa, registra o fato ("p99,9 escolhido, FPR de 10,2% no teste"), mas a informação não foi propagada para o artefato nem para o manifesto.
- **Por que importa:** quem ler apenas o `validacao_report.json` — que é o artefato citável do capítulo — recebe "p99" e nenhum falso positivo, e conclui um ponto de operação bem mais conservador do que o efetivamente usado. É a diferença entre 1% e 10% de alarme falso na leitura da orientadora.
- **Correção:** copiar `threshold_effective_percentile`, `fp_calib_pct` e `fp_test_pct` do `limiar.json` para o `__meta__` do `validacao_report.json` e para `parameters` do manifesto; renomear o campo legado para `threshold_family` ou similar, já que `test_limiar.py` fixa `threshold_method == "p99"` como contrato.
- **Invalida artefatos:** não

---
