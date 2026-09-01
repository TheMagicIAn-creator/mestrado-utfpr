# ALIAdo — Especificação Mestre de Implementação para Codex

**Projeto:** Mestrado UTFPR — ALIAdo  
**Repositório:** `TheMagicIAn-creator/mestrado-utfpr`  
**Finalidade deste documento:** instruir o Codex a auditar, alinhar e implementar no repositório a arquitetura e a metodologia atualmente aprovadas pelo pesquisador, sem reabrir decisões já tomadas, sem apagar resultados válidos e sem introduzir complexidade sem justificativa científica ou de engenharia.

---

## 0. INSTRUÇÃO PRINCIPAL AO CODEX

Este documento é uma **especificação executiva** e não apenas material de consulta.

Ao recebê-lo, o Codex deve:

1. ler integralmente este documento antes de modificar qualquer arquivo;
2. inspecionar o estado atual do repositório e identificar divergências entre esta especificação e o código/documentação existentes;
3. preservar as regras essenciais de segurança, sem expor credenciais em código, logs, prompts ou artefatos;
4. não substituir o `AGENTS.md` atual por este documento;
5. armazenar esta especificação no repositório, preferencialmente em:
   `docs/exec-plans/active/ALIADO_ALINHAMENTO_ARQUITETURA_METODOLOGIA.md`;
6. acrescentar ao `AGENTS.md` apenas um ponteiro conciso para esta especificação, sem remover as instruções existentes;
7. executar a implementação em fases, validando testes e consistência documental após cada fase;
8. não considerar documentação antiga mais autoritativa que as decisões aqui consolidadas;
9. não fabricar dados, parâmetros estatísticos, referências, resultados de ML ou resultados de confiabilidade para “fechar” a implementação;
10. quando uma decisão científica depender de evidência ainda não existente, implementar a infraestrutura de forma correta e marcar o ponto como **pendente de validação**, sem inventar valores;
11. concluir o trabalho com um relatório objetivo contendo:
    - arquivos modificados;
    - decisões implementadas;
    - divergências encontradas;
    - testes executados;
    - resultados dos testes;
    - pendências metodológicas reais;
    - qualquer ponto que ainda exija decisão explícita do pesquisador.

A tarefa não é “propor uma nova arquitetura”. A tarefa é **implementar e alinhar a arquitetura e a metodologia definidas neste documento**.

---

# 1. HIERARQUIA DE AUTORIDADE

Em caso de conflito, obedecer à seguinte ordem:

## P0 — Decisões explícitas do pesquisador

As decisões do pesquisador consolidadas neste documento têm prioridade funcional sobre versões anteriores do código, documentação, Work, Gemini ou decisões tomadas por agentes.

Isso significa que o Codex não deve preservar uma decisão antiga apenas porque ela já está implementada.

## P1 — Integridade científica e rastreabilidade

Nenhuma decisão do pesquisador deve ser “implementada” mediante fabricação de evidência.

Quando o objetivo desejado exigir dados que o projeto ainda não possui, o Codex deve:

- preservar o objetivo;
- implementar a interface/pipeline necessária quando útil;
- impedir geração de resultado científico inválido;
- registrar explicitamente o dado ou fundamento ausente.

## P2 — Consolidação do Work

Utilizar como contexto secundário as decisões consolidadas no ambiente Work:
- Dense Autoencoder versus AE-LSTM;
- GPVS-Faults;
- top-k;
- `k=5` e p99,9 somente como referência histórica reproduzível;
- grade descritiva `k={5,10,20}` por `{p99,p99,5,p99,9}`;
- separação estrita entre métricas dos detectores e FMECA;
- RCM + FMECA;
- necessidade de consistência entre código, resultados e dissertação.

Esses elementos não podem ser promovidos a “verdade científica” sem rastreabilidade.

## P3 — Arquitetura consolidada com Gemini / relatório de arquitetura

Adotar a nova arquitetura:
- dois provedores operacionais: **OpenAI + Google Gemini**;
- Provider Gateway;
- Router;
- modelos como recursos de inferência;
- roteamento por tarefa, custo, qualidade, latência e risco metodológico;
- validação cruzada entre provedores quando a criticidade justificar;
- suporte estrutural para provedores futuros sem integrar Anthropic agora.

## P4 — Estado atual do repositório

O código atual é ponto de partida, não fonte final de verdade.

Documentos como:
- `docs/arquitetura.md`;
- `docs/metodologia_ml.md`;
- `docs/fmeca.md`;
- `docs/mapa_de_resultados.md`;
- `docs/memoria_agentes.md`;
- `src/conhecimento/provedores.py`;
- `src/conhecimento/multiagente.py`;
- `src/conhecimento/agente.py`;
- `src/webapp/agent_adapter.py`;

devem ser atualizados se estiverem incompatíveis com esta especificação.

---

# 2. OBJETIVO CIENTÍFICO DO PROJETO

O ALIAdo é um assistente de pesquisa e uma plataforma de apoio ao mestrado em Engenharia Elétrica da UTFPR.

O núcleo científico é:

> **análise preditiva / detecção de falhas em inversores fotovoltaicos on-grid, integrando RCM, FMECA e Machine Learning.**

A comparação principal de ML é:

- **Dense Autoencoder (AE Denso)** — modelo proposto/avaliado;
- **LSTM Autoencoder (AE-LSTM)** — referência comparativa associada à literatura de Ibrahim.

O dataset experimental atualmente adotado é:

- **GPVS-Faults**.

O dataset Paderborn não deve voltar a ser dataset principal do pipeline sem uma decisão explícita posterior.

---

# 3. PERGUNTA CENTRAL DE VALIDAÇÃO

Toda decisão de ML, toda tabela principal e todo gráfico de validação devem ajudar a responder:

> **“Quantas falhas reais o meu modelo detecta sem gerar alarmes falsos demais?”**

Essa pergunta é o filtro metodológico central.

Qualquer métrica, figura ou processamento adicional deve ter pelo menos uma das seguintes funções:

- quantificar capacidade de detecção;
- quantificar falsos alarmes;
- explicar erro do modelo;
- comparar AE Denso e AE-LSTM;
- conectar o resultado à manutenção / confiabilidade;
- comprovar robustez, reprodutibilidade ou validade da comparação.

Se não cumprir nenhuma dessas funções, não deve ser promovido a saída principal.

---

# 4. PRINCÍPIO DE ENXUGAMENTO

O projeto não deve acumular métricas, gráficos, tabelas ou modelos “porque são comuns em ML”.

Regra:

> **Nada entra na camada principal de resultados sem justificativa metodológica, bibliográfica ou operacional.**

É aceitável manter métricas auxiliares internamente para teste/diagnóstico, desde que:

- não sejam apresentadas como pilares da dissertação;
- não confundam o usuário;
- não alterem a decisão de modelo sem justificativa;
- não gerem tabelas e painéis desnecessários.

---

# 5. NOVA ARQUITETURA LLM OBRIGATÓRIA

## 5.1. Princípio arquitetural

O ALIAdo não deve ser “um agente Gemini”.

A arquitetura correta é:

```text
                         ALIAdo
                            |
                      LLM / Router
                            |
                    Provider Gateway
                            |
             +--------------+--------------+
             |                             |
           OpenAI                         Google
             |                             |
        +----+----+                    +---+---+
        |    |    |                    |       |
      Luna Terra Sol              Flash-Lite Flash
```

Definição:

> **O Router é o cérebro operacional. Os modelos são recursos de inferência.**

## 5.2. Provedores operacionais

Implementar inicialmente:

- OpenAI;
- Google Gemini.

Não integrar Anthropic nesta fase.

Entretanto, o contrato deve permitir:

```python
gateway.register_provider(...)
```

ou equivalente, sem exigir reescrita do restante do agente.

## 5.3. Modelos lógicos

Utilizar aliases lógicos de capacidade:

### OpenAI
- `luna`
- `terra`
- `sol`

### Google
- `flash_lite`
- `flash`

Os aliases não devem obrigar hardcode de IDs de API eternos.

Os IDs concretos devem ser configuráveis por variáveis de ambiente ou registry.

Exemplo conceitual:

```text
AL_IADO_OPENAI_MODEL_LUNA
AL_IADO_OPENAI_MODEL_TERRA
AL_IADO_OPENAI_MODEL_SOL
AL_IADO_GEMINI_MODEL_FLASH_LITE
AL_IADO_GEMINI_MODEL_FLASH
```

A implementação deve utilizar apenas identificadores realmente disponíveis no ambiente/API.

## 5.4. Contrato unificado

Criar uma abstração equivalente a:

```python
@dataclass
class LLMRequest:
    task_type: str
    messages: list
    context: object | None
    tools: list | None
    structured_output: object | None
    reasoning_level: str | None
    multimodal: bool
    methodological_risk: str
    max_cost: float | None
    max_latency: float | None
```

e uma resposta equivalente a:

```python
@dataclass
class LLMResult:
    content: str
    provider: str
    model: str
    task_type: str
    latency_ms: float | None
    estimated_cost: float | None
    fallback_used: bool
    validation_used: bool
```

Os nomes podem ser adaptados ao padrão atual do repositório, mas os conceitos devem existir.

## 5.5. Roteamento inicial

Política inicial:

| Tipo de tarefa | Preferência operacional |
|---|---|
| conversa simples / consulta factual curta | OpenAI Luna |
| extração documental em volume | Gemini Flash-Lite |
| documento multimodal / tabelas / figuras | Gemini Flash |
| raciocínio técnico-científico | OpenAI Terra |
| raciocínio crítico excepcional | OpenAI Sol |
| cálculo determinístico | sem LLM |
| BM25 / RRF / hashing / métricas | sem LLM |

O Router não deve transformar essa tabela em uma “hierarquia universal de inteligência”.

A seleção deve ser baseada em requisitos da tarefa.

## 5.6. Escalonamento

Fluxo recomendado:

```text
modelo econômico
      |
 resolve com confiança suficiente?
      |
   sim -> resposta
   não
      |
 modelo superior / mais adequado
```

Para tarefa metodológica de alto risco:

```text
Terra
  |
conclusão crítica
  |
Gemini como revisor independente
  |
concordância?
  |       |
 sim     não
  |       |
final   escalonamento / Sol / sinalização de conflito
```

A validação cruzada deve ser seletiva. Não duplicar toda chamada em dois provedores.

## 5.7. Fallback

Separar:

1. retry de erro transitório;
2. fallback para outro modelo do mesmo provedor;
3. fallback para outro provedor;
4. escalonamento por complexidade.

Não confundir fallback por indisponibilidade com escolha por qualidade.

## 5.8. Modelos operacionais e experimentais

Manter registry com status:

```text
operational
experimental
disabled
```

Um modelo experimental só entra no caminho crítico depois de avaliação.

## 5.9. Observabilidade

Registrar, sem expor segredos ou conteúdo sensível:

- provider;
- modelo;
- task type;
- motivo do roteamento;
- fallback;
- latência;
- custo estimado quando possível;
- validação cruzada;
- erro;
- tentativas.

Não gravar API keys.

## 5.10. Refatoração do estado atual

O arquivo atual `src/conhecimento/provedores.py` declara uma “Equipe 100% Gemini”.

Esse acoplamento deve ser removido.

Preferência estrutural:

```text
src/conhecimento/
    provedores/
        __init__.py
        base.py
        openai.py
        gemini.py
        registry.py
    roteador_llm.py
    contratos_llm.py
```

Se uma migração total criar risco desnecessário, manter uma fachada compatível em `provedores.py`, mas toda chamada nova deve passar pelo Gateway/Router.

Nenhum módulo de negócio deve depender diretamente do SDK OpenAI ou Gemini se puder depender do contrato comum.

---

# 6. RAG, MEMÓRIA E RACIOCÍNIO CIENTÍFICO

Preservar:

- ChromaDB;
- busca lexical;
- RRF / reranking quando existentes;
- memória validada;
- integração Obsidian quando funcional;
- rastreabilidade das fontes.

Regra fundamental:

> O LLM não substitui o mecanismo de evidências.

Em consulta científica:

```text
pergunta
  -> recuperação
  -> pacote de evidências
  -> modelo
  -> auditoria quando necessária
  -> resposta com rastreabilidade
```

Cálculo determinístico deve permanecer fora do LLM.

---

# 7. DATASET E PROTOCOLO DE ML

## 7.1. Dataset oficial

O dataset ativo da comparação é o **GPVS-Faults**.

Antes de modificar o pipeline:

- confirmar arquivos esperados;
- confirmar classes/ensaios;
- confirmar frequência de amostragem;
- confirmar colunas;
- confirmar processo de janelamento;
- confirmar ausência de leakage.

## 7.2. Modelos

Comparação obrigatória:

1. AE Denso;
2. AE-LSTM.

Nenhum terceiro modelo deve ser colocado como competidor principal sem decisão explícita.

Outros algoritmos podem existir em código legado ou laboratório, mas não devem diluir a comparação central.

## 7.3. Justiça experimental

A comparação precisa ser controlada.

Garantir, tanto quanto tecnicamente aplicável:

- mesma base de dados;
- mesmo split;
- mesma política de normalização;
- treinamento apenas com informação permitida;
- mesma regra de uso de holdout;
- orçamento comparável de treinamento;
- mesmas sementes quando comparáveis;
- threshold calibrado sem olhar o conjunto de falhas de teste;
- nenhuma seleção de arquitetura usando resultado final do holdout.

O AE-LSTM pode exigir sequências; isso não autoriza vazamento temporal.

---

# 8. MÉTRICAS DE VALIDAÇÃO — PRIORIDADE DO PESQUISADOR

## 8.1. Métricas principais

A camada principal deve priorizar:

1. **Recall**
2. **F1 Score**
3. **Precision**
4. **AUC** como métrica complementar

Além disso:

5. **Matriz de Confusão é obrigatória.**

## 8.2. Interpretação

Recall responde principalmente:

> “Das falhas existentes, quantas foram detectadas?”

Precision responde:

> “Dos alarmes gerados, quantos realmente correspondiam a falha?”

F1 equilibra Recall e Precision.

A matriz de confusão torna TP, FP, TN e FN transparentes.

## 8.3. Acurácia

Acurácia não deve ser critério principal.

Em cenário de classes desbalanceadas, pode ser enganosa.

Se continuar sendo calculada por compatibilidade, deve aparecer apenas como auxiliar.

## 8.4. AUC

O código deve ser capaz de distinguir explicitamente:

- ROC-AUC;
- PR-AUC.

Não chamar genericamente qualquer área de “AUC” em artefatos científicos sem informar qual curva a originou.

O pesquisador não deseja que uma curva ROC seja obrigatoriamente uma figura principal.

Portanto:

- AUC pode ser numérica;
- ROC curve não é saída obrigatória;
- PR curve não é saída obrigatória;
- curvas só entram como principais se houver justificativa bibliográfica/metodológica.

O repositório atual trata AUC-PR como métrica principal. Essa decisão deve ser rebaixada da condição de “principal obrigatória” até que a literatura consolidada confirme essa preferência ou o pesquisador a aprove novamente.

## 8.5. Alarmes falsos

A pergunta central exige quantificação de falsos alarmes.

Portanto, mesmo sem promover FPR a “métrica principal”, o pipeline deve registrar:

- FP;
- TN;
- taxa de falso positivo quando matematicamente pertinente;
- número/razão de alarmes falsos na condição saudável;
- unidade de análise (janela, ensaio, sequência etc.).

Não esconder falsos positivos atrás de um F1 agregado.

---

# 9. MATRIZES DE CONFUSÃO — NÃO REMOVER

As matrizes de confusão foram removidas em uma alteração anterior e devem voltar.

Requisitos mínimos:

- matriz para AE Denso;
- matriz para AE-LSTM;
- rótulos explícitos;
- TP, FP, TN, FN rastreáveis;
- versão normalizada somente como adicional, nunca substituindo completamente os valores absolutos;
- dados-fonte tabulares;
- PNG acadêmico;
- PDF vetorial quando a infraestrutura de publicação já exigir;
- mesma convenção de eixo nos dois modelos.

Se for metodologicamente válido, incluir:

- matriz agregada;
- tabela por ensaio/falha.

Não gerar dezenas de matrizes redundantes sem necessidade.

---

# 10. TABELAS DE RESULTADOS

As tabelas devem ser estáveis, acadêmicas e comparáveis.

## 10.1. Tabela principal AE Denso x AE-LSTM

Colunas mínimas recomendadas:

| Modelo | Recall | F1 | Precision | AUC (tipo explícito) | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Se houver IC95%, informar método.

## 10.2. Tabela por falha/ensaio

Quando houver suporte dos dados:

| Falha/ensaio | Modelo | Recall | F1 | Precision | FP | FN |
|---|---|---:|---:|---:|---:|---:|

## 10.3. Regras

- não arredondar internamente antes de calcular;
- registrar unidade;
- manter número de casas consistente na apresentação;
- não misturar percentuais e proporções sem indicação;
- não preencher célula ausente com zero;
- usar `N/A`/`não aplicável` quando apropriado;
- toda tabela publicada deve ter dados-fonte versionáveis.

---

# 11. LIMIAR DE DETECÇÃO

Implementar:

- threshold configurável;
- método de calibração rastreável;
- p99,9 como referência histórica reproduzível;
- grade pré-fixada p99, p99,5 e p99,9;
- proibição de escolher o threshold olhando o resultado final de falhas do holdout.

Exemplo:

```text
threshold_method = healthy_percentile
threshold_percentile = 99.9
```

A saída deve registrar o threshold real de cada modelo.

---

# 12. ESCORE TOP-K

Implementar como componente configurável e reproduzível.

Regras:

- `k=5` é referência histórica, não constante científica universal;
- a grade descritiva usa `k={5,10,20}`;
- documentar fórmula;
- documentar qual dimensão recebe top-k;
- justificar ou avaliar alternativas;
- não selecionar `k` usando o holdout final;
- produzir ablação somente se ela responder a uma pergunta metodológica real.

Se o pipeline atual não usa top-k, não inserir silenciosamente sem testes de regressão e documentação.

---

# 13. CONFIABILIDADE E MANUTENÇÃO

## 13.1. Funções fundamentais

A implementação deve distinguir corretamente:

\[
R(t) = P(T > t)
\]

\[
F(t) = P(T \le t) = 1-R(t)
\]

\[
f(t) = \frac{dF(t)}{dt}
\]

\[
h(t) = \lambda(t) = \frac{f(t)}{R(t)}
\]

Não usar `f(lambda(t))` como sinônimo de densidade de falha.

## 13.2. Gráficos desejados pelo pesquisador

O sistema deve suportar, quando houver fundamento/dados:

- curva de confiabilidade `R(t)`;
- curva acumulada de falha `F(t)`;
- densidade de falha `f(t)`;
- taxa de falha / hazard `h(t)` ou `lambda(t)`;
- distribuição de eventos/falhas quando houver observações reais;
- Weibull de 2 parâmetros;
- distribuição Gaussiana/Normal somente quando justificada;
- Lognormal somente quando justificada.

## 13.3. Restrição científica

O GPVS-Faults não deve ser tratado como base de vida útil se ele não possui:

- tempos individuais até falha;
- exposição de frota;
- censura;
- histórico de sobrevivência por ativo.

Portanto:

> Não ajustar Weibull, Normal, Lognormal, bathtub curve ou RUL ao GPVS apenas para gerar figuras.

Se a literatura fornecer parâmetros Weibull válidos para IGBT, a curva pode ser
implementada como **cenário bibliográfico**, com fonte, parâmetros e hipótese
explicitamente registrados. Sem `beta` e `eta` rastreáveis, permanece bloqueada.

## 13.4. Histograma e “distribuição de taxa de falha”

Se os dados disponíveis forem apenas participações de chamados/ocorrências por componente:

- chamar de frequência, participação ou distribuição de ocorrências;
- não chamar automaticamente de distribuição de taxa de falha.

Um histograma de tempo até falha só deve existir com observações de tempo até falha.

## 13.5. Princípio de preservação

Se um gráfico desejado não puder ser cientificamente produzido agora:

- não remover sua capacidade do projeto;
- manter função/interface/teste quando útil;
- registrar “não gerado: dados insuficientes”;
- não inventar dados.

---

# 14. COMPONENTES CRÍTICOS E FMECA

A análise deve preservar a conexão com componentes internos do inversor.

Escopo canônico vigente:

- IGBT, associado a F1;
- sistema de sensor/realimentação, associado a F2;
- sistema/circuito de controle do inversor, associado a F6/F7.

F6/F7 são anomalias funcionais do controle, não falhas físicas de PCB. O
recorte histórico do TCC não transfere valores para o novo escopo.

A FMECA não deve ser reduzida a “falha de tensão”, “falha de corrente” ou anomalias genéricas de sinal.

Sinais são observáveis.

Modo de falha pertence ao componente/sistema.

O Codex deve preservar rastreabilidade:

```text
componente
 -> função
 -> modo de falha
 -> efeito
 -> observável
 -> detector
 -> resultado
 -> impacto de manutenção
```

---

# 15. FMEA x FMECA E NPR

Não atribuir genericamente NPR à FMEA quando o trabalho está utilizando FMECA.

Preservar os campos:

- Severidade `S`;
- Ocorrência `O`;
- Detectabilidade `D`;
- NPR conforme escala/metodologia adotada.

Enquanto o pesquisador não fornecer valores e fontes compatíveis com os três
itens atuais, todos permanecem `null`, com estado `awaiting_user_fmeca`.

Não confundir `D` com NPR.

Não sobrescrever os valores originais da FMECA.

---

# 16. SEPARAÇÃO ENTRE DETECTOR E FMECA

A extensão que transformava desempenho do detector em detectabilidade ordinal
foi revogada pela decisão metodológica de 2026-09-01. Ela não deve permanecer
na publicação científica vigente nem ser tratada como capacidade futura.

Recall, F1, Precision, matrizes de confusão e falso positivo saudável descrevem
os detectores. Nenhuma dessas métricas altera S, O, D ou NPR.

---

# 17. MÉTRICAS DO MONITORAMENTO

As métricas experimentais devem manter unidade inferencial, denominador,
estimativa e incerteza explícitos. O ensaio é a unidade do bootstrap E3; janelas
agregadas em matrizes têm uso descritivo.

Não converter automaticamente Recall global em probabilidade física de
detecção por componente ou em escala de manutenção.

---

# 18. RESULTADOS E FIGURAS

## 18.1. Família principal de ML

Obrigatório:

- tabela comparativa AE Denso x AE-LSTM;
- Recall;
- F1;
- Precision;
- AUC com tipo explícito;
- matriz de confusão;
- falsos positivos/alarmes falsos;
- resultados por falha/ensaio quando suportados.

## 18.2. Família de confiabilidade

Obrigatório quando sustentado por dados/parâmetros:

- `R(t)`;
- `F(t)`;
- `f(t)`;
- `h(t)`/`lambda(t)`;
- cenários por componente;
- parâmetros e fontes.

## 18.3. Estilo acadêmico

Manter o contrato atual de publicação quando aplicável:

- PNG 300 dpi;
- PDF vetorial;
- dados-fonte;
- JSON metodológico;
- manifesto;
- hash/proveniência.

## 18.4. Não recalcular no frontend

O frontend não deve recalcular métricas científicas.

Ele apenas lê artefatos publicados.

---

# 19. FRONTEND / PAINÉIS

Painéis devem refletir o núcleo metodológico.

Prioridade:

1. comparação AE Denso x AE-LSTM;
2. matrizes de confusão;
3. confiabilidade/manutenção;
4. rastreabilidade;
5. estado do pipeline.

Não transformar o frontend em dashboard genérico de ML.

Se métricas auxiliares permanecerem internamente, não precisam ocupar espaço principal.

---

# 20. ALTERAÇÕES DOCUMENTAIS OBRIGATÓRIAS

Após implementar o código, revisar no mínimo:

- `README.md`;
- `docs/arquitetura.md`;
- `docs/metodologia_ml.md`;
- `docs/datasets.md`;
- `docs/fmeca.md`;
- `docs/mapa_de_resultados.md`;
- `docs/memoria_agentes.md`;
- `docs/reproducibilidade.md`;
- `docs/comandos.md`;
- `.env.example`;
- `src/README.md` se existir;
- qualquer documento que ainda declare “100% Gemini”.

Objetivo:

> não pode haver duas arquiteturas “canônicas” contraditórias dentro do repositório.

---

# 21. DOCUMENTAÇÃO DO CODEX

O `AGENTS.md` deve permanecer limitado às instruções essenciais do projeto e ao ponteiro para esta especificação.

Não adicionar scanners externos obrigatórios ao fluxo local de leitura de arquivos.

Após incorporar este documento ao repositório, acrescentar apenas uma seção curta semelhante a:

```markdown
## Projeto ALIAdo — especificação ativa

Antes de alterar arquitetura LLM, pipeline ML, FMECA, confiabilidade,
resultados ou documentação científica, leia:

`docs/exec-plans/active/ALIADO_ALINHAMENTO_ARQUITETURA_METODOLOGIA.md`
```

Não copiar este documento inteiro para o `AGENTS.md`.

---

# 22. SEGURANÇA E SEGREDOS

Preservar o scanner de segredos exigido pelo repositório.

Nunca:

- commitar API key;
- imprimir segredo em log;
- colocar chave em fixture;
- salvar `.env` real;
- enviar chave para modelo;
- copiar segredo para documentação.

Atualizar somente `.env.example` com nomes de variáveis e placeholders.

---

# 23. TESTES — ARQUITETURA LLM

Criar/atualizar testes para:

- seleção do Router;
- roteamento por task type;
- multimodal;
- fallback mesmo provedor;
- fallback cruzado;
- retry transitório;
- erro permanente;
- registry;
- modelo experimental não utilizado por padrão;
- structured output;
- compatibilidade da fachada antiga;
- logs sem segredo;
- nenhuma chamada real de API em teste unitário;
- mocks OpenAI;
- mocks Gemini.

---

# 24. TESTES — MACHINE LEARNING

Validar:

- split sem leakage;
- scaler ajustado apenas no conjunto permitido;
- threshold calibrado no conjunto permitido;
- p99,9 configurável;
- top-k configurável;
- AE Denso;
- AE-LSTM;
- Recall;
- Precision;
- F1;
- matriz de confusão;
- ROC-AUC quando usado;
- PR-AUC quando usado;
- FP/TN/FN/TP;
- igualdade da convenção positivo=anomalia/falha;
- reprodutibilidade de sementes;
- artefatos publicados;
- manifestos.

---

# 25. TESTES — CONFIABILIDADE E FMECA

Validar matematicamente:

\[
F(t)=1-R(t)
\]

e, para modelos parametrizados válidos:

\[
f(t) = \frac{dF}{dt}
\]

\[
h(t)=\frac{f(t)}{R(t)}
\]

Validar:

- parâmetros positivos;
- unidades;
- monotonicidade de `R(t)` quando aplicável;
- `F(t)` entre 0 e 1;
- consistência de fonte;
- S/O/D/NPR nulos enquanto faltar decisão do pesquisador;
- ausência de qualquer conversão de métrica do detector em criticidade FMECA.

---

# 26. QUALIDADE DE CÓDIGO

Manter ou melhorar:

- cobertura de código novo >= 80%;
- zero nova duplicação relevante;
- nenhuma nova vulnerabilidade;
- nenhuma nova exposição de segredo;
- nenhum erro de lint introduzido;
- funções pequenas e testáveis;
- separação de domínio, infraestrutura e apresentação.

Não perseguir cobertura artificial por testes sem valor.

---

# 27. COMANDOS DE VERIFICAÇÃO

Utilizar os comandos canônicos existentes no repositório e atualizá-los quando necessário.

Executar, quando compatíveis:

```powershell
python scripts/verificar_projeto.py
python scripts/auditar_resultados.py
python scripts/avaliar_agente.py
python -m pytest -p no:cacheprovider -q -W ignore -m "not pesado"
python -m pytest -p no:cacheprovider -q -W ignore tests/test_torch_smoke.py tests/test_modelos_autoencoder_canonicos.py
python -m ruff check --select F821,F822,F823 src tests scripts
```

Adicionar verificações específicas da nova arquitetura sem quebrar a suíte existente.

---

# 28. ORDEM DE EXECUÇÃO

## Fase 0 — Preflight

Sem alterar código:

- verificar branch;
- verificar worktree;
- ler `AGENTS.md`;
- executar scanner obrigatório antes de ler arquivos, conforme instruções do repositório;
- mapear estrutura;
- localizar chamadas diretas ao Gemini;
- localizar qualquer chamada OpenAI existente;
- localizar métricas atuais;
- localizar geração de matrizes;
- localizar confiabilidade;
- localizar FMECA;
- localizar documentação divergente;
- rodar testes baseline.

Registrar o baseline.

## Fase 1 — Contratos e Provider Gateway

Implementar:

- contrato comum;
- registry;
- OpenAI provider;
- Gemini provider;
- compatibilidade;
- env vars;
- testes.

## Fase 2 — Router

Implementar:

- task classification explícita;
- regras iniciais;
- escalonamento;
- fallback;
- validação cruzada seletiva;
- observabilidade;
- testes.

## Fase 3 — Migração do agente

Remover acoplamento “100% Gemini”.

Migrar:

- conversa;
- auditoria;
- memória;
- extração documental;
- multimodal;
- raciocínio científico.

Não migrar cálculos determinísticos para LLM.

## Fase 4 — Alinhamento ML

Auditar o pipeline existente.

Corrigir:

- hierarquia de métricas;
- matrizes de confusão;
- falsos alarmes;
- threshold configurável;
- p99,9 e `k=5` como referência histórica;
- grade descritiva 3×3;
- tabelas;
- documentação.

Não quebrar justiça experimental.

## Fase 5 — Confiabilidade/FMECA

Restaurar/garantir:

- funções de confiabilidade;
- gráficos;
- FMECA atual com IGBT, sensor/realimentação e sistema de controle;
- S/O/D/NPR anuláveis, sem herança do recorte histórico;
- bloqueios científicos quando faltarem dados.

## Fase 6 — Frontend

Expor somente artefatos calculados no backend/pipeline.

Garantir presença das matrizes e tabelas principais.

## Fase 7 — Documentação

Eliminar contradições.

## Fase 8 — Qualidade

Executar testes, lint e auditorias.

## Fase 9 — Relatório final

Produzir resumo de implementação e lista de pendências reais.

---

# 29. CONFLITOS JÁ CONHECIDOS QUE DEVEM SER VERIFICADOS

O Codex deve procurar especificamente os seguintes desalinhamentos:

## C1 — Gemini-only versus OpenAI + Gemini

Estado antigo:
- `src/conhecimento/provedores.py` declara equipe 100% Gemini;
- `docs/arquitetura.md` menciona aquecimento de papéis Gemini.

Estado desejado:
- Provider Gateway;
- OpenAI + Gemini;
- Router.

## C2 — AUC-PR principal versus prioridade do pesquisador

Estado antigo:
- `docs/metodologia_ml.md` define AUC-PR como principal.

Estado desejado:
- Recall principal;
- F1;
- Precision;
- AUC complementar;
- matriz de confusão obrigatória.

## C3 — calibração e resolução empírica

Estado antigo:
- documentação atual registra p99.

Estado desejado:
- p99,9 como referência histórica;
- grade p99/p99,5/p99,9 com ordem e percentil efetivo;
- threshold configurável;
- sem otimização usando holdout final.

## C4 — Matrizes de confusão

Estado observado pelo pesquisador:
- foram retiradas.

Estado desejado:
- devem existir e permanecer.

## C5 — FMECA e métricas do detector

Estado antigo:
- detector não recalcula NPR.

Estado desejado:
- novo escopo FMECA com campos nulos até decisão do pesquisador;
- nenhuma transformação de desempenho do detector em NPR;
- recorte antigo preservado somente como histórico.

## C6 — Distribuições de confiabilidade

Estado atual:
- exponencial por limitação de dados.

Desejo do pesquisador:
- suporte a Weibull 2P e demais distribuições/gráficos relevantes.

Solução:
- suportar modelos com contrato explícito;
- não ajustar sem dados;
- usar cenário bibliográfico quando fonte fornecer parâmetros.

---

# 30. O QUE NÃO FAZER

Não:

- redesenhar o projeto inteiro sem necessidade;
- adicionar terceiro provedor nesta fase;
- adicionar dezenas de modelos;
- transformar o Router em LLM sem necessidade;
- usar LLM para cálculo determinístico;
- usar Paderborn como dataset principal;
- incluir algoritmos extras como novos competidores principais;
- remover matriz de confusão;
- usar Accuracy como critério principal;
- chamar AUC sem definir o tipo;
- ajustar threshold com dados de teste de falha;
- criar Weibull/Normal/Lognormal com dados inventados;
- inferir vida útil do GPVS sem dados de vida;
- transferir valores do recorte histórico para a FMECA vigente;
- inferir S/O/D/NPR a partir de métricas dos detectores;
- apagar resultados negativos;
- ocultar falsos positivos;
- quebrar rastreabilidade;
- remover verificações essenciais de testes, lint ou integridade;
- commitar segredos;
- criar documentação canônica contraditória;
- fazer refatoração estética sem benefício para a tarefa.

---

# 31. DEFINIÇÃO DE PRONTO — DEFINITION OF DONE

A tarefa só pode ser considerada concluída quando:

- o ALIAdo não estiver mais arquiteturalmente preso ao Gemini;
- OpenAI e Gemini passarem pelo mesmo Gateway;
- o Router estiver testado;
- aliases/modelos estiverem configuráveis;
- o projeto suportar cinco papéis lógicos de modelo sem espalhar SDKs;
- chamadas determinísticas permanecerem locais;
- Dense AE e AE-LSTM continuarem comparáveis;
- GPVS-Faults permanecer como dataset ativo;
- Recall, F1 e Precision estiverem na camada principal;
- AUC tiver tipo explícito;
- matriz de confusão estiver restaurada;
- falsos alarmes estiverem quantificados;
- threshold estiver configurável;
- p99,9 e k=5 estiverem registrados somente como referência histórica;
- grade 3×3 estiver rastreável e calibrada apenas no saudável;
- R(t), F(t), f(t) e h(t) estiverem matematicamente corretos;
- Weibull/Normal/Lognormal não forem fabricados;
- FMECA vigente estiver com o novo escopo e campos nulos sem fonte;
- métricas do detector não recalcularem NPR;
- frontend não recalcular ciência;
- documentação refletir a mesma arquitetura;
- `.env.example` refletir os dois provedores;
- suíte de testes passar;
- lint passar;
- checagens do projeto passarem;
- cobertura de código novo for >= 80%;
- não houver novo vazamento de segredo;
- relatório final do Codex explicar exatamente o que mudou.

---

# 32. FORMATO DO RELATÓRIO FINAL DO CODEX

Ao final, responder ao pesquisador com estas seções:

## A. Implementado
Somente fatos efetivamente implementados.

## B. Arquivos alterados
Arquivo + finalidade.

## C. Testes
Comando + resultado.

## D. Divergências corrigidas
Listar C1–C6 e status.

## E. Pendências metodológicas
Somente as que realmente exigem literatura, dados ou decisão do pesquisador.

## F. Riscos
Problemas que ainda podem afetar validade científica, custo, disponibilidade ou reprodutibilidade.

## G. Próxima ação recomendada
No máximo 3 ações, em ordem.

---

# 33. REGRA FINAL

O objetivo não é maximizar quantidade de funcionalidades.

O objetivo é:

> **maximizar coerência científica, rastreabilidade, capacidade de detecção e eficiência computacional, mantendo o ALIAdo alinhado à dissertação.**

Sempre que houver dúvida entre:

```text
mais recursos
```

e:

```text
mais coerência com a hipótese do mestrado
```

priorizar a segunda opção.

E sempre que houver dúvida sobre a validação de Machine Learning, retornar à pergunta:

> **“Quantas falhas reais o meu modelo detecta sem gerar alarmes falsos demais?”**

---

## Fontes internas que originaram esta especificação

Este documento consolida:

- decisões explícitas do pesquisador nas conversas do Projeto Mestrado;
- consolidação metodológica do ambiente Work;
- direcionamento metodológico elaborado com Gemini;
- nova arquitetura OpenAI + Gemini;
- estado do repositório `TheMagicIAn-creator/mestrado-utfpr` observado antes da geração deste documento;
- documentação canônica já existente no repositório.

Quando houver conflito entre esta especificação e documentação antiga, tratar a documentação antiga como item a ser auditado e corrigido, observando a hierarquia definida na Seção 1.

**Fim da especificação.**
