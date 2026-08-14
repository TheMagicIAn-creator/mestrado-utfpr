# Aplicação web V2

## Objetivo

A aplicação V2 substitui o Streamlit como interface principal do ALIAdo PV. Ela
separa três responsabilidades que antes compartilhavam o mesmo ciclo de vida:

1. resultados científicos são produzidos por scripts reproduzíveis;
2. `src/webapp_v2/contracts.py` valida e publica uma projeção somente leitura;
3. o navegador apresenta os contratos e inicializa o agente apenas quando há
   uma pergunta.

Abrir o dashboard não treina modelos, não altera artefatos e não carrega Torch,
ChromaDB ou embeddings.

## Execução

```powershell
python -m src.webapp_v2
```

O endereço padrão é `http://127.0.0.1:8000`. Durante desenvolvimento:

```powershell
uvicorn src.webapp_v2.app:app --reload
```

Variáveis do servidor:

| Variável | Padrão | Finalidade |
|---|---:|---|
| `AL_IADO_HOST` | `127.0.0.1` | interface de rede no launcher V2 |
| `PORT` | `8000` | porta HTTP |
| `AL_IADO_LOG_LEVEL` | `info` | nível de log do Uvicorn |

## Contratos HTTP

| Rota | Método | Contrato |
|---|---|---|
| `/` | GET | aplicação HTML semântica |
| `/api/dashboard` | GET | autoencoder, confiabilidade, FMECA e evidências V2 |
| `/api/reliability/curves` | GET | 2.005 pontos previamente calculados |
| `/api/version` | GET | identidade inequívoca da aplicação e schema V2 |
| `/api/health` | GET | integridade dos contratos e estado do agente |
| `/api/chat` | POST | mensagem JSON ou multipart com até quatro anexos |
| `/api/agent/initialize` | POST | aquece Gemini, embeddings, ChromaDB e BM25 |
| `/api/agent/reset` | POST | reinicializa o runtime RAG sob demanda |
| `/vendor/plotly.min.js` | GET | Plotly servido pelo próprio ambiente Python |

O frontend não estima limiares, métricas, Weibull ou confiabilidade. Ele filtra
e apresenta os valores dos contratos. As probabilidades usam eixo `[0, 1]`; a
densidade e a taxa de falha usam escala logarítmica com domínio derivado das
séries selecionadas.

Perguntas sobre resultados recebem um resumo autoritativo produzido pelo mesmo
`dashboard_contract()` que alimenta as figuras. O resumo impede que memórias
antigas, artefatos legados ou conhecimento genérico do LLM substituam as
métricas publicadas. Diferenças AE/PCA não são tratadas como causalidade,
equivalência ou significância estatística sem evidência específica.

## Fronteiras científicas

- GPVS-Faults é o único dataset experimental V2.
- F1-F7 são classes de ensaio, não componentes FMECA.
- a confiabilidade física usa cenários bibliográficos sob hipótese exponencial;
- Weibull físico e RUL não são estimados sem vidas, exposição e censura;
- `D_campo` do FMECA não é a detectabilidade experimental do monitor.

## Agente

O agente é a primeira tela da aplicação. `src/conhecimento/base_runtime.py`
substitui o ciclo de inicialização da UI antiga. Ao abrir essa tela, ele restaura
snapshots portáteis quando necessário,
escolhe o backend de embeddings, sincroniza o índice lexical BM25 e entrega a
base ao adaptador HTTP. A auditoria de evidências e a guarda de citações também
permanecem ativas.

Cada aba do navegador mantém um identificador de sessão próprio. Os turnos são
gravados em `notas/sessoes/*_sessao_web_v2.md` e reindexados na coleção de
sessões, preservando a memória sem reutilizar estado ou funções do Streamlit.
O histórico imediato também permanece no `sessionStorage` da aba.

O deploy público deve adicionar autenticação e persistência de sessão na
plataforma de hospedagem. O servidor local assume um único pesquisador e não
expõe dados brutos, modelos ou estado local do Obsidian como arquivos estáticos.

## Interface legada

Os módulos em `src/interface/` permanecem congelados para testes e histórico de
migração. Nenhum módulo de `src/webapp_v2/` pode importá-los. Eles não fazem
parte de `requirements-ui.txt` e não devem ser usados em novos deploys.
`streamlit run app.py` é bloqueado deliberadamente; `python app.py` apenas
delega ao launcher V2 por compatibilidade.
