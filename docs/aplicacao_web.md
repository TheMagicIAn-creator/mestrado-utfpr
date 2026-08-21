# Aplicação web do ALIAdo

## Objetivo

A aplicação ASGI separa três responsabilidades:

1. scripts reproduzíveis produzem os resultados científicos;
2. `src/webapp/contracts.py` valida projeções somente leitura;
3. o navegador apresenta os contratos e conversa com o mesmo agente RAG.

Abrir a aplicação não treina modelos nem altera artefatos. O HTML é entregue
imediatamente; contratos, embeddings, ChromaDB, BM25 e equipe multiagente
aquecem em segundo plano.

## Execução

```powershell
python -m src.webapp
```

O endereço padrão é `http://127.0.0.1:8000`. Durante o desenvolvimento:

```powershell
uvicorn src.webapp.app:app --reload
```

| Variável | Padrão | Finalidade |
|---|---:|---|
| `AL_IADO_HOST` | `127.0.0.1` | interface de rede |
| `PORT` | `8000` | porta HTTP |
| `AL_IADO_LOG_LEVEL` | `info` | nível de log do Uvicorn |

## Contratos HTTP

| Rota | Método | Contrato |
|---|---|---|
| `/` | GET | aplicação HTML chat-first |
| `/api/chat/stream` | POST | eventos SSE `status`, `delta`, `done` e `error` |
| `/api/status` | GET | estado barato do agente e dos contratos |
| `/api/health` | GET | alias operacional de status |
| `/api/results/e3` | GET | comparação GPVS Denso × AE-LSTM |
| `/api/results/e2` | GET | detectabilidade FMECA no eixo `a_det` |
| `/api/reliability` | GET | confiabilidade física bibliográfica |
| `/api/sources` | GET | dataset, PDFs, relatórios e manifestos |
| `/api/version` | GET | identidade da aplicação e versão da API |

O frontend não estima limiares, métricas, Weibull ou confiabilidade. Figuras
acadêmicas são carregadas somente quando o painel é aberto e permanecem
disponíveis como PNG 300 dpi e PDF vetorial; os dados-fonte CSV/JSON têm links
de download e hash.

## Fronteiras científicas

- GPVS-Faults é o único dataset experimental ativo.
- E3 compara apenas Autoencoder Denso e AE-LSTM nos ensaios F1L–F7M.
- E2 usa magnitude sintética adimensional; `a_det` não é tempo.
- Weibull E2 é diagnóstico de detectabilidade, não modelo de vida útil.
- Confiabilidade física usa taxas bibliográficas ou cenários derivados.
- Weibull físico não é estimado sem vidas, exposição e censura por ativo.

## Agente

`src/webapp/agent_adapter.py` reutiliza o pipeline completo: Gemini, recuperação
híbrida semântica/BM25, ChromaDB, sessões, Obsidian, auditoria multiagente,
anexos e memória. Saudações são respondidas localmente antes do aquecimento.
Perguntas acadêmicas recebem o contexto autoritativo produzido pelos mesmos
contratos que alimentam os painéis.

Cada conversa mantém um identificador próprio. Os turnos são gravados em
`notas/sessoes/*_sessao_web.md`, reindexados fora do caminho crítico e podem ser
exportados em Markdown pelo navegador.

## Desempenho e segurança

- HTML/CSS/JavaScript são servidos sem bundle analítico pesado.
- `/api/status` não lê artefatos nem inicializa o modelo.
- anexos são limitados em quantidade, tamanho e nome de arquivo seguro;
- Markdown do agente é renderizado sem HTML bruto;
- logs registram ID, duração e rota, nunca prompt, anexo ou chave;
- cabeçalhos CSP, `nosniff`, `DENY` e política de permissões são aplicados.

O servidor local assume um único pesquisador. Uma implantação pública deve
adicionar autenticação e persistência de sessão adequada à plataforma.
