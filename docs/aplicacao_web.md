# Aplicação web do ALIAdo

## Objetivo

A aplicação ASGI separa três responsabilidades:

1. scripts reproduzíveis produzem os resultados científicos;
2. `src/webapp/contracts.py` valida projeções somente leitura;
3. o navegador oferece uma experiência de conversa e Biblioteca; o agente usa
   os contratos conforme a intenção identificada pelo Router LLM.

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
| `AL_IADO_LOG_FILE` | desligado | habilita arquivo rotativo local |

## Contratos HTTP

| Rota | Método | Contrato |
|---|---|---|
| `/` | GET | aplicação HTML chat-first |
| `/api/chat/stream` | POST | eventos SSE `status`, `delta`, `done` e `error` |
| `/api/render` | POST | Markdown e matemática seguros para sessões restauradas |
| `/api/conversations` | GET | histórico ativo ou arquivado |
| `/api/conversations/{id}` | GET/PATCH/DELETE | leitura, renomeação e ocultação auditável |
| `/api/conversations/{id}/archive` | POST | arquivamento sem apagar o transcrito |
| `/api/conversations/{id}/restore` | POST | restauração de conversa arquivada |
| `/api/status` | GET | estado barato do agente e dos contratos |
| `/api/health` | GET | alias operacional de status |
| `/api/library` | GET/POST | catálogo e inclusão local de referências |
| `/api/library/{id}` | PATCH | edição local de metadados bibliográficos |
| `/api/library/{id}/reindex` | POST | reindexação assíncrona da fonte |
| `/api/results/e3` | GET | contrato científico interno Denso × AE-LSTM |
| `/api/reliability` | GET | contrato científico interno de confiabilidade |
| `/api/sources` | GET | dataset, PDFs, relatórios e manifestos |
| `/api/version` | GET | identidade da aplicação e versão da API |

O frontend não apresenta dashboards científicos fixos e não estima limiares,
métricas, Weibull ou confiabilidade. Quando solicitado na conversa, o agente
consulta ou executa a ferramenta adequada e pode anexar as figuras acadêmicas
publicadas. Os contratos HTTP continuam disponíveis para auditoria e automação.

## Fronteiras científicas

- GPVS-Faults é o único dataset experimental ativo.
- E3 compara apenas Autoencoder Denso e AE-LSTM nos ensaios F1L–F7M.
- FMECA orienta criticidade e manutenção; não gera uma curva sintética ativa.
- Confiabilidade física usa taxas bibliográficas ou cenários derivados.
- Weibull físico não é estimado sem vidas, exposição e censura por ativo.

## Agente

`src/webapp/agent_adapter.py` reutiliza o pipeline completo: Gemini, recuperação
híbrida semântica/BM25, ChromaDB, sessões, Obsidian, auditoria multiagente,
anexos e memória. Saudações são respondidas localmente antes do aquecimento.
Perguntas acadêmicas recebem o contexto autoritativo produzido pelos mesmos
contratos científicos internos.

Cada conversa mantém um identificador próprio. Os turnos são gravados em
`notas/sessoes/*_sessao_web.md`, reindexados fora do caminho crítico e podem ser
renomeados, arquivados, restaurados, ocultados ou exportados em Markdown. A
exclusão na interface não apaga o transcrito auditável.

O Router LLM interpreta pedidos normais antes de selecionar ferramentas. Regras
determinísticas ficam restritas à confirmação literal de operações destrutivas
e às guardas de integridade científica. O fallback textual só é usado quando o
provedor semântico está indisponível.

## Desempenho e segurança

- HTML/CSS/JavaScript são servidos sem bundle analítico pesado.
- `/api/status` não lê artefatos nem inicializa o modelo.
- anexos são limitados em quantidade, tamanho e nome de arquivo seguro;
- Markdown do agente é renderizado sem HTML bruto;
- logs registram ID, duração e rota, nunca prompt, anexo ou chave;
- cabeçalhos CSP, `nosniff`, `DENY` e política de permissões são aplicados.

O servidor local assume um único pesquisador. Uma implantação pública deve
adicionar autenticação e persistência de sessão adequada à plataforma.
