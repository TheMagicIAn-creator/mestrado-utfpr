# Arquitetura canônica

O ALIAdo é uma aplicação ASGI e um pipeline científico independente. A página
abre sem carregar Torch, ChromaDB ou embeddings; esses recursos são aquecidos
em segundo plano apenas para o chat.

```text
src/
|-- core/                 configuração, segurança, tempo, logs e formatação
|-- conhecimento/         Gemini, RAG híbrido, memória, Obsidian e ferramentas
|-- ml/                   GPVS, Denso/AE-LSTM E3 e confiabilidade física
|-- webapp/               Starlette, contratos HTTP e frontend responsivo
`-- orquestrador.py       coordenação explícita das operações

scripts/
|-- auditar_resultados.py valida manifestos, arquivos e hashes
|-- avaliar_agente.py     avalia roteamento e salvaguardas científicas
|-- manter_base.py        mantém literatura, sessões e Obsidian
`-- verificar_projeto.py  verifica ambiente, GPVS, árvore e publicação
```

## Fluxo web

1. `src.webapp` entrega HTML/CSS/JavaScript imediatamente.
2. `/api/status` informa `iniciando`, `pronto` ou `degradado` sem inicializar o
   agente.
3. `agent_adapter` aquece índice lexical, ChromaDB, embeddings e papéis Gemini
   em thread de fundo.
4. Saudações são respondidas localmente; perguntas acadêmicas usam eventos SSE
   `status`, `delta`, `done` e `error`.
5. O Router LLM decide entre resposta, RAG e ferramenta; figuras e resultados
   aparecem na própria conversa somente quando solicitados.

A navegação visual contém apenas Conversa e Referências. Histórico, busca,
renomeação, arquivamento, restauração, exclusão não destrutiva e exportação são
recursos da conversa. As figuras continuam em PNG acadêmico de 300 dpi e PDF
vetorial; o frontend não recalcula métrica, limiar ou ajuste estatístico.

## Fluxo científico

`dados_gpvs.py` é o único contrato de ingestão. Ele valida os 16 ensaios,
extrai 24 features e cria blocos saudáveis disjuntos. `treino_comparacao.py`
treina Denso e AE-LSTM com o mesmo protocolo. `avaliacao_comparativa.py` produz
a comparação E3 de bancada. `publicacao_comparacao.py` grava tabelas, figuras e
manifesto.

`confiabilidade_componentes.py` mantém as equações e os cenários bibliográficos;
`publicacao_confiabilidade.py` publica curvas temporais separadas dos resultados
do detector.

`pipeline.py` registra somente `comparacao` e `confiabilidade` e determina
`ready`, `stale` ou `pending` a partir dos manifestos v2.

## Armazenamento

- `dados/brutos/gpvs/`: único dataset ativo, local e ignorado.
- `dados/processados/`: cache local de features, ignorado.
- `artefatos/modelos/{ae_denso,ae_lstm}`: pesos e scalers locais, ignorados.
- `resultados/comparacao/`: comparação Denso versus AE-LSTM versionável.
- `resultados/confiabilidade/`: cenários físicos versionáveis.
- `resultados/manifestos/`: proveniência e hashes.
- `base_conhecimento/`: ChromaDB local, ignorado.
- `artefatos/*indexado*`: snapshots portáteis do RAG.

## Dependências

Os grupos `core`, `rag`, `ui` e `ml` contêm apenas dependências diretas. A
aplicação não usa bibliotecas de dashboard ou gráficos interativos no bundle
inicial. `requirements.txt` instala o ambiente completo.
