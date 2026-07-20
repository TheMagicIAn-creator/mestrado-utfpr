# Memória dos agentes

## O que significa "aprender"

Gemini e Groq não alteram os próprios pesos durante uma sessão. O aprendizado
do Al IAdo PV é externo e auditável: uma informação durável declarada pelo
pesquisador pode ser transformada em um item estruturado, validada pelo Groq e
recuperada pelo Gemini em conversas futuras.

Isso evita três problemas de uma memória livre: transformar resposta do modelo
em fato, perpetuar resultado científico já recalculado e carregar toda a
conversa em cada prompt.

## Fluxo

1. O pesquisador declara uma preferência, correção, decisão ou contexto estável.
2. Uma heurística local detecta o gatilho. Perguntas comuns não chamam o Groq.
3. O Groq recebe a mensagem e um resumo curto da resposta, retornando JSON.
4. Regras locais rejeitam segredo, métrica volátil, baixa confiança ou conteúdo
   sem ancoragem no texto do pesquisador.
5. O item aprovado é gravado atomicamente em
   `notas/memorias/agentes/memoria_validada.json`.
6. Uma projeção Markdown é atualizada em
   `notas/Cerebro/Memorias validadas/<id>.md` para navegação no Obsidian.
7. Em uma pergunta futura, somente os itens lexicalmente pertinentes entram no
   prompt do Gemini, identificados como dados e acompanhados de proveniência.

## Schema

Cada item contém:

| Campo | Função |
|---|---|
| `id` | hash estável para deduplicação |
| `tipo` | preferência, decisão metodológica, correção ou contexto do projeto |
| `escopo` | conversa, literatura, ML ou compartilhado |
| `conteudo` | formulação autocontida aprovada |
| `evidencia_usuario` | trecho que ancora a memória na fala do pesquisador |
| `origem` | fluxo que criou o item |
| `validado_por` | agente que aprovou a gravação |
| `confianca` | confiança normalizada entre 0 e 1 |
| `status` | ativo ou superado |
| `criado_em_utc` | data de criação auditável |

Resultados como AUC, F1, MTTF, B10 e limiares não pertencem à memória. O agente
deve lê-los sempre dos artefatos atuais do pipeline.

## Persistência

No PC, o JSON está no repositório e pode ser versionado com Git. Esse é o modo
durável e auditável. No Streamlit Community Cloud, o sistema de arquivos da
instância é efêmero: itens aprendidos durante uma execução permanecem enquanto
a instância vive, mas desaparecem em um redeploy ou reinício. O conteúdo já
commitado no JSON reaparece em toda implantação.

Para persistência de escrita contínua na nuvem, será necessário conectar no
futuro um armazenamento externo transacional. A interface atual não simula essa
garantia.

## Relação com o Obsidian

O JSON é a memória normativa e o Markdown é uma visão derivada. Editar a nota
gerada não altera a memória aprovada; correções devem ser declaradas no chat e
passar novamente pelo Groq.

Todo Markdown útil do vault participa da coleção `obsidian_pv` por padrão. A
origem é inferida pela pasta e preservada nos metadados:

| Pasta | Classe de memória |
|---|---|
| `Cerebro/` | nota curada ou memória validada |
| `sessoes/` | conversa atual |
| `sessoes_arquivadas/` | registro conversacional histórico |
| `memorias/` | memória consolidada |
| `Conceitos/` e `Experimentos/` | conhecimento manual do projeto |
| `Literatura/` | nota auxiliar de leitura, nunca fonte científica |

Frontmatter é recomendado, mas não obrigatório. `tipo`, `confianca`,
`nivel_evidencia`, `status`, data e tags refinam o reranking. Para excluir uma
nota deliberadamente, use `al_iado: false` ou `privado: true`; diretórios
técnicos, templates e segredos aparentes também são ignorados.

Esse conteúdo nunca gera citação no rodapé. Literatura citável continua vindo
dos PDFs e números continuam vindo dos artefatos e manifestos vigentes. Uma
sessão arquivada prova somente que algo foi dito naquele momento. No PC,
qualquer arquivo novo ou alterado é percebido no próximo turno. Para atualizar
o snapshot que alimenta o Streamlit Cloud, execute
`python scripts/reconstruir_cerebro_obsidian.py`; uma raiz externa pode ser
informada com `--vault` ou `AL_IADO_OBSIDIAN_VAULT_DIR`.
