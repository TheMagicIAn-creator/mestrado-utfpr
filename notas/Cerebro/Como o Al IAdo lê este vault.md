---
al_iado: true
titulo: "Como o Al IAdo lê este vault"
tipo: contexto
status: ativo
confianca: alta
nivel_evidencia: projeto
tags: [cerebro, vault, al-iado-fluxo]
---

# Como o Al IAdo lê este vault

O fluxo completo, do arquivo `.md` até a resposta no chat.

## 1. ENTRADA — como uma nota entra no cérebro

Toda nota Markdown "útil" do vault é indexada na coleção **`obsidian_pv`**
(ChromaDB) por `src/conhecimento/obsidian.py` → `sincronizar_obsidian()`.

**Quando roda:** a cada turno do chat e na inicialização do app
(`streamlit_app.py` e `agente.py` chamam a sincronização). Notas novas ou
editadas entram no turno seguinte — não precisa reindexar à mão.

**O que NÃO entra:**

| Excluído | Por quê |
|---|---|
| `al_iado: false` no frontmatter | exclusão deliberada |
| `privado: true` | nota pessoal |
| `.obsidian/`, `Templates/` | diretório técnico |
| conteúdo com aparência de segredo | segurança |

**Classe de origem:** cada nota é marcada como *curada* (`Cerebro/`), *sessão*,
*memória consolidada*, *nota de literatura* etc. É isso que impede o agente de
confundir um rascunho de sessão com conhecimento validado.

## 2. CONSULTA — como ele acha a nota certa

Quando você pergunta algo, roda o pipeline RAG (`agente.py`):

```
pergunta
  → expansão de query (sinônimos do domínio, PT/EN)
  → busca HÍBRIDA: semântica (embeddings) + lexical (BM25/FTS5)
  → fusão RRF  → reranking (peso por pasta e sobreposição)
  → top-K trechos  → Gemini sintetiza a resposta
```

O vault entra nessa busca **junto** com a literatura — mas com peso e rótulo
próprios, para o agente saber *de onde* veio cada trecho.

## 3. SAÍDA — o que ele pode e não pode afirmar

Hierarquia de autoridade (regra do projeto, vale sempre):

| Tipo de afirmação | Fonte legítima |
|---|---|
| Métrica, resultado numérico | **artefatos** em `resultados/` — nunca a memória |
| Citação bibliográfica | **PDF** em `literatura/` — nunca ficha do vault |
| Decisão / preferência do pesquisador | memória validada + notas do `Cerebro/` |
| Contexto e histórico | sessões e memórias consolidadas |

> **Nota de vault nunca vira citação.** As fichas em `Literatura/` servem para
> *encontrar* o artigo; a citação sai do PDF. Ver [[Literatura a revisar]].

Em conflito, prevalecem: artefato atual > nota curada ativa > fonte primária >
sessão antiga (que registra o que foi dito, inclusive respostas já superadas).

## 4. ESCRITA — o caminho de volta

O agente também **escreve** no vault:

- **sessões** → `notas/sessoes/` a cada conversa
- **memória validada** → JSON em `notas/memorias/agentes/` + projeção legível
  em `Cerebro/Memorias validadas/` (só passa o que o auditor aprova:
  preferência, decisão, correção, contexto estável — nunca métrica)
- **consolidação** → periodicamente resume e arquiva sessões

## 5. O que VOCÊ deve fazer para o vault funcionar

1. Conhecimento durável (decisão, conceito, resultado) → nota no **`Cerebro/`**
   com as tags-nó do [[00 - Painel do cerebro]].
2. Sessão e memória são **histórico** — não organize nada lá.
3. Tag só se **separa** grupos. Se vai aparecer em quase toda nota, é contexto,
   não tag (foi o que criou o "novelo" — ver [[Como usar o grafo]]).

## Conexões

- [[00 - Painel do cerebro]]
- [[Como usar o grafo]]
- [[Literatura a revisar]]
- [[Arquitetura da equipe de modelos]]
