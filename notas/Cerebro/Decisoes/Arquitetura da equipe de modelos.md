---
titulo: Arquitetura da equipe de modelos
tipo: decisao
status: ativo
confianca: alta
nivel_evidencia: projeto
al_iado: true
tags: [cerebro, decisao, arquitetura, gemini]
---

# Arquitetura da equipe de modelos

> Substitui a nota "Arquitetura Gemini e Groq". **O Groq foi removido do
> projeto**; a equipe é 100% Gemini, um modelo por nível de tarefa.

#arquitetura #al-iado

## Níveis

| Nível | Modelo | Papel |
|---|---|---|
| 1 — conversa/síntese | `gemini-flash-latest` | única voz do chat; interpreta ferramentas e produz a resposta final |
| 2 — auditoria | `gemini-flash-latest` | auditor de evidências e porteiro da memória (entrada/saída em JSON) |
| 3 — fundo/lote | `gemini-flash-lite-latest` | metadados de PDF e consolidação de memória |

Os modelos **não são retreinados** durante a conversa. O aprendizado durável
ocorre em memória externa validada e auditável.

## Python

Executa indexação, busca, cálculos, treinamento, gráficos, tabelas e leitura de
artefatos. **Nenhum LLM recalcula nem aprova o próprio resultado.**

## Obsidian

Interface navegável do cérebro externo. Recebe notas curadas e a projeção
legível das memórias aprovadas — mas **não substitui** PDFs, datasets,
artefatos nem o JSON de memória validada. Nota de vault nunca vira citação.

## Conexões

- [[00 - Painel do cerebro]]
- [[Níveis de evidência]]
