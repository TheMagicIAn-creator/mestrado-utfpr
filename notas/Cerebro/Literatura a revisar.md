---
titulo: Literatura a revisar
tipo: contexto
status: ativo
confianca: alta
nivel_evidencia: projeto
al_iado: true
tags: [literatura, cerebro, vault, confiabilidade, manutencao]
---

# Literatura a revisar

Varredura automática das 89 fichas de `Literatura/`.
**14 têm metadado problemático** — vieram da extração automática do PDF,
não de leitura. Cada uma está marcada com `revisar: true` no frontmatter
(busque `revisar:true` para filtrar).

> **Nada aqui foi "corrigido no chute".** Autor, ano e título corretos precisam
> ser conferidos na fonte (o PDF). Esta nota lista o que conferir.

## O que cada problema significa

| Marca | Significa |
|---|---|
| `autor-invalido` | "Autor Desconhecido", vazio, ou nome que é fragmento de texto |
| `ano-invalido` | ano `0000` ou fora de 1900–2099 |
| `titulo-suspeito` | título é fragmento de frase da 1ª página, não o título real |
| `abstract-editorial` | o "abstract" é ficha catalográfica do livro, sem valor |

## Fichas a conferir

| Nota | Problema | Autor registrado | Ano |
|---|---|---|---|
| [[autor-desconhecido_analise-confiabilidade-sistemas-potencia_0000]] | autor-invalido | Autor Desconhecido | 1998 |
| [[autor-desconhecido_analise-confiabilidade-sistemas-potencia_0000_1]] | autor-invalido, ano-invalido | Autor Desconhecido | 0 |
| [[universidade-federal_prof-benjamim-rodrigues-menezes-orientador_2008]] | titulo-suspeito | Universidade Federal | 2008 |
| [[autor-desconhecido_steve-voss-tassos-golnas-steve_2009]] | autor-invalido | Autor Desconhecido | 2009 |
| [[autor-desconhecido_universidade-federal-do-par-a_2024]] | autor-invalido, titulo-suspeito | autor-desconhecido | 2024 |
| [[autor-desconhecido_manual-de-confiabilidade-mantenabilidade-e-disponibilidade_0000]] | autor-invalido, ano-invalido | autor-desconhecido | 0000 |
| [[autor-desconhecido_processes-these-disturbances-impede-ability_2025]] | autor-invalido | Autor Desconhecido | 2025 |
| [[autor-desconhecido_zoelzer-pdf_2005]] | autor-invalido, titulo-suspeito | autor-desconhecido | 2005 |
| [[digital-twin_therefore-distinct-between-digital-model_2018]] | titulo-suspeito | Digital Twin | 2018 |
| [[gonzalez_digital-image-processing_2008]] | abstract-editorial | Rafael C. Gonzalez, Richard E. Woods | 2008 |
| [[autor-desconhecido_engenharia-de-sistemas-de-potencia_2019]] | autor-invalido | autor-desconhecido | 2019 |
| [[eletrica_subestacoes-de-energia-definicoes-conceitos-e-aplicacoes_0000]] | ano-invalido | Aprender Elétrica | s.d. |
| [[gonzalez_0136095577-pdf_2007]] | abstract-editorial | Rafael C. Gonzalez | 2007 |
| [[smith_the-scientist-and-engineer-s-guide-to-digital-signal-process_1999]] | titulo-suspeito | Smith, Steven W | 1999 |

## Regra

Ficha de `Literatura/` **nunca** é citação — cite sempre o PDF em `literatura/`.
Estas fichas servem só para busca. Antes de citar qualquer uma delas na
dissertação, confira autor/ano/título no próprio PDF.

## Conexões

- [[00 - Painel do cerebro]]
- [[Como usar o grafo]]
