---
al_iado: true
titulo: "Painel do cérebro Al IAdo PV"
tipo: contexto
status: ativo
confianca: alta
nivel_evidencia: projeto
tags: [cerebro, moc]
---

# Painel do cérebro Al IAdo PV

Mapa de conteúdo curado que conecta decisões, conceitos e resultados do
mestrado. A presença de uma nota aqui **não a transforma em evidência
científica**: artigos são citados dos PDFs indexados e resultados são lidos dos
artefatos atuais do pipeline.

## Nós comuns da dissertação

Estas tags são os **pontos de encontro** do grafo — clique para ver tudo que
toca cada tema:

| Tag | Tema |
|---|---|
| #deteccao-anomalia | detecção por modelagem de normalidade |
| #escore-localizado | a contribuição metodológica própria |
| #autoencoder | o detector e sua arquitetura |
| #fmeca | modos de falha, NPR e assinaturas elétricas |
| #weibull-rul | confiabilidade e vida útil remanescente |
| #paderborn | dataset de normalidade |
| #evidencia-e2 | nível de evidência sintético |
| #comparacao-literatura | benchmark frente aos artigos-base |
| #metodologia | decisões de método |
| #contator-ac · #igbt · #fusivel-ac | os três componentes CA |

## Fundamentos

- [[Níveis de evidência]]
- [[Separação dos domínios de dados]]
- [[Modelagem de normalidade]]

## Conceitos do método

- [[Escore localizado]]
- [[FMECA e detectabilidade empírica]]

## Resultados

- [[Correção do escore — antes e depois]]

## Decisões

- [[Macro-códigos de comparação]]
- [[Arquitetura da equipe de modelos]]

## Como o vault se organiza

| Pasta | Papel | Vira citação? |
|---|---|---|
| `Cerebro/` | conhecimento **curado** do projeto — o núcleo | não (aponta para as fontes) |
| `Literatura/` | fichas auxiliares dos PDFs, geradas na indexação | **não** — cite o PDF |
| `memorias/` | consolidações automáticas de sessão | não |
| `sessoes*/` | registro conversacional | não |

**Fluxo:** o que for decisão, conceito ou resultado durável deve virar nota no
`Cerebro/` com as tags acima. O resto é histórico pesquisável.
