---
titulo: Painel do cérebro Al IAdo PV
tipo: contexto
status: ativo
confianca: alta
nivel_evidencia: projeto
al_iado: true
tags: [cerebro, moc, weibull-rul, fmeca, inversor-pv, paderborn, igbt, fusivel-ac]
---

# Painel do cérebro Al IAdo PV

Mapa de conteúdo curado que conecta decisões, conceitos e resultados do
mestrado. A presença de uma nota aqui **não a transforma em evidência
científica**: artigos são citados dos PDFs indexados e resultados são lidos dos
artefatos atuais do pipeline.

## Nós comuns da dissertação

Toda nota do vault tem **pelo menos uma tag**: uma *estrutural* (pela pasta) e,
quando o conteúdo justifica, as de *tópico* abaixo. São os pontos de encontro do
grafo — clique numa para ver tudo que toca o tema.

### Tópico (os pontos principais)

| Tag | Tema | Notas |
|---|---|---|
| #fmea · #fmeca | modos de falha, criticidade, NPR | 71 · 66 |
| #rcm | manutenção centrada em confiabilidade | 21 |
| #manutencao | manutenção e preditiva | 105 |
| #confiabilidade | confiabilidade, MTBF/MTTF | 80 |
| #weibull-rul | Weibull, censura, vida útil remanescente | 77 |
| #inversor-pv | inversor e sistema fotovoltaico | 154 |
| #contator-ac · #igbt · #fusivel-ac | os três componentes CA da FMECA | 24 · 25 · — |
| #autoencoder | o detector | 86 |
| #deteccao-anomalia | detecção por modelagem de normalidade | 77 |
| #escore-localizado | a contribuição metodológica própria | — |
| #machine-learning | modelos e algoritmos | 58 |
| #sinais-eletricos | THD, harmônicos, FFT, RMS | 36 |
| #paderborn | dataset de normalidade | 46 |
| #evidencia-e2 | nível de evidência sintético | — |

### Estrutural (de onde a nota vem)

`#cerebro` · `#literatura` · `#sessao` · `#sessao-arquivada` · `#memoria`

### Sinalização de metadado

`#autor-invalido` · `#ano-invalido` · `#titulo-invalido` — fichas de literatura
cuja extração automática falhou. Ver [[Literatura a revisar]].

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
