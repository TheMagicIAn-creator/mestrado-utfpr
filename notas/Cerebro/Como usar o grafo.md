---
titulo: Como usar o grafo
tipo: contexto
status: ativo
confianca: alta
nivel_evidencia: projeto
al_iado: true
tags: [cerebro, vault, weibull-rul, inversor-pv, igbt, fusivel-ac, fmeca, deteccao-anomalia]
---

# Como usar o grafo

O grafo estava um "novelo" (tudo ligado a tudo) por dois motivos, ambos
corrigidos:

1. **Tags universais** — `#al-iado-pv` (117 notas), `#mestrado` (116),
   `#sessao` (93), `#streamlit` (85) apareciam em quase toda nota. Cada uma
   virava um hub ligado a ~120 nós. Como não informam nada (tudo aqui é do
   mestrado), foram **removidas de 212 notas**.
2. **Configuração** — `linkStrength: 0` e zoom em 0,059 achatavam tudo num
   círculo. Ajustado para forças que separam clusters.

## Filtro padrão

O grafo abre com `-path:sessoes_arquivadas` — as 121 sessões arquivadas ficam
**fora** por padrão (são histórico, têm só 19 links entre si e só poluíam).
Para vê-las, limpe o campo de busca do grafo.

## Cores

| Cor | O que é |
|---|---|
| Verde | notas do `Cerebro/` — o conhecimento curado |
| Azul | #deteccao-anomalia · #escore-localizado · #autoencoder |
| Laranja | #fmeca · #contator-ac · #igbt · #fusivel-ac |
| Roxo | #weibull-rul · #evidencia-e2 |
| Cinza | `Literatura/` e `memorias/` |

## Como manter assim

- Tag nova só se **separa** as notas em grupos úteis. Se ela vai aparecer em
  quase tudo, não é tag — é contexto.
- Conhecimento durável vira nota no `Cerebro/` com os nós comuns (as tags da
  tabela acima). Sessão e memória são histórico, não estrutura.

## Conexões

- [[00 - Painel do cerebro]]
