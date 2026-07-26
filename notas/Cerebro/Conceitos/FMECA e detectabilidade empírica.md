---
al_iado: true
titulo: "FMECA e detectabilidade empírica"
tipo: conceito
status: ativo
confianca: alta
nivel_evidencia: projeto
tags: [al-iado, cerebro, conceito, fmeca, resultados]
---

# FMECA e detectabilidade empírica

Relação — e divergência — entre o índice **D** da FMECA (detecção em campo) e a
detectabilidade medida do Autoencoder. É resultado a discutir, não defeito.

#fmeca #deteccao-anomalia #evidencia-e2

## As três falhas injetadas

| Id | Componente | S·O·D | NPR | Assinatura elétrica |
|----|---|---|---|---|
| 1 | #contator-ac | 5·7·9 | **315** | transiente/ruído de comutação |
| 2 | #igbt | 5·6·3 | 90 | harmônicos 5/7/11/13 + THD ↑ |
| 3 | #fusivel-ac | 5·3·2 | 30 | perda parcial de fase |

Fonte única: `docs/fmeca.md` (ancorada no TCC — Torres, 2024).

## A divergência a discutir

O #contator-ac tem o **maior NPR** e é, empiricamente, **o mais detectável** —
casa perfeitamente. Já #igbt e #fusivel-ac têm **D baixo** na FMECA (fáceis de
detectar *em campo*, com proteção dedicada), mas são os **mais difíceis** para o
detector por modelagem de normalidade.

Ou seja: **detectabilidade de campo ≠ detectabilidade por ML no sinal**. São
conceitos distintos e a relação entre eles é material de capítulo.

## Cuidado conceitual (registrar na dissertação)

- **NPR** mede prioridade de manutenção (S×O×D).
- **β de Weibull** descreve a forma da distribuição do tempo até falha.
- **Não existe relação teórica entre NPR e β.** Esperar "NPR maior → β maior" é
  misturar conceitos.
- Poucas amostras num histograma decorrem de **censura** (o detector não
  disparou), não de criticidade baixa.

## Conexões

- [[00 - Painel do cerebro]]
- [[Correção do escore — antes e depois]]
- [[Escore localizado]]

Fonte: `docs/fmeca.md` · Injeção: `src/ml/injecao_falhas.py`
