---
titulo: Macro-códigos de comparação
tipo: decisao
status: ativo
confianca: alta
nivel_evidencia: projeto
al_iado: true
tags: [cerebro, decisao, comparacao, arquitetura, autoencoder, machine-learning, fmeca, sinais-eletricos, escore-localizado]
---

# Macro-códigos de comparação

Decisão de arquitetura: substituir o framework de experimentos por **dois
scripts legíveis** com avaliação e saída idênticas.

#comparacao-literatura #metodologia #autoencoder

## Motivo

O framework antigo gerava `metricas.csv` com **33 colunas**, matrizes de
confusão **enganosas** (limiar de prevalência rara + teste balanceado → recall
0,5%, matriz com uma coluna vazia) e protocolos incomparáveis entre si.

## A decisão

| | `macro_proposto.py` | `macro_ibrahim.py` |
|---|---|---|
| Modelo | AE denso + [[Escore localizado]] | AE-LSTM temporal (Ibrahim 2022) |
| Features | espectrais #fmeca | **as mesmas** |
| Avaliação | E2: injeção FMECA por severidade | **a mesma** |
| Saída | tabela de 5 colunas + gráfico | **o mesmo módulo** |

**Contrato:** cada macro fornece só um *scorer* (janelas → escores);
`macro_comum.py` faz toda a avaliação. Comparação maçã-com-maçã por construção.

Escolhas do pesquisador: avaliação nossa (E2) para os dois; só o AE-LSTM no
macro do Ibrahim; scripts **importam e orquestram** (sem duplicar lógica), para
serem legíveis e citáveis na dissertação.

## Disciplina metodológica

Calibração e avaliação em blocos **disjuntos** com purga: o limiar sai do 1º
bloco; FP, AUC e injeção vêm do 2º, nunca visto. O AE-LSTM também é treinado só
no bloco de calibração — sem isso haveria vazamento e a comparação seria
inválida.

## Conexões

- [[00 - Painel do cerebro]]
- [[Modelagem de normalidade]]
- [[Níveis de evidência]]

Implementação: `src/ml/macro_{proposto,ibrahim,comum,comparar}.py` · Auditoria: §27–§28
