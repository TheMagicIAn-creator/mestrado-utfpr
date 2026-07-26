---
al_iado: true
titulo: "Escore localizado (top-k dos resíduos padronizados)"
tipo: conceito
status: ativo
confianca: alta
nivel_evidencia: projeto
<<<<<<< HEAD
tags: [cerebro, conceito, escore, autoencoder]
=======
tags: [al-iado, cerebro, conceito, escore, autoencoder]
>>>>>>> origin/main
---

# Escore localizado

Contribuição metodológica própria da dissertação. Define **como** o erro do
Autoencoder vira um escore de anomalia.

#deteccao-anomalia #escore-localizado #autoencoder #fmeca

## Problema que resolve

O escore original era o **MSE médio sobre ~109 features**. Falhas localizadas no
espectro — harmônicos do #igbt, perda de fase do #fusivel-ac — mexem em poucas
features e eram **diluídas** pelas ~100 intactas. O detector só enxergava o
#contator-ac (transiente de banda larga).

## Definição

Para cada janela: padroniza o |resíduo| de cada feature contra a régua saudável
(z = (|r| − μ)/σ, com μ/σ do bloco de calibração) e agrega pela **média dos k
maiores z** (k=5 por padrão).

## Fundamentação

- Erro de reconstrução como sinal de anomalia — Ibrahim (2022), eq. 3
- Padronização por variável — Francisti (2025), Z-score/Shewhart ±3σ, aqui
  aplicado ao **resíduo do AE** e não ao sinal bruto
- Agregação top-k — generalização da regra de Shewhart / SPC multivariável e da
  contribuição por feature (Narayanan, 2023)

O k reflete a cardinalidade da assinatura FMECA (uma falha toca ~3–9 features).

## Por que importa para a dissertação

Como as features são **nomeadas**, o escore diz *qual* feature desviou — ligando
a detecção ao modo de falha da #fmeca. Um AE sobre sinal bruto perderia essa
rastreabilidade. É o diferencial do método frente ao AE-LSTM do Ibrahim.

## Ressalva conhecida

É uma estatística de **cauda** — mais sensível a ruído amostral que uma média.
Com calibração pequena, o falso positivo fica instável; por isso o limiar é
auto-calibrado ao FP alvo. Ver [[Correção do escore — antes e depois]].

## Conexões

- [[00 - Painel do cerebro]]
- [[Modelagem de normalidade]]
- [[Correção do escore — antes e depois]]
- [[FMECA e detectabilidade empírica]]

Implementação: `src/ml/escore_anomalia.py` · Auditoria: `docs/auditoria_pipeline_ml.md` §3.1, §13
