---
al_iado: true
titulo: "Modelagem de normalidade com Autoencoder"
tipo: conceito
status: ativo
confianca: alta
nivel_evidencia: projeto
<<<<<<< HEAD
tags: [cerebro, conceito, autoencoder, metodologia]
=======
tags: [al-iado, cerebro, conceito, autoencoder, metodologia]
>>>>>>> origin/main
---

# Modelagem de normalidade

Estratégia central do método: treinar apenas em operação **saudável** e detectar
desvios, em vez de classificar falhas rotuladas.

#deteccao-anomalia #autoencoder #paderborn #metodologia

## Justificativa

Em manutenção preditiva real raramente há dados de falha. Modela-se o
comportamento normal e detectam-se desvios (Ibrahim, 2022; Ahirwar, 2025). O
dataset de #paderborn fornece a referência de normalidade (inversor IGBT
trifásico saudável).

## O que é fundamentado na literatura × o que é escolha nossa

| Peça | Origem |
|---|---|
| Treinar só no saudável, sem rótulo | Ibrahim (2022) |
| Erro de reconstrução como escore | Ibrahim (2022), eq. 3 |
| **Arquitetura densa sobre features FMECA** | **escolha nossa** — Ibrahim usa AE-LSTM temporal |
| **Escore localizado (top-k)** | **contribuição nossa** |
| Limiar por percentil auto-calibrado | escolha nossa (controla FP ≈ 1%) |

## Justificativa da arquitetura densa (para o capítulo de metodologia)

Optou-se por AE **denso sobre features espectrais** (e não pelo AE-LSTM temporal
do artigo) porque: (i) a dinâmica intra-janela relevante já está condensada nas
features de ~102 ms, tornando a recorrência redundante; (ii) o espaço de
features nomeadas habilita o [[Escore localizado]], interpretável via #fmeca;
(iii) o AE-LSTM temporal é mantido como **concorrente** na comparação, de modo
que a escolha é justificada empiricamente por AUC.

## Pendência

Hiperparâmetros (latente=16, épocas=150, lr=1e-3) ainda são defaults. Falta uma
varredura documentada de latente ∈ {8,16,32} para não ficarem sem justificativa.

## Conexões

- [[00 - Painel do cerebro]]
- [[Escore localizado]]
- [[Macro-códigos de comparação]]
- [[Níveis de evidência]]

Implementação: `src/ml/autoencoder.py` · Auditoria: `docs/auditoria_pipeline_ml.md` §9, §21
