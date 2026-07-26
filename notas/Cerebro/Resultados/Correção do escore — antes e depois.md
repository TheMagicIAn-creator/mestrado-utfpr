---
al_iado: true
titulo: "Correção do escore — antes e depois"
tipo: resultado
status: ativo
confianca: alta
nivel_evidencia: E2
tags: [al-iado, cerebro, resultado, escore, weibull]
---

# Correção do escore — antes e depois

Validação empírica no dataset completo de #paderborn. Antes/depois da troca do
MSE médio pelo [[Escore localizado]].

#escore-localizado #weibull-rul #evidencia-e2 #deteccao-anomalia

## Resultado medido

| | #igbt | #fusivel-ac |
|---|---|---|
| Censura | 70% → **13%** | 98% → **0%** |
| Eventos observados | 13 → **34** | 1 → **39** |
| β de Weibull | 5,71 (artefato) → **2,74** | não estimável → **6,33** |
| Detecção @sev 1,0 | 34% → **86%** | 4,5% → **100%** |

As três falhas passam a ter Weibull **convergente**, sem extrapolação além do
horizonte e sem a marca "não confiável".

## Por que o β antigo era falso

β=5,71 do #igbt era ajustado em 13 eventos empilhados na borda da censura, com
η=143,6 **além** do horizonte de 120 (extrapolação pura). Não era propriedade
física — era artefato de censura.

## Ressalva honesta (limitação a documentar)

O falso positivo no teste isolado subiu de 1,1% (MSE) para **6,8%** (localizado)
— o top-k é estatística de cauda, mais sensível a ruído com calibração pequena.
Mitigação implementada: limiar **auto-calibrado** ao FP alvo. Falta confirmar no
rerun.

## Nível de evidência

**E2** — sintético orientado pela FMECA. Não é prova de desempenho industrial.
Ver [[Níveis de evidência]].

## Conexões

- [[00 - Painel do cerebro]]
- [[Escore localizado]]
- [[FMECA e detectabilidade empírica]]

Artefatos: `resultados/autoencoder/weibull_results.json` · Auditoria: `docs/auditoria_pipeline_ml.md` §16
