---
titulo: Validação experimental GPVS-Faults
tipo: resultado
status: ativo
confianca: alta
nivel_evidencia: E3
al_iado: true
tags: [cerebro, resultados, gpvs, inversor-pv, evidencia-e3, autoencoder]
---

# Validação experimental GPVS-Faults

O GPVS-Faults foi executado como protocolo independente do conjunto Stender. Os 14 ensaios de falha em IPPT/MPPT são a unidade de inferência; janelas do mesmo ensaio não são tratadas como replicações independentes.

## Decisão metodológica

O limiar aprendido apenas nos ensaios saudáveis F0L/F0M não transfere sem perda aos demais ensaios, por deslocamento de distribuição.

O que o artefato registra (`resultados/gpvs/validacao_gpvs_e3.json`) é `adaptation_per_experiment: false`: **pesos do Autoencoder e limiar permanecem CONGELADOS** em F1–F7. O que é local por ensaio é apenas a **normalização de comissionamento** — a primeira metade das janelas pré-falha define a referência, e a segunda metade, que não entra na normalização, mede a especificidade. Um PCA de reconstrução usa o mesmo split como baseline.

A distinção não é preciosismo: "ajustar o limiar por ensaio" seria recalibrar contra o dado que se quer julgar. Não foi isso que se fez, e a redação anterior desta nota sugeria que sim.

## Interpretação

F1, F2 e F5 são os cenários mais detectáveis. F3 é intermitente; F4, F6 e F7 preservam limitações importantes no limiar p99. O detector identifica desvio, não prova a causa do componente.

Os números vigentes devem ser lidos de `resultados/gpvs/validacao_gpvs_e3.json`. O escopo é E3 experimental de bancada, não campo, e não autoriza Weibull/RUL físico.

## Conexões

- [[Níveis de evidência]]
- [[Separação dos domínios de dados]]
