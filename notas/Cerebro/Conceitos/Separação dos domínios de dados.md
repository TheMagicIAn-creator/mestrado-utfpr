---
al_iado: true
titulo: "Separação dos domínios de dados"
tipo: conceito
status: ativo
confianca: alta
nivel_evidencia: projeto
tags: [al-iado, datasets, paderborn, pv-farms, metodologia]
---

# Separação dos domínios de dados

Paderborn e PV Farms exercem papéis diferentes e não são fundidos em um único problema de aprendizado.

## Paderborn — domínio CA

Representa o inversor trifásico em operação saudável. É usado para modelagem de normalidade e para o pipeline de detecção de anomalias no lado CA. A validação de falhas depende de injeção sintética e deve declarar o nível de evidência correspondente.

## PV Farms — domínio CC

É um dataset rotulado para classificação supervisionada de falhas conhecidas no lado CC. Seus resultados não diagnosticam falhas CA do inversor.

## Regra metodológica

O uso combinado é conceitual e arquitetural. Métricas, classes, features e conclusões de um domínio não são transferidas automaticamente para o outro.

## Conexões

- [[00 - Painel do cerebro]]
- [[Níveis de evidência]]
