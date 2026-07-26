---
al_iado: true
titulo: "Níveis de evidência"
tipo: conceito
status: ativo
confianca: alta
nivel_evidencia: projeto
tags: [evidencia, metodologia, banca]
---

# Níveis de evidência

Os resultados do projeto são qualificados pela origem da validação. O nível não mede apenas desempenho numérico; ele limita o alcance da conclusão.

## E0 — hipótese

Proposição ainda não testada. Pode orientar um experimento, mas não sustenta conclusão de desempenho.

## E1 — benchmark exploratório

Resultado em dataset rotulado ou perturbação genérica, incluindo experimentos por artigo. Permite comparar métodos sob um protocolo declarado, sem afirmar validade industrial.

## E2 — validação sintética orientada pela FMECA

Resultado do pipeline principal com injeção de falhas fundamentada na FMECA. Sustenta a coerência interna da metodologia, mas não equivale a falha real de campo.

## E3 — validação experimental externa

Validação em bancada ou campo com falhas reais e protocolo externo. É o nível necessário para afirmações fortes de generalização operacional.

## Uso no agente

O nível acompanha a interpretação de resultados. Uma nota Obsidian pode registrar uma decisão sobre E1 ou E2, mas não eleva a evidência original nem substitui o artefato que contém os números.

## Conexões

- [[00 - Painel do cerebro]]
- [[Separação dos domínios de dados]]
