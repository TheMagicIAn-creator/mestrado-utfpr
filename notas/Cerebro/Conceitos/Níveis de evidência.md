---
titulo: Níveis de evidência
tipo: conceito
status: ativo
confianca: alta
nivel_evidencia: projeto
al_iado: true
tags: [cerebro, evidencia, metodologia, banca, fmeca, evidencia-e2]
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

O GPVS-Faults realizou E3 de bancada em protocolo temporal separado. Isso não equivale a campo: o alcance continua limitado aos ensaios experimentais e deve ser lido no artefato `resultados/gpvs/validacao_gpvs_e3.json`.

## Uso no agente

O nível acompanha a interpretação de resultados. Uma nota Obsidian pode registrar uma decisão sobre E1, E2 ou E3, mas não eleva a evidência original nem substitui o artefato que contém os números.

## Conexões

- [[00 - Painel do cerebro]]
- [[Separação dos domínios de dados]]
