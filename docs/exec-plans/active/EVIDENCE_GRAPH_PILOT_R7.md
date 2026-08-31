# Evidence Graph — piloto R7

> Estado: piloto leve posterior à promoção R6; não substitui o RAG híbrido.

## Escopo

- Taxonomia curada com 25 entidades de componentes, métodos, dados, métricas,
  modos de falha e estratégias de manutenção.
- Nós de documento, autor, evidência e entidades literalmente mencionadas.
- Arestas `STUDIES` e `SUPPORTED_BY`.
- Toda aresta exige ao menos um `evidence_id` e um `chunk_id`.
- Memória, Obsidian e sessões não entram no grafo científico.

## Uso

O grafo é construído sobre o Evidence Package da pergunta atual somente quando
a consulta pede relação ou encadeamento multi-hop. O resumo enviado ao Router
mantém os marcadores `[E#]`. Consultas simples continuam no RAG híbrido promovido
na R6.

## Limites explícitos

- Não é GraphRAG completo.
- Não há banco de grafo externo.
- Nenhum LLM extrai ou inventa relações.
- Ausência de termo literal não gera entidade.
- RAPTOR continua fora do caminho crítico.
- O texto original do PDF permanece a fonte científica.
