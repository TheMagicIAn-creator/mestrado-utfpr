# Evidence RAG — Evidence Guard R5

> Estado: guarda determinística integrada; promoção do retrieval permanece para R6.

## Integridade de evidência

| Métrica | Resultado |
|---|---:|
| Citation validity | 100.0% |
| Rejeição de claims inválidos | 100.0% |
| Acerto de abstenção | 100.0% |
| Rejeição de memória como fonte | 100.0% |
| Unsupported claim rate após guarda | 0.0% |

## Contrato

- Cadeia obrigatória: claim → evidence_id → chunk_id → document_id → página → PDF.
- Quotes são conferidas somente contra `raw_text` normalizado.
- Memória, Obsidian e sessões não podem se tornar fonte científica.
- Ausência de evidência produz abstenção explícita.
- Nenhum segundo LLM é usado como verificador documental.
