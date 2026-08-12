# Retroalimentação da FMECA — NPR projetado (E2)

Severidade de referência: **1.0** · Limiar operacional: **0.8577015399932861** — mse / percentil efetivo 99

| Componente | S | O | D_campo | NPR oficial | POD_mon | não detecta | D_mon | D_proj | NPR projetado |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Contator AC | 5 | 7 | 9 | **315** | 1.000 | 0.0% | 1 (Remota) | 1 | **35** |
| IGBT | 5 | 6 | 3 | **90** | 1.000 | 0.0% | 1 (Remota) | 1 | **30** |
| Fusível AC | 5 | 3 | 2 | **30** | 1.000 | 0.0% | 1 (Remota) | 1 | **15** |

Ordem oficial: contator_ac > igbt > fusivel_ac
Ordem projetada: contator_ac > igbt > fusivel_ac

> Evidência **E2**. NPR projetado sob validação sintética, não NPR de campo. A FMECA oficial permanece `docs/fmeca.md`.

> Escala de detecção transcrita de Torres (2024), Tabela 4.8, p. 50 — "probabilidade de não detectar a falha".