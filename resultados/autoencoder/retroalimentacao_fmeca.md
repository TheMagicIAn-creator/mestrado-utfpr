# Retroalimentação da FMECA — NPR projetado (E2)

Severidade de referência: **1.0** · Limiar operacional: **7.826175715408156** — percentil 99.9 (auto-calibrado)

| Componente | S | O | D_campo | NPR oficial | POD_mon | não detecta | D_mon | D_proj | NPR projetado |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Contator AC | 5 | 7 | 9 | **315** | 1.000 | 0.0% | 1 | 1 | **35** |
| IGBT | 5 | 6 | 3 | **90** | 0.850 | 15.0% | 2 | 2 | **60** |
| Fusível AC | 5 | 3 | 2 | **30** | 1.000 | 0.0% | 1 | 1 | **15** |

Ordem oficial: contator_ac > igbt > fusivel_ac
Ordem projetada: igbt > contator_ac > fusivel_ac

> **A ordem de criticidade inverte.** Não é artefato: o monitoramento entrega mais onde a detecção em campo era pior. Um componente cujo NPR era carregado por D_campo alto cai muito; um cuja criticidade vem de S×O quase não se move — e passa à frente.

> Evidência **E2**. NPR projetado sob validação sintética, não NPR de campo. A FMECA oficial permanece `docs/fmeca.md`.

> As faixas intermediárias da escala D são reconstrução aritmética — conferir na Tab. 4.8 do TCC (ver `docs/nomenclatura_deteccao.md`).