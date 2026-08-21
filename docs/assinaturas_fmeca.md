# Assinaturas sintéticas da FMECA

Contrato operacional: `src/ml/assinaturas_fmeca.py`.

`a_det` é uma magnitude adimensional entre 0 e 1. Janelas, magnitudes e
sementes são compartilhadas entre Denso e AE-LSTM para garantir comparação
pareada.

| Componente | Colunas | Fórmula resumida |
|---|---|---|
| Contator AC | `ia` | `ia += N(0, a_det * std(ia) * 0,30)` |
| IGBT | `ia`, `ib`, `ic` | soma harmônica ponderada em 5, 7, 11 e 13 vezes 50 Hz |
| Fusível AC | `ia` | `ia *= 1 - 0,12 * a_det` |

As assinaturas atuam no sinal bruto e as 24 features são extraídas depois da
injeção. A ordem evita perturbar diretamente atributos estatísticos sem uma
hipótese de sinal.

O contator usa uma aproximação estocástica de chattering; o IGBT usa conteúdo
harmônico; o fusível usa perda parcial de fase. Todas são hipóteses de
degradação incipiente e exigem calibração física antes de aplicação industrial.

O limiar de cada modelo permanece congelado. SMD95 só é reportada quando o
limite inferior do IC95% de detecção alcança 95%. Weibull 2P é opcional,
diagnóstica e não substitui as funções empíricas.
