# FMECA consolidada

Esta é a fonte canônica dos três componentes usados na validação sintética E2.
Os índices foram estipulados pelo pesquisador com base no TCC e nas referências
cruzadas. Eles não são recalculados a partir do desempenho dos detectores.

## Conceitos

- FMEA identifica modos e efeitos de falha.
- FMECA acrescenta criticidade.
- `NPR = S x O x D_campo`.
- `D_campo` mede dificuldade de detecção no processo de manutenção e não deve
  ser confundido com probabilidade de detecção do monitoramento.

## Componentes

| Componente | Função | S | O | D_campo | NPR |
|---|---|---:|---:|---:|---:|
| Contator AC | conectar a saída CA à rede | 5 | 7 | 9 | 315 |
| IGBT | realizar o chaveamento da conversão CC-CA | 5 | 6 | 3 | 90 |
| Fusível AC | proteger o lado CA contra sobrecorrente | 5 | 3 | 2 | 30 |

Ordem de criticidade: Contator AC, IGBT, Fusível AC.

## Fundamentação da seleção

O TCC registra participações de chamados de 12% para Contator AC, 6% para IGBT
e 4% para Fusível AC. Software e PCB não entram no recorte porque o objetivo é
avaliar assinaturas elétricas observáveis pelos canais disponíveis. As
participações sustentam cenários bibliográficos, mas não são taxas de falha dos
componentes.

## Ponte para E2

| Componente | Hipótese elétrica incipiente | Limitação |
|---|---|---|
| Contator AC | conteúdo transitório na corrente `ia` | ruído gaussiano é proxy e requer calibração física |
| IGBT | aumento de harmônicos 5, 7, 11 e 13 nas três correntes | amplitudes são proxies, não envelhecimento medido |
| Fusível AC | redução parcial de `ia` e aumento de desbalanceamento | não reproduz abertura abrupta completa |

As fórmulas executáveis e limites pertencem a
`src/ml/assinaturas_fmeca.py`. A perturbação modela precursor incipiente, não a
ocorrência física do modo terminal.

## Fronteira inferencial

E2 mede a resposta do detector a hipóteses controladas. Ela não altera S, O,
D_campo ou NPR e não demonstra prevalência, tempo de vida ou desempenho em
campo. O resultado deve ser discutido como detectabilidade sintética.
