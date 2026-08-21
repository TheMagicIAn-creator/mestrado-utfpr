# Nomenclatura de detecção

## Campo e monitoramento

`D_campo` é o índice da FMECA e cresce com a dificuldade de detectar a falha no
processo de manutenção. Ele é uma avaliação de engenharia.

A saída dos Autoencoders é medida diretamente por erro de reconstrução,
limiar, decisão binária e métricas com intervalo de confiança. O projeto não
converte automaticamente esse desempenho em um novo NPR.

## Magnitude sintética

- `a_det`: fator aplicado à assinatura sintética, adimensional.
- taxa de detecção: proporção de janelas detectadas em uma magnitude.
- SMD95: menor magnitude com limite inferior do IC95% de detecção maior ou
  igual a 95%.
- primeiro cruzamento: magnitude do primeiro evento persistente por trajetória.

Evite "severidade" sem qualificador: S da FMECA e `a_det` não possuem a mesma
escala nem a mesma origem.

## Tempo físico

Somente as curvas bibliográficas usam `t` em horas ou anos. `R(t)`, `F(t)`,
`f(t)` e `h(t)` não podem ser calculadas a partir de `a_det`. Termos como vida
útil, RUL, MTTF e taxa de falha são proibidos para gráficos E2.
