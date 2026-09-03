# Níveis de evidência

| Nível | Definição | Situação no projeto |
|---|---|---|
| E0 | hipótese ou proposta não testada | ideias futuras e calibração física pendente |
| E1 | demonstração computacional preliminar | testes unitários e dados artificiais pequenos |
| E2 | varredura de severidade sobre falha sintética | injeção nos três itens da FMECA vigente sobre o sinal saudável F0 |
| E3 | validação experimental de bancada | F1L-F7M do GPVS-Faults |
| E4 | validação de campo | não realizada |

E2 não é um degrau intermediário entre E1 e E3 — é outra pergunta. E3 mede
detecção em falha real, e é binária: o ensaio tem falha ou não tem. E2 mede a
magnitude em que a detecção começa, sobre falha construída, e por isso tem
eixo contínuo. Um número de E2 nunca substitui um de E3, e vice-versa.

Dentro de E2, o método de injeção precisa acompanhar o número: assinatura
elétrica (IGBT, sensor/realimentação) é fundamentada na física da falha;
interpolação entre estados medidos (sistema de controle) não é simulação
física e não pode ser apresentada como tal.

E3 de bancada não implica desempenho industrial. Confiabilidade física
bibliográfica é cenário de sensibilidade e deve ser rotulada por sua origem,
sem receber nível experimental que os dados não sustentam.

Toda afirmação de resultado deve informar dataset, protocolo, unidade de
análise e nível de evidência. Toda extrapolação deve ser identificada como tal.
