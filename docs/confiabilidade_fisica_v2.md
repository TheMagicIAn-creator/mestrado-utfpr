# Confiabilidade física V2

**Data da análise:** 13/08/2026  
**Natureza:** sensibilidade bibliográfica, não estimativa do GPVS-Faults  
**Modelo comum:** exponencial com taxa constante  
**Unidade canônica:** ano, adotando `1 ano = 8.760 h`

## Pergunta respondida

A camada compara o comportamento matemático de referências de confiabilidade de
inversores quando suas quantidades são convertidas para uma unidade comum. Ela
não usa o erro de reconstrução do autoencoder, a magnitude de injeção E2 ou os
ensaios GPVS para inferir vida útil.

O GPVS-Faults permanece como único dataset experimental principal do detector,
mas não possui o contrato necessário para confiabilidade física: tempos de vida
por ativo, exposição, censura, modo de falha e histórico de reparo.

## Fontes verificadas

| Cenário | Local auditado | Quantidade usada | Natureza |
|---|---|---:|---|
| Torres (2024), adaptado de Colli (2015) | Tabela 3.4, PDF p. 35 | `1,75 × 10^-4 falha/h` | transcrição bibliográfica secundária |
| Cristaldi et al. (2017) | PDF p. 5 | `0,125 falha/ano` | hipótese para o inversor no modelo string-BoS |
| Obeidat e Shuttleworth (2015), alta qualidade | Tabela III(a), PDF p. 5 | `8,069 falhas/10^6 h` | predição MIL-HDBK-217F N2 |
| Obeidat e Shuttleworth (2015), baixa qualidade | Tabela III(b), PDF p. 5 | `50,76 falhas/10^6 h` | predição MIL-HDBK-217F N2 |
| Dhople e Dominguez-Garcia (2012) | estudo de caso, PDF p. 6 | MTTF ilustrativo de `10 anos` | parâmetro de modelo Markov reparável |

Os quatro PDFs canônicos vivem em `literatura/inversores-pv/`; seus hashes
SHA-256 são gravados no resultado e no manifesto.

## Auditoria dimensional

O TCC apresenta posteriormente `1/(1,8 × 10^-4) = 5.555,55 anos`. Como a taxa
está em falhas por hora, o recíproco está em horas: `5.555,55 h`, aproximadamente
`0,634 ano`. A V2 não corrige o texto-fonte em silêncio. Ela usa o valor exato da
Tabela 3.4, calcula `1/λ = 0,652 ano` e publica a divergência como ressalva.

Em Cristaldi, `1/0,125 = 8 anos` corresponde ao inversor isolado sob a hipótese
exponencial. O MTTF próximo de seis anos mencionado no artigo pertence ao
string-BoS completo. Os dois números não são contraditórios: possuem escopos
diferentes.

As taxas de Obeidat são predições por componentes, qualidade e temperatura. O
próprio artigo informa que não há evidência de campo para confirmar que os
microinversores falham nessas frequências. O cenário de Dhople é ilustrativo e
inclui reparo; não é estimativa de frota.

## Funções e figuras

Para `t` em anos e `λ` em ano⁻¹:

- `R(t) = exp(-λt)`;
- `F(t) = 1 - exp(-λt)`;
- `f(t) = λ exp(-λt)` em ano⁻¹;
- `h(t) = λ` em ano⁻¹;
- `B_p = -ln(1-p)/λ`.

A densidade `f(t)` é uma curva analítica suave. Um gráfico de probabilidade com
pontos dispersos exigiria uma amostra de tempos de falha. Como ela não existe no
dataset atual, a V2 não publica `β`, `η`, papel de Weibull físico ou RUL.

As figuras respondem a perguntas diferentes:

1. `confiabilidade_cenarios`: probabilidade de operação sem falha;
2. `probabilidade_falha_cenarios`: probabilidade acumulada de falha;
3. `densidade_taxa_falha`: diferença entre densidade e taxa instantânea;
4. `marcos_confiabilidade`: B1, B10, mediana e tempo recíproco `1/λ`.

Cada figura é exportada em PNG a 300 dpi e PDF vetorial. As escalas logarítmicas
são declaradas no eixo e na nota metodológica.

## Reprodução

```powershell
.\.venv\Scripts\python.exe -m scripts.gerar_confiabilidade_fisica_v2
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q -W ignore `
  tests\test_confiabilidade_fisica_v2.py `
  tests\test_confiabilidade_fisica_resultados.py
```

Os artefatos citáveis ficam em `resultados/v2/confiabilidade/`. O JSON consolidado
é o contrato para a aplicação web; o frontend não recalcula valores científicos.

