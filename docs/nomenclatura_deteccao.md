# Nomenclatura da detecção — FONTE ÚNICA

> **Por que este documento existe.** O projeto tem DUAS grandezas ligadas à
> palavra "detecção", e elas apontam para lados opostos. Enquanto ambas se
> chamarem "D", qualquer frase da dissertação é ambígua e a banca tem razão em
> cobrar. Aqui elas ganham nomes distintos, e a conversão entre elas deixa de
> ser uma escolha nossa.

## As quatro grandezas

| Símbolo | Nome | Escala | Direção | Origem |
|---|---|---|---|---|
| **D_campo** | Dificuldade de detecção em campo | ordinal 1–10 | maior = pior | **julgada** (TCC, Tab. 4.8) |
| **POD_mon(s)** | Probabilidade de detecção pelo monitoramento proposto | [0, 1] | maior = melhor | **medida** (E2) |
| **D_mon** | O mesmo índice da Tab. 4.8, obtido por medição | ordinal 1–10 | maior = pior | derivado de POD_mon |
| **SMD** | Severidade mínima detectável | severidade (0–1) | menor = melhor | medida (E2) |

### D_campo — o índice da FMECA

Índice de **dificuldade de detecção** atribuído por julgamento de literatura e
engenharia (Torres, 2024, Tab. 4.8). É o `D` de `NPR = S × O × D`. Refere-se
aos meios de detecção **já existentes em campo**: inspeção, alarme do inversor,
rotina de manutenção.

O subscrito `campo` não é decoração. A Tab. 4.8 define o índice em **percentual
de NÃO detectar** (D=1 → 0–5%; D=10 → 86–100%), ou seja: apesar do nome
"Detecção", `D` cresce com o **fracasso** em detectar. Escrever `D_campo`
lembra ao leitor de qual detecção se fala e em que sentido ela corre.

**Os valores da FMECA consolidada não mudam** (`docs/fmeca.md` continua a fonte
única): Contator AC 9, IGBT 3, Fusível AC 2. Só o rótulo ganhou subscrito.

### POD_mon(s) — o que o detector proposto mede

Fração de janelas com a falha injetada na severidade `s` que o Autoencoder
sinaliza acima do **limiar operacional congelado**, sob o protocolo E2.

Definição operacional completa — sem estes quatro qualificadores o símbolo é
ambíguo por construção:

1. **protocolo**: E2 — injeção sintética orientada pela FMECA no sinal bruto;
2. **ponto de operação**: limiar operacional de `limiar.json`, congelado antes
   de ver qualquer falha (não é um ponto escolhido no teste);
3. **severidade**: `s`, um ponto da grade de injeção — POD_mon é uma **curva**,
   não um escalar;
4. **estimador**: proporção amostral sobre as janelas do holdout, sempre
   reportada com IC95 (n pequeno; ver `validacao_report.json`).

Quando um escalar for inevitável (uma tabela, um índice derivado), usa-se a
**severidade de referência `s_ref = 1,0`** — a assinatura incipiente plenamente
desenvolvida, a mais próxima do modo terminal que a FMECA classifica. Isso é
declarado, nunca implícito, e a curva completa acompanha o número.

### D_mon — a ponte

`D_mon` é o índice da **mesma Tab. 4.8**, lido a partir de `1 − POD_mon(s_ref)`.
Não é um índice rival: é o mesmo índice, obtido por **medição** em vez de
julgamento, para um meio de detecção diferente (o monitoramento proposto).

### SMD — já existia, agora com ancestral

`SMD` é a menor severidade em que a falha é detectada. É o análogo direto do
**a₉₀** do MIL-HDBK-1823A — o menor tamanho de defeito detectado com
probabilidade 90% em ensaios não destrutivos. O projeto já usava o conceito;
agora ele tem linhagem citável.

---

## A conversão POD_mon → D_mon NÃO é uma escolha nossa

Este é o ponto que dispensa a decisão metodológica que estava pendente.

A Tab. 4.8 do TCC define o índice em **percentual de não detectar**.
`1 − POD_mon` é exatamente essa grandeza. Logo a conversão é a **leitura da
escala**, não uma régua a calibrar:

| D | Não detecta | | D | Não detecta |
|---:|---:|---|---:|---:|
| 1 | 0 – 5% | | 6 | 46 – 55% |
| 2 | 6 – 15% | | 7 | 56 – 65% |
| 3 | 16 – 25% | | 8 | 66 – 75% |
| 4 | 26 – 35% | | 9 | 76 – 85% |
| 5 | 36 – 45% | | 10 | 86 – 100% |

> ⚠️ **A CONFERIR na Tab. 4.8 do TCC.** `docs/fmeca.md` registra apenas os dois
> extremos (D=1 → 0–5%; D=10 → 86–100%). As oito faixas intermediárias acima
> são **reconstrução aritmética** forçada por esses extremos: 80 pontos
> percentuais (6–85) divididos em 8 faixas de 10. Se a Tab. 4.8 usar outras
> faixas, valem as dela — e `src/ml/retroalimentacao_fmeca.py` muda em uma
> constante.

**Por que isso encerra a circularidade.** A objeção que travava a
retroalimentação era: "se vocês escolheram as faixas depois de ver os
resultados, calibraram a régua para o resultado desejado". A objeção morre
porque as faixas **não são nossas** — são do TCC, publicado em 2024, antes de
qualquer medição do detector. Não há o que congelar, porque não há o que
escolher.

---

## Letras descartadas, com o motivo

| Candidata | Veredito | Motivo |
|---|---|---|
| **U** | **fatal** | `U` é o símbolo IEC/ABNT de **tensão** (IEC 60027). Numa dissertação sobre o lado CA de um inversor, `U = 0,87` ao lado de `U_ab = 380 V` é indefensável. |
| **E** | fatal | Triplamente ocupada: `E1/E2/E3` (níveis de evidência do próprio projeto), energia, campo elétrico. |
| **POD** *(sem subscrito)* | descartada | Em sistemas de potência, **POD = Power Oscillation Damping** — e o uso corrente é justamente em conversores conectados à rede, o domínio desta dissertação. O subscrito `mon` é obrigatório, não opcional. |
| **PD** | descartada | **Partial Discharge** / descarga parcial — colisão viva, já que um dos modos de falha da FMECA é degradação de isolamento. |
| letra única qualquer | inviável | O alfabeto está esgotado neste projeto: S, O, D, C (FMECA), R (confiabilidade/RUL), P, I, U, V, F, Z (eletrotécnica), β, η (Weibull), k (top-k do escore). |

**Conclusão:** não há letra única livre. A saída é sigla consagrada com
subscrito obrigatório — que também é a saída mais defensável, porque adota
nomenclatura existente em vez de inventar.

## Procedência da raiz POD

**MIL-HDBK-1823A** (7 abr. 2009), *Nondestructive Evaluation System Reliability
Assessment* — o manual que padroniza a **curva POD** e a métrica **a₉₀/₉₅** (o
tamanho de defeito detectado com 90% de probabilidade e 95% de confiança) para
o programa de integridade estrutural da USAF.

A correspondência de forma é exata, e é o que justifica adotar a raiz:

| Ensaios não destrutivos (END) | Este projeto |
|---|---|
| POD em função do tamanho de defeito `a` | POD_mon em função da severidade `s` |
| curva POD(a) | curva de detecção por severidade (já plotada) |
| a₉₀ / a₉₀/₉₅ | SMD (com IC95) |
| confiabilidade do ENSAIO, não do componente | capacidade do MONITORAMENTO, não do inversor |

Essa última linha é a que mais importa: em END, POD mede a confiabilidade do
**método de inspeção** — exatamente o que se quer aqui, e exatamente o que o
`D` da FMEA sempre tentou capturar por julgamento.

**Não há termo consagrado para a grandeza exata** ("capacidade empírica, por
modo de falha e resolvida por severidade, de um detector de ML enxergar a
assinatura elétrica da falha"). Os vizinhos foram considerados e recusados:

- **DC — Diagnostic Coverage** (IEC 61508 / ISO 26262): é fração de *taxas de
  falha perigosa* coberta por diagnóstico automático, não probabilidade por
  modo. Domínio de segurança funcional, com implicações de SIL/ASIL que não se
  aplicam aqui.
- **d′** (teoria de detecção de sinal): índice de sensibilidade independente de
  limiar — é parente da AUC, não do recall num ponto de operação.
- **FDR / FIR** (MIL-STD-2165, IEEE Std 1522): percentuais agregados de
  testabilidade por BIT, sem eixo de severidade.
- **Diagnosability** (Sampath et al., 1995): propriedade booleana de sistemas a
  eventos discretos. Não é escalar.
- **Monitorability** (verificação em tempo de execução): homônimo tentador, sem
  relação com a grandeza.
- **Recall / TPR**: é literalmente o mesmo número, e continua sendo o nome
  correto **dentro** da avaliação de ML. `POD_mon` é o mesmo valor lido como
  **propriedade do método de inspeção**, com eixo de severidade — que é o que
  permite a ponte com a FMECA. Usar os dois nomes não é redundância: é dizer em
  que papel o número está sendo usado.

## Como escrever

- Primeira ocorrência no texto: *"probabilidade de detecção pelo monitoramento
  proposto (POD_mon, na acepção do MIL-HDBK-1823A)"*.
- Em tabelas, cabeçalho por extenso — nunca a sigla nua, para não colidir com
  *Power Oscillation Damping* na leitura de quem passa os olhos.
- Em equações e gráficos: `POD_mon(s)`, sempre com o subscrito.
- Nunca escrever `D` sozinho: ou `D_campo`, ou `D_mon`.

## Referências internas

- `docs/fmeca.md` — FMECA consolidada, fonte única de S/O/D_campo
- `docs/retroalimentacao_fmeca.md` — como o NPR projetado é calculado
- `docs/decisao_retroalimentacao_fmeca.md` — a decisão e seu registro
- `docs/glossario.md` — verbetes curtos
- `src/ml/retroalimentacao_fmeca.py` — a conversão, implementada e testada
