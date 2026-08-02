# Folha de decisão — retroalimentação do D da FMECA

**Para:** Profª. Fernanda Cristina Correa
**De:** Rodolfo Torres
**Status:** decisão pendente — **congelar antes de olhar os números**

---

## A decisão em uma frase

O TCC atribuiu o índice **D** (Detecção) por julgamento de literatura. A
dissertação **mede** a capacidade de detecção do monitoramento proposto. A
pergunta é: **o D medido substitui o D julgado, recalculando o NPR?**

---

## Por que isso precisa de decisão, e não de implementação

Existe hoje uma **contradição entre dois documentos do projeto**, e ela não foi
resolvida por conveniência de nenhum lado:

| Documento | Posição |
|---|---|
| `docs/fmeca.md` | D é dificuldade de detecção **em campo/manutenção** (Tab. 4.8 do TCC), conceitualmente **distinta** da detectabilidade empírica do Autoencoder. A relação entre as duas é "um resultado a discutir", não uma substituição. |
| `docs/retroalimentacao_fmeca.md` | Os resultados do detector **substituem** o D julgado por um D medido, recalculando o NPR e fechando o ciclo do RCM. |

Um componente pode ter D baixo (fácil de detectar em campo, por inspeção ou
alarme do inversor) e ainda assim o Autoencoder ter dificuldade com sua
assinatura no sinal elétrico — ou o oposto. **São grandezas diferentes.**

---

## A régua proposta (a congelar)

De `docs/retroalimentacao_fmeca.md`, ainda **não** validada:

| Recall no limiar operacional | D novo |
|---|---|
| ≥ 0,90 | 2 – 3 |
| 0,50 – 0,89 | 4 – 6 |
| < 0,50 **ou** SMD nula | manter o D original |

`S` e `O` permanecem inalterados: `O` só mudaria com dados de campo (E3), e `S`
é invariante à detecção.

---

## Consequência aritmética — o que a banca vai perguntar

`S×O` é fixo por componente, então `NPR = (S×O) × D`:

| Componente | S×O | NPR hoje | D=2 | D=3 | D=4 | D=6 | manter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Contator AC | 35 | **315** | 70 | 105 | 140 | 210 | 315 |
| IGBT | 30 | **90** | 60 | 90 | 120 | 180 | 90 |
| Fusível AC | 15 | **30** | 30 | 45 | 60 | 90 | 30 |

Dois efeitos que não são óbvios e precisam de decisão consciente:

**1. O NPR pode SUBIR.** IGBT e Fusível AC já têm D julgado baixo (3 e 2). Se o
detector os colocar na faixa intermediária (D=4–6), a criticidade **aumenta** —
IGBT vai de 90 para até 180. Um componente que o detector trata razoavelmente
bem ficaria classificado como *mais* crítico do que antes.

**2. A ordem de criticidade pode inverter.** Testando todas as combinações de
faixa para os três componentes: **28 das 64 invertem a ordem atual**
(Contator AC > IGBT > Fusível AC).

Exemplo concreto: se o Contator cair em D=2 e o IGBT em D=4, a ordem vira
**IGBT (120) > Contator AC (70) > Fusível AC (30)**.

Isso não é detalhe de tabela. **A prioridade de injeção de falhas da dissertação
segue o NPR** — o Contator AC é hoje a primeira falha injetada por ter NPR=315.
Inverter a ordem depois de os resultados estarem produzidos exigiria reescrever
a justificativa metodológica do capítulo.

---

## A salvaguarda que torna isso defensável

**Congelar a régua antes de olhar os números.**

Se as faixas forem escolhidas depois de conhecer o recall de cada componente, a
retroalimentação vira circular: a régua terá sido calibrada para produzir o
resultado desejado, e uma única pergunta da banca desmonta o argumento.

Por isso **nenhuma linha de código implementa essa conversão**, e os números de
recall por componente **não** estão nesta folha. A tabela acima é aritmética
pura sobre a régua — não diz em que faixa cada componente cai.

---

## Nossa recomendação

Levamos uma proposta, não só um leque. Duas partes.

### Parte 1 — emendar a régua: `D_novo = min(D_original, D_medido)`

A régua como está tem um **defeito**: faz o NPR **subir** em **56 das 64**
combinações. Um componente que o detector trata bem ficaria classificado como
*mais* crítico do que antes — o oposto do que retroalimentação deveria produzir.

A emenda vem da lógica da situação, não dos resultados: **o monitoramento
proposto é adicional ao que já existe em campo.** Acrescentar um detector não
torna nenhuma falha mais difícil de detectar. Portanto D só pode melhorar
(diminuir) ou ficar igual — nunca piorar.

Efeito da emenda:

| Régua | NPR sobe | Ordem de criticidade inverte |
|---|---:|---:|
| Substituir D (como proposto) | **56/64** | 28/64 |
| `min(D_original, D_medido)` | **0/64** | **12/64** |

O caso `SMD nula → manter D original` já é um caso particular do `min`, então a
emenda **simplifica** a régua em vez de acrescentar exceção.

### Parte 2 — adotar a opção C (cenário paralelo)

A FMECA original permanece **oficial**; o NPR recalculado entra como **análise
de sensibilidade**, rotulada "NPR projetado sob validação sintética (E2)".

Por quê, e não a substituição direta:

- **Não exige assumir o que `fmeca.md` nega.** A substituição direta pressupõe
  que detectabilidade em sinal e em campo são a mesma grandeza. Como cenário
  paralelo, a pergunta vira "o que aconteceria com a criticidade *se* o
  monitoramento proposto estivesse instalado?" — que é respondível sem igualar
  as duas grandezas.
- **Preserva a narrativa metodológica.** A prioridade de injeção segue o NPR
  original; nada do que já foi produzido precisa ser rejustificado.
- **Fecha o ciclo do RCM mesmo assim** — que é a contribuição que liga o TCC à
  dissertação.
- **É prática padrão** apresentar recálculo sob evidência mais fraca como
  análise de sensibilidade, não como substituição do valor de referência.

O custo é ter duas tabelas. Numa dissertação, isso é uma tabela a mais, não um
problema.

### O que ainda precisa da orientadora

Mesmo com a recomendação, três pontos são dela:

1. **Aceitar ou recusar a emenda `min`** — é decisão metodológica, não técnica.
2. **Congelar a régua agora**, antes de qualquer número ser olhado. Se a
   decisão vier depois dos resultados, o argumento fica circular e a banca
   desmonta.
3. **Decidir o que fazer com a ressalva de `fmeca.md`** — mantê-la como está
   (as grandezas são distintas, e o cenário paralelo não as iguala) ou revisá-la.

---

## Opções

**A. Substituir D pelo medido.** Fecha o ciclo do RCM de forma explícita e é o
argumento mais forte de "o mestrado estende o TCC". Custo: assume que
detectabilidade em sinal e detectabilidade em campo são a mesma grandeza — o que
`fmeca.md` hoje nega. Exige assumir e defender essa equivalência.

**B. Não substituir; reportar lado a lado.** Mantém `fmeca.md` como está e
apresenta uma coluna "detectabilidade empírica (E2)" ao lado do D julgado, sem
recalcular NPR. Mais conservador e sem risco de reordenação. Custo: o ciclo do
RCM não fecha numericamente — fica como discussão qualitativa.

**C. Substituir, mas como cenário paralelo.** Mantém a FMECA original como
oficial e apresenta o NPR recalculado como **análise de sensibilidade**, deixando
claro que é projeção sob evidência E2. Preserva a narrativa atual e mostra o
ciclo fechando. Custo: duas tabelas para a banca acompanhar.

---

## O que fica registrado independentemente da escolha

Qualquer NPR recalculado herda **evidência E2** — validação sintética orientada
pela FMECA, não medição de campo. Deve ser rotulado "NPR projetado sob validação
sintética", nunca NPR de campo (E3).

E o caso de **SMD nula** (falha não detectada em nenhuma severidade) não deve
ser lido como "o componente está coberto": o ciclo do RCM recomendaria, nesse
caso, revisar o **método de detecção**, não a criticidade do item.

---

## Referências internas

- `docs/fmeca.md` — FMECA consolidada, fonte única de S/O/D
- `docs/retroalimentacao_fmeca.md` — a proposta (E0) em detalhe
- `docs/evidence_levels.md` — o que E2 sustenta e o que não sustenta
- `resultados/macro/` — desempenho do detector por componente
