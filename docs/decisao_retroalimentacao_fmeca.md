# Registro de decisão — retroalimentação da FMECA

**Decidido por:** Rodolfo Torres · **Data:** 2026-08-03
**Status:** **DECIDIDO e implementado** (`src/ml/retroalimentacao_fmeca.py`)
**Para a orientadora:** este documento é o *registro* da decisão e da sua
justificativa, para revisão — não mais um pedido de decisão.

> **Por que a decisão mudou de dono.** A pergunta original — "qual régua
> converte a detectabilidade medida em índice D?" — pressupunha que houvesse
> uma régua a escolher. **Não há.** A Tab. 4.8 do TCC define o índice D em
> **percentual de não detectar**, e `1 − POD_mon` é exatamente essa grandeza. A
> conversão é a leitura da escala, publicada em 2024. Não sendo uma escolha
> metodológica, não precisa de arbitragem.

---

## A decisão em uma frase

O TCC atribuiu **D_campo** por julgamento de literatura; a dissertação **mede**
**POD_mon**. As duas grandezas **continuam distintas**, cada uma com nome
próprio, e o NPR recalculado é publicado como **cenário de sensibilidade em
tabela separada** — a FMECA oficial não é substituída.

---

## A contradição que existia, e como foi resolvida

| Documento | Posição anterior |
|---|---|
| `docs/fmeca.md` | D é dificuldade de detecção **em campo/manutenção** (Tab. 4.8 do TCC), conceitualmente **distinta** da detectabilidade empírica do Autoencoder. |
| `docs/retroalimentacao_fmeca.md` | Os resultados do detector **substituem** o D julgado, recalculando o NPR. |

Um componente pode ter D_campo baixo (fácil de detectar em campo, por inspeção
ou alarme do inversor) e ainda assim o Autoencoder ter dificuldade com sua
assinatura no sinal elétrico — ou o oposto. **São grandezas diferentes.**

**A resolução não foi escolher um lado.** Foi reconhecer que a contradição
vinha de as duas se chamarem "D": enquanto compartilhassem o nome, ou uma
substituía a outra, ou nada podia ser dito. Com nomes distintos
(`docs/nomenclatura_deteccao.md`), as duas coexistem e ambas as posições ficam
verdadeiras — `fmeca.md` está certo de que são distintas, e é justamente por
isso que o NPR projetado vai em tabela própria.

---

## A conversão — não é régua, é a escala

A régua por faixas que este documento propunha (`recall ≥ 0,90 → D = 2–3` etc.)
foi **descartada**. Ela tinha dois defeitos além de ser arbitrária: faixas com
dois valores exigem um segundo critério, não declarado, para escolher dentro da
faixa; e o intervalo `[2,3]` do topo era suficiente para decidir sozinho se a
ordem de criticidade invertia.

No lugar dela, a Tab. 4.8: `D_mon = faixa(1 − POD_mon(s_ref))`.

`S` e `O` permanecem inalterados: `O` só mudaria com dados de campo (E3), e `S`
é invariante à detecção.

---

## A emenda `min` — a única escolha que sobrou, e por quê

`D_proj = min(D_campo, D_mon)`.

A justificativa vem da lógica da situação, não dos resultados: **o
monitoramento proposto é adicional ao que já existe em campo.** Acrescentar um
detector não torna nenhuma falha mais difícil de detectar. Portanto o índice só
pode melhorar (diminuir) ou ficar igual — nunca piorar.

Sem a emenda, um componente com `D_campo` já baixo (IGBT = 3, Fusível = 2)
poderia sair **mais** crítico depois de instalado o monitoramento, o que é o
oposto do que retroalimentação deveria produzir.

O caso "falha nunca detectada → manter o D original" é **caso particular** do
`min` (`POD_mon = 0` → `D_mon = 10` → `min(D_campo, 10) = D_campo`), então a
emenda **simplifica** a regra em vez de acrescentar exceção. Está coberto por
teste (`tests/test_retroalimentacao_fmeca.py`).

---

## O resultado, com os artefatos vigentes

Severidade de referência `s_ref = 1,0`; limiar operacional congelado (p99);
`POD_mon` lido de `validacao_report.json`:

| Componente | S | O | D_campo | NPR oficial | POD_mon | não detecta | D_mon | D_proj | NPR projetado |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Contator AC | 5 | 7 | 9 | **315** | 1,000 | 0,0% | 1 | 1 | **35** |
| IGBT | 5 | 6 | 3 | **90** | 0,850 | 15,0% | 2 | 2 | **60** |
| Fusível AC | 5 | 3 | 2 | **30** | 1,000 | 0,0% | 1 | 1 | **15** |

**A ordem de criticidade inverte:** `IGBT > Contator AC > Fusível AC`.

### Por que isso é resultado, e não artefato

O NPR do Contator AC era **315 carregado quase todo pelo `D_campo` = 9** — uma
falha que a manutenção existente quase não pega. O detector proposto a pega
inteira, e a criticidade desaba para 35.

O IGBT já tinha `D_campo` = 3: sua criticidade **nunca veio de
detectabilidade**, e sim de `S×O` = 30, que nenhum monitoramento altera. Como o
detector também é quem menos o enxerga (0,850 → `D_mon` = 2), o `min` não cede,
e ele passa à frente.

Leitura para manutenção: **depois de instalado o monitoramento proposto, a
prioridade muda de componente.** É o ciclo do RCM fechando de verdade — não uma
tabela recalculada por formalidade.

### O que NÃO se pode concluir

- A inversão **não é mecânica**. Com `POD_mon = 1` nos três, a ordem se
  **preserva** (35 > 30 > 15). Quem inverte é o componente que o detector trata
  pior. Há teste para os dois casos.
- É `s_ref = 1,0`. Em severidade 0,3 o quadro se inverte de novo (`POD_mon` de
  0,600 / 0,150 / 0,125). O escalar **exige** a severidade declarada ao lado.
- Evidência **E2**. É "NPR projetado sob validação sintética", nunca NPR de
  campo (E3).

---

## Consequência sobre a prioridade de injeção

**Nada do que já foi produzido precisa ser rejustificado.** A prioridade de
injeção de falhas seguiu — e continua seguindo — o **NPR oficial** da FMECA
(Contator AC = 315 primeiro). O NPR projetado é *posterior ao detector*: ele
diz o que a manutenção deveria priorizar **depois** de instalado o
monitoramento, não o que a dissertação deveria ter injetado antes de tê-lo.

Usar o NPR projetado para escolher o que injetar seria circular — a saída do
detector decidindo a entrada do detector.

---

## Registro de honestidade metodológica

A salvaguarda anterior era "congelar a régua **antes** de olhar os números".
Ela não foi cumprida no sentido literal: quando a decisão foi tomada, os
recalls por componente já estavam nos artefatos do repositório e já haviam sido
lidos.

Isso é declarado aqui de propósito, e a defesa não é "não olhamos" — é que
**não havia régua a escolher**. As faixas são a Tab. 4.8 do TCC (2024); o único
grau de liberdade que restava, a emenda `min`, é justificado por um argumento
que independe de qualquer número (monitoramento é aditivo). Nenhuma escolha
deste projeto foi feita em função do resultado que produziria.

A régua descartada, por contraste, **tinha** esse grau de liberdade: com faixas
de dois valores (`D = 2–3`), escolher 2 ou 3 para o Contator decidia sozinho se
a ordem invertia. Era exatamente o tipo de folga que a salvaguarda existia para
impedir.

---

## Pendência factual (não metodológica)

`docs/fmeca.md` registra apenas os extremos da Tab. 4.8 (D=1 → 0–5%; D=10 →
86–100%). As oito faixas intermediárias em
`src/ml/retroalimentacao_fmeca.py::BORDAS_D` são **reconstrução aritmética**
forçada por esses extremos (80 pontos em 8 faixas de 10).

**Conferir na Tab. 4.8 do TCC.** Se divergirem, muda uma constante — e a
conclusão pode mudar com ela, já que o IGBT cai a 15,0% de não detecção, perto
da borda de 15%.

---

## O que fica registrado independentemente

Qualquer NPR recalculado herda **evidência E2** — validação sintética orientada
pela FMECA, não medição de campo. Deve ser rotulado "NPR projetado sob
validação sintética", nunca NPR de campo (E3).

E o caso de **SMD nula** (falha não detectada em nenhuma severidade) não deve
ser lido como "o componente está coberto": o ciclo do RCM recomendaria, nesse
caso, revisar o **método de detecção**, não a criticidade do item.

---

## Referências internas

- `docs/nomenclatura_deteccao.md` — os símbolos e por que estes
- `docs/fmeca.md` — FMECA consolidada, fonte única de S/O/D_campo
- `docs/retroalimentacao_fmeca.md` — o método, em detalhe
- `docs/evidence_levels.md` — o que E2 sustenta e o que não sustenta
- `src/ml/retroalimentacao_fmeca.py` — a conversão, implementada e testada
- `resultados/autoencoder/retroalimentacao_fmeca.md` — a saída vigente
