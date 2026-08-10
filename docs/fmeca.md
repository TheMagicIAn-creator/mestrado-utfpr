# FMECA consolidada — FONTE ÚNICA DE VERDADE

> **Esta tabela é a fonte única da análise de falhas do projeto.** Todos os
> arquivos (injeção de falhas, assinaturas, CLAUDE.md, gráficos, respostas do
> agente) devem usar EXATAMENTE estes componentes, modos, índices e NPR.
> Em caso de conflito entre arquivos, vale esta tabela. Alterou aqui →
> propague a `src/ml/injecao_falhas.py`, `src/ml/protocolos_artigos.py`,
> `docs/assinaturas_fmeca.md` e `CLAUDE.md`.

## Distinção FMEA × FMECA (Torres, 2024, Seção 2.2 e Eq. 4.19–4.21)

- **FMEA** = Failure Mode and Effects Analysis — modos e efeitos de falha.
- **FMECA** = FMEA + **C**riticidade. `FMECA = FMEA + C`.
- **NPR** (Número de Prioridade de Risco) = **S × O × D** — índice da **FMECA**
  (criticidade quantitativa), **nunca** atribuído genericamente à FMEA nem
  igualado a um índice isolado (D não é NPR).
- **C** (criticidade, Eq. 4.20) = **S + O**.

## Escalas (Torres, 2024, Tabelas 4.6, 4.7, 4.8)

- **S — Severidade** (escala 1–5): 1 = dificilmente detectada, sem influência;
  2 = leve deterioração; 3 = deterioração de peças/desempenho; 4 = deterioração
  extensa + perda relevante; 5 = não-operação ou perda severa.
- **O — Ocorrência** (escala 1–10): 1 = remota (10⁻⁷/ano); 2–3 = baixa (10⁻⁶);
  4–5 = moderada (10⁻⁵); 6–7 = alta (10⁻⁴); 8–10 = muito alta (10⁻²).
- **D_campo — Detecção em campo** (escala 1–10), transcrita da Tab. 4.8
  (p. 50), colunas "Probabilidade de Não Detectar a Falha" / "Probabilidade do
  Defeito Afetar o Cliente (%)" / "Rank":

  | D | Não detecta | Rótulo | | D | Não detecta | Rótulo |
  |---:|---:|---|---|---:|---:|---|
  | 1 | 0 – 5% | Remota | | 6 | 46 – 55% | Moderada |
  | 2 | 6 – 15% | Baixa | | 7 | 56 – 65% | Alta |
  | 3 | 16 – 25% | Baixa | | 8 | 66 – 75% | Alta |
  | 4 | 26 – 35% | Moderada | | 9 | 76 – 85% | Muito alta |
  | 5 | 36 – 45% | Moderada | | 10 | 86 – 100% | Muito alta |

> **Leia com atenção o que a escala mede.** Apesar do nome "Detecção", a Tab.
> 4.8 é definida em **percentual de NÃO detectar**: o índice cresce com o
> FRACASSO em detectar. Isso tem duas consequências. (1) O subscrito `campo`
> passa a ser obrigatório, para separar este índice **julgado** da
> detectabilidade **medida** do detector proposto (`POD_mon`) — nunca escrever
> `D` sozinho. (2) A conversão de uma para a outra **não é uma régua que
> escolhemos**: `1 − POD_mon` é exatamente a grandeza em que a escala está
> escrita. Fonte única da nomenclatura: `docs/nomenclatura_deteccao.md`.

## Seleção dos componentes (justificativa bibliográfica)

Os três componentes abaixo são os **componentes internos CA-elétricos do
inversor que mais contribuem para falhas**, segundo a **Tabela 3.3 do TCC**
(adaptada de Cristaldi, Khalil & Soulatintork, 2017), reforçada por Golnas
(2012, Tab. 3.2 — inversor = 43% dos tickets, 36% da energia perdida) e Voss
et al. (2009, Fig. 3.16–3.17 — inversor como componente crítico):

| Componente | % tickets (Tab. 3.3) | % kWh perdidos | Detectável no sinal CA? |
|---|---:|---:|---|
| Contator AC | 12% | 13% | Sim (transientes de comutação) |
| IGBT | 6% | 6% | Sim (harmônicos de chaveamento) |
| Fusível AC | 4% | 12% | Sim (perda parcial de fase) |

Software de Controle (28%) e Placa de Circuito/PCB (13%), embora liderem a
Tab. 3.3, **não se manifestam em sinais elétricos CA** (lógica/placa) — por
isso ficam fora do escopo de detecção por Autoencoder no sinal.

## Tabela FMECA consolidada

Modo de falha / Efeito / Causa: **campos reservados para preenchimento por
Rodolfo Torres** (deixados em branco por decisão do autor).

| Id | Componente | Função | Modo de falha                                                                                                   | Efeito                                                                                                        | Causa                                                                    | S | O | D | **NPR** | **C** |
|----|-----------|--------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|:-:|:-:|:-:|:---:|:-:|
| 1 | **Contator AC** | Chavear/conectar a saída CA do inversor à rede | Fuga de corrente; Injeção contínua de energia ainda com a falta de energia da concessionária ou oscilação severa. | Paralisação do sistema por falha no isolamento; Riscos de eletrocussão em técnicos operando na rede externa.| Arco elétrico e desgaste mecânico; Degradação da bobina e do isolamento. | 5 | 7 | 9 | **315** | 12 |
| 2 | **IGBT** | Comutar a conversão CC→CA (chaveamento PWM) | Não comutação CC→CA; Curto-circuito permanente entre os terminais.                          | Interrupção imediata no forncecimento de energia e (possível) disparo de alarme de hardware no display.       | Estresse termodinâmico e Surtos de sobretensão.                          | 5 | 6 | 3 | **90** | 11 |
| 3 | **Fusível AC** | Proteger o lado CA contra sobrecorrente | Interrupção da condução de corrente (abertura do elo fusível).  | Isolamento de uma ou mais fases da saída AC; Interrupção no fornecimento de energia; Desarme do inversor por desbalanceamento de fases. | Fadiga Térmica por ciclos de carga; Surtos de rede.                      | 5 | 3 | 2 | **30** | 8 |

**Ordem de criticidade (NPR): Contator AC (315) > IGBT (90) > Fusível AC (30).**

- Fonte dos índices S/O/D: **estipulados pelo pesquisador (Rodolfo Torres)**
  com base no TCC (Torres, 2024) e nas suas referências cruzadas — Tab. 3.3
  (Cristaldi et al., 2017), Golnas (2012), Voss et al. (2009) — seguindo as
  escalas das Tabelas 4.6/4.7/4.8. NPR = S×O×D calculado.
- Nível de evidência: **E1/E2** (fundamentação bibliográfica + engenharia +
  injeção sintética), **não** medição de campo (E3).

## Assinatura elétrica de cada falha (ponte FMECA → injeção sintética)

A injeção sintética (`src/ml/injecao_falhas.py`) perturba janelas saudáveis F0
do GPVS-Faults com a assinatura elétrica coerente com cada componente:

| Componente | Assinatura elétrica injetada | Mecanismo no código |
|---|---|---|
| Contator AC | Transiente/ruído de comutação (contatos desgastados/chattering) na corrente CA | ruído gaussiano em `i_a` (proxy do transiente) |
| IGBT | Harmônicos 5ª/7ª/11ª/13ª + THD ↑ (chaveamento imperfeito) | harmônicos aditivos nas correntes CA |
| Fusível AC | Redução de amplitude de uma fase (perda parcial) → desbalanceamento ↑ | redução multiplicativa da amplitude de `i_a` |

**Modo incipiente vs. modo terminal (declaração metodológica).** Os modos de
falha da tabela FMECA acima são **terminais/catastróficos** (IGBT em curto
permanente, contator com fuga/injeção indevida, fusível aberto). A injeção
sintética, porém, modela deliberadamente a **assinatura elétrica INCIPIENTE**
— o precursor de degradação que ANTECEDE o modo terminal (harmônicos
crescentes do IGBT, transientes de comutação do contator, perda parcial de
fase do fusível). Isso é intencional e alinhado ao propósito da manutenção
preditiva: **detectar a degradação antes da falha catastrófica**, quando ainda
há RUL a estimar. Não há, portanto, contradição entre a FMECA (que classifica
a criticidade do modo terminal) e a injeção (que treina o detector na fase
incipiente) — são estágios distintos e complementares do mesmo modo de falha.

## Ressalva metodológica (importante para a banca)

O índice **D_campo** (dificuldade de detecção **em campo/manutenção**, Tab.
4.8) e a **detectabilidade empírica do Autoencoder** (`POD_mon`) são conceitos
distintos. Um componente pode ter D_campo baixo (fácil de detectar em campo) e,
ainda assim, o Autoencoder ter dificuldade com sua assinatura no sinal — ou
vice-versa. Essa relação (o detector proposto melhora, iguala ou fica aquém do
D_campo?) é um resultado a discutir na dissertação, não uma inconsistência.

> **Tensão RESOLVIDA (2026-08-03).** `docs/retroalimentacao_fmeca.md` propunha
> **substituir** o D julgado pelo D medido; o parágrafo acima trata as duas
> grandezas como distintas. A saída não foi escolher um lado: foi **dar nomes
> distintos** e manter as duas tabelas.
>
> - `D_campo` — o índice desta tabela, julgado. **Nada muda aqui**: os S/O/D da
>   FMECA consolidada seguem como estipulados pelo pesquisador, e a FMECA
>   oficial da dissertação é esta.
> - `POD_mon(s)` — a detectabilidade medida, grandeza própria, com nome próprio.
> - `D_mon` e o **NPR projetado** — cenário de sensibilidade sob E2, calculado
>   por `src/ml/retroalimentacao_fmeca.py`, publicado em
>   `resultados/autoencoder/retroalimentacao_fmeca.md`, **separado** desta
>   tabela.
>
> As grandezas continuam distintas — é justamente por isso que uma não
> substitui a outra. A conversão `POD_mon → D_mon` deixou de ser uma régua a
> calibrar: a Tab. 4.8 é definida em % de não detectar, e `1 − POD_mon` é essa
> mesma grandeza. Ver `docs/nomenclatura_deteccao.md`.
