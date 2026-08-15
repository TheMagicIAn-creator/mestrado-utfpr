# Mapa de resultados — qual artefato sustenta qual afirmação

**Para quê:** a dissertação está entrando em redação. Este arquivo responde a uma
pergunta só, para cada número que o texto vai citar: **de onde ele sai, e o que
exatamente ele mede.**

Ele **não** repete valores. Métrica citada de documento envelhece em silêncio; a
regra do projeto é ler sempre o artefato vigente. Aqui só há endereço, grandeza
e nível de evidência.

---

## ⚠️ A colisão mais cara do repositório

Existem **duas** famílias de curva chamadas "confiabilidade", "falha" e "taxa de
falha". Elas medem coisas diferentes, em eixos diferentes, com hipóteses
diferentes. Trocá-las numa banca é o erro mais caro possível aqui.

| | **Detectabilidade** | **Confiabilidade física** |
|---|---|---|
| Onde | `resultados/autoencoder/weibull_*`<br>`resultados/macro/weibull/<modelo>/` | `resultados/v2/confiabilidade/` |
| Eixo | `a` — **fração da assinatura nominal injetada**, em [0; 1] | `t` — **anos** |
| Curva "confiabilidade" | `S_D(a)` = P(o detector **ainda não detectou**) | `R(t)` = P(o componente **ainda não falhou**) |
| Curva "taxa de falha" | `h_D(a)` = intensidade de **primeiro cruzamento** do limiar | `h(t) = λ` = taxa de falha do componente |
| Modelo | Weibull 2P com censura intervalar na grade | Exponencial, `λ` constante por cenário |
| Origem do número | **medido** — varredura sobre o GPVS | **bibliográfico** — taxas publicadas |
| Nível | **E2** (injeção FMECA no sinal) | sensibilidade bibliográfica, `status: bibliographic_sensitivity_not_dataset_estimate` |
| O que **não** é | não é vida do componente, não é RUL em tempo | não é estimativa do GPVS |

**A frase segura para o texto:** *o GPVS-Faults sustenta a detectabilidade; a
confiabilidade física entra como sensibilidade bibliográfica, e as duas nunca
compartilham eixo.* O próprio artefato v2 declara
`dataset_role: detector_evaluation_only_not_physical_reliability` — a ressalva
já viaja com o dado, e o texto precisa honrá-la.

---

## Onde ler cada afirmação

### Detecção em falha REAL — a evidência mais forte que existe hoje

| Afirmação | Artefato | Nível |
|---|---|---|
| O detector separa ensaio saudável de ensaio com falha real | `resultados/gpvs/validacao_gpvs_e3.json` → `macro_summary` | **E3 de bancada** |
| Desempenho ensaio a ensaio (F1–F7, duas condições) | mesmo arquivo → `scenario_results` | E3 de bancada |
| O que essa evidência **não** cobre | mesmo arquivo → `limitations` | — |

Unidade de reamostragem é o **ensaio**, não a janela. Janelas vizinhas não são
réplicas independentes, e o intervalo publicado já respeita isso.

**E3 de bancada não é E3 de campo.** Pesos e limiar congelados sobre falha
experimental real é o teto do que este trabalho demonstra. Desempenho industrial
em campo continua não realizado, e o texto deve dizer isso com todas as letras.

### Detecção em falha SINTÉTICA fundamentada na FMECA

| Afirmação | Artefato | Nível |
|---|---|---|
| A partir de que magnitude cada falha da FMECA é detectada | `resultados/macro/comparacao_tabela.md` (AUC, SMD@FPR=10%) | **E2** |
| Comparação com a literatura — AE denso × AE-LSTM (Ibrahim, 2022) | `resultados/macro/comparacao_resultado.json` | E2 |
| As quatro curvas de detectabilidade, **por modelo** | `resultados/macro/weibull/detectabilidade_por_modelo.{json,csv,md}` + `<modelo>/weibull_*.png` | E2 |
| As mesmas curvas para o detector do pipeline | `resultados/autoencoder/weibull_results.json` + `weibull_*.png` | E2 |

Os dois lados da comparação passam pelo **mesmo** holdout, a **mesma** injeção e
as **mesmas** realizações de ruído. O que difere é a arquitetura e o limiar — e
o limiar difere **de propósito**: escores de detectores distintos não são
comparáveis em escala. Fonte única da regra: `macro_comum.calibrar_limiar`.

### Matrizes de confusão

| Onde | O que é |
|---|---|
| `resultados/autoencoder/validacao_matriz.png` | validação do detector do pipeline |
| `resultados/autoencoder/validacao_matrizes_severidades.png` | uma matriz por severidade injetada |
| `resultados/v2/autoencoder/matrizes_confusao.png` | experimento v2 |
| `resultados/classificacao_pv/matriz_confusao.png` | PV Farms, lado CC, E1 |

**A comparação macro não tem matriz de confusão, e isso é decisão, não falta.**
Sob prevalência rara a matriz no limiar calibrado é enganosa: com FP alvo de 1%,
quase toda a massa cai na diagonal do negativo e a figura sugere um desempenho
que a AUC não confirma. É por isso que o comparativo ranqueia por **AUC** e
**SMD@FPR=10%**, que independem do limiar. Registrado em `macro_comum.py` e
`macro_comparar.py`.

### Criticidade e priorização

| Afirmação | Fonte |
|---|---|
| Quais componentes, com que S/O/D_campo e NPR | `docs/fmeca.md` — **fonte única** |
| O que `D_campo`, `POD_mon`, `D_mon` e `D_proj` significam | `docs/nomenclatura_deteccao.md` |
| O NPR projetado pelo monitoramento | tabela **separada**, de `retroalimentacao_fmeca.py` |

A FMECA oficial de `docs/fmeca.md` **não muda** com o resultado do detector. O
NPR projetado é uma leitura adicional, em tabela própria.

### Classificação supervisionada (eixo complementar)

`resultados/classificacao_pv/metricas.json` — PV Farms, falhas do lado **CC**.
Não é o método da dissertação e não entra na mesma tabela que o pipeline CA.

---

## O que ainda não é defensável, e por quê

Escrito aqui para não ser descoberto na banca.

1. **O ajuste Weibull 2P vem sendo rejeitado.** O teste de aderência quantizada
   marca `resumo_parametrico_recomendado = False`. Os parâmetros β e η existem
   no artefato, mas **não** têm direito de aparecer como resumo. A leitura
   defensável é a curva empírica de Kaplan-Meier. O código já desenha a
   paramétrica tracejada e rotulada "exploratória" — o texto tem de acompanhar.

2. **`a_det` não é tempo.** Chamava-se TTF até 08/08/2026, nome que prometia
   hora onde há fração de assinatura. Qualquer conversão para RUL em tempo
   precisa de uma hipótese de taxa de degradação que este trabalho não mediu.

3. **Escopo CA × features CC.** Oito das 24 features do vetor E3 são do lado CC
   (`Ipv`, `Vpv`, `Vdc`, `p_dc`), e a dissertação declara foco no lado CA. É
   justificável — o GPVS tem falhas de origem CC — mas a justificativa precisa
   estar escrita no Capítulo 3. Registrado como decisão pendente em
   `docs/handoff/`.

4. **Amostra pequena e grade discreta.** Os valores de E2 são consistentes, não
   precisos. Um SMD de 0,50 significa "falhou em 0,30, passou em 0,50" — a grade
   de severidade não tem resolução para afirmar mais.

---

## Regra de precedência, quando dois artefatos discordarem

1. Artefato vigente com manifesto em `resultados/manifestos/` **prevalece**.
2. Entre dois artefatos vigentes, prevalece o de **maior nível de evidência**
   para a afirmação em causa — e E3 de bancada só vale para detecção em falha
   real, nunca para detectabilidade sintética.
3. Nota curada do Obsidian **não** sobrepõe artefato. Sessão arquivada registra
   o que foi dito, não prova fato.
4. `CLAUDE.md` e este mapa dizem **onde ler**, nunca **quanto deu**.

## Como recalcular

```bash
python -m src.ml.exec_etapa_isolada features_gpvs
python -m src.ml.exec_etapa_isolada autoencoder
python -m src.ml.macro_comparar            # AUC e SMD, proposto × Ibrahim
python -m src.ml.macro_weibull             # as quatro curvas, por modelo
```

Tudo isso exige `dados/brutos/gpvs/`, que fica fora do Git. Na nuvem o agente lê
os artefatos versionados e **nunca** afirma que treinou.
