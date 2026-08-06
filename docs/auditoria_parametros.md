# Auditoria de parâmetros — autoencoder, cálculo e gráficos

Reavaliação de **todo** parâmetro que decide um número ou uma figura do
pipeline. Cada linha responde: qual é o valor, de onde ele veio, o que ele
afeta, e se ele **descreve o que o sistema faz**.

Critério de reprovação, aplicado sem exceção: um parâmetro que declara uma
coisa e produz outra é defeito, mesmo quando o resultado final parece bom.

| Gravidade | Achados |
|---|---|
| ⛔ **Crítico** | 1 (§1) |
| ⚠️ **Sério** | 3 (§2, §3, §4) |
| 🔸 **Menor** | 4 (§5–§8) |
| ✅ **Justificado** | os demais (§9) |

---

## ⛔ §1. A busca de F0 não cobre o dataset

```python
# src/ml/features_ca.py
F0 = 60          # Hz — frequência fundamental nominal (Brasil)
...
def estimar_f0(freqs, amps, f0_nominal=F0, faixa_hz=40.0):
    f_min = max(5.0, f0_nominal - faixa_hz)   # 20 Hz
    f_max = f0_nominal + faixa_hz             # 100 Hz
```

A fundamental é procurada em **[20, 100] Hz**, faixa dimensionada para a rede
elétrica brasileira de 60 Hz.

**O Paderborn não é rede: é acionamento de motor de velocidade variável.** A
própria docstring de `estimar_f0` reconhece isso ("Necessário para datasets de
acionamento de motor (Paderborn) onde F0 varia com a velocidade"). Mas a faixa
de busca continuou a da rede.

### A evidência

A medição de F0 por bloco (levantada pela PR #94):

| Bloco | Mediana de F0 |
|---|---:|
| calibração | 51,25 Hz |
| teste | **100,19 Hz** |

**Uma mediana de bloco pousada em 100,19 Hz, com o teto da busca em 100 Hz, não
é um regime operacional — é a assinatura de um estimador saturado.** Os poucos
décimos acima do teto são a interpolação parabólica, que pode ultrapassar o
último candidato; ela não pode ir muito além.

### Por que isso é crítico e não cosmético

`F0` é a âncora de **toda** feature espectral:

- `amplitude_harmonica(freqs, amps, f0, ordem)` procura energia em `ordem × f0`
  com tolerância de ±15 Hz. Com `f0` saturado em 100 quando o verdadeiro é,
  digamos, 130 Hz, o 5º harmônico é procurado em 500 Hz em vez de 650 Hz —
  **150 Hz fora**, dez vezes a tolerância. A função devolve o máximo de uma
  janela que não contém o harmônico.
- `calcular_thd` usa as mesmas amplitudes → THD errada.
- O erro **cresce com a ordem**: a tolerância é fixa em ±15 Hz, mas o desvio do
  alvo é `ordem × erro_f0`. No 13º harmônico, 2 Hz de erro em F0 já jogam o
  alvo 26 Hz fora — além da tolerância.

### O que isso explica

Três coisas que estavam sendo atribuídas a outras causas:

1. **O FPR de 10% no bloco de teste.** Features harmônicas calculadas a partir
   de um F0 saturado são artefato de medição. O autoencoder vê vetores que não
   correspondem a nenhum estado físico e o erro de reconstrução sobe — **sem
   falha alguma**. A PR #94 tratou isso como limiar mal calibrado; eu tratei
   como cobertura de dados. Ambos são consequência, não causa.
2. **Por que o IGBT é a falha mais difícil de detectar** (0,850 contra 1,000
   das outras). A assinatura do IGBT é *exatamente* harmônicos de 5ª, 7ª, 11ª e
   13ª ordem — as features mais corrompidas pelo erro de F0, e mais quanto
   maior a ordem.
3. **Por que apertar o limiar destrói o recall.** Se parte do escore saudável
   alto é artefato de medição, subir o corte até eliminá-lo consome também a
   margem que separava falha de normalidade.

### Como confirmar (na máquina com o dataset)

```
python scripts/diagnostico_limiar.py
```

Reporta a fração de janelas com F0 no teto, por bloco. **Acima de ~10% em
qualquer bloco confirma a saturação.**

### A correção, e por que ela NÃO foi aplicada aqui

A correção é pequena — derivar a faixa dos dados, ou ampliá-la para cobrir a
rotação do Paderborn. Mas mudar `features_ca.py` **muda todas as features**, e
com elas: modelo, limiar, SMD, validação, Weibull e a retroalimentação da
FMECA. Tudo o que está publicado seria invalidado de uma vez.

Isso é uma decisão sua, com rodada planejada e comparação antes/depois — não
uma edição silenciosa de constante. **Recomendo fortemente fazê-la**, e é o
item de maior retorno do projeto hoje: se a hipótese se confirmar, o FPR cai
sem custar recall, e o recall do IGBT sobe — o oposto do que a #94 conseguia.

---

## ⚠️ §2. Janela dimensionada para uma frequência que o dataset não tem

```python
FS = 10_000; JANELA = 1024   # "~6 ciclos a 60 Hz"
```

A justificativa registrada é "≈ 6 ciclos a 60 Hz". No dataset:

| F0 | Ciclos por janela |
|---:|---:|
| 51,25 Hz (calibração) | 5,2 |
| 100,19 Hz (teste) | 10,3 |

O número de ciclos por janela **varia por um fator de 2** ao longo do dataset.
Não é fatal (5 ciclos ainda bastam para estimar a fundamental), mas a
justificativa escrita é falsa, e a variação é fonte de dispersão nas features
que hoje não está declarada.

Resolução espectral: `Δf = FS/JANELA = 9,77 Hz`. A fundamental de 51 Hz cai no
bin 5,2 — a interpolação parabólica é o que salva a estimativa. Vale registrar
que ela é essencial, não um refinamento.

---

## ⚠️ §3. `N_JANELAS_SMD = 100` quando existem 40

```python
# src/ml/injecao_falhas.py
N_JANELAS_SMD = 100  # limitado pelo numero de janelas nao sobrepostas do holdout
```

O artefato vigente registra `n_janelas_disponiveis: 40`. O comentário admite o
limite, mas o valor declarado continua 100 — e é ele que aparece em revisão
rápida. Mesmo defeito de `FP_ALVO`: **o número declarado não descreve a
execução.**

Correção sugerida: derivar o teto das janelas realmente disponíveis, ou
registrar `n_efetivo` ao lado do teto em todo artefato que cite o parâmetro.

---

## ⚠️ §4. Réplicas de bootstrap divergem entre módulos

| Local | Réplicas |
|---|---:|
| `src/ml/estatistica.py` | 500 |
| `src/ml/escore_anomalia.py::incerteza_do_limiar` | 500 |
| `src/ml/rul_weibull.py::N_BOOTSTRAP` | **250** |

Os IC do Weibull são calculados com metade das réplicas dos demais, sem
justificativa registrada. Ou o custo computacional justifica (e deve estar
escrito), ou deve ser 500 como o resto. Não muda conclusão; muda comparabilidade
da largura dos intervalos entre seções da dissertação.

---

## 🔸 §5. `DPI = 150` é insuficiente para figura impressa

```python
# src/ml/estilo_graficos.py
DPI = 150
```

Dissertação impressa/PDF costuma exigir **300 DPI** para imagem rasterizada.
Em 150, texto de eixo e legenda ficam visivelmente amaciados na impressão.

Duas melhorias, ambas baratas e sem risco (só refazem PNG):

1. `DPI = 300` para as figuras que vão ao texto;
2. salvar também em **vetor** (PDF ou SVG) os gráficos de linha e barra — curva
   ROC, PR, detecção por severidade, confiabilidade. Vetor não tem DPI e
   imprime perfeito em qualquer escala.

Não altera número nenhum: gráfico é renderização.

---

## 🔸 §6. Grades de severidade diferentes entre etapas

| Etapa | Severidades |
|---|---|
| `injecao_falhas.SEVERIDADES` | 0,05 · 0,1 · 0,2 · 0,3 · 0,5 · 0,7 · 1,0 (7) |
| `validacao.SEVS_VALIDACAO` | 0,30 · 0,50 · 1,00 (3) |

A SMD é buscada em 7 pontos; a validação reporta 3. É defensável (a validação
é mais cara), mas a curva `POD_mon(s)` que sustenta a retroalimentação da FMECA
tem **3 pontos**, e a `SMD` tem 7. Os dois números não vêm da mesma grade, e
isso precisa estar dito onde ambos aparecem juntos.

---

## 🔸 §7. `EPOCHS = 150` é teto, não configuração

`limiar.json` registra `epochs_treinadas: 75`, `epoca_melhor: 55`, com
`PACIENCIA = 20`. O early stopping decide; 150 nunca é alcançado. Correto —
mas o parâmetro merece o rótulo "teto", porque hoje sugere um treino de 150
épocas que não acontece.

---

## 🔸 §8. `DROPOUT = 0.2` no encoder, junto do gargalo

O dropout é aplicado nas camadas do encoder e do decoder. Em autoencoder de
detecção de anomalia isso é escolha legítima, mas soma-se ao item já registrado
em `docs/auditoria_pipeline_ml.md` §23 (ReLU no gargalo, que impede
representação negativa). Os dois são hiperparâmetros de arquitetura sem
varredura, e devem ser decididos **na mesma rodada** — como já registrado.

---

## ✅ §9. Parâmetros conferidos e defensáveis

| Parâmetro | Valor | Veredito |
|---|---|---|
| `SEED = 42` / `SEED_BOOTSTRAP = 42` | fixo | ✅ Determinismo verificado: retreino em 02/08 reproduziu a rodada anterior **bit a bit**, só o carimbo de data mudou |
| `TRAIN/CALIB/TEST = 0,60/0,20/0,20` | fixo | ✅ Split temporal com purga de 2 janelas, coerente com sobreposição de 50% |
| `SOBREPOSICAO = 512` (50%) | fixo | ✅ Purga dimensionada para a sobreposição; sem vazamento |
| `K_LOCALIZADO = 5` | env | ✅ Varredura executada (PR #85): `k=5` e `k=10` empatam por teto, `k=15` degrada o fusível |
| `SIGMA = 3.0` | fixo | ✅ Só referência comparativa; nunca é o limiar operacional, e isso está escrito |
| `PERSISTENCIA_CRUZAMENTO = 3` | fixo | ✅ Exigir 3 passos consecutivos evita TTF disparado por ruído de um passo |
| `MIN_EVENTOS_WEIBULL = 10` | fixo | ✅ Piso razoável para MLE de 2 parâmetros |
| `MAX_CENSURA_RUL_PCT = 50` | fixo | ✅ Guarda explícita; acima disso a RUL paramétrica é sinalizada |
| `PREVALENCIA_RARA = 0.05` | fixo | ✅ Reprojeção declarada e explicada no artefato; AUC/recall não dependem dela |
| `LATENTE_DIM = 16` | fixo | ⏸️ Sem varredura — **já registrado** como adiado (§23 da auditoria), amarrado ao ReLU |
| `HARMONICOS = [3,5,7,11,13]` | fixo | ✅ Cobre a assinatura do IGBT (5/7/11/13) e mais a 3ª; superconjunto é correto |
| `PALETA` / `COR_METODO` | fixo | ✅ Contraste validado, ordem fixa, cor codifica entidade e não rank |
| `TAM` (tamanhos de figura) | fixo | ✅ Conjunto fechado; proíbe `figsize` avulso |

---

## Ordem recomendada de ação

1. **Confirmar §1** com `scripts/diagnostico_limiar.py`. É medição, custa uma
   execução e não muda nada.
2. Se confirmado, **corrigir a faixa de busca de F0** numa rodada planejada,
   com comparação antes/depois de FPR, recall por falha, SMD e censura do
   Weibull. É o item de maior retorno do projeto.
3. **§5 (DPI e vetor)** a qualquer momento — não toca em número.
4. **§3, §4, §7** — higiene de declaração; baratos, sem risco.
5. **§8 + latente** juntos, na rodada de arquitetura já adiada.

## Referências internas

- `scripts/diagnostico_limiar.py` — confirma §1 e §2
- `docs/decisao_fpr_1pct.md` — por que o corte de 1% não foi adotado
- `docs/auditoria_pipeline_ml.md` §22, §23 — teto amostral e rodada de arquitetura
- `docs/metodologia_ml.md` §3 — definição do limiar operacional
