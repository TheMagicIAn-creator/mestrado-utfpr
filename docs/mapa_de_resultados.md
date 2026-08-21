# Mapa canônico de resultados

Este documento indica qual artefato sustenta cada afirmação da dissertação.
Este mapa não repete valores: eles devem ser lidos dos arquivos vigentes.

## Regra central

Há três famílias publicadas e elas não podem ser misturadas:

| Família | Pasta | Grandeza | Evidência |
|---|---|---|---|
| Comparação experimental | `resultados/comparacao/` | desempenho do AE Denso e do AE-LSTM nos ensaios GPVS | E3 de bancada |
| Detectabilidade sintética | `resultados/comparacao/` | resposta do detector em função de `a_det` | E2 FMECA |
| Confiabilidade física | `resultados/confiabilidade/` | `R(t)`, `F(t)`, `f(t)` e `h(t)` no tempo | sensibilidade bibliográfica |

`a_det` é a fração da assinatura nominal injetada, uma magnitude adimensional
de perturbação. `S_D(a)` representa a probabilidade de o detector ainda não ter
detectado e `h_D(a)` a intensidade discreta do primeiro cruzamento. Nenhuma das
duas é tempo, vida útil, RUL, MTTF ou taxa de falha física. Já `R(t)` e `h(t)`
usam tempo em horas, com conversão explícita para anos.

## E3 experimental

| Afirmação | Fonte |
|---|---|
| Métricas macro com IC95% | `e3_metricas_macro.csv` |
| Métricas dos 14 ensaios F1L-F7M | `e3_metricas_por_ensaio.csv` |
| Comparação gráfica dos modelos | `e3_metricas_macro.{png,pdf}` e `e3_resultados_por_ensaio.{png,pdf}` |
| Curvas ROC e precisão-revocação | `e3_curvas_discriminacao.{png,pdf}` |
| Matrizes de confusão | `e3_matrizes_confusao.{csv,png,pdf}` |
| Estabilidade em cinco sementes | `e3_estabilidade_sementes.csv` |
| Diferenças pareadas | `e3_diferencas_pareadas.csv` |

AUC-PR é a métrica principal. O bootstrap usa o ensaio como unidade de
reamostragem; janelas vizinhas não são tratadas como réplicas independentes.
Os dois modelos usam o mesmo pré-processamento e os mesmos ensaios, mas cada um
mantém seu próprio limiar p99.

## E2 FMECA

| Afirmação | Fonte |
|---|---|
| Detecção por modelo, componente e magnitude | `e2_deteccao_por_magnitude.csv` |
| Menor magnitude com detecção conservadora de 95% | `e2_resumo.csv` e `e2_smd95.{png,pdf}` |
| Primeiro cruzamento observado | `e2_primeiro_cruzamento.csv` |
| Sobrevivência, incidência e risco discretos | `e2_funcoes_empiricas.{csv,png,pdf}` |
| Pontos e diagnóstico Weibull | `e2_weibull_pontos.csv`, `e2_weibull_ajustes.csv` e `e2_diagnostico_weibull.{png,pdf}` |

Quando o limite inferior do IC95% não alcança 95%, SMD95 é publicado como
“não atingido”. Weibull 2P é apenas diagnóstico e só pode ser resumida quando o
critério formal de aceitação estiver satisfeito; a leitura principal permanece
empírica.

## Confiabilidade física

| Afirmação | Fonte |
|---|---|
| Taxas diretas e derivadas, origem e ressalvas | `cenarios.csv` e `metodologia.json` |
| Curvas horárias e anuais | `curvas.csv` |
| Confiabilidade e probabilidade acumulada | `confiabilidade_probabilidade_falha.{png,pdf}` |
| Densidade e taxa de falha | `densidade_taxa_falha.{png,pdf}` |
| Comparação das taxas | `taxas_componentes.{png,pdf}` |

O modelo físico publicado é exponencial: `R(t)=exp(-lambda*t)`,
`F(t)=1-R(t)`, `f(t)=lambda*exp(-lambda*t)` e `h(t)=lambda`. As taxas de
Contator AC e IGBT são cenários derivados, não medições. A taxa direta do
fusível permanece separada. O GPVS não estima essas taxas e não autoriza
parâmetros Weibull físicos.

## Proveniência e precedência

- `resultados/manifestos/comparacao_autoencoders.json` protege a comparação.
- `resultados/manifestos/confiabilidade_componentes.json` protege a confiabilidade.
- Cada manifesto registra código, entradas, parâmetros, outputs e hashes.
- Os arquivos consolidados `comparacao_autoencoders.json` e `metodologia.json`
  carregam os contratos metodológicos usados pelo agente e pela interface.
- Notas e sessões registram contexto, mas não sobrepõem um artefato publicado.

## Verificação e regeneração

```powershell
python scripts/auditar_resultados.py
python -m src.ml.comparacao_autoencoders
python scripts/gerar_confiabilidade.py
```

As duas regenerações exigem o ambiente científico local; a comparação também
exige os 16 CSVs GPVS ignorados pelo Git. A aplicação web apenas consulta os
artefatos publicados.
