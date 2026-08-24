# Mapa canônico de resultados

Há somente duas famílias publicadas. Elas respondem a perguntas diferentes e
não podem ser combinadas estatisticamente.
O mapa não repete valores numéricos: cada resultado deve ser lido do artefato
vigente indicado na tabela.

| Família | Pasta | Pergunta | Evidência |
|---|---|---|---|
| Comparação dos modelos | `resultados/comparacao/` | Autoencoder Denso ou AE-LSTM detecta melhor os ensaios avaliados? | E3 de bancada |
| Confiabilidade e manutenção | `resultados/confiabilidade/` | Como evoluem `R(t)`, `F(t)`, `f(t)` e `h(t)` nos cenários bibliográficos? | sensibilidade bibliográfica |

O nome GPVS-Faults identifica a proveniência da base experimental. Não existe
uma família autônoma de “resultados GPVS”.

## Comparação Denso versus AE-LSTM

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
reamostragem. Cada modelo mantém seu próprio limiar p99, aprendido antes dos
ensaios de falha.

## Confiabilidade e manutenção

| Afirmação | Fonte |
|---|---|
| Taxas, origem, fórmulas e ressalvas | `cenarios.csv` e `metodologia.json` |
| Séries horárias e anuais | `curvas.csv` |
| Curva de confiabilidade `R(t)` | `curva_confiabilidade.{png,pdf}` |
| Curva da probabilidade acumulada de falha `F(t)` | `curva_probabilidade_falha.{png,pdf}` |
| Densidade de probabilidade de falha `f(t)` | `curva_densidade_falha.{png,pdf}` |
| Taxa de falha `h(t)` | `curva_taxa_falha.{png,pdf}` |
| Comparação das taxas | `taxas_componentes.{png,pdf}` |

O modelo publicado é exponencial, usa tempo em horas com conversão para anos e
mantém eixos lineares:
`R(t)=exp(-lambda*t)`, `F(t)=1-R(t)`, `f(t)=lambda*exp(-lambda*t)` e
`h(t)=lambda`. As taxas derivadas são cenários, não medições. Sem uma amostra
de tempos individuais de falha e censura, não há base para histograma normal,
Weibull físico, curva de banheira ou RUL.

## Proveniência

- `resultados/manifestos/comparacao_autoencoders.json` protege a comparação.
- `resultados/manifestos/confiabilidade_componentes.json` protege a confiabilidade.
- Os manifestos registram código, entradas, parâmetros, saídas e hashes.
- A aplicação apenas lê os contratos; nunca recalcula ao abrir um painel.

```powershell
python scripts/auditar_resultados.py
python -m src.ml.comparacao_autoencoders
python -m src.ml.publicacao_confiabilidade
```
