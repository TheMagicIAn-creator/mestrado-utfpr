# Mapa canônico de resultados

Há três famílias publicadas. Elas respondem a perguntas diferentes e não podem
ser combinadas estatisticamente.
O mapa não repete valores numéricos: cada resultado deve ser lido do artefato
vigente indicado na tabela.

| Família | Pasta | Pergunta | Evidência |
|---|---|---|---|
| Comparação dos modelos | `resultados/comparacao/` | Quantas falhas reais cada modelo detecta sem falsos alarmes operacionalmente inadequados? | E3 de bancada |
| Detectabilidade por magnitude | `resultados/detectabilidade/` | A partir de que fração da assinatura nominal cada modelo passa a detectar? | E2 sintética |
| Confiabilidade e manutenção | `resultados/confiabilidade/` | Como evoluem `R(t)`, `F(t)`, `f(t)` e `h(t)` nos cenários bibliográficos? | sensibilidade bibliográfica |

**A colisão de símbolos é o risco permanente deste mapa.** A detectabilidade
produz curvas em `a` — fração da assinatura nominal, adimensional, em `[0,1]`.
A confiabilidade produz curvas em `t` — tempo, em horas ou anos. As duas usam
ajuste de distribuição e podem sair no mesmo formato de gráfico. Nenhum número
de uma pode ser lido na escala da outra, e `a_det` nunca é vida útil.

O nome GPVS-Faults identifica a proveniência da base experimental. Não existe
uma família autônoma de “resultados GPVS”.

## Comparação Autoencoder Denso versus AE-LSTM

| Afirmação | Fonte |
|---|---|
| Métricas macro com IC95% | `e3_metricas_macro.csv` |
| Métricas dos 14 ensaios F1L-F7M | `e3_metricas_por_ensaio.csv` |
| Comparação gráfica dos modelos | `e3_metricas_macro.{png,pdf}` e `e3_resultados_por_ensaio.{png,pdf}` |
| Curvas ROC e precisão-revocação | `e3_curvas_discriminacao.{png,pdf}` |
| Matrizes de confusão | `e3_matrizes_confusao.{csv,png,pdf}` |
| Estabilidade em cinco sementes | `e3_estabilidade_sementes.csv` |
| Diferenças pareadas | `e3_diferencas_pareadas.csv` |
| Ablação do contexto temporal | `e3_ablacao_temporal.csv`, `e3_ablacao_temporal_por_ensaio.csv` e `e3_ablacao_temporal.{png,pdf}` |
| Sensibilidade de top-k e percentil | `e3_sensibilidade_escore_limiar.csv` e `e3_sensibilidade_escore_limiar.{png,pdf}` |

Recall, F1 e Precision são as métricas principais; ROC-AUC e PR-AUC são
complementares. O bootstrap usa o ensaio como unidade de reamostragem. Cada
modelo mantém seu próprio limiar saudável, aprendido antes dos ensaios de
falha, com ordem estatística, percentil efetivo e resolução registrados. A
ponto canônico usa `k=5` e p99: com 210 observações de calibração, a ordem
208/210 e p99,05 efetivo. A referência histórica `k=5` com p99,9 cairia na ordem
210/210 — o máximo amostral — e permanece apenas como reprodutibilidade, marcada
como degenerada. A grade descritiva usa `k={5,10,20}` por `{p99,p99,5,p99,9}`,
sem selecionar configuração com F1-F7.
A análise causal de transição e falha sustentada foi inconclusiva e não sustenta
superioridade inequívoca do AE-LSTM.

## Detectabilidade por magnitude

| Afirmação | Fonte |
|---|---|
| `a_det` empírico, censura e ajuste Weibull por item e modelo | `detectabilidade_resumo.csv` |
| Curvas POD sobre a grade de severidade | `pod_curvas.csv` |
| Distância entre a janela injetada em `a=1` e o ensaio real | `ancoras.csv` |
| Confronto POD(a=1) versus recall da E3 | `fidelidade_injecao.csv` |
| Curvas POD por item da FMECA | `e2_pod_curvas.{png,pdf}` |
| Fidelidade da injeção em `a=1` | `e2_fidelidade.{png,pdf}` |
| Contrato, metodologia e portão de adoção da Weibull | `detectabilidade.json` |

`a` é fração da assinatura nominal, em `[0,1]`. O percentil empírico vem primeiro
e sempre; o paramétrico só acompanha quando o ajuste é adotado, e o portão recusa
também o ajuste cujo percentil publicável escape de `[0,1]`. Trajetória que não
cruza até `a=1` é censurada, não `a_det=1`.

**POD e recall da E3 não vivem na mesma escala.** A POD sai de janelas F0
normalizadas pela baseline saudável — a escala em que o limiar foi calibrado — e o
recall da E3 sai dos ensaios reais normalizados por comissionamento. Em `a=1` a
injeção representa a assinatura nominal completa, então as duas deveriam
conversar; onde não conversam, quem está em questão é a fidelidade da injeção. É
isso que `fidelidade_injecao.csv` publica, e a divergência não é erro de nenhuma
das duas etapas nem substitui a E3.

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

`metodologia.json` também publica contratos nulos para Weibull 2P, Normal,
Lognormal e histograma de vidas. A busca no corpus não encontrou `beta` e `eta`
rastreáveis para IGBT. A FMECA vigente cobre seis itens — IGBT,
sensor/realimentação, controle, PCB, contatores CA/CC e ventiladores — com
`NPR = S * O * D` validado e proveniência mista registrada no artefato. Nenhum
desses escores é inferido das métricas dos detectores.

## Proveniência

- `resultados/manifestos/comparacao_autoencoders.json` protege a comparação.
- `resultados/manifestos/detectabilidade.json` protege a detectabilidade.
- `resultados/manifestos/confiabilidade_componentes.json` protege a confiabilidade.
- Os manifestos registram código, entradas, parâmetros, saídas e hashes.
- A aplicação apenas lê os contratos; nunca recalcula ao abrir um painel.

```powershell
python scripts/auditar_resultados.py
python -m src.ml.comparacao_autoencoders
python -m src.ml.campanha_detectabilidade
python -m src.ml.publicacao_confiabilidade
```
