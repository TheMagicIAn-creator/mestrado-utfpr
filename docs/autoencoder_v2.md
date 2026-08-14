# Autoencoder denso V2: protocolo e resultados

## Escopo

Esta é a execução congelada da reconstrução do detector no **GPVS-Faults**
(DOI `10.17632/n76t439f65.1`). Nenhum outro dataset é concatenado. F0L/F0M
são usados para seleção e calibração; F1L-F7M permanecem fechados até a
avaliação final.

A descrição oficial informa que as falhas foram introduzidas manualmente na
metade dos ensaios. Os CSVs não fornecem um canal instrumentado de disparo;
assim, 50% do registro é uma **fronteira nominal**, não um instante exato
medido. Fonte: <https://data.mendeley.com/datasets/n76t439f65/1>.

## Escore físico balanceado

As 24 features são divididas em quatro famílias com peso total idêntico de
25%: operação CC, corrente CA, tensão CA e potência CA. Dentro de cada família,
o peso é dividido entre as respectivas features. A medida evita que uma
família domine o erro somente por possuir mais colunas correlacionadas.

O escore por janela é

`s(x) = soma_j w_j [x_j - x_hat_j]^2`, com `soma_j w_j = 1`.

O scaler, a rede e os pesos são ajustados somente em F0. Em cada ensaio de
falha, apenas o baseline de comissionamento local é ajustado no início do
trecho nominalmente pré-falha.

## Seleção sem falhas

Foram executadas cinco sementes (`13`, `29`, `42`, `71`, `101`) para cada uma
das três arquiteturas. O critério primário é a mediana da perda na validação
saudável; a regra de parcimônia escolheria a menor rede dentro de 2% da melhor.

| Arquitetura | Parâmetros | Mediana da perda saudável | Resultado |
|---|---:|---:|---|
| `24-12-4-12-24` | 724 | 0,1657 | Não selecionada |
| `24-16-8-16-24` | 1.088 | 0,1255 | **Selecionada** |
| `24-16-8-4-8-16-24` | 1.164 | 0,3012 | Não selecionada |

A semente canônica `42` é a mais próxima da mediana da arquitetura escolhida.
Ela não foi escolhida pelo desempenho em F1-F7.

## Limiar saudável

O limiar usa a ordem estatística finita da calibração com cauda nominal de 1%,
sem interpolação: ordem 209 de 210, `score > 0,4404987423`. Houve 1/210
excedência na calibração e 3/281 no teste saudável independente, isto é,
**1,07%** (IC95% de Wilson: 0,36%-3,09%). O IC por janela é descritivo e não
elimina autocorrelação temporal.

## Avaliação nos 14 ensaios

A unidade inferencial é o ensaio. Os IC95% macro usam 20.000 reamostragens dos
14 ensaios; curvas por janela são publicadas apenas como descrição.

| Método congelado | AUC-ROC macro | Sensibilidade | Especificidade | Acurácia balanceada |
|---|---:|---:|---:|---:|
| Autoencoder V2 | 0,778 [0,695; 0,859] | 0,455 [0,268; 0,645] | 0,953 [0,916; 0,978] | 0,704 [0,615; 0,796] |
| PCA (8 componentes) | 0,789 [0,707; 0,866] | 0,416 [0,221; 0,618] | 0,984 [0,972; 0,994] | 0,700 [0,604; 0,801] |

Na comparação pareada AE-PCA, o PCA teve AUC maior em 0,011
(IC95% 0,002-0,022) e especificidade maior em 0,032. O autoencoder teve
sensibilidade maior em 0,039 (IC95% 0,016-0,068). A diferença de acurácia
balanceada foi 0,004 (IC95% -0,010 a 0,019). Portanto, não há superioridade
global: os métodos apresentam compromissos operacionais distintos.

F1, F2 e F5 possuem assinaturas fortes. F4 quase não se separa da condição
saudável. F3, F6 e F7 são mais difíceis e dependentes do modo. Esses resultados
negativos são mantidos; não há seleção de cenários por desempenho.

## Limites de interpretação

- F1 é o único cenário diretamente rotulado como falha total em IGBT.
- F2-F7 não podem ser renomeados como contator ou fusível da FMECA.
- O GPVS demonstra detecção em bancada, não prevalência ou confiabilidade de campo.
- O dataset não possui vidas até falha, censura, reparos ou exposição por unidade.
- Nenhum eixo GPVS estima Weibull temporal, taxa física de falha, MTTF ou RUL.
- O atraso publicado é contado desde a fronteira nominal de 50% do registro.

## Artefatos citáveis

Os arquivos ficam em `resultados/v2/autoencoder/`. JSON, CSV, Markdown e PNG
são versionados. Checkpoints PyTorch, scaler, PCA e normalização local são
regeneráveis e permanecem ignorados pelo Git conforme a política do projeto.

- `contrato_experimento.json`: split, seleção, treino e hashes locais.
- `limiar_v2.json`: famílias, pesos, ordem do limiar e robustez por semente.
- `avaliacao_experimental.json`: protocolo, métricas macro e comparações pareadas.
- `avaliacao_cenarios.csv`: todos os métodos nos 14 ensaios.
- `avaliacao_seeds.csv`: sensibilidade a cinco inicializações.
- `avaliacao_scores.csv`: série por janela para figuras e auditoria.
- `contribuicoes_familias.csv`: decomposição física do escore.
- Figuras: seleção, calibração, desempenho, ponto operacional, matrizes,
  ROC/PR macro, séries temporais e contribuições por família.
