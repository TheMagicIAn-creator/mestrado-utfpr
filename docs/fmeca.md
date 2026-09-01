# FMECA consolidada

A FMECA organiza a criticidade dos componentes do lado CA e orienta a análise
de manutenção. Ela não é recalculada pelo desempenho dos Autoencoders e não
produz uma campanha sintética de resultados.

## Conceitos

- FMEA identifica modos e efeitos de falha.
- FMECA acrescenta criticidade.
- `NPR = S x O x D_campo`.
- `D_campo` mede a dificuldade de detecção no processo de manutenção; não é a
  sensibilidade do Autoencoder.

## Componentes

| Componente | Função | S | O | D_campo | NPR |
|---|---|---:|---:|---:|---:|
| Contator AC | conectar a saída CA à rede | 5 | 7 | 9 | 315 |
| IGBT | realizar o chaveamento da conversão CC-CA | 5 | 6 | 3 | 90 |
| Fusível AC | proteger o lado CA contra sobrecorrente | 5 | 3 | 2 | 30 |

Ordem de criticidade: Contator AC, IGBT e Fusível AC.

## Fundamentação e uso

O TCC registra participações de chamados de 12% para Contator AC, 6% para IGBT
e 4% para Fusível AC. Essas participações apoiam cenários bibliográficos de
sensibilidade, mas não são taxas medidas dos componentes. Software e PCB ficam
fora do recorte porque a dissertação prioriza componentes CA relacionados às
assinaturas elétricas e ao planejamento de manutenção.

A FMECA deve ser lida junto às curvas `R(t)`, `F(t)`, `f(t)` e `h(t)` para
discutir prioridade, inspeção e ressalvas da evidência. Ela não transforma
taxas bibliográficas em observações de campo, não cria distribuição normal e
não altera limiares ou métricas da comparação Denso versus AE-LSTM.

## Ponte com a evidência GPVS

Os 14 ensaios F1L-F7M permanecem com as categorias nativas do GPVS-Faults. O
pipeline os trata como condições experimentais de falha para comparar os dois
detectores; não os relabela como Contator AC, IGBT ou Fusível AC e não injeta
assinaturas sintéticas desses componentes.

Consequentemente, a E3 sustenta a detecção de anomalias no conjunto de bancada
avaliado, mas não uma probabilidade de detecção individual para cada componente
da FMECA. Contator AC, IGBT e Fusível AC entram na dissertação pela priorização
de manutenção e pelos cenários bibliográficos, em uma família separada.

## Extensão de monitoramento

O contrato versionado preserva `D_campo` e `NPR_base` e expõe como nulos:

- `POD_mon`;
- `D_mon`;
- `D_proj`;
- `NPR_proj`.

A regra candidata `D_proj=min(D_campo,D_mon)` só poderá ser ativada depois de
definir POD por componente, unidade inferencial, denominador, IC e um mapeamento
bibliograficamente validado para a escala ordinal `D_mon`. Até lá, a publicação
bloqueia NPR projetado e mantém os valores 315, 90 e 30 imutáveis.
