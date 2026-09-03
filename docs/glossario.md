# Glossário canônico

## Símbolos que colidem

| Termo | Significado |
|---|---|
| `F0 = 50 Hz` | frequência fundamental nominal (`GRID_FREQUENCY_HZ`) |
| `F0L`, `F0M` | ensaios saudáveis da base experimental |
| `F1L`-`F7M` | 14 ensaios experimentais de falha |

Use sempre a unidade ao escrever a frequência e o sufixo L/M ao escrever o
ensaio. Os nomes dos ensaios vêm do dataset e não significam frequência.

## FMECA

- **FMEA:** análise de modos e efeitos de falha.
- **FMECA:** FMEA com análise de criticidade.
- **NPR:** produto entre severidade, ocorrência e detectabilidade, calculável
  somente quando os três valores pertencem ao mesmo escopo documentado.
- **Escopo FMECA vigente:** IGBT, sistema de sensor/realimentação e
  sistema/circuito de controle do inversor.
- **`validated`:** S, O, D e NPR foram definidos pelo pesquisador e a identidade
  `NPR = S * O * D` foi validada.
- **Base das escalas FMECA:** as escalas numéricas são fundamentadas no TCC e em
  outras referências; os valores adotados para os itens são critério do
  pesquisador.
- **`pending_source_documentation`:** os escores estão vigentes, mas as fontes,
  páginas, tabelas e critérios das escalas ainda precisam ser catalogados.
- **`awaiting_user_fmeca`:** estado histórico anterior a 2026-09-01, quando S,
  O, D e NPR ainda eram `null`; não descreve o contrato atual.

## Detectores e evidência E3

- **Autoencoder Denso:** arquitetura `24-16-8-16-24`.
- **AE-LSTM:** modelo temporal com sequência 8, oculto 32 e latente 8.
- **Escore top-k:** média dos `k` maiores erros quadráticos por feature; no
  AE-LSTM, calculada em `W_t` após contexto causal `[W_(t-7), ..., W_t]`.
- **Grade de sensibilidade:** nove combinações de `k={5,10,20}` com percentis
  `{99;99,5;99,9}` por modelo e semente, sem seleção pelas falhas.
- **Limiar p99:** percentil solicitado próprio de cada modelo na calibração
  saudável; o contrato também informa o percentil empírico efetivamente
  realizável. Com `n=210`, p99 cai na ordem 208/210, p99,05 efetivo e
  resolução de 0,476 ponto percentual.
- **Máximo amostral (`threshold_is_sample_maximum`):** marca que o limiar
  coincide com o maior escore da calibração. Nesse caso o percentil declarado
  não é representável e o limiar tem a variância de um máximo, não a de um
  quantil. Um percentil `p` só é representável com `n >= (p/100 − 2)/(p/100 − 1)`:
  p99 exige 101 observações, p99,5 exige 201 e p99,9 exige 1001.
- **p99,9 (histórico):** ponto operacional publicado até 2026-09-03. Com as 210
  janelas de calibração ele selecionava a ordem 210/210 — era o máximo amostral.
  Permanece na grade de sensibilidade como referência de reprodutibilidade,
  marcado por `is_historical_reference_configuration`.
- **E3 de bancada:** avaliação dos modelos congelados nos 14 ensaios F1L-F7M.
- **Recall, F1 e Precision:** métricas principais da comparação.
- **ROC-AUC e PR-AUC:** métricas complementares de discriminação.
- **Bootstrap por ensaio:** reamostragem cuja unidade é o ensaio, não cada janela.

## Confiabilidade física

- **`R(t)`:** probabilidade de sobrevivência até o tempo t.
- **`F(t)`:** probabilidade acumulada de falha.
- **`f(t)`:** densidade temporal de falha.
- **`h(t)`:** taxa instantânea de falha.
- **Cenário derivado:** taxa calculada por hipótese de alocação; não é medição.
- **Taxa direta:** valor transcrito de fonte bibliográfica identificada.

O contrato atual usa modelo exponencial de taxa constante. Curvas Normal,
Lognormal ou Weibull exigiriam tempos individuais de falha, exposição e censura
que não estão disponíveis.

## Agente e publicação

- **RAG:** recuperação de literatura antes da síntese.
- **Busca híbrida:** combinação lexical e vetorial.
- **Manifesto v2:** parâmetros, entradas, código, dependências, saídas e hashes.
- **`ready` / `stale` / `pending`:** publicação compatível, desatualizada ou ausente.
