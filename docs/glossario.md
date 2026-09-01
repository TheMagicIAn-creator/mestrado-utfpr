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
- **`awaiting_user_fmeca`:** S, O, D e NPR aguardam valores e fontes aprovados
  pelo pesquisador; `null` não significa zero.

## Detectores e evidência E3

- **Autoencoder Denso:** arquitetura `24-16-8-16-24`.
- **AE-LSTM:** modelo temporal com sequência 8, oculto 32 e latente 8.
- **Escore top-k:** média dos `k` maiores erros quadráticos por feature; no
  AE-LSTM, calculada em `W_t` após contexto causal `[W_(t-7), ..., W_t]`.
- **Grade de sensibilidade:** nove combinações de `k={5,10,20}` com percentis
  `{99;99,5;99,9}` por modelo e semente, sem seleção pelas falhas.
- **Limiar p99,9:** percentil solicitado próprio de cada modelo na calibração
  saudável; o contrato também informa o percentil empírico efetivamente
  realizável. Na execução vigente, `n=210` implica ordem 210/210, p100 efetivo
  e resolução de 0,476 ponto percentual.
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
