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
- **NPR:** `S x O x D_campo`.
- **D_campo:** dificuldade de detecção no processo de manutenção, não uma
  métrica do Autoencoder.
- **Contator AC, IGBT e Fusível AC:** componentes do recorte de manutenção.

## Detectores e evidência E3

- **Autoencoder Denso:** arquitetura `24-16-8-16-24`.
- **AE-LSTM:** modelo temporal com sequência 8, oculto 32 e latente 8.
- **Limiar p99:** percentil empírico próprio de cada modelo na calibração saudável.
- **E3 de bancada:** avaliação dos modelos congelados nos 14 ensaios F1L-F7M.
- **AUC-PR:** métrica principal da comparação.
- **Bootstrap por ensaio:** reamostragem cuja unidade é o ensaio, não cada janela.

## Confiabilidade física

- **`R(t)`:** probabilidade de sobrevivência até o tempo t.
- **`F(t)`:** probabilidade acumulada de falha.
- **`f(t)`:** densidade temporal de falha.
- **`h(t)`:** taxa instantânea de falha.
- **Cenário derivado:** taxa calculada por hipótese de alocação; não é medição.
- **Taxa direta:** valor transcrito de fonte bibliográfica identificada.

O contrato atual usa modelo exponencial de taxa constante. Uma curva normal ou
Weibull exigiria tempos individuais de falha, exposição e censura que não estão
disponíveis.

## Agente e publicação

- **RAG:** recuperação de literatura antes da síntese.
- **Busca híbrida:** combinação lexical e vetorial.
- **Manifesto v2:** parâmetros, entradas, código, dependências, saídas e hashes.
- **`ready` / `stale` / `pending`:** publicação compatível, desatualizada ou ausente.
