# Glossário canônico

## Símbolos que colidem

| Termo | Significado |
|---|---|
| `F0 = 50 Hz` | frequência fundamental nominal (`GRID_FREQUENCY_HZ`) |
| `F0L`, `F0M` | ensaios saudáveis do GPVS-Faults |
| `F1L`-`F7M` | 14 ensaios experimentais de falha |

Use sempre a unidade ao escrever a frequência e o sufixo L/M ao escrever o
ensaio. Os nomes F0L/F0M vêm do dataset e não devem ser reinterpretados.

## FMECA

- **FMEA:** análise de modos e efeitos de falha.
- **FMECA:** FMEA com análise de criticidade.
- **NPR:** `S x O x D_campo`.
- **D_campo:** dificuldade de detectar no processo de manutenção, conforme a
  escala do TCC. Não é uma métrica do Autoencoder.
- **Contator AC, IGBT e Fusível AC:** componentes do recorte canônico.

## Detectores

- **Autoencoder Denso:** arquitetura `24-16-8-16-24`.
- **AE-LSTM:** modelo temporal com sequência 8, oculto 32 e latente 8.
- **Erro de reconstrução:** distância entre entrada e reconstrução.
- **Limiar p99:** percentil empírico próprio de cada modelo na calibração
  saudável, calculado antes de avaliar falhas.
- **Falso positivo saudável:** excedência do limiar no teste F0 isolado.

## Evidência E2

- **`a_det`:** magnitude adimensional da assinatura sintética, no intervalo
  avaliado de 0 a 1. Não é tempo nem a severidade S da FMECA.
- **SMD95:** menor `a_det` cujo limite inferior do IC95% de detecção alcança
  95%; caso contrário, `não atingido`.
- **Primeiro cruzamento:** menor magnitude em que uma trajetória satisfaz a
  persistência do detector.
- **`S_D(a)`:** sobrevivência empírica da não detecção no eixo de magnitude.
- **`h_D(a)`:** risco discreto de primeiro cruzamento por magnitude.
- **Weibull 2P E2:** diagnóstico intervalar com censura; só pode ser resumido
  quando os critérios formais são aceitos.

Essas grandezas descrevem detectabilidade. Não representam confiabilidade
física, MTTF, RUL ou taxa de falha temporal.

## Evidência E3

- **E3 de bancada:** avaliação dos modelos congelados em F1L-F7M.
- **AUC-PR:** métrica principal da comparação.
- **Bootstrap por ensaio:** reamostragem cuja unidade independente é o ensaio,
  não cada janela autocorrelacionada.
- **E4:** validação de campo; ainda ausente.

## Confiabilidade física

- **`R(t)`:** probabilidade de sobrevivência até o tempo t.
- **`F(t)`:** probabilidade acumulada de falha.
- **`f(t)`:** densidade temporal de falha.
- **`h(t)`:** taxa instantânea de falha.
- **Cenário derivado:** taxa calculada a partir de taxa global e participação de
  chamados; não é medição do componente.
- **Taxa direta:** valor transcrito de uma fonte bibliográfica identificada.

O contrato atual usa modelo exponencial de taxa constante. O GPVS-Faults não
estima essas funções.

## Agente

- **RAG:** recuperação de literatura antes da síntese.
- **Busca híbrida:** combinação lexical e vetorial.
- **Manifesto v2:** parâmetros, entradas, código, dependências, saídas e hashes.
- **`ready` / `stale` / `pending`:** publicação compatível / desatualizada /
  ainda não produzida.
