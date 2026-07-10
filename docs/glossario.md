# Glossário canônico — Al IAdo PV

Ponto único de definição dos termos usados em CLAUDE.md, docs/, código e
dissertação. Em caso de conflito entre documentos, vale a definição daqui.

## Confiabilidade e manutenção

- **RCM** (Reliability Centred Maintenance): metodologia que orienta o plano
  de manutenção pela criticidade funcional dos componentes. Base do TCC de
  Torres (2024) e da seleção de falhas da dissertação.
- **FMEA / FMECA**: análise de modos e efeitos de falha; a variante FMECA
  adiciona a análise de criticidade (NPR). O FMECA de referência do projeto
  é o do CEAMAZON (Torres, 2024, Apêndice E).
- **NPR** (Número de Prioridade de Risco): S × O × D. Valores de referência
  do projeto: inversor 210, subsistema CA 150 (fonte estática: TCC — não são
  medições do pipeline).
- **S / O / D**: Severidade, Ocorrência, Detecção (1–10; D=10 significa
  falha muito difícil de detectar).
- **Weibull (2 parâmetros)**: distribuição de vida com forma **beta** (β>1 →
  desgaste progressivo; β≈1 → falhas aleatórias; β<1 → mortalidade infantil)
  e escala **eta** (vida característica, 63,2% de falhas acumuladas).
- **MTTF / B10**: tempo médio até a falha; tempo em que 10% da população
  falhou. No projeto, medidos em PASSOS de simulação, não em horas de campo.
- **TTF**: tempo até a falha de uma trajetória de degradação simulada —
  passo em que o erro de reconstrução cruza o limiar operacional.
- **Teste KS (Kolmogorov–Smirnov)**: teste de aderência entre os TTF
  simulados e a Weibull ajustada. p ≤ 0,05 → ajuste REJEITADO → MTTF/B10
  indicativos, não conclusivos (campo `ajuste_weibull_adequado`).
- **RUL** (Remaining Useful Life): vida útil remanescente estimada a partir
  da curva de confiabilidade.

## Detecção de anomalias

- **Modelagem de normalidade**: treinar o modelo SÓ com operação saudável e
  tratar desvios de reconstrução como anomalia — abordagem central da
  dissertação (não requer dados rotulados de falha).
- **Erro de reconstrução**: distância entre a janela de entrada e a saída do
  Autoencoder; o score de anomalia do pipeline.
- **Limiar operacional (p99)**: percentil 99 do erro de reconstrução no
  conjunto saudável de validação — congelado ANTES de ver qualquer falha.
  μ+3σ é referência comparativa, não o limiar em uso.
- **SMD** (Severidade Mínima Detectável): menor severidade injetada em que o
  erro médio cruza o limiar. `SMD nula` = falha não detectada em nenhuma
  severidade testada (achado de limitação, não erro de execução).
- **Severidade**: fator 0–1 que escala a intensidade da falha injetada
  (grade do pipeline: 0.05–1.0 em 7 níveis).
- **Injeção sintética orientada pelo FMEA**: perturbação apenas das
  grandezas que a física de cada modo de falha afeta (ver
  docs/assinaturas_fmea.md) — fornece ground truth para validar o detector.
- **Split temporal com purga**: divisão treino/teste por blocos contíguos no
  tempo, descartando janelas na fronteira (janelas com 50% de sobreposição →
  purga de 2) para impedir vazamento temporal.
- **Protocolo por artigo**: cada experimento usa a regra de decisão do
  próprio paper (Francisti: Shewhart 3σ; Ibrahim: contaminação a priori +
  p99 do treino congelado). F1 NÃO é comparável entre protocolos; AUC é.
- **Degradação honesta**: modelo cuja dependência não está instalada aparece
  como "requer <lib>" em vez de sumir silenciosamente do resultado.

## Níveis de evidência (ver docs/evidence_levels.md)

- **E0** hipótese · **E1** benchmark exploratório · **E2** validação
  sintética orientada pelo FMEA · **E3** validação experimental externa
  (ainda não realizada). Nenhum resultado E1/E2 é prova de desempenho
  industrial.

## Sistema/agente (RAG)

- **RAG** (Retrieval-Augmented Generation): recuperar trechos relevantes da
  base indexada e injetá-los no prompt antes da geração da resposta.
- **Camadas 1–3**: expansão de query (local, por regras) → busca híbrida
  (semântica + keyword) → reranking (local, heurístico). Nenhuma chamada de
  LLM ocorre nas camadas; o LLM só gera a resposta final.
- **Chunk**: fragmento de texto indexado. Literatura: ~1800 chars com
  sobreposição de 200; sessões/memórias: 500/50.
- **PERFIL_COMPACTO**: identidade do agente injetada no prompt (hardcoded em
  agente.py). O CLAUDE.md completo NÃO entra no prompt (>6000 chars).
- **Modo revisão bibliográfica**: pergunta classificada como revisão amplia
  o orçamento de busca (mais termos e mais chunks finais).
- **Manifesto de proveniência**: JSON por etapa do pipeline com parâmetros,
  hashes e commit — base dos estados ready/stale/pending.
- **ready / stale / pending**: etapa atualizada / desatualizada em relação às
  dependências / nunca executada.
