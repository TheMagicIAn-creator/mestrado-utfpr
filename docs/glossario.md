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
- **NPR** (Número de Prioridade de Risco): S × O × D — índice da **FMECA**
  (nunca da FMEA; D isolado NUNCA é o NPR). FMECA aplicada do TCC (Apêndice E):
  inversor 210, subsistema CA 150. FMECA consolidada da dissertação
  (docs/fmeca.md, fonte única): Contator AC 315, IGBT 90, Fusível AC 30.
- **S / O / D_campo**: Severidade (**1–5**), Ocorrência (1–10) e dificuldade de
  **detecção em campo** (1–10; maior = pior). Apesar do nome "Detecção", a Tab. 4.8 do TCC
  define o índice em **percentual de NÃO detectar** (D=1 → 0–5%; D=10 →
  86–100%) — ele cresce com o fracasso em detectar. O subscrito `campo` separa
  esse índice **julgado** da detectabilidade **medida** do detector proposto.
  Nunca escrever `D` sozinho. Fonte única: `docs/nomenclatura_deteccao.md`.
- **POD_mon(s)**: probabilidade de detecção pelo **monitoramento proposto** na
  severidade `s`, medida sob E2 no limiar operacional congelado (0–1, maior =
  melhor). Raiz consagrada: curva POD do **MIL-HDBK-1823A** (ensaios não
  destrutivos). O subscrito é obrigatório — em sistemas de potência, `POD` nu é
  *Power Oscillation Damping*. É o mesmo número que o `recall` da validação,
  lido como propriedade do método de inspeção.
- **D_mon**: o índice da mesma Tab. 4.8 obtido de `1 − POD_mon(s_ref)`. Não é
  índice rival do D_campo: é o mesmo índice, **medido** em vez de julgado, para
  outro meio de detecção. `D_proj = min(D_campo, D_mon)` — o monitoramento é
  adicional ao que já existe, logo nunca piora o índice.
- **NPR projetado**: `S × O × D_proj`, sob evidência **E2**. Análise de
  sensibilidade; a FMECA oficial continua sendo `docs/fmeca.md`.
- **Weibull (2 parâmetros)**: distribuição de vida com forma **beta** (β>1 →
  desgaste progressivo; β≈1 → falhas aleatórias; β<1 → mortalidade infantil)
  e escala **eta** (vida característica, 63,2% de falhas acumuladas).
- **a_det — magnitude de detecção**: o eixo do Weibull do projeto. Numa
  trajetória, `a_inj` cresce de 0 a 1 sobre a MESMA janela saudável, e `a_det`
  é a magnitude em que o escore fica acima do limiar por `PERSISTENCIA_
  CRUZAMENTO` avaliações seguidas. Mesma unidade de `a_inj` e da SMD, o que
  permite ler Weibull e injeção na mesma régua. Fonte única:
  `src/ml/rul_weibull.py`, bloco "O EIXO NÃO É TEMPO".
  **Substituiu o nome TTF em 08/08/2026**: o eixo nunca foi tempo, e "TTF"/
  "passo de degradação" prometiam hora onde há fração de assinatura. As chaves
  `ttf_*` sobrevivem nos artefatos como alias, já apontando para a unidade nova.
- **MTTF / B10**: média e décimo percentil da distribuição ajustada. Os nomes
  são os da Weibull, mas **no projeto saem em fração da assinatura nominal**,
  não em horas. `B10 = 0,12` lê-se: em 10% das trajetórias a falha já é
  detectada com 12% da assinatura nominal.
- **Indetectabilidade no teto × censura genuína**: censura à direita é
  acompanhamento interrompido — o evento viria depois. **Indetectabilidade no
  teto** é a grade de magnitude varrida INTEIRA, até `a_inj = 1,0`, sem o
  detector confirmar: não há "depois" dentro do experimento. No desenho atual
  toda não detecção é do segundo tipo; tratá-la como censura no MLE pressupõe
  que a falha real possa ter assinatura maior que a nominal — hipótese
  declarada no campo `desfechos` do artefato, não suposição tácita.
- **Teste KS (Kolmogorov–Smirnov)**: teste de aderência entre os `a_det`
  simulados e a Weibull ajustada. p ≤ 0,05 → ajuste REJEITADO → MTTF/B10
  indicativos, não conclusivos (campo `ajuste_weibull_adequado`).
- **RUL** (Remaining Useful Life): vida útil remanescente. No eixo `a_det` a
  grandeza calculada é a **margem de magnitude até detectar** —
  `E[a_det − a | a_det > a]` —, não vida em tempo; o nome RUL é mantido porque
  é o da literatura de prognóstico. O projeto distingue **RUL restrita KM**
  (não paramétrica, limitada ao horizonte observado) de **RUL Weibull**
  (paramétrica e extrapolativa, com ressalva explícita sob alta censura).

## Detecção de anomalias

- **Modelagem de normalidade**: treinar o modelo SÓ com operação saudável e
  tratar desvios de reconstrução como anomalia — abordagem central da
  dissertação (não requer dados rotulados de falha).
- **Erro de reconstrução**: distância entre a janela de entrada e a saída do
  Autoencoder; é a referência MSE do pipeline.
- **Escore operacional**: estatística usada para decidir anomalia
  (`score_method`). Na execução vigente é o escore localizado: média dos top-k
  resíduos padronizados por feature.
- **Limiar operacional**: `score_threshold` do escore operacional, congelado
  ANTES de ver qualquer falha. `mse_p99` é a referência do erro médio de
  reconstrução; μ+3σ é referência comparativa, não o limiar em uso.
- **SMD** (Severidade Mínima Detectável): menor severidade injetada em que o
  erro médio cruza o limiar. `SMD nula` = falha não detectada em nenhuma
  severidade testada (achado de limitação, não erro de execução). É o análogo
  do **a₉₀** do MIL-HDBK-1823A — o menor defeito detectado com 90% de
  probabilidade em ensaios não destrutivos.
- **a_inj — magnitude da assinatura injetada**: fator adimensional em
  [0,05; 1,0] que escala a amplitude da perturbação injetada no sinal saudável
  (grade do pipeline: 7 níveis). **NÃO é o S da FMECA** — são grandezas
  distintas que até 07/08/2026 dividiam o nome "severidade", distinguidas só
  pela caixa da letra. O nome vem do tamanho de defeito `a` da curva POD(a) do
  MIL-HDBK-1823A; com ele a SMD é `a_inj,95`, análogo do a₉₀. Fonte única:
  `docs/auditoria_total_src.md` §1.
- **Injeção sintética orientada pela FMECA**: perturbação apenas das
  grandezas que a física de cada modo de falha afeta (ver
  docs/assinaturas_fmeca.md) — fornece ground truth para validar o detector.
- **Split em blocos intercalados com purga**: a série é dividida em 15 blocos
  contíguos, distribuídos alternadamente entre treino/calibração/teste
  (`T E T V T T E T V T T E T V T`), com **purga** de 2 janelas em toda
  fronteira onde o destino muda — janelas com 50% de sobreposição nunca cruzam
  conjuntos. Substituiu, em 09/08/2026, os **três blocos contíguos**, que
  fatiavam a rampa de rotação do Paderborn em três faixas de velocidade e
  deixavam a calibração num regime só (IQR de F0 de 1,46 Hz contra 83 Hz do
  treino), tornando o limiar congelado inaplicável ao teste. Consequência para
  a redação: o teste **não é "o futuro"**, é generalização entre regimes. Fonte
  única: `src/ml/split_temporal.py`; detalhamento em `docs/metodologia_ml.md` §5.
  O split contíguo (`split_temporal_com_purga`) segue disponível e é o que o
  protocolo E1 por artigo usa.
- **Protocolo por artigo**: o experimento executável vigente usa a regra de
  decisão do Ibrahim/AE-LSTM (p99 do erro em calibração temporal, congelado
  antes do teste). F1 depende do ponto de operação; AUC é a métrica comparável.
- **Degradação honesta**: modelo cuja dependência não está instalada aparece
  como "requer <lib>" em vez de sumir silenciosamente do resultado.

## Níveis de evidência (ver docs/evidence_levels.md)

- **E0** hipótese · **E1** benchmark exploratório · **E2** validação
  sintética orientada pela FMECA · **E3** validação experimental externa
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
