---
data: 2026-07-09
tipo: memoria-consolidada
sessoes_incluidas: 6
interacoes_totais: 21
periodo: 2026-06-19 a 2026-07-09
tags: [memoria, consolidado]
---

# Memória Consolidada — 09/07/2026

> 6 sessões | 21 interações | 2026-06-19 a 2026-07-09

---

---
data: 2026-07-09
tipo: memoria-consolidada
sessoes_incluidas: 18
interacoes_totais: 105
periodo: 2026-05-30 a 2026-07-09
tags: [al-iado-pv, memoria, consolidado, mestrado, fmea, autoencoder, machine-learning, preditiva, inversores-fotovoltaicos, paderborn, pv-farms, rul, weibull, anomalias-sinteticas, pipeline-execution, agent-configuration, weibull-rejection, metric-update, model-evaluation, statistical-analysis]
---

> [!warning] DIVERGÊNCIA WEIBULL CONHECIDA
> Os parâmetros Weibull citados nesta memória podem divergir do artefato
> vigente (weibull_results.json): até 2026-07-06 o jitter dos TTF censurados
> usava RNG global sem semente, tornando beta/eta irreprodutíveis entre
> execuções. Corrigido em src/ml/rul_weibull.py (semente derivada do índice
> da falha). Para valores citáveis, use SEMPRE o JSON vigente.

# Memória Consolidada — 09/07/2026

> 18 sessões | 105 interações | 2026-05-30 a 2026-07-09

---

## 1. AÇÕES CONCRETAS REALIZADAS

*   **Execução e Re-execução do Pipeline de Análise Preditiva:**
    *   **O que foi pedido:** Rodolfo solicitou "rode o pipeline" em 2026-07-06 (23:21) e 2026-07-09 (13:27). Também solicitou "rodar as etapas do pipeline" em 2026-07-06 (19:32).
    *   **O que foi feito:** O pipeline completo de análise preditiva foi executado com sucesso em múltiplas ocasiões. Em 2026-07-09, todas as etapas (`Features CA`, `Autoencoder`, `Injeção de Falhas`, `Validação Formal`, `RUL / Weibull`) foram reportadas como "OK - concluido com sucesso" ou "SKIP - ja esta pronto".
    *   **Por que foi feito assim:** A execução do pipeline é fundamental para gerar e atualizar os artefatos e métricas do projeto, garantindo que os resultados apresentados estejam sempre alinhados com a última versão do código e dos dados processados.

*   **Geração e Exibição de Gráficos do Pipeline:**
    *   **O que foi pedido:** Rodolfo solicitou "Mostre os gráficos gerados" (2026-07-06, 19:32), "Mostre-me os gráficos do autoencoder" (2026-07-06, 20:02), "Me mostre do Rul agora" (2026-07-06, 20:02), "Mostre os resultados gráficos do pipeline" (2026-07-06, 20:02) e "Traga os gráficos dos experimentos somente" (2026-07-06, 20:10).
    *   **O que foi feito:** O agente exibiu e listou os caminhos para diversos gráficos gerados pelo pipeline, incluindo:
        *   **Autoencoder:** `curva_treino.png`, `distribuicao_erro.png`, `erro_temporal.png`.
        *   **Injeção de Falhas Sintéticas:** `injecao_falhas_resultados.png`, `injecao_falhas_comparacao.png`.
        *   **Validação Formal:** `validacao_roc.png`, `validacao_pr.png`, `validacao_matriz.png`, `validacao_metricas.png`.
        *   **RUL / Weibull:** `weibull_ttf.png`, `weibull_confiabilidade.png`, `weibull_rul.png`.
        *   **Experimentos por Artigo (Ahirwar, Francisti, Ibrahim):** `comparacao_metricas.png`, `anomalias_detectadas.png`, e gráficos individuais de métricas para cada modelo (e.g., `modelo_isolation_forest_metricas.png`, `modelo_z_score_estatistico_metricas.png`).
    *   **Por que foi feito assim:** A visualização dos gráficos é essencial para a compreensão e análise dos resultados do pipeline, permitindo a Rodolfo avaliar o desempenho dos modelos e a progressão das falhas.

## 2. DECISÕES ARQUITETURAIS TOMADAS

*   **Definição e Implementação da Persona e Capacidades do Agente (Al IAdo PV):**
    *   **Contexto:** Em 2026-06-19, Rodolfo solicitou esclarecimentos sobre "mudanças estruturais no código fonte" do agente.
    *   **Decisão:** Foi confirmada a implementação de uma persona específica e um conjunto de capacidades operacionais que moldam a interação do agente. Isso inclui:
        1.  **Persona e Papel:** Configuração para atuar como Al IAdo PV, pesquisador sênior e coorientador técnico de Rodolfo Torres, influenciando a voz e profundidade técnica das respostas.
        2.  **Capacidade Operacional Direta:** Habilitação para executar etapas do pipeline (rodar, treinar, recalcular) e proibição de afirmar incapacidade de fazê-lo, visando uma interação mais ativa.
        3.  **Rigor Metodológico e Níveis de Evidência:** Introdução dos Níveis de Evidência (E0 a E3) e a regra de sempre informá-los, além da proibição de inventar dados, autores ou equações, para estruturar a apresentação e validação de informações técnicas.
        4.  **Uso Ativo e Contínuo do Histórico:** Implementação da regra de ler ativamente o histórico, usar consentimento para propostas e evitar repetições ou ciclos na conversa, para garantir fluidez e progressão.
        5.  **Conhecimento Contextualizado do Projeto:** Memorização do "CONTEXTO DO PROJETO" (tema, TCC base, datasets, pipeline, etc.) como base de conhecimento estruturada para todas as interações.
        6.  **Raciocínio Estruturado e Calibração da Confiança:** Instrução para raciocinar em etapas explícitas (hipótese → evidência → limitações → conclusão → próximo passo) e calibrar a confiança das afirmações (distinguindo dados locais, literatura e opinião), além de recomendar caminhos em cenários de trade-offs.
    *   **Por que foi feito assim:** Essas decisões visam transformar o agente em um coorientador técnico eficaz, capaz de interagir de forma proativa, tecnicamente rigorosa e alinhada aos objetivos da dissertação, superando limitações de agentes genéricos.

## 3. PROBLEMAS ENCONTRADOS E SOLUÇÕES

*   **Experimento de Ghoneim, Rashed & Elkalashy (2021) não listado como executável:**
    *   **Problema:** Em 2026-06-19, Rodolfo solicitou "rode o experimento do ghoneim", mas o agente respondeu listando apenas os experimentos de Francisti, Ibrahim, Stender e Ahirwar, omitindo Ghoneim. A memória anterior já havia reportado um "erro de dependência no modelo CN2: No module named 'bottleneck'" para este experimento.
    *   **Diagnóstico e Solução:** A omissão do experimento da lista de executáveis sugere que ele não está atualmente integrado para execução direta via o comando "rode o experimento do [autor]". A causa raiz original (dependência `bottleneck` ausente para o modelo CN2) permanece não resolvida, e a integração do experimento como um todo parece estar pendente ou desativada. **Solução:** O problema persiste, e o experimento não está disponível para execução direta.

*   **Falha "Desbalanceamento de Fase" não detectada pelo Autoencoder no limiar operacional:**
    *   **Problema:** A partir de 2026-07-06, a etapa de "Injeção de Falhas Sintéticas" do pipeline passou a reportar "⚠️ não detectada" para a falha "Desbalanceamento de Fase". O erro médio máximo para esta falha na severidade 1.0 foi de 0.7744, o que representa apenas 0.37x do limiar operacional (2.0785). Consequentemente, na "Validação Formal", esta falha apresentou Recall de 0.020 e F1 de 0.039, apesar de um AUC-ROC de 0.937.
    *   **Diagnóstico e Solução:** O Autoencoder, calibrado com o limiar p99 (2.0785), não consegue distinguir a falha de Desbalanceamento de Fase do comportamento normal no ponto de operação escolhido, mesmo que o modelo seja capaz de ranquear bem as anomalias (alto AUC). Isso indica que o limiar é muito conservador para esta falha específica, ou que a injeção de falha sintética para "Desbalanceamento de Fase" não gera um erro de reconstrução suficientemente distinto para cruzar o limiar. **Solução:** Este é um achado relevante para a dissertação e requer investigação futura para ajustar o limiar, refinar a injeção de falhas ou explorar outros detectores para esta falha.

*   **Rejeição do Ajuste Weibull pelo Teste de Kolmogorov-Smirnov (KS):**
    *   **Problema:** A partir de 2026-07-06 (23:21), a etapa "RUL / Weibull" passou a incluir a coluna "Ajuste (KS)" e reportou "⚠️ rejeitado" para o ajuste Weibull de todas as falhas (Degradação Filtro LCL, Desbalanceamento de Fase, Falha de Sensor CA) com p-valores de 0.0014, <0.0001 e 0.0004, respectivamente.
    *   **Diagnóstico e Solução:** Embora o parâmetro `beta` > 1 ainda sustente a hipótese de degradação progressiva, o teste KS indica que a distribuição Weibull não é a melhor representação estatística para os Tempos Até a Falha (TTF) simulados. Isso significa que as estimativas de MTTF e B10 são indicativas, mas não estatisticamente conclusivas sob a premissa de uma distribuição Weibull. **Solução:** Este é um achado crítico que exige uma ressalva na dissertação sobre a validade estatística do ajuste Weibull ou a exploração de outras distribuições de confiabilidade para modelar os TTF simulados.

## 4. RESULTADOS E MÉTRICAS OBTIDOS

Os seguintes resultados foram consolidados a partir dos artefatos do pipeline, com atualizações significativas desde a última consolidação:

### Autoencoder - Modelo de Normalidade
*   **Limiar p99:** 2.0785
*   **Média baseline:** 0.2309
*   **Desvio baseline:** 0.4528
*   **Falsos positivos validação:** 1.10%
*   **Épocas treinadas:** 150
*   **Interpretação:** O detector está calibrado por erro de reconstrução. Quanto maior a distância entre o erro de reconstrução de uma falha e o limiar, mais clara é a anomalia.

### Injeção de Falhas Sintéticas
*   **Limiar:** 2.0785
*   **Baseline:** 0.2052 ± 0.2433
| Falha | NPR | SMD | Erro na SMD | Margem |
|---|---:|---:|---:|---:|
| Degradação Filtro LCL | 210 | 1.0 | 2.2238 | 1.07x |
| Desbalanceamento de Fase | 150 | ⚠️ não detectada | - | - |
| Falha de Sensor CA | - | 0.7 | 3.2743 | 1.58x |
*   **Atualização:** A falha "Desbalanceamento de Fase" não é mais detectada pelo Autoencoder no limiar operacional (erro médio máximo 0.7744 na severidade 1.0, margem 0.37x do limiar). A SMD e o erro na SMD para "Falha de Sensor CA" foram atualizados de 0.1 para 0.7 e de 11.6359 para 3.2743, respectivamente.

### Validação Formal
| Falha | Severidade | AUC-ROC | Recall | F1 (50%) | F1 (raro 5%) |
|---|---:|---:|---:|---:|---:|
| Degradação Filtro LCL | 1.0 | 0.943 | 0.480 | 0.649 | 0.649 |
| Desbalanceamento de Fase | 1.0 | 0.937 | 0.020 | 0.039 | 0.039 |
| Falha de Sensor CA | 1.0 | 1.000 | 1.000 | 1.000 | 1.000 |
*   **Atualização:** As métricas para "Desbalanceamento de Fase" foram significativamente atualizadas, com Recall e F1 muito baixos no ponto de operação, apesar de um AUC-ROC ainda alto. Isso indica que, embora a falha seja bem ranqueada, o limiar p99 a deixa passar.

### RUL / Weibull
| Falha | NPR | beta | eta | MTTF | B10 | Ajuste (KS) | Interpretação |
|---|---:|---:|---:|---:|---:|---|---|
| Degradação Filtro LCL | 210 | 3.449 | 55.0 | 49.5 | 28.6 | ⚠️ rejeitado (p=0.0014) | desgaste progressivo |
| Desbalanceamento de Fase | 150 | 2.298 | 102.1 | 90.4 | 38.3 | ⚠️ rejeitado (p<0.0001) | desgaste progressivo |
| Falha de Sensor CA | D=10 | 4.626 | 37.3 | 34.1 | 22.9 | ⚠️ rejeitado (p=0.0004) | desgaste progressivo |
*   **Atualização:** Os parâmetros `beta`, `eta` e `B10` para "Desbalanceamento de Fase" foram ligeiramente atualizados (beta de 5.866 para 2.298, eta de 29.7 para 102.1, B10 de 20.3 para 38.3). Os parâmetros para "Falha de Sensor CA" também foram atualizados (beta de 3.959 para 4.626, eta de 6.0 para 37.3, MTTF de 5.5 para 34.1, B10 de 3.4 para 22.9). Mais importante, foi adicionada a coluna "Ajuste (KS)" indicando que o ajuste Weibull foi **rejeitado** para todas as falhas pelo teste de Kolmogorov-Smirnov, com os p-valores correspondentes.

### Experimentos por Artigo
| Experimento | Modelo | Accuracy | Precision | Recall | F1 | AUC | Specificity | Anomalias detectadas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ahirwar & Nandanwar (2025) | Isolation Forest | 0.508 | 0.556 | 0.082 | 0.144 | 0.626 | 0.934 | 27 |
| Ahirwar & Nandanwar (2025) | AE-LSTM | 0.503 | 0.545 | 0.033 | 0.062 | 0.731 | 0.973 | 11 |
| Ahirwar & Nandanwar (2025) | Facebook Prophet | 0.505 | 0.667 | 0.022 | 0.043 | 0.553 | 0.989 | 6 |
| Ahirwar & Nandanwar (2025) | Híbrido (voto) | 0.505 | 0.562 | 0.049 | 0.091 | 0.508 | 0.962 | 16 |
| Francisti et al. (2025) | Z-score (estatístico) | 0.555 | 0.672 | 0.214 | 0.325 | 0.744 | 0.896 | 58 |
| Ibrahim et al. (2022) | Isolation Forest | 0.508 | 0.556 | 0.082 | 0.144 | 0.626 | 0.934 | 27 |
| Ibrahim et al. (2022) | AE-LSTM | 0.503 | 0.545 | 0.033 | 0.062 | 0.731 | 0.973 | 11 |
| Ibrahim et al. (2022) | Facebook Prophet | 0.505 | 0.667 | 0.022 | 0.043 | 0.553 | 0.989 | 6 |
*   **Atualização:** As métricas para os experimentos de Ahirwar, Francisti e Ibrahim foram atualizadas. A coluna "Anomalias detectadas" foi adicionada, fornecendo o número de anomalias identificadas por cada modelo no ponto de operação. O experimento de Ghoneim, Rashed & Elkalashy (2021) não foi listado como executável nas sessões recentes, embora seus resultados anteriores ainda estejam na memória consolidada.

## 5. INSIGHTS TÉCNICOS E ACADÊMICOS

*   **Diferença entre "Fonte de Conhecimento" e "Código Fonte" do Agente:** A interação de 2026-06-19 (Interações 1 e 2) revelou uma distinção importante na interpretação do agente sobre sua "fonte". Inicialmente, ele se referiu à sua base de conhecimento (documentos RAG), mas Rodolfo esclareceu que se referia às "mudanças estruturais no código fonte" do próprio agente. Isso levou à explicitação das decisões arquiteturais sobre a persona e capacidades do Al IAdo PV, que são fundamentais para o seu funcionamento como coorientador técnico.
*   **Limitação do Autoencoder na Detecção de "Desbalanceamento de Fase":** Apesar de o modelo Autoencoder apresentar um alto AUC-ROC (0.937) para a falha "Desbalanceamento de Fase", indicando boa capacidade de ranquear anomalias, ele falha em detectá-la no ponto de operação definido pelo limiar p99 (Recall de 0.020). Este é um insight crítico: um alto AUC não garante detecção efetiva na prática se o limiar de decisão for muito conservador ou se a anomalia não gerar um desvio suficientemente grande do comportamento normal para cruzar o limiar. Isso sugere que a injeção de falhas para "Desbalanceamento de Fase" pode precisar de revisão ou que um limiar adaptativo ou outro modelo pode ser mais adequado para esta falha.
*   **Ressalva Estatística na Análise Weibull:** A rejeição do ajuste Weibull pelo teste de Kolmogorov-Smirnov para todas as falhas (Degradação Filtro LCL, Desbalanceamento de Fase, Falha de Sensor CA) é um achado estatístico significativo. Embora o parâmetro `beta` > 1 ainda indique um padrão de desgaste progressivo, a inadequação da distribuição Weibull para os TTF simulados implica que as estimativas de MTTF e B10 devem ser tratadas como indicativas e não conclusivas. Isso reforça a necessidade de cautela na interpretação desses valores e pode levar à exploração de outras distribuições de confiabilidade ou à revisão do método de simulação de TTF.
*   **Impacto da Reproducibilidade nos Parâmetros Weibull:** As pequenas variações nos parâmetros `beta`, `eta` e `B10` para "Desbalanceamento de Fase" entre as execuções do pipeline (e.g., beta de 2.300 para 2.298) reforçam a importância da correção de RNG global sem semente mencionada no alerta de "DIVERGÊNCIA WEIBULL CONHECIDA". Isso sublinha a sensibilidade dos resultados a fatores de reprodutibilidade e a necessidade de usar sempre o artefato `weibull_results.json` vigente para valores citáveis.

## 6. ESTADO ATUAL DO PIPELINE

*   **Pipeline Principal:** Todas as etapas do pipeline (`Features CA`, `Autoencoder`, `Injeção de Falhas`, `Validação Formal`, `RUL / Weibull`) estão funcionando e foram executadas com sucesso nas últimas sessões (2026-07-09).
*   **Autoencoder:** Calibrado e funcionando, mas com limitação na detecção da falha "Desbalanceamento de Fase" no limiar p99.
*   **Injeção de Falhas Sintéticas:** As falhas são injetadas e avaliadas, mas a "Desbalanceamento de Fase" não está sendo detectada pelo Autoencoder no limiar atual. Os parâmetros de detecção para "Falha de Sensor CA" foram atualizados.
*   **Validação Formal:** As métricas de validação são geradas, com destaque para a baixa performance de Recall/F1 para "Desbalanceamento de Fase" no ponto de operação.
*   **RUL / Weibull:** A análise de Weibull é realizada, mas o ajuste da distribuição aos dados simulados é estatisticamente rejeitado pelo teste KS para todas as falhas, indicando uma limitação na modelagem.
*   **Experimentos por Artigo:** Os experimentos de Ahirwar, Francisti e Ibrahim estão integrados e seus resultados foram atualizados. O experimento de Ghoneim, Rashed & Elkalashy (2021) não está listado como executável no momento.
*   **Artefatos Gerados:** Todos os gráficos e tabelas de métricas correspondentes às etapas do pipeline e aos experimentos por artigo foram gerados e estão acessíveis nos diretórios `resultados/autoencoder/` e `resultados/experimentos/`.

## 7. PRÓXIMOS PASSOS IDENTIFICADOS

1.  **Investigar e Melhorar a Detecção de "Desbalanceamento de Fase":**
    *   **Prioridade:** Alta.
    *   **Ação:** Analisar a injeção de falhas sintéticas para "Desbalanceamento de Fase" para garantir que ela gere um sinal de anomalia mais pronunciado. Avaliar a possibilidade de ajustar o limiar de detecção do Autoencoder especificamente para esta falha ou explorar modelos alternativos que sejam mais sensíveis a ela.
2.  **Abordar a Rejeição do Ajuste Weibull:**
    *   **Prioridade:** Média-Alta.
    *   **Ação:** Investigar por que o teste KS rejeita o ajuste Weibull. Isso pode envolver a exploração de outras distribuições de confiabilidade (e.g., Log-Normal, Exponencial) para modelar os TTF simulados ou a revisão do processo de simulação de TTF para que se ajuste melhor a uma distribuição Weibull.
3.  **Definir o Status do Experimento de Ghoneim:**
    *   **Prioridade:** Baixa-Média.
    *   **Ação:** Esclarecer se o experimento de Ghoneim, Rashed & Elkalashy (2021) será reativado para execução direta, e, em caso afirmativo, resolver a dependência `bottleneck` e integrá-lo à lista de experimentos executáveis.

## 8. REFERÊNCIAS E FONTES CITADAS

*   **Administration (2008).** *Nasa Reliability Centered Maintenance Guide For Facilities A*.
*   **Ahirwar & Nandanwar (2025).** (Referência inspiradora para modelos de anomalia).
*   **Carpinetti (2016).** *Gestao Da Qualidade Cap 6*.
*   **Francisti et al. (2025).** (Referência inspiradora para modelos de anomalia e protocolo de injeção de falhas).
*   **Ghoneim, Rashed & Elkalashy (2021).** (Referência para modelos de anomalia, atualmente não executável).
*   **Ibrahim et al. (2022).** (Referência inspiradora para modelos de anomalia).
*   **Lafraia (s.d.).** *Manual De Confiabilidade Mantenabilidade E Disponibilidade Cap4*.
*   **Sakurada (1998).** *As Tecnicas De Analise Do Modos De Falhas E Seus Efeitos E A*.
*   **Silva (2008).** *Avaliacao Da Confiabilidade Em Sistemas Eletricos Com Base N*.
*   **Stender, Wallscheid & Böcker (2020).** (Artigo de descrição do dataset de Paderborn, referência de normalidade).
*   **Torres (2024).** *Aplicacao Da Metodologia Reliability Centred Maintenance A S*.
---