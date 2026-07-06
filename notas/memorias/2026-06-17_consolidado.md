---
data: 2026-06-17
tipo: memoria-consolidada
sessoes_incluidas: 10
interacoes_totais: 30
periodo: 2026-06-02 a 2026-06-17
tags: [al-iado-pv, memoria, consolidado, mestrado]
---

> [!warning] DIVERGÊNCIA WEIBULL CONHECIDA
> Os parâmetros Weibull citados nesta memória podem divergir do artefato
> vigente (weibull_results.json): até 2026-07-06 o jitter dos TTF censurados
> usava RNG global sem semente, tornando beta/eta irreprodutíveis entre
> execuções. Corrigido em src/ml/rul_weibull.py (semente derivada do índice
> da falha). Para valores citáveis, use SEMPRE o JSON vigente.

# Memória Consolidada — 17/06/2026

> 10 sessões | 30 interações | 2026-06-02 a 2026-06-17

---

---
data: 2026-06-14
tipo: memoria-consolidada
sessoes_incluidas: 12
interacoes_totais: 84
periodo: 2026-05-30 a 2026-06-14
tags: [al-iado-pv, memoria, consolidado, mestrado, fmea, autoencoder, machine-learning, preditiva, inversores-fotovoltaicos, paderborn, pv-farms, rul, weibull, anomalias-sinteticas]
---

# Memória Consolidada — 14/06/2026

> 12 sessões | 84 interações | 2026-05-30 a 2026-06-14

---

## 1. AÇÕES CONCRETAS REALIZADAS

*   **Melhoria e Validação Interna do Agente:**
    *   **O que foi pedido:** Não houve um pedido explícito do usuário, mas o sistema passou por avaliações técnicas internas.
    *   **O que foi feito:** O agente passou por três rodadas de avaliação técnica (em 2026-06-11 e 2026-06-13). Na primeira avaliação de 2026-06-11, 58 testes falharam (501/559 passaram), com falhas nas categorias `autor_trigger_rag`, `proveniencia_topica` (0/49 e 0/7, respectivamente) e `roteamento_ferramentas` (68/70). Na segunda avaliação de 2026-06-11, apenas 1 teste falhou (558/559 passaram), especificamente em `roteamento_ferramentas` (69/70). Na avaliação de 2026-06-13, todos os 559 testes passaram (559/559), indicando que as falhas anteriores foram corrigidas.
    *   **Por que foi feito assim:** Essas avaliações são parte do processo de desenvolvimento e garantia de qualidade do agente, assegurando que suas funcionalidades internas, como recuperação de informação e roteamento de ferramentas, operem corretamente.

*   **Execução do Experimento de Francisti et al. (2025):**
    *   **O que foi pedido:** Rodolfo solicitou "rode o experimento do francisti."
    *   **O que foi feito:** O experimento `francisti2025_spc_rf` foi executado. Os resultados para os modelos Z-score (estatístico) e Random Forest (anomalia) foram gerados e apresentados. As imagens resultantes (`modelo_z_score_estatistico_metricas.png`, `modelo_z_score_estatistico_matriz_confusao.png`, `modelo_random_forest_anomalia_metricas.png`, `modelo_random_forest_anomalia_matriz_confusao.png`, `comparacao_metricas.png`, `anomalias_detectadas.png`, `matriz_confusao.png`) foram salvas no diretório `resultados/experimentos/francisti/`.
    *   **Por que foi feito assim:** A execução do experimento foi solicitada para comparar o desempenho de diferentes abordagens de detecção de anomalias (estatística vs. ML) em um cenário de injeção de falhas sintéticas, seguindo o protocolo do artigo de Francisti et al. (2025).

## 2. DECISÕES ARQUITETURAIS TOMADAS

*   **Uso do FMEA para Orientar a Injeção de Falhas Sintéticas:** A metodologia FMEA (Failure Mode and Effects Analysis) foi adotada como base para a identificação e modelagem de modos de falha potenciais em componentes do inversor fotovoltaico. Esta decisão permite que as anomalias sintéticas injetadas no dataset Paderborn sejam guiadas por falhas reais ou plausíveis, como a degradação do filtro LCL, desbalanceamento de fase e falha de sensor CA, tornando os experimentos mais relevantes para o domínio de aplicação.
*   **Priorização de Métricas para Comparação de Modelos de Anomalia:** Foi decidido que, ao comparar experimentos de detecção de anomalias de diferentes artigos, o AUC (Área sob a Curva ROC) é uma métrica mais apropriada do que o F1, especialmente devido às variações nos protocolos de decisão e nos limiares de classificação entre os estudos. O AUC oferece uma medida da capacidade de separação do modelo independentemente do ponto de corte escolhido, enquanto o F1 deve ser analisado em conjunto com o número de anomalias detectadas para entender a operação prática do modelo.

## 3. PROBLEMAS ENCONTRADOS E SOLUÇÕES

*   **Falhas em Testes de Avaliação Técnica do Agente:**
    *   **Problema:** Durante as avaliações técnicas de 2026-06-11, o agente apresentou falhas em categorias como `autor_trigger_rag`, `proveniencia_topica` e `roteamento_ferramentas`.
    *   **Diagnóstico e Solução:** Embora os detalhes específicos do diagnóstico e das correções não estejam explícitos nas interações, o fato de todas as categorias terem atingido 100% de sucesso na avaliação de 2026-06-13 indica que os problemas subjacentes foram identificados e resolvidos, provavelmente através de ajustes na lógica de recuperação de informação e roteamento de ferramentas do agente.
*   **Erro de Dependência no Modelo CN2:**
    *   **Problema:** Ao tentar executar o modelo CN2 (indução de regras) no experimento de Ghoneim, Rashed & Elkalashy (2021), foi reportado um "erro no Orange/CN2: No module named 'bottleneck'".
    *   **Diagnóstico e Solução:** Este erro indica uma dependência de software ausente (`bottleneck`). A solução seria instalar o módulo `bottleneck` no ambiente de execução do Orange/CN2. O problema permanece não resolvido nas sessões atuais, resultando em métricas ausentes para este modelo.

## 4. RESULTADOS E MÉTRICAS OBTIDOS

Os seguintes resultados foram consolidados a partir dos artefatos do pipeline:

### Autoencoder - Modelo de Normalidade
*   **Limiar p99:** 2.0785
*   **Média baseline:** 0.2309
*   **Desvio baseline:** 0.4528
*   **Falsos positivos validação:** 1.10%
*   **Épocas treinadas:** 150

### Injeção de Falhas Sintéticas
*   **Limiar:** 2.0785
*   **Baseline:** 0.2052 ± 0.2433
| Falha | NPR | SMD | Erro na SMD | Margem |
|---|---:|---:|---:|---:|
| Degradação Filtro LCL | 210 | 1.0 | 2.2238 | 1.07x |
| Desbalanceamento de Fase | 150 | 0.3 | 2.6079 | 1.25x |
| Falha de Sensor CA | - | 0.1 | 11.6359 | 5.60x |

### Validação Formal
| Falha | Severidade | AUC-ROC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Degradação Filtro LCL | 1.0 | 0.943 | 0.649 | 0.480 | 1.000 |
| Desbalanceamento de Fase | 0.3 | 1.000 | 1.000 | 1.000 | 1.000 |
| Falha de Sensor CA | 0.3 | 1.000 | 1.000 | 1.000 | 1.000 |

### RUL / Weibull
| Falha | NPR | beta | eta | MTTF | B10 | Interpretação |
|---|---:|---:|---:|---:|---:|---|
| Degradação Filtro LCL | 210 | 3.449 | 55.0 | 49.5 | 28.6 | desgaste progressivo |
| Desbalanceamento de Fase | 150 | 5.866 | 29.7 | 27.5 | 20.3 | desgaste progressivo |
| Falha de Sensor CA | D=10 | 3.959 | 6.0 | 5.5 | 3.4 | desgaste progressivo |

### Experimentos por Artigo
| Experimento | Modelo | Accuracy | Precision | Recall | F1 | AUC | Specificity | Anomalias detectadas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ahirwar & Nandanwar (2025) | Isolation Forest | 0.519 | 0.613 | 0.104 | 0.178 | 0.628 | 0.934 | 31 |
| Ahirwar & Nandanwar (2025) | AE-LSTM | 0.500 | 0.500 | 0.027 | 0.052 | 0.729 | 0.973 | 10 |
| Ahirwar & Nandanwar (2025) | Facebook Prophet | 0.500 | 0.500 | 0.011 | 0.022 | 0.572 | 0.989 | 4 |
| Ahirwar & Nandanwar (2025) | Híbrido (voto) | 0.500 | 0.500 | 0.038 | 0.071 | 0.518 | 0.962 | 14 |
| Francisti et al. (2025) | Z-score (estatístico) | 0.566 | 0.694 | 0.236 | 0.352 | 0.741 | 0.896 | 62 |
| Francisti et al. (2025) | Random Forest (anomalia) | 0.852 | 0.890 | 0.802 | 0.844 | 0.939 | 0.901 | 164 |
| Ghoneim, Rashed & Elkalashy (2021) | Random Forest | 0.870 | 0.887 | 0.870 | 0.868 | 0.987 | 0.957 | - |
| Ghoneim, Rashed & Elkalashy (2021) | AdaBoost | 0.370 | 0.194 | 0.370 | 0.242 | 0.747 | 0.790 | - |
| Ghoneim, Rashed & Elkalashy (2021) | Regressão Logística | 0.710 | 0.638 | 0.710 | 0.643 | 0.815 | 0.903 | - |
| Ghoneim, Rashed & Elkalashy (2021) | Naive Bayes | 0.810 | 0.876 | 0.810 | 0.792 | 0.889 | 0.937 | - |
| Ghoneim, Rashed & Elkalashy (2021) | CN2 (indução de regras) | - | - | - | - | - | - | - |
| Ibrahim et al. (2022) | Isolation Forest | 0.519 | 0.613 | 0.104 | 0.178 | 0.628 | 0.934 | 31 |
| Ibrahim et al. (2022) | AE-LSTM | 0.500 | 0.500 | 0.027 | 0.052 | 0.729 | 0.973 | 10 |
| Ibrahim et al. (2022) | Facebook Prophet | 0.500 | 0.500 | 0.011 | 0.022 | 0.572 | 0.989 | 4 |
| Sharma et al. (2026) | Isolation Forest | 0.500 | 0.500 | 0.057 | 0.102 | 0.620 | 0.943 | 10 |
| Sharma et al. (2026) | KNN | 0.739 | 1.000 | 0.477 | 0.646 | 0.832 | 1.000 | 42 |
| Sharma et al. (2026) | SVM | 0.778 | 0.915 | 0.614 | 0.735 | 0.935 | 0.943 | 59 |
| Sharma et al. (2026) | ANN (MLP) | 0.881 | 0.959 | 0.795 | 0.870 | 0.927 | 0.966 | 73 |
| Sharma et al. (2026) | RNN | 0.500 | 0.000 | 0.000 | 0.000 | 0.531 | 1.000 | 0 |
| Sharma et al. (2026) | CNN | 0.557 | 0.778 | 0.159 | 0.264 | 0.740 | 0.955 | 18 |
| Sharma et al. (2026) | Isolation Forest + PPO | 0.500 | 0.500 | 1.000 | 0.667 | 0.620 | 0.000 | 176 |

### Resultados Específicos do Experimento Francisti et al. (2025)
*   **Melhor Modelo:** Random Forest (anomalia) com F1=0.8439.
*   **Protocolo:** `split temporal_com_purga` (purga=2), 273 janelas de treino, 182 de teste. Injeção de falhas `fmea_espaco_features` (LCL, desbalanceamento, sensor) com severidade 1.0. Decisão Z-score: `|z| > 3.0σ` por variável (Shewhart, fixo a priori). Decisão Random Forest: `probabilidade nativa ≥ 0,5`.
*   **Detecção por família de falha (recall):**
    *   **Z-score (estatístico):** LCL: 37%, Desbalanceamento: 14%, Sensor: 12%.
    *   **Random Forest (anomalia):** LCL: 99%, Desbalanceamento: 42%, Sensor: 95%.

## 5. INSIGHTS TÉCNICOS E ACADÊMICOS

*   **Definição e Contexto do FMEA:** Segundo Sakurada (1998, p. 3), o FMEA é uma técnica analítica para garantir que modos potenciais de falha e suas causas/mecanismos sejam considerados e localizados. É um método qualitativo de análise de confiabilidade que estuda modos de falha e seus efeitos em itens e funções. No contexto do projeto, o FMEA é aplicado para identificar e avaliar modos de falha potenciais em componentes CA do inversor fotovoltaico on-grid trifásico, como o inversor NPR=210 e o subsistema CA NPR=150, identificados como críticos no TCC base da UFPA (2024). Ele é usado em conjunto com modelagem de normalidade e detecção de anomalias.
*   **Interpretação dos Resultados do Autoencoder:** O detector de anomalias baseado em Autoencoder é calibrado pelo erro de reconstrução. Uma distância maior entre o erro de falha e o limiar indica uma anomalia mais clara.
*   **Significado da SMD (Severidade Mínima Detectável):** A SMD representa a menor severidade de uma falha na qual o Autoencoder consegue cruzar o limiar de detecção. Isso é crucial para entender a sensibilidade do modelo a diferentes níveis de degradação.
*   **Validação Formal e AUC-ROC:** Um valor de AUC-ROC próximo de 1 na validação formal indica uma separação muito forte entre o comportamento saudável e as falhas injetadas, demonstrando a robustez do modelo em distinguir essas condições.
*   **Análise RUL / Weibull e Degradação Progressiva:** Os valores de `beta > 1` nos resultados da análise Weibull para as falhas de Degradação do Filtro LCL, Desbalanceamento de Fase e Falha de Sensor CA sustentam a hipótese de degradação progressiva. Isso é um achado importante, pois valida a aplicabilidade de estratégias de manutenção preditiva baseadas em Machine Learning para essas falhas.
*   **Comparação de Abordagens de Machine Learning para Anomalias:**
    *   **Modelos de Aprendizado de Máquina vs. Estatísticos:** Os resultados dos experimentos por artigo sugerem que modelos de Machine Learning, como Random Forest e ANN (MLP), geralmente apresentam melhor desempenho na detecção de anomalias em dados do Paderborn (com anomalias sintéticas) em comparação com modelos estatísticos como o Z-score.
    *   **Interpretação de Métricas:** Um AUC alto indica boa capacidade de separação, mas deve ser complementado com F1/Accuracy e o número de anomalias detectadas. Um AUC ou recall alto com poucas ou zero anomalias detectadas pode indicar que o modelo ranqueia bem, mas opera de forma muito conservadora no ponto de decisão escolhido.
    *   **Limitações dos Experimentos Atuais:** Os experimentos são classificados como "benchmark exploratório" (E1), utilizando injeção sintética de falhas. Isso significa que os resultados podem não ser generalizáveis para dados reais e há uma lacuna na validação formal e no desempenho industrial.
*   **Abordagens de ML na Dissertação:**
    *   **Supervisionada (PV Farms - CC):** Foca na classificação de falhas conhecidas com dados rotulados (RF, AdaBoost, LogReg, Naive Bayes, CN2). Requer muitos dados rotulados.
    *   **Não Supervisionada (Paderborn - CA):** Aprende a normalidade para detectar desvios (Autoencoder, Isolation Forest). Detecta anomalias, mas não garante diagnóstico causal.
    *   **Sintética (FMEA):** Valida assinaturas de falhas modeladas com dados artificiais. Depende de calibração física.
    *   **Inconsistência:** As abordagens não se fundem; por exemplo, o classificador PV Farms não diagnostica falhas CA do inversor, nem suas métricas são transferíveis diretamente ao pipeline CA. A comparação entre protocolos de artigos é desafiadora, favorecendo o uso do AUC para comparação de capacidade de separação.
*   **Características do Dataset Paderborn (Stender, 2020):** O dataset pode ser usado para treinar modelos de inversor e esquemas de compensação, resultando em altas precisões de tensão. Suas variáveis podem ser definidas como entradas e alvos para esses modelos.

## 6. ESTADO ATUAL DO PIPELINE

*   **Modelo de Normalidade (Autoencoder):** Calibrado com um limiar p99 de 2.0785 e validado com 1.10% de falsos positivos.
*   **Mecanismo de Injeção de Falhas Sintéticas:** Implementado e utilizado para gerar anomalias guiadas pelo FMEA, com resultados de SMD e margens para diferentes tipos de falha.
*   **Validação Formal:** Concluída para as falhas de Degradação do Filtro LCL, Desbalanceamento de Fase e Falha de Sensor CA, mostrando alta capacidade de detecção (AUC-ROC próximo de 1).
*   **Análise RUL / Weibull:** Realizada, confirmando a hipótese de degradação progressiva para as falhas analisadas.
*   **Benchmarking de Experimentos da Literatura:** Vários modelos de ML e estatísticos foram avaliados usando dados do Paderborn (com anomalias sintéticas) e PV Farms (dados rotulados CC), fornecendo uma base comparativa de desempenho.
*   **Execução de Experimentos Específicos:** O experimento de Francisti et al. (2025) foi executado com sucesso, gerando resultados detalhados e gráficos.
*   **Agente Al IAdo PV:** Completamente funcional em termos de testes internos, com 100% de aprovação nas últimas avaliações técnicas.
*   **Pendências:** O modelo CN2 do experimento de Ghoneim, Rashed & Elkalashy (2021) não pôde ser executado devido a um erro de dependência (`No module named 'bottleneck'`).

## 7. PRÓXIMOS PASSOS IDENTIFICADOS

*   **Prioridade Alta:**
    *   **Resolver dependência do modelo CN2:** Instalar o módulo `bottleneck` para permitir a execução e avaliação completa do experimento de Ghoneim, Rashed & Elkalashy (2021).
*   **Prioridade Média:**
    *   **Aprofundar a validação dos modelos promissores:** Realizar estudos adicionais para validar os modelos de Machine Learning mais promissores (como Random Forest e ANN) em cenários mais próximos da realidade industrial, buscando superar as limitações da validação exploratória atual.
    *   **Investigar a generalização dos modelos:** Avaliar como os modelos treinados com dados sintéticos se comportam com dados reais de inversores fotovoltaicos, se disponíveis.

## 8. REFERÊNCIAS E FONTES CITADAS

*   **Sakurada (1998) — As Tecnicas De Analise Do Modos De Falhas E Seus Efeitos E A:** Citado para a definição e contextualização do FMEA, especificamente na página 3.
*   **Torres (2024) — Aplicacao Da Metodologia Reliability Centred Maintenance A S:** Mencionada em contexto de FMECA.
*   **Carpinetti (2016) — Gestao Da Qualidade Cap 6:** Mencionada em contexto de FMEA.
*   **Karim (2025) — A Review On Risk And Reliability Analysis In Photovoltaic Po:** Mencionada em contexto de métodos de análise de risco e confiabilidade.
*   **TCC base da UFPA (2024):** Citado como fonte para a identificação do inversor NPR=210 e subsistema CA NPR=150 como críticos no contexto do projeto.
*   **Stender (2020) — Data Set Description Three Phase Igbt Two Level Inverter For:** Referenciado para a descrição do dataset Paderborn, indicando seu uso para treinar modelos de inversor e esquemas de compensação.
*   **Francisti (2025) — Predictive Modeling And Anomaly Detection In Solar Pv Invert:** Artigo base para um dos experimentos de detecção de anomalias executados, utilizando modelos Z-score e Random Forest.
*   **Ghoneim (2021) — Fault Detection Algorithms For Achieving Service Continuity:** Artigo base para experimentos de detecção de falhas, utilizando modelos como Random Forest, AdaBoost, Regressão Logística, Naive Bayes e CN2, com dados do PV Farms.
*   **Ahirwar & Nandanwar (2025):** Artigo que inspira modelos e metodologia para experimentos de anomalia no dataset Paderborn (Isolation Forest, AE-LSTM, Facebook Prophet, Híbrido).
*   **Ibrahim et al. (2022):** Artigo que inspira modelos e metodologia para experimentos de anomalia no dataset Paderborn (Isolation Forest, AE-LSTM, Facebook Prophet).
*   **Sharma et al. (2026):** Artigo que inspira modelos e metodologia para experimentos de anomalia no dataset Paderborn (Isolation Forest, KNN, SVM, ANN, RNN, CNN, Isolation Forest + PPO).