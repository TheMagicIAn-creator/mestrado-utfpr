---
titulo: 2026 07 16 consolidado
tipo: memoria-consolidada
tags: [memoria, consolidado, deteccao-anomalia, fmea, igbt, fusivel-ac, contator-ac, weibull-rul]
data: 2026-07-16
sessoes_incluidas: 2
interacoes_totais: 2
periodo: 2026-07-11 a 2026-07-16
---

# Memória Consolidada — 16/07/2026

> 2 sessões | 2 interações | 2026-07-11 a 2026-07-16

---

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
| Contator AC | 315 | 0.7 | 3.2743 | 1.58x |
| IGBT | 90 | 1.0 | 2.2238 | 1.07x |
| Fusível AC | 30 | ⚠️ não detectada | - | - |

### Validação Formal
| Falha | Severidade | AUC-ROC | Recall | F1 (50%) | F1 (raro 5%) |
|---|---:|---:|---:|---:|---:|
| Contator AC | 1.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| IGBT | 1.0 | 0.943 | 0.480 | 0.649 | 0.649 |
| Fusível AC | 1.0 | 0.937 | 0.020 | 0.039 | 0.039 |

### RUL / Weibull
| Falha | NPR | beta | eta | MTTF | B10 | Interpretação |
|---|---:|---:|---:|---:|---:|---|
| Contator AC | 315 | 4.626 | 37.3 | 34.1 | 22.9 | desgaste progressivo |
| IGBT | 90 | 3.449 | 55.0 | 49.5 | 28.6 | desgaste progressivo |
| Fusível AC | 30 | 2.299 | 102.1 | 90.4 | 38.3 | desgaste progressivo |

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

## 5. INSIGHTS TÉCNICOS E ACADÊMICOS

*   **Conexão entre FMEA e Injeção de Falhas Sintéticas:** A utilização do FMEA para guiar a injeção de falhas sintéticas permitiu que os experimentos sejam mais relevantes para o domínio de aplicação, aumentando a validade dos resultados.
*   **Importância da Escolha de Métricas:** A escolha da métrica AUC como principal medida de desempenho para a detecção de anomalias se mostrou apropriada, considerando as variações nos protocolos de decisão e limiares de classificação entre os estudos.

## 6. ESTADO ATUAL DO PIPELINE

*   **Componentes Funcionais:** O pipeline de processamento de dados, incluindo a injeção de falhas sintéticas, validação formal e cálculo de RUL/Weibull, está funcionando corretamente.
*   **Componentes Pendentes:** A resolução do erro de dependência no modelo CN2 permanece pendente, afetando a disponibilidade de métricas para este modelo.

## 7. PRÓXIMOS PASSOS IDENTIFICADOS

*   **Resolução do Erro de Dependência no Modelo CN2:** Instalar o módulo `bottleneck` no ambiente de execução do Orange/CN2 para resolver o erro de dependência.
*   **Análise Aprofundada dos Resultados:** Realizar uma análise mais detalhada dos resultados obtidos, especialmente em relação à performance dos diferentes modelos de detecção de anomalias e à interpretação dos parâmetros Weibull.

## 8. REFERÊNCIAS E FONTES CITADAS

*   Francisti et al. (2025) - Artigo que serviu de base para o experimento de detecção de anomalias utilizando Z-score e Random Forest.
*   Ghoneim, Rashed & Elkalashy (2021) - Artigo que apresentou os resultados dos modelos Random Forest, AdaBoost, Regressão Logística e Naive Bayes para detecção de anomalias.
*   Ahirwar & Nandanwar (2025) - Artigo que discutiu a aplicação de Isolation Forest, AE-LSTM, Facebook Prophet e um modelo híbrido para detecção de anomalias.