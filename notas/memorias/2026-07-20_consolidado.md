---
data: 2026-07-20
tipo: memoria-consolidada
sessoes_incluidas: 9
interacoes_totais: 22
periodo: 2026-07-09 a 2026-07-20
tags: [al-iado-pv, memoria, consolidado, mestrado]
---

# Memória Consolidada — 20/07/2026

> 9 sessões | 22 interações | 2026-07-09 a 2026-07-20

---

data: 2026-07-20
tipo: memoria-consolidada
sessoes_incluidas: 10
interacoes_totais: 25
periodo: 2026-07-11 a 2026-07-20
tags: [al-iado-pv, memoria, consolidado, mestrado, pipeline-recalibracao, limpeza-artefatos, novas-falhas-sinteticas, francisti-reexecucao, ibrahim-execucao]
---

# Memória Consolidada — 20/07/2026

> 10 sessões | 25 interações | 2026-07-11 a 2026-07-20

---

---

## 1. AÇÕES CONCRETAS REALIZADAS

*   **Re-execução e Recalibração do Pipeline Principal (Autoencoder, Injeção de Falhas, Validação E2, RUL/Weibull):**
    *   **O que foi pedido:** Rodolfo solicitou "rode o pipeline completo" ou "rode o pipeline".
    *   **O que foi feito:** O pipeline foi executado múltiplas vezes (em 2026-07-09 e novamente após uma limpeza em 2026-07-17). A execução mais recente (2026-07-17) incluiu:
        *   **Features CA:** Concluído com sucesso.
        *   **Autoencoder:** Treinado e calibrado.
        *   **Injeção de Falhas Sintéticas:** Executada com novos parâmetros e tipos de falha.
        *   **Validação Interna E2:** Realizada para as novas falhas e severidades.
        *   **RUL / Weibull:** Calculado com metodologia aprimorada.
    *   **Por que foi feito assim:** As re-execuções são parte do ciclo de desenvolvimento e refinamento do modelo, incorporando melhorias na metodologia de injeção de falhas, validação e análise de RUL.

*   **Execução de Experimentos de Anomalia (Francisti et al. 2025 e Ibrahim et al. 2022):**
    *   **O que foi pedido:** Rodolfo solicitou "rode os experimentos de anomalia" e "compare meu método com a literatura".
    *   **O que foi feito:**
        *   O experimento `francisti2025_spc` foi executado (em 2026-07-09 e re-executado em 2026-07-17/20). Os resultados para o modelo Z-score (estatístico) foram gerados.
        *   O experimento `ibrahim` foi executado (em 2026-07-17/20), incluindo os modelos Isolation Forest e AE-LSTM.
        *   As imagens resultantes foram salvas nos diretórios `resultados/experimentos/francisti/` e `resultados/experimentos/ibrahim/`.
    *   **Por que foi feito assim:** Para comparar o desempenho do método proposto com abordagens da literatura em cenários de benchmark exploratório (Evidência E1), utilizando o dataset Paderborn com injeção de falhas sintéticas orientadas pelo FMEA no espaço de features.

*   **Limpeza de Artefatos do Pipeline:**
    *   **O que foi pedido:** Rodolfo solicitou "apague todos os resultados e gráficos do pipeline".
    *   **O que foi feito:** Em 2026-07-17, 28 artefatos foram removidos a partir da etapa "Features CA", invalidando as etapas subsequentes para recalculo. Arquivos como `features_paderborn.parquet`, `modelo_autoencoder.pt`, `limiar.json`, e diversos gráficos e relatórios foram apagados.
    *   **Por que foi feito assim:** Para garantir que uma nova execução do pipeline começasse do zero, com dados e modelos atualizados, evitando inconsistências de versões anteriores.

## 2. DECISÕES ARQUITETURAIS TOMADAS

*   **Refinamento da Metodologia de Injeção de Falhas Sintéticas:** A metodologia de injeção de falhas foi aprimorada para incluir a métrica **SMD95** (Severidade Mínima Detectável com 95% de confiança) e seus respectivos intervalos de confiança (IC95%). Esta decisão permite uma avaliação mais robusta e probabilística da capacidade de detecção do Autoencoder em diferentes severidades, em vez de apenas um ponto de detecção binário.
*   **Aprimoramento da Análise de RUL/Weibull:** A análise de RUL foi atualizada para **preservar a censura** nos dados e calcular os **intervalos de confiança (IC95%)** para os parâmetros beta, eta, MTTF e B10 utilizando **bootstrap**. Esta melhoria aumenta a robustez estatística da estimativa de vida útil restante, embora os resultados ainda sejam considerados exploratórios devido à natureza sintética dos dados.
*   **Priorização de Métricas para Comparação de Modelos de Anomalia (Reafirmado):** O AUC (Área sob a Curva ROC) continua sendo a métrica prioritária para comparação entre diferentes modelos e artigos em benchmarks exploratórios (E1), pois mede a capacidade de separação independentemente do ponto de corte. O F1-score é analisado em conjunto com o número de anomalias detectadas para entender a operação prática do modelo no ponto de decisão escolhido.

## 3. PROBLEMAS ENCONTRADOS E SOLUÇÕES

*   **Erro de Dependência no Modelo CN2 (Persistente):**
    *   **Problema:** Ao tentar executar o modelo CN2 (indução de regras) no experimento de Ghoneim, Rashed & Elkalashy (2021) (mencionado na memória anterior), foi reportado um "erro no Orange/CN2: No module named 'bottleneck'".
    *   **Diagnóstico e Solução:** Este erro indica uma dependência de software ausente (`bottleneck`). A solução seria instalar o módulo `bottleneck` no ambiente de execução do Orange/CN2. O problema permanece não resolvido nas sessões atuais, resultando em métricas ausentes para este modelo.
*   **Erro de Execução no Experimento 'ibrahim' (Resolvido):**
    *   **Problema:** Em 2026-07-09, o experimento 'ibrahim' reportou "Nao executado - erro no experimento: 'Prophet' object has no attribute 'stan_backend'".
    *   **Diagnóstico e Solução:** Este erro sugere um problema com a biblioteca Prophet ou seu backend (Stan). Embora a solução específica não tenha sido registrada, o fato de o experimento 'ibrahim' ter sido executado com sucesso posteriormente (em 2026-07-17/20) com modelos Isolation Forest e AE-LSTM (e sem mencionar Prophet) indica que o problema foi contornado, possivelmente pela remoção do modelo Prophet ou pela correção do ambiente.
*   **Cálculo Indisponível no Ambiente Web:**
    *   **Problema:** Rodolfo tentou "rodar o pipeline completo" no ambiente web, mas recebeu a mensagem "Cálculo indisponível neste ambiente".
    *   **Diagnóstico e Solução:** O ambiente web está configurado em modo de consulta e não possui o dataset bruto (`dados/brutos/Inverter_Data_Set.csv`) necessário para re-executar o pipeline pesado. A solução é realizar os cálculos no PC local e publicar os artefatos para consulta no site.

## 4. RESULTADOS E MÉTRICAS OBTIDOS

Os seguintes resultados foram consolidados a partir dos artefatos do pipeline, refletindo a última execução completa em 2026-07-17/20:

### Autoencoder - Modelo de Normalidade
*   **Limiar p99:** 2.5454
*   **Média baseline:** 0.3536
*   **Desvio baseline:** 0.6341
*   **Janelas de treino:** 274
*   **Janelas de calibração:** 91
*   **Janelas de teste:** 88
*   **Falsos positivos no teste isolado:** 1.14%
*   **Épocas treinadas:** 75

### Injeção de Falhas Sintéticas
*   **Limiar:** 2.5454
*   **Baseline:** 0.5022 ± 0.9813
| Falha | NPR | SMD95 | Taxa (IC95%) | n | Erro mediano |
|---|---:|---:|---:|---:|---:|
| Contator AC | 315 | 0.7 | 0.955 [0.849; 0.987] | 44 | 9.4537 |
| IGBT | 90 | ⚠️ alvo não atingido | - | - | - |
| Fusível AC | 30 | ⚠️ alvo não atingido | - | - | - |
*   **Observações:**
    *   **IGBT:** taxa máxima de detecção 0.341 na severidade 1.0; o alvo probabilístico de 95% não foi atingido.
    *   **Fusível AC:** taxa máxima de detecção 0.045 na severidade 1.0; o alvo probabilístico de 95% não foi atingido.

### Validação Sintética Interna E2
| Falha | Sev. | AUC-ROC (IC95%) | Recall (IC95%) | FNR | Especificidade | n/classe |
|---|---:|---:|---:|---:|---:|---:|
| Contator AC | 0.3 | 0.953 [0.881; 1.000] | 0.075 [0.026; 0.199] | 0.925 | 0.975 | 40 |
| Contator AC | 0.5 | 0.975 [0.929; 1.000] | 0.650 [0.495; 0.779] | 0.350 | 0.975 | 40 |
| Contator AC | 1.0 | 0.997 [0.990; 1.000] | 1.000 [0.912; 1.000] | 0.000 | 0.975 | 40 |
| IGBT | 0.3 | 0.602 [0.466; 0.730] | 0.025 [0.004; 0.129] | 0.975 | 0.975 | 40 |
| IGBT | 0.5 | 0.836 [0.728; 0.920] | 0.025 [0.004; 0.129] | 0.975 | 0.975 | 40 |
| IGBT | 1.0 | 0.944 [0.880; 0.991] | 0.350 [0.221; 0.505] | 0.650 | 0.975 | 40 |
| Fusível AC | 0.3 | 0.552 [0.422; 0.670] | 0.025 [0.004; 0.129] | 0.975 | 0.975 | 40 |
| Fusível AC | 0.5 | 0.751 [0.630; 0.862] | 0.025 [0.004; 0.129] | 0.975 | 0.975 | 40 |
| Fusível AC | 1.0 | 0.903 [0.813; 0.971] | 0.050 [0.014; 0.165] | 0.950 | 0.975 | 40 |
*   **Observações:** O recall é baixo para Contator AC (sev. 0.3), IGBT (sev. 0.3, 0.5), Fusível AC (sev. 0.3, 0.5, 1.0), indicando que o limiar conservador p99 perde a maior parte dessas falhas.

### RUL / Weibull
| Falha | NPR | Eventos/Censura | beta (IC95%) | eta (IC95%) | MTTF (IC95%) | B10 (IC95%) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Contator AC | 315 | 43/1 | 1.69 [0.62; 5.34] | 56.5 [46.4; 66.1] | 50.4 [46.1; 70.7] | 14.9 [1.4; 37.7] | exploratório |
| IGBT | 90 | 15/29 | 0.73 [0.25; 7.99] | 480.7 [130.1; 9061.8] | 586.7 [121.0; 239738.1] | 21.9 [1.0; 102.2] | alta censura; RUL omitida |
| Fusível AC | 30 | 2/42 | - [-; -] | - [-; -] | - [-; -] | - [-; -] | não estimável |
*   **Observações:** A censura é preservada e os intervalos vêm de bootstrap. Os tempos continuam sendo passos de degradação sintética E2. MTTF, B10 e RUL descrevem o experimento computacional e não podem ser apresentados como vida útil física ou de campo.

### Experimentos por Artigo (Evidência E1: Benchmark Exploratório)
| Experimento | Modelo | Accuracy | Precision | Recall | F1 | AUC | Specificity | Anomalias detectadas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Francisti et al. (2025) | Z-score (estatístico) | 0.511 | 0.548 | 0.126 | 0.205 | 0.572 | 0.896 | 42 |
| Ibrahim et al. (2022) | Isolation Forest | 0.500 | 0.500 | 0.071 | 0.125 | 0.589 | 0.929 | 26 |
| Ibrahim et al. (2022) | AE-LSTM | 0.500 | 0.500 | 0.005 | 0.011 | 0.659 | 0.995 | 2 |
*   **Origem dos dados:** Ambos os experimentos usam features locais do Paderborn extraídas de `Inverter_Data_Set.csv`. As anomalias avaliadas são sintéticas, geradas no pipeline para criar ground truth.
*   **Nível de evidência:** E1 — benchmark exploratório (injeção sintética orientada pelo FMEA no espaço de features, com protocolo de decisão do próprio artigo); não é validação formal nem desempenho industrial.

## 5. INSIGHTS TÉCNICOS E ACADÊMICOS

*   **Impacto da Calibração do Autoencoder:** O Autoencoder foi recalibrado, resultando em um novo `Limiar p99` de 2.5454 (anteriormente 2.0785) e `Falsos positivos no teste isolado` de 1.14% (anteriormente 1.10%). Isso indica um ajuste fino para manter a taxa de falsos positivos baixa, o que é crucial para a aceitação industrial, mas pode impactar a sensibilidade a certas falhas.
*   **Variação na Detecção de Falhas Sintéticas:**
    *   O modelo demonstra alta eficácia na detecção de **Contator AC** em severidades mais altas (SMD95 de 0.7, Recall de 1.000 na severidade 1.0).
    *   No entanto, falhas como **IGBT** e **Fusível AC** são significativamente mais difíceis de detectar, com o alvo probabilístico de 95% de detecção (SMD95) não sendo atingido mesmo na severidade máxima (1.0). O recall para essas falhas permanece muito baixo em todas as severidades testadas (e.g., Fusível AC sev 1.0 com Recall de 0.050).
*   **Diferença entre AUC e Recall no Ponto de Operação:** A "leitura honesta" dos resultados de validação reforça a importância de distinguir entre a capacidade de ranqueamento do modelo (medida pelo AUC-ROC) e sua capacidade de detecção efetiva no ponto de operação (medida pelo Recall). Um AUC-ROC alto não garante um bom Recall se o limiar de decisão for muito conservador, como observado para IGBT e Fusível AC.
*   **Refinamento da Análise de RUL:** A inclusão de censura e intervalos de confiança por bootstrap na análise Weibull representa um avanço metodológico significativo, fornecendo uma estimativa mais robusta e transparente, embora a natureza sintética dos dados ainda exija cautela na interpretação como vida útil física.
*   **Contextualização com a Literatura (E1):** Os experimentos de Francisti et al. (2025) e Ibrahim et al. (2022) fornecem um benchmark exploratório. Os resultados mostram que o Autoencoder proposto no pipeline principal (com AUC de 0.793 para o banco comum, conforme sessão de 2026-07-09) pode superar abordagens mais simples como Z-score (AUC 0.572) e modelos como Isolation Forest (AUC 0.589) e AE-LSTM (AUC 0.659) em cenários de injeção de falhas no espaço de features.

## 6. ESTADO ATUAL DO PIPELINE

*   **Componentes Funcionais e Recalculados:** Todas as etapas do pipeline principal (Features CA, Autoencoder, Injeção de Falhas Sintéticas, Validação Interna E2, RUL / Weibull) foram executadas com sucesso e seus artefatos estão disponíveis, refletindo as últimas configurações e dados.
*   **Experimentos de Benchmark Funcionais:** Os experimentos de comparação com a literatura (Francisti et al. 2025 e Ibrahim et al. 2022) foram executados e seus resultados estão disponíveis.
*   **Componentes Pendentes:**
    *   **Erro de Dependência no Modelo CN2:** A resolução do erro de dependência (`No module named 'bottleneck'`) para o modelo CN2 (experimento de Ghoneim, Rashed & Elkalashy 2021) permanece pendente.
*   **Limitação do Ambiente Web:** O ambiente web está em modo de consulta e não permite a re-execução do pipeline completo devido à ausência do dataset bruto.

## 7. PRÓXIMOS PASSOS IDENTIFICADOS

*   **Investigar Baixo Recall para IGBT e Fusível AC:** Analisar as características dessas falhas e as features extraídas para entender por que o Autoencoder tem dificuldade em detectá-las no limiar p99. Considerar:
    *   Ajuste do limiar de detecção (trade-off com falsos positivos).
    *   Exploração de features mais discriminativas para essas falhas.
    *   Avaliação de modelos complementares ou ensemble para falhas de difícil detecção.
*   **Refinar a Interpretação do RUL/Weibull:** Dada a natureza exploratória e a alta censura para algumas falhas, aprofundar a discussão sobre as limitações e o valor indicativo dos resultados de MTTF e B10 na dissertação.
*   **Resolver Erro do CN2:** Instalar a dependência `bottleneck` para permitir a execução completa do experimento de Ghoneim, Rashed & Elkalashy (2021) e obter suas métricas.

## 8. REFERÊNCIAS E FONTES CITADAS

*   **Francisti et al. (2025):** Artigo que inspira um dos experimentos de benchmark exploratório (E1), utilizando o dataset Paderborn com injeção de anomalias sintéticas e um detector Z-score estatístico.
*   **Ibrahim et al. (2022):** Artigo que inspira outro experimento de benchmark exploratório (E1), utilizando o dataset Paderborn com injeção de anomalias sintéticas e modelos como Isolation Forest e AE-LSTM.
*   **Ghoneim, Rashed & Elkalashy (2021):** Artigo cujo experimento de indução de regras (modelo CN2) está pendente de execução devido a um erro de dependência.