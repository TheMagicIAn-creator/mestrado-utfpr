---
data: 2026-05-23
tipo: memoria-consolidada
sessoes_incluidas: 8
periodo: 2026-05-21 a 2026-05-23
tags: [al-iado-pv, memoria, consolidado, mestrado]
---

# Memória Consolidada — 23/05/2026

> Gerado automaticamente a partir de 8 sessões (2026-05-21 a 2026-05-23)

---

Prezado Rodolfo,

Com base nas transcrições das sessões de pesquisa, compilei um resumo consolidado detalhado, abrangendo os principais tópicos, conclusões, insights e próximos passos para a sua dissertação de mestrado.

---

## RESUMO CONSOLIDADO DAS SESSÕES DE PESQUISA

### 1. Principais Tópicos Discutidos

As sessões de pesquisa abordaram os seguintes temas técnicos:

*   **Matriz FMEA (Failure Mode and Effects Analysis) para o Lado CA do Inversor Fotovoltaico:** Discussão sobre a necessidade de construir uma matriz FMEA detalhada para os componentes do lado CA do inversor (Filtro LCL, IGBTs, Contactores, Sensores, Transformadores), com foco na identificação de modos de falha, causas, efeitos e, crucialmente, as **assinaturas elétricas** de cada falha.
*   **Assinaturas Elétricas de Falhas:** Ênfase na importância de descrever concretamente como as falhas se manifestam nos sinais elétricos (corrente CA, tensão CA, tensão CC, duty cycle PWM) para fundamentar a detecção por Machine Learning.
*   **Componente com Maior Taxa de Falha em SFVs:** Identificação do inversor como o componente mais crítico em sistemas fotovoltaicos em termos de ocorrência de falhas e perda de energia.
*   **Análise FMECA (Failure Mode, Effects, and Criticality Analysis) do Sistema Fotovoltaico do CEAMAZON:** Revisão dos resultados do TCC de graduação de Rodolfo (Torres, 2024) sobre a aplicação da FMECA, focando na criticidade do inversor.
*   **Índices de NPR (Número de Prioridade de Risco):** Discussão sobre o cálculo (NPR = Severidade x Ocorrência x Detecção) e a interpretação dos NPRs para priorizar modos de falha.
*   **Dataset da Universidade de Paderborn:** Descrição do dataset utilizado para modelar o comportamento saudável do inversor, incluindo os sinais contidos, taxa de amostragem e volume de dados.
*   **Modelos de Machine Learning para Detecção de Anomalias:** Justificativa técnica para a escolha do Autoencoder (AE), incluindo arquiteturas híbridas como AE-LSTM e a combinação com Isolation Forest, para a detecção de anomalias no filtro LCL.

### 2. Conclusões e Decisões Tomadas

As seguintes conclusões e decisões foram estabelecidas ao longo das sessões:

*   **Inversor como Componente Mais Crítico:** Foi consistentemente confirmado que o **inversor** é o componente que apresenta a maior taxa de falha em sistemas fotovoltaicos. Dados adaptados de Golnas (citado em Torres, 2024) indicam que o inversor é responsável por **43% dos "tickets" de falha** e **36% da perda de energia (kWh)**.
*   **Resultados da FMECA do CEAMAZON:** A análise FMECA do TCC (Torres, 2024) revelou a alta criticidade do inversor e do subsistema CA. O modo de falha **"Problema de conexão com a rede" no inversor** apresentou o **NPR mais elevado (210)**, com índices de Severidade (S)=3, Ocorrência (O)=7 e Detecção (D)=10. O modo de falha "Curto-circuito em dispositivos de proteção" no subsistema CA obteve um NPR de 150 (S=5, O=3, D=10).
*   **Adequação do Dataset de Paderborn:** O dataset da Universidade de Paderborn (`Inverter_Data_Set.csv`) é considerado **especialmente adequado** para modelar o comportamento saudável do inversor. Ele contém 26 colunas de sinais elétricos (tensão CC, correntes CA trifásicas, duty cycle PWM, tensões CA, velocidade) e foi coletado com uma **taxa de amostragem de 10 kHz**, totalizando aproximadamente 235 mil amostras, e **não contém falhas**.
*   **Autoencoder como Modelo Principal para Detecção de Anomalias:** O **Autoencoder (AE)** foi justificado como o modelo mais adequado para detectar anomalias no filtro LCL do inversor de Paderborn. Sua capacidade de **aprendizado não supervisionado** é ideal para o dataset sem falhas, e sua habilidade de capturar dependências temporais (especialmente em arquiteturas **AE-LSTM**) é crucial para sinais de alta taxa de amostragem.
*   **Abordagem Híbrida Recomendada:** A combinação de **AE-LSTM** com **Isolation Forest** é uma estratégia robusta e recomendada para a detecção de anomalias, visando maior precisão e robustez.

### 3. Insights Técnicos Relevantes

Os pontos mais importantes para a dissertação incluem:

*   **Foco na Assinatura Elétrica:** A coluna "assinatura elétrica" na matriz FMEA é a mais importante, pois será a base para a engenharia de *features* e a detecção de falhas por Machine Learning. A especificidade (e.g., "aumento de THD na corrente", "componente CC na corrente de fase", "queda de amplitude em uma fase") é fundamental.
*   **Metodologia FMECA como Fundamento:** A FMECA é uma ferramenta sistemática para identificar e priorizar riscos, sendo essencial para a Manutenção Centrada em Confiabilidade (MCC). A quantificação da criticidade via NPR é um diferencial.
*   **NPR como Priorizador de Falhas:** O NPR (Severidade x Ocorrência x Detecção) é um índice crucial para priorizar quais modos de falha merecem maior atenção na modelagem e detecção de anomalias. Um valor alto de NPR indica uma falha com grande impacto, alta probabilidade e/ou difícil detecção.
*   **Aprendizado Não Supervisionado para Normalidade:** A ausência de dados de falha no dataset de Paderborn direciona a pesquisa para modelos de aprendizado não supervisionado, como Autoencoders, que aprendem o comportamento "normal" e sinalizam desvios como anomalias através do erro de reconstrução.
*   **Importância das Dependências Temporais:** Sinais elétricos de inversores são séries temporais. Modelos como LSTM (integrados em Autoencoders) são essenciais para capturar as dependências sequenciais e padrões dinâmicos desses dados de alta frequência (10 kHz).
*   **Interpretabilidade de Modelos (SHAP):** A correlação clara entre as assinaturas elétricas e os modos de falha é vital para a interpretabilidade dos modelos de ML, especialmente com ferramentas como SHAP, que serão utilizadas para explicar as detecções de anomalias.

### 4. Próximos Passos Identificados

As seguintes ações foram identificadas para a continuidade da pesquisa:

*   **Elaborar a Matriz FMEA Detalhada do Lado CA:** Construir a matriz FMEA para cada componente do lado CA do inversor (Filtro LCL, IGBTs, Contactores, Sensores, Transformadores), preenchendo as colunas de Componente CA, Função, Modo de Falha, Causa, Efeito, Assinatura Elétrica e Índices FMECA (S, O, D, NPR com justificativa). A profundidade e a especificidade das assinaturas elétricas são cruciais.
*   **Mapear Assinaturas Elétricas para Modos de Falha:** Detalhar como cada modo de falha se manifesta concretamente nos sinais elétricos, utilizando exemplos específicos e quantificáveis sempre que possível.
*   **Conectar FMEA com Injeção de Falhas Sintéticas:** Utilizar os modos de falha identificados na FMEA e seus respectivos NPRs como base para a modelagem e injeção de falhas sintéticas no dataset, garantindo que as assinaturas elétricas geradas sejam realistas e representativas.
*   **Revisitar o TCC Completo de Rodolfo (Torres, 2024):** Consultar as tabelas FMEA/FMECA completas no TCC para extrair todos os valores numéricos de Severidade, Ocorrência, Detecção e NPR para os modos de falha do inversor e subsistema CA. Isso servirá como um *ground truth* e ponto de partida para a FMEA do mestrado.
*   **Definir/Adaptar Critérios de S, O, D:** Estabelecer claramente os escores e critérios para Severidade, Ocorrência e Detecção para os componentes do lado CA do inversor, alinhando-os com o contexto da dissertação.
*   **Documentar os NPRs Obtidos:** Registrar os valores de NPR para cada modo de falha identificado no lado CA do inversor na nova FMEA, justificando a priorização das falhas para a injeção de falhas sintéticas e a validação do modelo de detecção de anomalias.

### 5. Referências Citadas

As seguintes referências foram mencionadas e utilizadas nas discussões:

*   **Ahirwar (2025)** — A Review on Anomaly Detection in Solar PV Plants using Machine Learning.
*   **Autor Desconhecido (2024)** — Universidade Federal Do Par A.
*   **Autor Desconhecido (2024)** — Utfpr Universidade Tecnologica Federal Do Parana.
*   **Branco P (2024)** — Identifying Critical Failures In Pv Systems Based On Pv Inve.
*   **Carpinetti (2016)** — Gestao Da Qualidade Cap 6.
*   **Colli (2015)** — (citado por Torres, 2024, sobre criticidade do inversor).
*   **Cristaldi (2017)** — A Root Cause Analysis And A Risk Evaluation Of Pv Balance Of.
*   **Eletrica (s.d.)** — Subestacoes De Energia Definicoes Conceitos E Aplicacoes.
*   **Francisti (2025)** — Anomaly Detection in Photovoltaic Inverters using Machine Learning.
*   **Frontin (2013)** — Equipamentos de Alta Tensão.
*   **Golnas (2012)** — (citado por Torres, 2024, para distribuição de falhas).
*   **Ibrahim (2022)** — Unsupervised Anomaly Detection in Time Series Data using Variational Autoencoder.
*   **Karim (2025)** — A Review On Risk And Reliability Analysis In Photovoltaic Po.
*   **Sakurada (1998)** — As Tecnicas De Analise Do Modos De Falhas E Seus Efeitos E A.
*   **Sakurada et al. (2001)** — (citado por Torres, 2024, para definição de FMECA).
*   **Sayed et al. (2019)** — (citado por Torres, 2024, sobre criticidade do inversor).
*   **Torres (2024)** — Aplicacao Da Metodologia Reliability Centred Maintenance A Sistemas Fotovoltaicos (TCC de Rodolfo).
*   **Voss et al. (2009)** — (citado por Torres, 2024, sobre criticidade do inversor).
*   **Xavier (2005)** — Analise De Confiabilidade Em Sistemas De Potencia.

---