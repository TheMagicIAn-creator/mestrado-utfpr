# Resultados da Fase 5 - Pipeline de ML

> Gerado em 02/06/2026 09:50

Aqui está o que já existe nos artefatos do pipeline.

## Autoencoder - modelo de normalidade

| Métrica | Valor |
|---|---:|
| Limiar p99 | 2.0785 |
| Média baseline | 0.2309 |
| Desvio baseline | 0.4528 |
| Falsos positivos validação | 1.10% |
| Épocas treinadas | 150 |

Leitura rápida: o detector está calibrado por erro de reconstrução. Quanto maior a distância entre erro de falha e limiar, mais clara é a anomalia.

## Injeção de falhas sintéticas
Limiar: **2.0785**. Baseline: **0.2052 ± 0.2433**.
| Falha | NPR | SMD | Erro na SMD | Margem |
|---|---:|---:|---:|---:|
| Degradação Filtro LCL | 210 | 1.0 | 2.2238 | 1.07x |
| Desbalanceamento de Fase | 150 | 0.3 | 2.6079 | 1.25x |
| Falha de Sensor CA | - | 0.1 | 11.6359 | 5.60x |

Leitura rápida: a SMD é a menor severidade em que o Autoencoder cruza o limiar.

## Validação formal

| Falha | Severidade | AUC-ROC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Degradação Filtro LCL | 1.0 | 0.943 | 0.649 | 0.480 | 1.000 |
| Desbalanceamento de Fase | 0.3 | 1.000 | 1.000 | 1.000 | 1.000 |
| Falha de Sensor CA | 0.3 | 1.000 | 1.000 | 1.000 | 1.000 |

Leitura rápida: AUC próximo de 1 indica separação muito forte entre comportamento saudável e falha injetada.

## RUL / Weibull

| Falha | NPR | beta | eta | MTTF | B10 | Interpretação |
|---|---:|---:|---:|---:|---:|---|
| Degradação Filtro LCL | 210 | 3.449 | 55.0 | 49.5 | 28.6 | desgaste progressivo |
| Desbalanceamento de Fase | 150 | 5.866 | 29.7 | 27.5 | 20.3 | desgaste progressivo |
| Falha de Sensor CA | D=10 | 3.959 | 6.0 | 5.5 | 3.4 | desgaste progressivo |

Leitura rápida: beta > 1 sustenta a hipótese de degradação progressiva, coerente com manutenção preditiva.

## Experimentos por artigo

| Experimento | Modelo | Accuracy | Precision | Recall | F1 | AUC | Specificity | Anomalias detectadas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ahirwar & Nandanwar (2025) | Isolation Forest | 0.607 | 0.753 | 0.317 | 0.446 | 0.759 | 0.896 | 77 |
| Ahirwar & Nandanwar (2025) | AE-LSTM | 0.877 | 0.867 | 0.891 | 0.879 | 0.913 | 0.863 | 188 |
| Ahirwar & Nandanwar (2025) | Facebook Prophet | 0.500 | 0.500 | 1.000 | 0.667 | 0.577 | 0.000 | 366 |
| Ahirwar & Nandanwar (2025) | Híbrido (voto) | 0.672 | 0.610 | 0.956 | 0.745 | 0.782 | 0.388 | 287 |
| Francisti et al. (2025) | Z-score (estatístico) | 0.732 | 0.667 | 0.929 | 0.776 | 0.815 | 0.536 | 255 |
| Francisti et al. (2025) | Random Forest (anomalia) | 0.926 | 0.970 | 0.880 | 0.923 | 0.984 | 0.973 | 166 |
| Ghoneim, Rashed & Elkalashy (2021) | Random Forest | 0.870 | 0.887 | 0.870 | 0.868 | 0.987 | 0.957 | - |
| Ghoneim, Rashed & Elkalashy (2021) | AdaBoost | 0.370 | 0.194 | 0.370 | 0.242 | 0.747 | 0.790 | - |
| Ghoneim, Rashed & Elkalashy (2021) | Regressão Logística | 0.710 | 0.638 | 0.710 | 0.643 | 0.815 | 0.903 | - |
| Ghoneim, Rashed & Elkalashy (2021) | Naive Bayes | 0.810 | 0.876 | 0.810 | 0.792 | 0.889 | 0.937 | - |
| Ghoneim, Rashed & Elkalashy (2021) | CN2 (indução de regras) (erro no Orange/CN2: No module named 'bottleneck') | - | - | - | - | - | - | - |
| Ibrahim et al. (2022) | Isolation Forest | 0.607 | 0.753 | 0.317 | 0.446 | 0.759 | 0.896 | 77 |
| Ibrahim et al. (2022) | AE-LSTM | 0.877 | 0.867 | 0.891 | 0.879 | 0.913 | 0.863 | 188 |
| Ibrahim et al. (2022) | Facebook Prophet | 0.500 | 0.500 | 1.000 | 0.667 | 0.577 | 0.000 | 366 |
| Sharma et al. (2026) | Isolation Forest | 0.607 | 0.753 | 0.317 | 0.446 | 0.759 | 0.896 | 77 |
| Sharma et al. (2026) | KNN | 0.503 | 1.000 | 0.005 | 0.011 | 0.596 | 1.000 | 1 |
| Sharma et al. (2026) | SVM | 0.929 | 0.934 | 0.923 | 0.929 | 0.972 | 0.934 | 181 |
| Sharma et al. (2026) | ANN (MLP) | 0.923 | 0.994 | 0.852 | 0.918 | 0.978 | 0.995 | 157 |
| Sharma et al. (2026) | RNN | 0.713 | 0.637 | 0.989 | 0.775 | 0.814 | 0.437 | 284 |
| Sharma et al. (2026) | CNN | 0.620 | 0.569 | 0.995 | 0.724 | 0.925 | 0.246 | 320 |
| Sharma et al. (2026) | Isolation Forest + PPO | 0.678 | 0.692 | 0.639 | 0.665 | 0.759 | 0.716 | 169 |

Origem dos dados usados nestes resultados:
- **Ahirwar & Nandanwar (2025)**: Usa features locais do Paderborn extraidas de Inverter_Data_Set.csv. Como o Paderborn e saudavel, as anomalias avaliadas sao sinteticas, geradas no pipeline para criar ground truth. O artigo inspira os modelos e a metodologia; os dados avaliados sao os do repositorio.
- **Francisti et al. (2025)**: Usa features locais do Paderborn extraidas de Inverter_Data_Set.csv. Como o Paderborn e saudavel, as anomalias avaliadas sao sinteticas, geradas no pipeline para criar ground truth. O artigo inspira os modelos e a metodologia; os dados avaliados sao os do repositorio.
- **Ghoneim, Rashed & Elkalashy (2021)**: Usa os arquivos locais train_data.csv e test_data.csv em dados/brutos. O artigo define a metodologia/base PV Farms; os numeros sao recalculados no repositorio, nao copiados do paper.
- **Ibrahim et al. (2022)**: Usa features locais do Paderborn extraidas de Inverter_Data_Set.csv. Como o Paderborn e saudavel, as anomalias avaliadas sao sinteticas, geradas no pipeline para criar ground truth. O artigo inspira os modelos e a metodologia; os dados avaliados sao os do repositorio.
- **Sharma et al. (2026)**: Usa features locais do Paderborn extraidas de Inverter_Data_Set.csv. Como o Paderborn e saudavel, as anomalias avaliadas sao sinteticas, geradas no pipeline para criar ground truth. O artigo inspira os modelos e a metodologia; os dados avaliados sao os do repositorio.

Leitura rapida: AUC alto mede separacao por score. Para operacao real, olhe junto F1/accuracy e a coluna de anomalias detectadas; AUC ou recall alto com poucas ou zero anomalias detectadas indica que o modelo pode estar ranqueando bem, mas operando conservador demais no ponto escolhido.
