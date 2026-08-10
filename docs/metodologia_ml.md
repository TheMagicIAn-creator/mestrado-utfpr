# Metodologia de ML - detector canônico GPVS-Faults

Este documento define a metodologia vigente da dissertação. Os valores
numéricos citáveis devem ser lidos dos artefatos em `resultados/`; relatórios
de auditorias anteriores registram decisões históricas e não substituem este
contrato.

## 1. Escopo e fonte de dados

O pipeline principal usa exclusivamente o **GPVS-Faults**, uma microrede
fotovoltaica experimental conectada à rede (DOI
`10.17632/n76t439f65.1`). Nenhuma amostra de Stender, PMSM, PV Farms,
telemetria residencial ou Bearing DataCenter é concatenada, usada para ajuste
ou incorporada às métricas canônicas.

- `F0L` e `F0M`: operação saudável em IPPT e MPPT.
- `F1L` a `F7M`: 14 ensaios com sete falhas experimentais.
- **E2:** falhas sintéticas orientadas pela FMECA sobre o teste saudável F0.
- **E3:** validação experimental pré/pós-falha nos 14 ensaios F1-F7.

E3 significa bancada experimental, não campo. O detector indica desvio da
normalidade; não prova automaticamente a causa física nem a prevalência
industrial.

Pipeline: `features_gpvs -> autoencoder -> injecao_falhas -> validacao E2+E3
-> rul_weibull`.

## 2. Amostragem, janelas e atributos

A taxa de amostragem é inferida da coluna `Time` de cada CSV. O valor observado
é aproximadamente 10 kHz, embora o manual declare período de 9,9989 us. O
processamento usa os dados observados e preserva a divergência como ressalva de
qualidade.

Cada janela contém 200 amostras, equivalentes a um ciclo nominal de 50 Hz, sem
sobreposição. São extraídos 24 atributos de sensores primários:

- mediana e IQR de `Ipv`, `Vpv` e `Vdc`;
- RMS e THD de corrente e tensão nas três fases;
- desbalanceamento RMS de corrente e tensão;
- média e desvio da potência CA;
- mediana e IQR da potência CC.

O contrato é implementado por `src/ml/gpvs_principal.py`; um teste compara a
extração por janela com a implementação de referência em `src/ml/gpvs.py`.

## 3. Partições F0 e prevenção de vazamento

`F0L` e `F0M` são particionados separadamente, preservando a ordem temporal, em
quatro papéis: treino 50%, validação 15%, calibração 15% e teste 20%. Há purga
de duas janelas nas fronteiras. Nenhuma janela é compartilhada entre papéis.

- **Treino:** ajusta normalização, scaler e pesos.
- **Validação:** early stopping; não calibra o limiar.
- **Calibração:** estima o MSE p99 operacional.
- **Teste:** estima falsos positivos saudáveis sem participar do ajuste.

Os índices completos ficam no diagnóstico NPZ; `limiar.json` publica contagens
e política do split sem inflar o artefato.

## 4. Normalização de comissionamento

Variações de regime entre ensaios provocam deslocamento de nível. O contrato
aplica, antes do `RobustScaler`, centralização pela mediana e escala pelo IQR.
O IQR recebe piso igual a 10% do IQR global observado no treino F0, evitando
amplificação de atributos quase constantes.

Em F0, as estatísticas são ajustadas somente no treino. Em cada ensaio F1-F7,
a primeira metade pré-falha fornece o baseline de comissionamento. A segunda
metade pré-falha permanece isolada para estimar especificidade; todo o trecho
pós-falha estima sensibilidade.

Isso é uma normalização por ensaio, mas **não** é retreino do Autoencoder nem
recalibração do limiar. Pesos, `RobustScaler`, método de escore e limiar ficam
congelados.

## 5. Autoencoder e limiar

O modelo é um Autoencoder denso `24 -> 16 -> 8 -> 16 -> 24`, treinado somente
com F0 saudável. O gargalo e a saída são lineares; validação temporal controla
o early stopping. O método canônico de decisão é o MSE de reconstrução.

- `score_method = mse`.
- `score_threshold = mse_p99` da calibração F0.
- `mu + 3 sigma` e p95 são referências, não pontos de decisão.
- O escore localizado permanece como ablação, nunca como limiar operacional.

O p99 é nominal e interpolado. A taxa de excedência observada deve ser
reportada separadamente, com intervalo de Wilson descritivo por janela.

## 6. Validação E2 orientada pela FMECA

As assinaturas sintéticas de Contator CA, IGBT e Fusível CA são aplicadas a
janelas não sobrepostas do teste F0. O limiar permanece congelado. Para cada
severidade são publicados taxa de detecção, IC95% de Wilson e SMD95, a menor
severidade cuja estimativa pontual alcança 95%.

E2 verifica detectabilidade das assinaturas modeladas; não é falha física
observada nem desempenho de campo. Proxies, como ruído de sensor para o
contator, exigem calibração física antes de qualquer afirmação industrial.

## 7. Validação E3 experimental

O mesmo detector é aplicado aos 14 ensaios F1L-F7M. Por cenário são publicados
AUC, sensibilidade pós-falha, especificidade no trecho pré-falha isolado,
acurácia balanceada e atraso para cinco excedências consecutivas.

A unidade inferencial macro é o **ensaio**, não cada janela autocorrelacionada.
Os IC95% macro usam 20.000 reamostragens dos 14 ensaios. ICs de Wilson por
janela são apenas descritivos. Todos os cenários são publicados, inclusive os
resultados negativos.

## 8. Weibull e confiabilidade

A análise atual usa `a_det`, a primeira magnitude sintética que produz três
excedências consecutivas na E2. A Weibull 2P descreve a distribuição de
detectabilidade; seu eixo não é tempo.

`F_D(a)` é probabilidade de detecção até a magnitude `a`; `S_D(a)` é não
detecção; `h_D(a)` é intensidade de primeiro cruzamento por magnitude. Esses
objetos não são probabilidade de falha, confiabilidade, taxa de falha, MTTF ou
RUL físico. Tais inferências exigiriam várias unidades, exposição temporal,
falhas e censuras observadas.

## 9. Proveniência e publicação

Cada etapa possui manifesto v2 com hash LF-normalizado do código, dependências
científicas, entradas, parâmetros e saídas. O manifesto E3 inclui os 16 CSVs,
modelo, scaler, limiar e normalização de baseline. Dados brutos, modelos e
estado local do Obsidian não são versionados; JSON, CSV, Markdown e figuras
acadêmicas verificáveis são publicados.

Estados `ready`, `stale` e `pending` são calculados por `src/ml/proveniencia.py`.
Uma alteração em código, entrada ou saída torna o estágio desatualizado até a
regeneração e o novo manifesto.

## 10. Leitura dos resultados

O resultado deve sempre informar dataset, nível de evidência e unidade de
análise. Alta especificidade com sensibilidade limitada significa detector
conservador, não sucesso irrestrito. Diferenças por falha são achados, não
cenários a serem omitidos. Métricas de experimentos legados permanecem
consultáveis apenas como histórico e não podem ser combinadas ao GPVS.
