# Assinaturas FMEA — justificativa físico-elétrica

Este documento traduz as assinaturas de falha implementadas no código
(`src/ml/injecao_falhas.py` — domínio do SINAL; `src/ml/protocolos_artigos.py`
`ASSINATURAS_FMEA` — domínio das FEATURES) em prosa técnica citável na
dissertação. As três famílias derivam do FMECA de Torres (2024, Apêndice E),
que apontou o inversor (NPR=210) e o subsistema CA (NPR=150) como os itens
mais críticos do sistema fotovoltaico do CEAMAZON.

Nível de evidência: **E2 no pipeline principal** (injeção no sinal bruto,
orientada pelo FMEA) e **E1 nos experimentos por artigo** (injeção no espaço
de features). Em ambos, as MAGNITUDES são plausíveis, não medidas em bancada
— ver limitações ao final.

## 1. Degradação do filtro LCL (NPR=210)

**Física da falha.** O filtro LCL na saída do inversor atenua os harmônicos
de chaveamento dos IGBTs antes da conexão com a rede. Dois mecanismos de
degradação dominam: (i) o capacitor do ramo paralelo perde capacitância e
ganha ESR (resistência série equivalente) com envelhecimento térmico do
eletrólito; (ii) os indutores saturam ou perdem indutância com degradação do
núcleo. Em ambos os casos a frequência de ressonância desloca e a atenuação
em alta frequência cai — os harmônicos de baixa ordem ímpares que o controle
não elimina (5º, 7º, 11º — os característicos de conversores trifásicos de
seis pulsos, ordens 6k±1) passam a aparecer com amplitude crescente nas
correntes de linha.

**Assinatura implementada.** Elevação dos harmônicos 5, 7 e 11 das três
correntes (`i_[abc]_harm_{5,7,11}`), do THD de corrente (`i_[abc]_thd`) e da
energia média espectral. No sinal bruto (`injecao_falhas.py`), soma-se
conteúdo harmônico proporcional à severidade; nas features, desloca-se cada
coluna em unidades do desvio-padrão do treino (evita escala arbitrária).

**Por que é detectável por modelagem de normalidade.** O Autoencoder aprende
a correlação saudável entre fundamental e harmônicos; harmônicos elevados com
fundamental normal violam essa correlação e inflam o erro de reconstrução.

## 2. Desbalanceamento de fase (NPR=150)

**Física da falha.** Perda parcial de uma fase — conexão degradada, contator
com resistência de contato elevada, ou falha em um braço do inversor — reduz
a corrente da fase afetada. HIPÓTESE DE MODELAGEM (explícita no código): o
controle do inversor redistribui corrente para as fases sãs para manter a
potência entregue, cenário típico de malhas de controle de corrente. A
assinatura CENTRAL e independente do controle é a métrica de desbalanceamento
subir; a compensação B/C é a parte dependente da estratégia de controle.

**Assinatura implementada.** Fase A enfraquece (RMS, pico a pico, desvio,
energia da fundamental, potência — modo multiplicativo, redução de 15–35% ×
severidade); fases B/C compensam parcialmente (modo aditivo); a feature
`desbalanceamento_corrente` (razão entre correntes de fase) sobe com o maior
peso da família (2–4 σ × severidade).

**Limitação honesta (registrada no código).** A hipótese NÃO cobre perda de
linha em que as três fases caem juntas, nem carga severamente desbalanceada
sem compensação. Resultado empírico relevante da execução vigente: no
pipeline principal, esta família apresentou o MENOR erro de reconstrução das
três — na execução de 2026-06-17 não cruzou o limiar operacional em nenhuma
severidade (SMD nula; ver `injecao_falhas_report.json`). Isso é um achado de
limitação do detector no ponto de operação, a discutir na dissertação, não a
omitir.

## 3. Falha de sensor de corrente (D=10 no FMEA)

**Física da falha.** Sensores de efeito Hall/shunt degradam por deriva
térmica, perda de blindagem ou envelhecimento do condicionamento de sinal. O
efeito dominante é ruído de medição: dispersão e conteúdo espectral de alta
frequência sobem NO CANAL MEDIDO, sem correspondência física nas outras
fases — a corrente real não mudou, a MEDIDA dela mudou.

**Assinatura implementada.** No canal da fase A: desvio-padrão, largura de
banda espectral, energia na banda de chaveamento, centroide espectral e THD
sobem (modo aditivo). As demais fases ficam intactas — é exatamente essa
incoerência entre canais que o detector explora.

**Limitação honesta (registrada no artefato).** O ruído gaussiano é um PROXY
da degradação real de sensor; a alta sensibilidade observada (SMD baixa)
exige calibração física antes de qualquer alegação de desempenho em campo
(`evidence_note` no report de injeção).

## Pesos de amostragem entre famílias

`PESOS_FALHAS = {lcl: 0.40, desbalanceamento: 0.35, sensor: 0.25}` — segue a
ordem de criticidade do FMECA (NPR 210 > NPR 150 > sensor sem NPR, D=10),
cumprindo o critério metodológico de priorização por NPR definido no perfil
do projeto.

## Limitações gerais (para a seção de ameaças à validade)

1. Magnitudes plausíveis, não calibradas em bancada (E1/E2, nunca E3).
2. Uma família por janela — falhas simultâneas não são modeladas.
3. A hipótese de compensação B/C do desbalanceamento depende do controle.
4. O dataset base (Paderborn) é de um inversor de bancada em operação
   saudável; generalização para inversores on-grid de campo não é testada.
