# Assinaturas FMECA — justificativa físico-elétrica

Este documento traduz as assinaturas de falha implementadas no código
(`src/ml/injecao_falhas.py` — domínio do SINAL; `src/ml/protocolos_artigos.py`
`ASSINATURAS_FMEA` — domínio das FEATURES) em prosa técnica citável na
dissertação.

> **Fonte única dos componentes/modos/índices: `docs/fmeca.md`.** Este arquivo
> é a justificativa física; os números (S, O, D, NPR) vêm de lá.

Os três componentes são os **componentes internos CA-elétricos do inversor que
mais falham** segundo a **Tabela 3.3 do TCC** (Cristaldi, Khalil & Soulatintork,
2017), reforçada por Golnas (2012) e Voss et al. (2009). NPR = S×O×D (índice da
**FMECA**, não FMEA; D nunca é o NPR). Índices S/O/D estipulados pelo
pesquisador (Torres, 2024).

Nível de evidência: **E2 no pipeline principal** (injeção no sinal bruto) e
**E1 nos experimentos por artigo** (injeção no espaço de features). Em ambos, as
MAGNITUDES são plausíveis, não medidas em bancada — ver limitações ao final.

## 1. Contator AC — NPR=315 (S=5, O=7, D=9) — componente mais crítico

**Física da falha.** O contator CA conecta/desconecta a saída do inversor à
rede. Com o envelhecimento, os contatos sofrem erosão por arco elétrico,
soldagem parcial e aumento da resistência de contato, produzindo comutação
deficiente e *chattering* (abre/fecha intermitente). O efeito elétrico
dominante é a injeção de transientes e conteúdo de alta frequência na corrente
CA no instante da comutação deficiente.

**Assinatura implementada.** Modelado como ruído no canal da fase A (proxy do
transiente de comutação): no sinal bruto, `i_a += N(0, sev·σ·0,3)`; nas
features, sobem desvio-padrão, largura de banda espectral, energia na banda de
chaveamento, centroide espectral e THD da fase A.

**Detecção.** O índice **D=9 da FMECA** refere-se à dificuldade de detecção EM
CAMPO/manutenção — distinta da detectabilidade empírica do Autoencoder (ver a
ressalva metodológica em `docs/fmeca.md`).

## 2. IGBT — NPR=90 (S=5, O=6, D=3)

**Física da falha.** O IGBT é o dispositivo de chaveamento da conversão CC→CA.
Com o envelhecimento (lift-off de bond wire, aumento de Vce(sat), fadiga de
solda por ciclagem térmica), a comutação torna-se imperfeita, elevando os
harmônicos ímpares de baixa ordem (5º, 7º, 11º — característicos de conversores
trifásicos de seis pulsos, ordens 6k±1) e o THD das correntes de linha.

**Assinatura implementada.** Elevação dos harmônicos 5, 7, 11 e 13 das três
correntes (`i_[abc]_harm_{5,7,11}`), do THD (`i_[abc]_thd`) e da energia média
espectral. No sinal bruto soma-se conteúdo harmônico proporcional à severidade;
nas features, desloca-se cada coluna em unidades do desvio-padrão do treino.

**Por que é detectável por modelagem de normalidade.** O Autoencoder aprende a
correlação saudável entre fundamental e harmônicos; harmônicos elevados com
fundamental normal violam essa correlação e inflam o erro de reconstrução.

## 3. Fusível AC — NPR=30 (S=5, O=3, D=2)

**Física da falha.** O fusível CA protege o lado CA contra sobrecorrente. Um
fusível degradado ou rompido causa perda parcial de uma fase, reduzindo a
amplitude da corrente dessa fase. HIPÓTESE DE MODELAGEM (explícita no código):
o controle do inversor redistribui corrente para as fases sãs para manter a
potência entregue. A assinatura CENTRAL e independente do controle é a métrica
de desbalanceamento subir; a compensação B/C depende da estratégia de controle.

**Assinatura implementada.** Fase A enfraquece (RMS, pico a pico, desvio,
energia da fundamental, potência — modo multiplicativo, redução de 15–35% ×
severidade); fases B/C compensam parcialmente (modo aditivo); a feature
`desbalanceamento_corrente` sobe com o maior peso da família.

**Limitação honesta.** A hipótese NÃO cobre perda de linha em que as três fases
caem juntas, nem carga severamente desbalanceada sem compensação. Em execuções
anteriores (com o rótulo antigo de "desbalanceamento"), esta família apresentou
o menor erro de reconstrução das três e chegou a não cruzar o limiar (SMD nula)
— achado de limitação do detector a discutir, não a omitir.

## Pesos de amostragem entre famílias

`PESOS_FALHAS = {contator_ac: 0.40, igbt: 0.35, fusivel_ac: 0.25}` — segue a
ordem de criticidade do NPR da FMECA (Contator AC 315 > IGBT 90 > Fusível AC
30), cumprindo o critério de priorização por NPR.

## Limitações gerais (para a seção de ameaças à validade)

1. Magnitudes plausíveis, não calibradas em bancada (E1/E2, nunca E3).
2. Uma família por janela — falhas simultâneas não são modeladas.
3. O ruído gaussiano do Contator AC é um PROXY do transiente de comutação.
4. A hipótese de compensação B/C do Fusível AC depende do controle.
5. O índice D (detecção em campo) da FMECA não equivale à detectabilidade
   empírica do Autoencoder — relação a discutir na dissertação.
6. O dataset base (Paderborn) é de um inversor de bancada saudável;
   generalização para inversores on-grid de campo não é testada.
