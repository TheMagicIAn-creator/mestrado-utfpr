# Reconstrucao V2 academica: contrato cientifico e arquitetura

**Data da auditoria:** 13/08/2026  
**Dataset principal:** GPVS-Faults, DOI `10.17632/n76t439f65.1`  
**Escopo:** autoencoder denso, confiabilidade, FMECA, figuras e aplicacao web.

## Decisao executiva

A V2 sera reconstruida em paralelo e somente substituira a aplicacao atual depois
de passar pelos testes cientificos e visuais. O historico Git e os artefatos atuais
permanecem como referencia de comparacao; eles nao serao apagados para simular um
recomeco.

O resultado final separa quatro perguntas que hoje aparecem misturadas:

1. **Deteccao:** o autoencoder reconhece desvio em relacao aos ensaios saudaveis?
2. **Validacao FMECA:** assinaturas eletricas sinteticas sao detectaveis e alteram a
   dificuldade de monitoramento dos modos priorizados pelo pesquisador?
3. **Confiabilidade:** qual e a confiabilidade calculada sob cenarios bibliograficos
   de taxa de falha e quais hipoteses sustentam cada curva?
4. **Decisao de manutencao:** como evidencia de deteccao, criticidade e referencia
   de confiabilidade devem ser lidas em conjunto, sem converter uma grandeza na
   outra?

## Auditoria do dataset e do modelo atual

O GPVS-Faults e a unica base experimental usada na reconstrucao principal. A fonte
oficial publica 16 ensaios: `F0L/F0M` saudaveis e `F1L-F7M` com falhas introduzidas
na metade da captura, nos modos IPPT e MPPT.

O artefato processado atual possui:

| Item | Valor auditado |
|---|---:|
| Janelas saudaveis F0 | 1.423 |
| F0L / F0M | 718 / 705 |
| Features por janela | 24 |
| Treino / validacao / calibracao / teste | 711 / 209 / 210 / 281 |
| Duracao nominal da janela | um ciclo de 50 Hz, aproximadamente 20 ms |
| Sobreposicao | 0% |

As 24 features nao possuem variancia nula, mas apresentam colinearidade forte. As
correlacoes absolutas chegam a `0,995` e o numero de condicao da matriz de
covariancia saudavel e aproximadamente `6,7 x 10^4`. Isso nao invalida o modelo,
mas impede tratar cada coluna correlacionada como evidencia independente e motiva
avaliar um escore balanceado por familias fisicas.

O detector publicado em `main` e a referencia, nao o vencedor antecipado da V2:

| Metrica por ensaio | Media macro | IC95% bootstrap, 14 ensaios |
|---|---:|---:|
| AUC | 0,773 | [0,691; 0,853] |
| Sensibilidade no limiar | 0,406 | [0,211; 0,615] |
| Especificidade pre-falha | 0,974 | [0,946; 0,992] |
| Acuracia balanceada | 0,690 | [0,596; 0,790] |

Esses intervalos usam o **ensaio** como unidade de reamostragem. Janelas vizinhas
nao serao apresentadas como replicas independentes.

## Protocolo do autoencoder denso V2

### Dados e papeis

- `F0L/F0M`: normalizacao, treino, early stopping, calibracao e teste saudavel.
- `F1L-F7M`: avaliacao final com pesos e limiar congelados.
- Nenhuma falha real ou sintetica participa da selecao da arquitetura ou do limiar.
- A normalizacao de comissionamento usa somente o inicio pre-falha de cada ensaio.
- O split permanece temporal por ensaio e inclui purga entre papeis.

### Selecao do modelo

A selecao comparara arquiteturas densas compactas e um baseline PCA. Cada
arquitetura sera repetida com os mesmos seeds predeclarados. O criterio primario e
a perda de reconstrucao no bloco saudavel de validacao; quando a diferenca relativa
for menor que 2%, vence o modelo com menos parametros. O seed canonico sera o que
ficar mais proximo da mediana de validacao da arquitetura escolhida, nunca o que
obtiver melhor resultado nos ensaios de falha.

O relatorio de selecao deve registrar, por execucao:

- arquitetura, ativacao, dropout, weight decay, otimizador e seed;
- numero de parametros e razao parametros/janela de treino;
- epoca selecionada e curvas de treino/validacao;
- perdas em treino e validacao;
- distribuicao do escore em calibracao e teste saudavel;
- hash do parquet, codigo, configuracao e artefatos.

### Limiar e escores

O limiar sera calculado exclusivamente na calibracao saudavel. A V2 publicara a
ordem estatistica finita usada no quantil, sua resolucao empirica e a excedencia no
teste saudavel com IC95% de Wilson. `p99` nominal nao sera descrito como garantia de
1% de falsos positivos.

Serao comparados, sem usar falhas para escolher:

- MSE global, referencia historica;
- erro de reconstrucao balanceado por familias fisicas;
- PCA de reconstrucao como baseline linear.

As familias fisicas preservam as 24 features e reduzem o peso acidental de blocos
altamente correlacionados: grandezas CC, corrente CA, tensao CA e potencia.

### Avaliacao final

Depois do congelamento, a avaliacao `F1L-F7M` reportara AUC, sensibilidade,
especificidade, acuracia balanceada e atraso sustentado por ensaio. O resumo macro
tera bootstrap por ensaio. Curvas ROC/PR agregadas por janela nao serao usadas como
figura principal, pois esconderiam a heterogeneidade entre os 14 experimentos.

## Contrato da FMECA

Os valores oficiais `315/90/30` permanecem inalterados porque foram estipulados
pelo pesquisador para a dissertacao:

| Componente | S | O | D_campo | NPR |
|---|---:|---:|---:|---:|
| Contator AC | 5 | 7 | 9 | 315 |
| IGBT | 5 | 6 | 3 | 90 |
| Fusivel AC | 5 | 3 | 2 | 30 |

Esses numeros **nao sao uma reproducao da tabela de Cristaldi et al. (2017)**. No
artigo anexado, a analise de criticidade do inversor apresenta, entre outros,
`RPN=150` para contatores AC/DC e `RPN=63` para IGBT. A selecao da dissertacao
recorta componentes CA eletricamente observaveis e aplica os indices julgados pelo
pesquisador segundo as escalas do TCC.

A injecao FMECA continua sendo uma validacao sintetica de detectabilidade. Sua
magnitude adimensional nao e tempo, desgaste, severidade de campo nem vida util.
O ajuste Weibull de primeiro cruzamento, quando mantido no apendice, sera chamado
de modelo de **magnitude de deteccao** e nunca de confiabilidade do componente.

## Contrato da confiabilidade fisica

### Achado de unidade no TCC

A Tabela 3.4 do TCC informa taxa de falha em falhas por hora. Na pagina seguinte
de resultados, `1 / (1,8 x 10^-4)` e descrito como `5.555,55 anos`. O valor
matematico e `5.555,55 horas`, aproximadamente `0,634 ano`. De modo analogo, se
`mu=0,0833` estiver em reparos por hora, `1/mu` vale 12 horas, nao 12 anos.

A V2 nao corrige a fonte silenciosamente. Ela registra:

- valor original;
- unidade declarada;
- conversao dimensional;
- valor derivado;
- ressalva de aplicabilidade.

### Cenarios bibliograficos comparaveis

| Cenario | Taxa ou MTBF da fonte | Leitura permitida |
|---|---:|---|
| Torres (2024), Tab. 3.4, adaptada de Colli | `1,75 x 10^-4 falha/h` | Sensibilidade bibliografica; ha inconsistencia de unidade no calculo posterior do TCC |
| Cristaldi et al. (2017) | `0,125 falha/ano` para o inversor | Modelo exponencial de BoS; fonte declara MTTF do BoS proximo de 6 anos |
| Obeidat e Shuttleworth (2015), alta qualidade | `8,069 falhas/10^6 h`; MTBF `123.938 h` | Predicao MIL-HDBK-217F N2 dependente de componentes e temperatura |
| Obeidat e Shuttleworth (2015), baixa qualidade | `50,76 falhas/10^6 h`; MTBF `19.700 h` | Cenario de qualidade, nao observacao de uma frota GPVS |
| Dhople e Dominguez-Garcia (2012) | MTTF ilustrativo de 10 anos | Parametro de estudo de caso Markov, nao estimativa do dataset atual |

Sob taxa constante, a V2 pode calcular legitimamente:

- confiabilidade: `R(t) = exp(-lambda t)`;
- probabilidade acumulada de falha: `F(t) = 1 - R(t)`;
- densidade: `f(t) = lambda exp(-lambda t)`;
- taxa de falha: `h(t) = lambda`;
- MTTF/MTBF de referencia: `1/lambda`, com a distincao reparavel explicitada.

O GPVS-Faults nao fornece tempos de vida, censura por unidade, reparos ou uma
frota de ativos. Portanto ele nao estimara `lambda`, beta/eta Weibull, RUL ou
confiabilidade fisica. Weibull fisico permanece bloqueado ate existir esse contrato
de dados.

## Figuras principais da V2

A aplicacao e o pacote de exportacao academica terao um conjunto pequeno de
figuras, cada uma respondendo a uma pergunta:

1. **Selecao do autoencoder denso em dados saudaveis GPVS-Faults.**
2. **Calibracao do limiar e excedencia no teste saudavel F0.**
3. **Desempenho por ensaio de falha com detector congelado.**
4. **Resposta temporal em ensaios representativos, com inicio da falha marcado.**
5. **Confiabilidade do inversor sob cenarios bibliograficos de taxa constante.**
6. **Probabilidade acumulada, densidade e taxa de falha sob o modelo exponencial.**
7. **Criticidade FMECA julgada e detectabilidade medida, em eixos separados.**

ROC/PR por severidade, matrizes repetidas, injecoes sinteticas e o papel Weibull
de magnitude ficam em um apendice auditavel. Nenhuma figura principal usara
"confiabilidade", "taxa de falha" ou "RUL" para o eixo de magnitude sintetica.

## Aplicacao web V2

A interface principal sera uma aplicacao ASGI com Starlette, HTML semantico,
CSS responsivo e JavaScript modular. Plotly sera usado para graficos interativos e
exportacao. O Streamlit deixara de ser a entrada principal; a versao antiga ficara
isolada apenas durante a transicao e sera removida depois da equivalencia funcional.

Vistas previstas:

- **Visao geral:** veredito atual, dataset e limites de evidencia;
- **Autoencoder:** arquitetura, calibracao, desempenho por ensaio e proveniencia;
- **Confiabilidade:** cenarios bibliograficos, formulas, unidades e sensibilidade;
- **FMECA:** tabela oficial do pesquisador e resultado de monitoramento separado;
- **Evidencias:** fontes, hashes, manifestos e downloads;
- **Agente:** conversa com o ALIAdo por um adaptador sem dependencia de Streamlit.

O frontend nunca recalcula numeros cientificos. Ele consome contratos JSON
validados pelo backend, e cada painel exibe dataset, unidade, nivel de evidencia,
data e hash da rodada.

## Criterios de aceite

- um unico dataset experimental principal: GPVS-Faults;
- selecao do AE sem olhar as falhas e com repeticao por seeds;
- baseline PCA e referencia do modelo anterior;
- limiar calibrado fora de treino/validacao e teste saudavel intocado;
- inferencia macro por ensaio, nao por janela;
- confiabilidade fisica apenas sob cenarios bibliograficos identificados;
- nenhuma conversao de `a_det` para tempo;
- figuras exportaveis em PNG e PDF, com titulos, unidades, fonte e ressalva;
- aplicacao responsiva verificada em desktop e celular;
- testes unitarios, integracao ASGI, lint e verificacao visual automatizada;
- manifestos com hashes de entradas, codigo, configuracao e saidas.

## Fontes primarias auditadas

- Bakdi et al. (2020), GPVS-Faults, DOI `10.17632/n76t439f65.1`.
- Torres (2024), TCC anexado, Tabelas 3.2-3.4 e Equacoes 4.8-4.17.
- Cristaldi, Khalil e Soulatiantork (2017), DOI `10.21014/acta_imeko.v6i4.425`.
- Obeidat e Shuttleworth (2015), DOI `10.1109/PVSC.2015.7356277`.
- Dhople e Dominguez-Garcia (2012), DOI `10.1109/TPWRS.2011.2165088`.
- Lafraia (2001), manual anexado, analise de confiabilidade e Weibull.
- Nketiah et al. (2021), DOI `10.22161/ijaers.89.21`.

