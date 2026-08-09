# Ingestao de literatura - confiabilidade, POD e RUL

Data da triagem: 2026-08-09

## Escopo

Esta ingestao foi limitada a fontes pertinentes a probabilidade de deteccao
(POD), confiabilidade, Weibull, censura e vida util remanescente (RUL). ZIPs de
datasets e pareceres de modelos nao foram tratados como literatura
bibliografica.

## Fontes aceitas

| Fonte recebida | Arquivo canonico | Paginas | SHA-256 indexado | Justificativa |
|---|---|---:|---|---|
| `2016-Practical_POD.pdf` | `literatura/confiabilidade/virkkunen_practical-experiences-in-pod-determination-for-airframe-et-inspection_2016.pdf` | 9 | `b7119c332bbe36e04401bc8b7ca6faafe4151f3c3bbb7a21b371930416a23c33` | Discute requisitos amostrais, hipoteses e falhas praticas de POD hit/miss e a-hat versus a. |
| `rezb84_890417.pdf` | `literatura/ml-preditivo/hu_remaining-useful-life-prediction-based-on-a-joint-model-with-degradation-failure-association-structures_2023.pdf` | 20 | `7f0f995923429b7ab69d017ccc88f13d7978da41b7ab3a2603362806195b2959` | Relaciona degradacao monitorada, risco de falha e RUL, com comparacao entre modelo conjunto e abordagem em duas etapas. |
| `30110100469575_3.pdf` | `literatura/confiabilidade/lien_unlocking-weibull-analysis_2013.pdf` | 9 | `006f3edcff170336618057370ffb915c575edeab78c3d5842b2c701710e7d211` | Apresenta censura, segregacao de modos de falha, diagnostico de ajuste, taxa de risco e B10. Foi indexada uma copia com camada OCR validada. |
| `monopoli10007438.pdf` | `literatura/confiabilidade/meyberg_aplicacao-de-metodos-probabilisticos-para-avaliacao-da-confiabilidade-de-funcoes-transmissao_2013.pdf` | 142 | `372e28882e14db2548b887e3aeb6ab2764e83cc95c17b6a63c7048049d513f51` | Fonte secundaria em portugues com secoes sobre Weibull, suspensoes, MTTF, confiabilidade, taxa de falha e B-life. |
| `21IJAERS-09202130-Parameter..pdf` | `literatura/confiabilidade/nketiah_parameter-estimation-of-the-weibull-distribution-comparison-of-ls-and-mle_2021.pdf` | 15 | `73be38c42a6045e4b972da72c2f05e02376050a02f9d5d3a9a4e47d0743b3706` | Compara estimacao de Weibull por maxima verossimilhanca e minimos quadrados em simulacao e dois conjuntos reais. |

## Fonte nao indexada

`t08049.pdf`, de Rausch (2008), foi lida para triagem, mas nao foi incorporada.
Seu foco principal e a otimizacao conjunta de manutencao baseada em condicao e
estoque de sobressalentes por processo gama. Alem da aderencia apenas indireta ao
pipeline atual, a camada textual do PDF insere o caractere `d` entre grande parte
das palavras, o que degradaria a busca semantica e lexical.

## Controle do OCR

O PDF de Lien e Nicholls era composto somente por imagens: a extracao original
retornou zero paginas com texto. A copia indexada preserva integralmente as nove
paginas e adiciona uma camada textual invisivel apenas nas regioes do artigo;
anuncios foram excluidos do OCR. O controle de renderizacao a 75 dpi apresentou
dimensoes iguais e diferenca de zero pixels em todas as paginas. A extracao final
produziu aproximadamente 19 mil caracteres e recuperou termos centrais como
`suspension`, `censored`, `hazard rate`, `goodness-of-fit`, `failure mode` e
`B10`.

O SHA-256 do scan recebido, preservado fora do repositorio, e
`758865c8f545685252e0de8d0c8b3e34c61fc725603896648522c124482649da`.

## Validacao do indice

A colecao foi reconstruida integralmente, sem reaproveitar o banco vetorial de
outro worktree. O resultado possui 44 documentos e 12.556 chunks, dos quais 302
vieram das cinco fontes desta ingestao. O snapshot portatil foi exportado com o
hash de corpus
`0ef91e96379c546c7bee42434a935e86d3711a3a31500476245274518c6612b0`.

A recuperacao hibrida foi exercitada com consultas sobre POD, estimacao MLE de
Weibull, modelos conjuntos de degradacao-falha/RUL e parametros de
confiabilidade. Virkkunen, Nketiah e Hu apareceram em primeiro lugar nas
respectivas consultas; Meyberg apareceu em terceiro na consulta geral de
Weibull. O artigo OCR de Lien foi o primeiro resultado lexical para
`suspensions censored Weibull` e apareceu entre os oito primeiros na consulta
geral sobre `beta`, `eta`, taxa de risco, MTTF e B10. Em uma formulacao ampla
apenas em portugues sobre censura, outras fontes em portugues ficaram acima de
Lien; portanto, a fonte esta recuperavel, mas nao se presume invariancia de
ranking entre idiomas.

## Materiais fora do indice bibliografico

- `13974425.zip`: PMSM Inverter Fault Dataset v1.0.0, candidato a avaliacao de
  dados experimentais de acionamento eletrico.
- `archive.zip`: telemetria temporal de inversor fotovoltaico, candidata a
  avaliacao de qualidade e rotulagem.
- pareceres anexados: material de auditoria, nao fonte bibliografica citavel.

Nenhum dataset, parecer, resultado de modelo ou texto gerado por LLM foi
inserido na colecao de literatura.
