# Auditoria acadêmica do Autoencoder e dos resultados

**Data:** 09/08/2026
**Escopo:** código, calibração, figuras, proveniência e comunicação dos
resultados E2/E3 disponíveis após o merge do PR #115.

## Conclusão executiva

O limiar operacional e as métricas científicas publicadas foram preservados.
O MSE p99 continua em `2,582821`; o teste saudável separado continua com uma
excedência em 60 janelas (`1,67%`). A auditoria não encontrou base para trocar
o ponto de operação sem um novo experimento.

Foram encontrados problemas de comunicação e rastreabilidade:

1. a ECDF mostrava somente a faixa 0,80-1,00 e o histograma sobreposto era
   instável para amostras pequenas;
2. o título destacava `1,67%` sem deixar igualmente visíveis o denominador, a
   incerteza e a sobreposição entre janelas;
3. o gráfico temporal afirmava mostrar as faixas do split, mas o código só
   entendia um intervalo por conjunto e ignorava os 14 blocos atuais;
4. manifestos downstream hasheavam PNGs/Markdown que não são entradas dos
   cálculos, enquanto o CSV bruto não aparecia como entrada de `features_ca`;
5. a memória consultável pelo ALIAdo estava anterior à rodada canônica e não
   incorporava a validação GPVS-Faults E3.

## Auditoria quantitativa da calibração

| Bloco | Janelas | Excedências acima do MSE p99 | Taxa por janela | Sem compartilhamento | Taxa nessa subamostra |
|---|---:|---:|---:|---:|---:|
| Calibração | 42 | 1 | 2,38% | 21 | 0,00% |
| Teste isolado | 60 | 1 | 1,67% | 32 | 3,12% |

O p99 foi calculado pelo `numpy.percentile` com interpolação linear. Em 42
observações, o corte `2,582821` fica entre os dois maiores erros de calibração
(`2,468290` e `2,662410`). Por isso:

- a probabilidade nominal de cauda é 1%;
- a menor frequência empírica não nula é `1/42 = 2,38%`;
- a ECDF empírica no corte é `41/42 = 97,62%`;
- a excedência da calibração não é estimativa externa de desempenho, pois o
  próprio bloco definiu o quantil;
- o teste separado é a estimativa relevante, mas seu IC95% de Wilson
  `[0,29%; 8,86%]` é descritivo por janela e não elimina dependência serial.

A subamostra sem compartilhamento retém uma janela a cada duas dentro de cada
bloco, coerente com o janelamento de 50%. Ela remove amostras brutas comuns,
mas não torna automaticamente as janelas independentes.

## Revisão das figuras

### Distribuição do erro

A figura passa a usar dois painéis:

- ECDF completa, de 0 a 1, sem truncamento do corpo da distribuição;
- probabilidade empírica de excedência em escala logarítmica, adequada para a
  cauda e para comparar o alvo nominal de 1% com as frequências observadas.

Cor e estilo de linha variam simultaneamente, o título é neutro e a anotação
mostra denominadores, IC95% e a análise sem compartilhamento. O histograma de
densidade foi removido porque 42 e 60 observações não sustentam uma leitura
estável de três densidades sobrepostas.

### Erro temporal

O sombreamento agora aceita a lista de intervalos do split intercalado e desenha
os 14 blocos. Antes, nenhuma faixa aparecia, embora a nota da figura dissesse o
contrário. O teste automatizado exige 14 faixas e apenas um rótulo de legenda
por conjunto.

## Proveniência científica

`depends_on` continua controlando a ordem de execução. O novo contrato
`input_artifacts` registra somente arquivos efetivamente lidos:

| Etapa | Entradas científicas principais |
|---|---|
| Features CA | CSV bruto Stender/Paderborn |
| Autoencoder | parquet de features |
| Injeção, validação e Weibull | CSV bruto, modelo, scaler, hash do scaler, estatística de resíduos e `limiar.json` |

PNG, CSV de apresentação e Markdown continuam sendo saídas publicáveis, mas
deixam de invalidar cálculos que não os consomem. `estatistica_residuo.npz` e
`scaler.pkl.sha256`, indispensáveis ao escore reproduzível, passam a integrar
formalmente as saídas da etapa Autoencoder.

## Leitura integrada E2 e E3

### Stender + falhas sintéticas orientadas pela FMECA

O Autoencoder de normalidade, a injeção, a validação e a Weibull permanecem
**E2**. O dado saudável é experimental, mas os modos de falha e as trajetórias
de degradação são sintéticos. SMD, POD, retroalimentação FMECA e Weibull devem
ser descritos como avaliação interna orientada por hipótese física, não como
desempenho industrial ou vida útil em horas.

### GPVS-Faults

O protocolo GPVS é **E3 de bancada externa**, com 14 ensaios de falha:

| Protocolo | AUC macro | Sensibilidade pós-falha | Especificidade | Acurácia balanceada |
|---|---:|---:|---:|---:|
| Transferência direta AE | 0,732 | 1,000 | 0,007 | 0,503 |
| AE adaptativo | 0,815 | 0,445 | 0,974 | 0,709 |
| PCA adaptativo | 0,794 | 0,431 | 0,972 | 0,701 |

A transferência direta do limiar F0 é rejeitada pela especificidade quase
nula. A adaptação local usa apenas o início saudável de cada ensaio e apresenta
resultado útil, mas sensibilidade moderada. Os intervalos são bootstrap de
ensaios, não de janelas. E3 significa bancada externa: não é campo, não prova a
causa do desvio e não fornece tempos de vida para Weibull/RUL físico.

## Política de artefatos

- **Versionar:** JSON/CSV/Markdown de resultados, figuras finais, diagnósticos
  compactos e manifestos necessários para auditar as afirmações.
- **Manter local e ignorado:** dados brutos, pesos `.pt`, scaler `.pkl` e bases
  locais de embeddings.
- **Não usar como entrada científica:** figuras e textos derivados. Eles podem
  ser regenerados sem alterar modelo, limiar ou métricas.
- **Não fabricar:** se dados, modelo ou dependências faltarem, registrar o
  bloqueio; nunca reconstruir números manualmente.

## Verificação reproduzível

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q -m "not pesado"
.venv\Scripts\python.exe -m ruff check --select F821,F822,F823 src tests scripts
.venv\Scripts\python.exe -c "from src.ml.graficos_autoencoder import regenerar_graficos_autoencoder as r; assert r()"
```

Os valores numéricos do modelo só devem mudar em uma nova rodada explícita de
treinamento/avaliação. A regeneração barata acima altera apenas figuras e
tabelas derivadas do diagnóstico persistido.
