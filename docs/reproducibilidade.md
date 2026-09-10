# Reprodutibilidade

## Manifestos v2

As três publicações canônicas — `comparacao`, `detectabilidade` e
`confiabilidade` — possuem manifesto v2 em `resultados/manifestos/`. Cada
manifesto registra:

- hash SHA-256 do código com texto normalizado para LF;
- dependências científicas por etapa;
- parâmetros e sementes;
- hashes das entradas;
- hashes de todas as saídas publicadas;
- commit Git e nível de evidência.

Manifestos v1 ainda podem ser lidos, mas são considerados `stale` quando não
contêm metadado necessário ao contrato v2. Um arquivo não é válido apenas por
existir.

## Dados e partições

`dataset_files()` exige exatamente `F0L.csv` a `F7M.csv` e rejeita arquivos CSV
extras na pasta ativa. F0L/F0M são particionados em treino, validação,
calibração e teste com purga. As janelas do AE-LSTM são construídas dentro de
cada papel, sem cruzar fronteiras.

A execução de referência usa semente 42; a estabilidade usa cinco sementes
pré-definidas. Seleção de modelo não consulta os rótulos de falha E3.

A análise temporal reutiliza exatamente os modelos e limiares congelados e usa
contexto causal contínuo, sem cruzar papel ou ensaio. A sensibilidade 3×3 usa
`k={5,10,20}` e percentis `{99;99,5;99,9}`; cada limiar continua vindo
exclusivamente da calibração saudável. Nenhuma das duas análises promove uma
configuração olhando as falhas.

O ponto operacional canônico é `k=5` com p99. A execução falha de propósito se
o percentil pedido degenerar no máximo da calibração — é o caso de p99,9 com as
210 janelas disponíveis, que exigiria `n >= 1001`. Para reproduzir o ponto
histórico p99,9, passe `--sem-limiar-estrito`.

## Artefatos

Somente tabelas-fonte, JSON metodológico, Markdown, PNG 300 dpi e PDF vetorial
são versionados. Dados brutos, caches, pesos, scalers, logs e estado local do
Obsidian ficam fora do Git. Os manifestos podem registrar seus hashes locais
sem publicar os arquivos.

**O hash não significa a mesma coisa para dado-fonte e para figura.** Em CSV,
JSON e Markdown ele prova duas coisas: que o arquivo não foi alterado depois de
gerado e que outra execução, com os mesmos dados, produz exatamente aqueles
bytes. Em PNG e PDF prova só a primeira: a rasterização depende da versão do
matplotlib, do freetype e das fontes instaladas, então o mesmo desenho, dos
mesmos dados-fonte, sai com bytes diferentes em máquinas diferentes. Para
figura, portanto, o hash é verificação de INTEGRIDADE, não de reprodutibilidade
independente — quem quiser reproduzir a figura reproduz o dado-fonte que a
origina, e esse é determinístico.

Campos metodologicamente indisponíveis são publicados como `null`, nunca zero.
Isso vale para os parâmetros de Weibull 2P, Normal e Lognormal, todos bloqueados
por falta de tempos individuais de falha e censura. O contrato informa o bloqueio
e os dados necessários antes de qualquer cálculo.

A FMECA **não** está nesse caso: o escopo vigente tem 6 itens com S, O e D
preenchidos e `NPR = S * O * D` calculado, e o contrato traz
`status: validated`. A proveniência dos escores é mista e viaja no artefato.

## Regeneração e validação

```powershell
python -m src.ml.comparacao_autoencoders
python -m src.ml.campanha_detectabilidade
python -m src.ml.publicacao_confiabilidade
python scripts/auditar_resultados.py
python scripts/verificar_projeto.py
```

A ordem é a de dependência: a campanha E2 consome os pesos, o scaler e o limiar
congelados pela `comparacao`, e não treina nem recalibra. Rodá-la contra um clone
desatualizado publica menos saídas do que a versão vigente da etapa — confira o
`output_count` antes de commitar.

Depois, execute:

```powershell
python -m pytest -p no:cacheprovider -q -W ignore -m "not pesado"
python -m pytest -p no:cacheprovider -q -W ignore tests/test_torch_smoke.py tests/test_modelos_autoencoder_canonicos.py
python -m ruff check --select F821,F822,F823 src tests scripts
```

O CI repete a suíte não pesada, valida a seleção por marcadores e executa um job
separado com Torch real em dados pequenos. O treino completo GPVS permanece
local porque os dados brutos não são publicados.

## Memória

Produção (`sessoes_pv`), avaliação (`avaliacoes_agente`) e Obsidian
(`obsidian_pv`) são coleções separadas. A avaliação offline não escreve memória.
Snapshots portáteis preservam o corpus, o modelo de embeddings e o hash da
fonte, sem transformar sessões em referências bibliográficas.
