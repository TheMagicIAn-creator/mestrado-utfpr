# Reprodutibilidade

## Manifestos v2

As publicações `comparacao` e `confiabilidade` possuem manifesto v2 em
`resultados/manifestos/`. Cada manifesto registra:

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

## Artefatos

Somente tabelas-fonte, JSON metodológico, Markdown, PNG 300 dpi e PDF vetorial
são versionados. Dados brutos, caches, pesos, scalers, logs e estado local do
Obsidian ficam fora do Git. Os manifestos podem registrar seus hashes locais
sem publicar os arquivos.

## Regeneração e validação

```powershell
python -m src.ml.comparacao_autoencoders
python -m src.ml.publicacao_confiabilidade
python scripts/auditar_resultados.py
python scripts/verificar_projeto.py
```

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
