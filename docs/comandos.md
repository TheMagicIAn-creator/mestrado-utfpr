# Comandos canônicos

Execute os comandos a partir da raiz do repositório, com `.venv` ativo.

## Aplicação

```powershell
python -m src.webapp
uvicorn src.webapp.app:app --reload
```

O endereço padrão é `http://127.0.0.1:8000`. Se a porta estiver ocupada:

```powershell
$env:PORT=8011
python -m src.webapp
```

## Pipeline científico

O pipeline tem apenas duas etapas. `comparacao` exige os 16 CSVs GPVS e Torch;
`confiabilidade` usa os valores bibliográficos rastreados no repositório.

```powershell
python -m src.ml.comparacao_autoencoders
python -m src.ml.publicacao_confiabilidade
```

Também podem ser chamadas pelo agente quando o pedido de recalcular for
explícito. Consultar resultados nunca dispara treino silenciosamente.

## Verificação

```powershell
python scripts/verificar_projeto.py
python scripts/auditar_resultados.py
python scripts/avaliar_agente.py
python -m pytest -p no:cacheprovider -q -W ignore -m "not pesado"
python -m pytest -p no:cacheprovider -q -W ignore tests/test_torch_smoke.py tests/test_modelos_autoencoder_canonicos.py
python -m ruff check --select F821,F822,F823 src tests scripts
```

`verificar_projeto.py --sem-resultados` valida apenas ambiente, árvore e GPVS.

## Manutenção da base

Há uma única entrada administrativa, com subcomandos explícitos:

```powershell
python scripts/manter_base.py reconstruir-literatura
python scripts/manter_base.py exportar-literatura
python scripts/manter_base.py reindexar-sessoes
python scripts/manter_base.py sincronizar-obsidian
python scripts/manter_base.py sincronizar-obsidian --vault C:\caminho\vault
python scripts/manter_base.py verificar-autores
```

Reconstrução e sincronização carregam embeddings e podem levar alguns minutos.
Não existe monitor de pasta em segundo plano; manutenção da base é deliberada e
observável.

## Logs

O console é o destino padrão. Para uma campanha que exija arquivo rotativo:

```powershell
$env:AL_IADO_LOG_FILE=1
python -m src.webapp
```

O arquivo local `logs/al_iado_pv.log` permanece ignorado pelo Git.
