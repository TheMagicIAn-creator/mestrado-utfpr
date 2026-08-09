# Comandos — Al IAdo PV

## Aplicação
```powershell
streamlit run app.py        # interface (use streamlit run, NÃO python app.py)
python main.py              # chat no terminal
```

## Verificação / diagnóstico
```powershell
python scripts/verificar_ambiente.py    # imports, versões, chaves, datasets, ChromaDB, pipeline
python scripts/verificar_datasets.py    # presença + SHA-256 + linhas dos datasets
python scripts/validar_gpvs.py          # validação E3 de bancada (16 CSVs GPVS; treino pesado)
python scripts/validar_gpvs.py --somente-graficos  # recompõe figuras/manifesto sem retreino
python scripts/verificar_resultados_fmeca.py  # cruza JSON, CSV, PNG e critérios metodológicos
python -m pytest                        # bateria de testes unitários (rápida, com fixtures)
python -m pytest -W ignore -q           # idem, sem warnings de limpeza de tmp
```

## Pipeline de ML (recalcular — exige `dados/brutos/` local)

> ⚠️ **Rodar os módulos direto NÃO grava manifesto de proveniência.** Nenhum
> bloco `__main__` das cinco etapas chama `registrar_manifesto`; a única chamada
> real está dentro de `pipeline.executar_etapa`. Consequência: os artefatos são
> regenerados, mas o manifesto continua apontando para os **anteriores**, e a
> etapa aparece **stale logo depois de ser recalculada**. Foi esse ciclo que
> levou, em 05/08, a manifestos reescritos à mão — as cinco etapas com
> `created_at` dentro de 0,55 s, para uma execução que leva 8 minutos.
> Ver `docs/auditoria_total_src.md` §2.

**Caminho recomendado — pelo pipeline, que registra proveniência:**

```powershell
python -m src.ml.exec_etapa_isolada features_ca
python -m src.ml.exec_etapa_isolada autoencoder
python -m src.ml.exec_etapa_isolada injecao_falhas
python -m src.ml.exec_etapa_isolada validacao
python -m src.ml.exec_etapa_isolada rul_weibull
```

**Execução direta dos módulos** — útil para depurar uma etapa, mas deixa o
manifesto defasado; depois de usar, recalcule pelo caminho acima:

```powershell
python src/ml/features_ca.py        # extrai features CA (Paderborn)
python src/ml/autoencoder.py        # treina o AE; grava limiar.json (score operacional + referências)
python src/ml/injecao_falhas.py     # injeta falhas FMECA (E2) + schema no report
python src/ml/validacao.py          # validação interna E2: ROC + PR + matrizes, limiar congelado
python src/ml/rul_weibull.py        # RUL / Weibull
```

**Forçar recálculo total** (ignora o estado `ready` de todas as etapas): pelo
chat, peça *"recalcule tudo do zero"*. Frases como *"rode o pipeline de novo"* e
*"retreine o autoencoder"* também forçam desde 06/08 — antes disso a etapa era
**pulada em silêncio**, e a resposta imprimia a tabela de resultados logo abaixo
de "já está pronto". Hoje, quando não recalcula, a resposta diz isso e carimba a
data do artefato.

## Experimentos por artigo
Cada experimento tem o SEU protocolo de decisão (src/ml/protocolos_artigos.py);
o runner abaixo é só o atalho de execução — não muda a metodologia nem os números.
```powershell
python scripts/rodar_experimentos.py              # lista o experimento Ibrahim
python scripts/rodar_experimentos.py ibrahim      # roda AE-LSTM temporal (Ibrahim)
python scripts/rodar_experimentos.py --todos      # hoje equivale a ibrahim
python src/ml/classificador_pv.py       # benchmark supervisionado PV Farms (CC)
python -c "from src.ml.classificador_pv_infer import treinar_e_salvar; treinar_e_salvar()"  # salva modelo/scaler/manifests/PNGs
```
Também pelo chat: "compare meu método com o AE-LSTM" ou "como estou frente ao Ibrahim?".

## Literatura local e snapshot da nuvem
```powershell
python scripts/reconstruir_literatura.py       # reconstrói o ChromaDB local a partir dos PDFs
python scripts/exportar_indice_literatura.py   # gera artefatos/literatura_indexada.jsonl.gz
```

O diretório `base_conhecimento/` é local, incremental e ignorado pelo Git. O
snapshot gzip é portátil e versionável: quando a coleção está vazia, o
Streamlit o restaura automaticamente. O botão de indexação da interface é
apenas um fallback para deploys sem snapshot válido.

## Bateria determinística do agente (RAG/roteamento)
```powershell
python scripts/avaliar_agente_100.py                 # 559 casos; por padrão não grava memória
python scripts/avaliar_agente_100.py --com-memoria   # grava a avaliação na coleção separada de testes
```

## Observações
- No Windows, `streamlit run app.py` (não `python app.py`, que roda em "bare mode" e sai sem servir).
- `KMP_DUPLICATE_LIB_OK=TRUE` é definido cedo (config/app/main) para evitar crash de OpenMP duplicado.
- Etapas aparecem como **stale/pending** até serem recalculadas com o código atual (cria o manifesto).
- O pipeline pesado roda no PC porque `dados/brutos/` não é publicado. Na
  nuvem, o app consulta os resultados versionados e não afirma ter retreinado modelos.
- O progresso do ML vai para `logs/al_iado_pv.log` (terminal silencioso no app);
  scripts rodados à mão reativam o eco automaticamente. Leia o log em UTF-8:
  `Get-Content logs\al_iado_pv.log -Tail 20 -Encoding utf8` (sem `-Encoding utf8`
  o PowerShell distorce acentos — o arquivo está correto). Emojis ficam só na
  interface do chat; o log é texto limpo.
