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
python -m pytest                        # bateria de testes unitários (rápida, com fixtures)
python -m pytest -W ignore -q           # idem, sem warnings de limpeza de tmp
```

## Pipeline de ML (recalcular — exige `dados/brutos/` local)
```powershell
python src/ml/features_ca.py        # extrai features CA (Paderborn) + manifesto
python src/ml/autoencoder.py        # treina o AE; grava limiar.json (p99) + manifesto
python src/ml/injecao_falhas.py     # injeta falhas FMEA (E2) + schema no report
python src/ml/validacao.py          # validação formal: ROC + PR + matriz, limiar congelado
python src/ml/rul_weibull.py        # RUL / Weibull
```

## Experimentos por artigo
```powershell
python src/ml/experimentos_artigos.py   # ou pelo chat ("rode o experimento do Ghoneim")
python src/ml/classificador_pv.py       # benchmark supervisionado PV Farms (CC)
python -c "from src.ml.classificador_pv_infer import treinar_e_salvar; treinar_e_salvar()"  # salva modelo/scaler/manifests/PNGs
```

## Bateria determinística do agente (RAG/roteamento)
```powershell
python scripts/avaliar_agente_100.py                 # 559 casos; por padrão não grava memória
python scripts/avaliar_agente_100.py --com-memoria   # grava a avaliação na coleção separada de testes
```

## Observações
- No Windows, `streamlit run app.py` (não `python app.py`, que roda em "bare mode" e sai sem servir).
- `KMP_DUPLICATE_LIB_OK=TRUE` é definido cedo (config/app/main) para evitar crash de OpenMP duplicado.
- Etapas aparecem como **stale/pending** até serem recalculadas com o código atual (cria o manifesto).
