# Al IAdo PV — Mestrado UTFPR

Predictive failure analysis of AC-side components in grid-connected
photovoltaic inverters using Machine Learning, with a
Reliability-Centered Maintenance (RCM) methodology.

Master's research project — Electrical Engineering, UTFPR.
Defense scheduled for March 2027.

## Overview

This repository hosts **Al IAdo PV**, an AI research assistant built
to support the master's dissertation. The system combines a
knowledge agent (RAG over the scientific literature) with a
Machine Learning pipeline for fault detection in PV inverters.

The methodology models the *healthy* behaviour of an inverter and
detects developing faults as deviations from that normality —
the standard approach in industrial predictive maintenance, where
real fault data is scarce.

## Architecture

The project is a modular Python package. The single entry point is
`app.py`, which launches a Streamlit interface and runs a backend
orchestrator on startup.

```
src/
├── core/            shared infrastructure (config, utils)
├── conhecimento/    knowledge agent — RAG pipeline
├── ml/              Machine Learning pipeline
└── orquestrador.py  backend flow coordinator
```

- **core** — central configuration and shared utilities
- **conhecimento** — PDF indexing, semantic + BM25 RAG, fixed-role all-Gemini team (Pro/Flash/Flash-Lite),
  memory consolidation
- **ml** — exploratory data analysis and fault classification
- **orquestrador** — runs pending steps on startup, skips
  what is already done (state verification)

## Tech stack

- Python 3.13
- Streamlit — local and cloud web interface
- ChromaDB — local vector database restored from a portable cloud snapshot
- sentence-transformers — multilingual embeddings
- LLM provider — Google Gemini (Pro for chat, Flash for auditing, Flash-Lite for background)
- scikit-learn, XGBoost, LightGBM — Machine Learning

## How to run

```
streamlit run app.py
```

A single Google Gemini API key must be set in a local `.env` file —
see `.env.example` for the template. The `.env` file is never
committed to the repository.

### One-time local Git setup

`resultados/` (pipeline artifacts) is regenerated only on the PC and is
git-tracked so the cloud deploy (query-only mode) can display it. Run this
**once** on any machine that will `git pull`/merge this repo, so a local
re-run of the pipeline never produces a merge conflict against the
already-committed artifacts (`.gitattributes` marks `resultados/** merge=ours`,
but the `ours` driver itself must be registered locally — it isn't something
a commit can carry):

```
git config merge.ours.driver true
```

## Status

| Phase | Description           | Status        |
|-------|-----------------------|---------------|
| 1     | Foundation            | Done          |
| 2     | RAG agent             | Done          |
| 3     | Streamlit interface   | Done          |
| 4     | Automation            | Done          |
| 5     | ML pipeline           | Implemented (E2 + E3 bench) |

Phase 5 status: all five stages (features, autoencoder, fault injection,
validation, Weibull/RUL) are implemented with provenance manifests.
Current stage state and metrics live in `resultados/` artifacts (E2 =
FMEA-guided synthetic validation). A separate GPVS-Faults protocol now provides
E3 experimental bench validation; field validation is still not performed.

## Documentação técnica

- [`docs/metodologia_ml.md`](docs/metodologia_ml.md) — decisões metodológicas e de integridade.
- [`docs/datasets.md`](docs/datasets.md) — Stender, PV Farms, GPVS-Faults E3 e separação de domínio.
- [`docs/evidence_levels.md`](docs/evidence_levels.md) — níveis de evidência E0–E3.
- [`docs/reproducibilidade.md`](docs/reproducibilidade.md) — manifestos, estados, memória, recálculo.
- [`docs/memoria_agentes.md`](docs/memoria_agentes.md) — aprendizado validado entre sessões e limites de persistência.
- [`docs/comandos.md`](docs/comandos.md) — todos os comandos.

## Verificação rápida

```powershell
python scripts/verificar_ambiente.py    # diagnóstico (imports, chaves, datasets, ChromaDB, pipeline)
python -m pytest                        # testes unitários
streamlit run app.py                    # interface (use 'streamlit run', não 'python app.py')
```

## Author

Rodolfo Torres — Master's student in Electrical Engineering, UTFPR.
Advisor: Prof. Fernanda Cristina Correa.
