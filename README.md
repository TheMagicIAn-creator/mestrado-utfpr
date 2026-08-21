# ALIAdo — Mestrado UTFPR

Predictive failure analysis of AC-side components in grid-connected
photovoltaic inverters using Machine Learning, with a
Reliability-Centered Maintenance (RCM) methodology.

Master's research project — Electrical Engineering, UTFPR.
Defense scheduled for March 2027.

## Overview

This repository hosts **ALIAdo**, an AI research assistant built
to support the master's dissertation. The system combines a
knowledge agent (RAG over the scientific literature) with a
Machine Learning pipeline for fault detection in PV inverters.

The methodology models the *healthy* behaviour of an inverter and
detects developing faults as deviations from that normality —
the standard approach in industrial predictive maintenance, where
real fault data is scarce.

## Architecture

The project is a modular Python package. The primary entry point is
`app.py`, an ASGI application that serves the academic dashboard and exposes
the knowledge agent through HTTP. Scientific results are read from validated,
versioned contracts; opening the page never retrains a model.

```
src/
├── core/            shared infrastructure (config, utils)
├── conhecimento/    knowledge agent — RAG pipeline
├── ml/              Machine Learning pipeline
├── webapp/          canonical Starlette + semantic HTML/CSS/JavaScript
└── orquestrador.py  backend flow coordinator
```

- **core** — central configuration and shared utilities
- **conhecimento** — PDF indexing, semantic + BM25 RAG, fixed-role all-Gemini team (Pro/Flash/Flash-Lite),
  memory consolidation
- **ml** — exploratory data analysis and fault classification
- **webapp** — read-only E2, E3 and reliability contracts, academic figures
  loaded on demand and an HTTP adapter for the ALIAdo agent
- **orquestrador** — executes explicitly requested indexing and ML operations;
  it is not run when the dashboard opens

## Tech stack

- Python 3.13
- Starlette + Uvicorn — local or cloud ASGI application
- Matplotlib — academic PNG 300 dpi and vector PDF figures loaded on demand
- ChromaDB — local vector database restored from a portable cloud snapshot
- sentence-transformers — multilingual embeddings
- LLM provider — Google Gemini (Pro for chat, Flash for auditing, Flash-Lite for background)
- scikit-learn, XGBoost, LightGBM — Machine Learning

## How to run

```powershell
python -m src.webapp
# development with reload:
uvicorn src.webapp.app:app --reload
```

Open `http://127.0.0.1:8000`. `python app.py` remains a compatibility alias.

A single Google Gemini API key must be set in a local `.env` file —
see `.env.example` for the template. The `.env` file is never
committed to the repository.

### Data and artifact policy

The 16 raw GPVS-Faults CSV files remain under the ignored `dados/` directory.
Reproducible JSON, CSV, Markdown and figures under `resultados/` are tracked so
the query-only cloud deployment can display the latest verified execution.
Models, scalers and local Obsidian state are not published.

## Status

| Phase | Description           | Status        |
|-------|-----------------------|---------------|
| 1     | Foundation            | Done          |
| 2     | RAG agent             | Done          |
| 3     | ASGI web application  | Done          |
| 4     | Automation            | Done          |
| 5     | ML pipeline           | Implemented (E2 + E3 bench) |

Phase 5 status: the five-stage pipeline uses GPVS-Faults as its single
canonical dataset. F0L/F0M train and calibrate the Autoencoder; FMECA-guided
synthetic injection on the F0 holdout provides E2 evidence, and F1L-F7M provide
E3 experimental bench validation. Weibull describes synthetic detectability
magnitude, not physical RUL. Field validation is still not performed.

## Documentação técnica

- [`docs/metodologia_ml.md`](docs/metodologia_ml.md) — decisões metodológicas e de integridade.
- [`docs/datasets.md`](docs/datasets.md) — contrato único GPVS-Faults, qualidade e limites.
- [`docs/evidence_levels.md`](docs/evidence_levels.md) — níveis de evidência E0–E3.
- [`docs/reproducibilidade.md`](docs/reproducibilidade.md) — manifestos, estados, memória, recálculo.
- [`docs/memoria_agentes.md`](docs/memoria_agentes.md) — aprendizado validado entre sessões e limites de persistência.
- [`docs/aplicacao_web.md`](docs/aplicacao_web.md) — aplicação ASGI, APIs e limites operacionais.
- [`docs/comandos.md`](docs/comandos.md) — todos os comandos.

## Verificação rápida

```powershell
python scripts/verificar_ambiente.py    # diagnóstico (imports, chaves, datasets, ChromaDB, pipeline)
python -m pytest                        # testes unitários
python -m src.webapp                     # interface em http://127.0.0.1:8000
```

## Author

Rodolfo Torres — Master's student in Electrical Engineering, UTFPR.
Advisor: Prof. Fernanda Cristina Correa.
