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
- **conhecimento** — PDF indexing, semantic + BM25 RAG, provider-neutral
  OpenAI/Gemini routing, evidence guards and memory consolidation
- **ml** — canonical GPVS ingestion, Denso versus AE-LSTM comparison,
  FMECA maintenance context and bibliographic physical reliability
- **webapp** — modern conversation workspace, managed chat history, editable
  reference library, and an HTTP adapter for the ALIAdo agent; scientific
  contracts and figures are invoked from the conversation when requested
- **orquestrador** — executes explicitly requested indexing and ML operations;
  it is not run when the dashboard opens

## Tech stack

- Python 3.13
- Starlette + Uvicorn — local or cloud ASGI application
- Matplotlib — academic PNG 300 dpi and vector PDF figures loaded on demand
- ChromaDB — local vector database restored from a portable cloud snapshot
- sentence-transformers — multilingual embeddings
- LLM providers — OpenAI and Google Gemini through one auditable Gateway/Router
- PyTorch + scikit-learn — Denso and AE-LSTM anomaly detectors

## How to run

```powershell
python -m src.webapp
# development with reload:
uvicorn src.webapp.app:app --reload
```

Open `http://127.0.0.1:8000`. Root `app.py` is the ASGI bridge used by hosts
that import `app:app`.

Configure at least one supported provider in a local `.env` file. Set both
`OPENAI_API_KEY` and `GOOGLE_API_KEY` to enable cross-provider fallback; model
aliases remain configurable as shown in `.env.example`. The `.env` file is
never committed to the repository.

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
| 5     | ML pipeline           | Implemented (E3 bench + bibliographic reliability) |

Phase 5 status: the two-stage scientific pipeline uses GPVS-Faults only as the
experimental base for comparing the Denso and AE-LSTM detectors. F0L/F0M train,
validate, calibrate and test the models; F1L-F7M provide E3 experimental bench
validation. FMECA supports maintenance prioritization, while temporal
reliability curves are independent bibliographic sensitivity scenarios. Field
validation is still absent.

## Documentação técnica

- [`docs/metodologia_ml.md`](docs/metodologia_ml.md) — decisões metodológicas e de integridade.
- [`docs/datasets.md`](docs/datasets.md) — contrato único GPVS-Faults, qualidade e limites.
- [`docs/evidence_levels.md`](docs/evidence_levels.md) — níveis de evidência E0–E3.
- [`docs/confiabilidade_fisica.md`](docs/confiabilidade_fisica.md) — curvas temporais, taxas e rastreabilidade bibliográfica.
- [`docs/mapa_de_resultados.md`](docs/mapa_de_resultados.md) — figuras, tabelas e contratos publicados.
- [`docs/reproducibilidade.md`](docs/reproducibilidade.md) — manifestos, estados, memória, recálculo.
- [`docs/memoria_agentes.md`](docs/memoria_agentes.md) — aprendizado validado entre sessões e limites de persistência.
- [`docs/aplicacao_web.md`](docs/aplicacao_web.md) — aplicação ASGI, APIs e limites operacionais.
- [`docs/comandos.md`](docs/comandos.md) — todos os comandos.

## Verificação rápida

```powershell
python scripts/verificar_projeto.py     # ambiente, GPVS, árvore e contratos publicados
python scripts/auditar_resultados.py    # contratos canônicos, outputs e hashes
python -m pytest -m "not pesado"        # testes unitários e de integração leves
python -m src.webapp                    # interface em http://127.0.0.1:8000
```

## Author

Rodolfo Torres — Master's student in Electrical Engineering, UTFPR.
Advisor: Prof. Fernanda Cristina Correa.
