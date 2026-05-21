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
- **conhecimento** — PDF indexing, RAG agent, multi-LLM provider,
  memory consolidation
- **ml** — exploratory data analysis and fault classification
- **orquestrador** — runs pending steps on startup, skips
  what is already done (state verification)

## Tech stack

- Python 3.13
- Streamlit — local web interface
- ChromaDB — local vector database
- sentence-transformers — multilingual embeddings
- LLM providers — Google Gemini and Groq
- scikit-learn, XGBoost, LightGBM — Machine Learning

## How to run

```
streamlit run app.py
```

API keys (Google and Groq) must be set in a local `.env` file —
see `.env.example` for the template. The `.env` file is never
committed to the repository.

## Status

| Phase | Description           | Status        |
|-------|-----------------------|---------------|
| 1     | Foundation            | Done          |
| 2     | RAG agent             | Done          |
| 3     | Streamlit interface   | Done          |
| 4     | Automation            | Done          |
| 5     | ML pipeline           | In progress   |

## Author

Rodolfo Torres — Master's student in Electrical Engineering, UTFPR.
Advisor: Prof. Fernanda Cristina Correa.