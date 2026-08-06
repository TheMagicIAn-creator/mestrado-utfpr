"""Acesso tardio ao Streamlit atual da fachada da interface."""

from __future__ import annotations


class _StreamlitProxy:
    def __getattr__(self, nome):
        from src.interface import streamlit_app

        return getattr(streamlit_app.st, nome)


st = _StreamlitProxy()
