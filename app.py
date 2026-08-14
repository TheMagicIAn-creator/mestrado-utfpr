"""Ponte ASGI da aplicacao canonica ALIAdo PV Web V2.

O comando recomendado e ``python -m src.webapp_v2``. Este modulo permanece
como compatibilidade para ``python app.py`` e servidores ASGI que importam
``app:app``. Ele nunca inicializa a interface Streamlit legada.
"""

from src.webapp_v2.launcher import bloquear_execucao_streamlit

if __name__ == "__main__":
    bloquear_execucao_streamlit()

from src.webapp_v2.app import app  # noqa: E402

if __name__ == "__main__":
    from src.webapp_v2.launcher import main

    main(app)
