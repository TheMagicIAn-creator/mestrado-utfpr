"""Ponte ASGI da aplicação canônica ALIAdo.

O comando recomendado é ``python -m src.webapp``. Este módulo também permite
``python app.py`` e servidores ASGI que importam ``app:app``.
"""

from src.webapp.app import app

if __name__ == "__main__":
    from src.webapp.launcher import main

    main(app)
