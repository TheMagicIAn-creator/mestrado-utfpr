"""
Sprint 1 — reprodutibilidade e portabilidade.

Garante que os artefatos gravam caminhos RELATIVOS ao projeto (sem 'C:\\...'),
e que a interface consegue resolver de volta para absoluto.
"""

import re
from pathlib import Path

from src.core.config import RAIZ_PROJETO
from src.core.utils import resolve_project_path, to_project_relative_path

_ABSOLUTO = re.compile(r"^[a-zA-Z]:[\\/]|^/")  # 'C:\\...' ou '/...'


def test_to_project_relative_remove_raiz():
    alvo = Path(RAIZ_PROJETO) / "resultados" / "experimentos" / "ghoneim" / "comparacao.png"
    rel = to_project_relative_path(alvo)
    assert not _ABSOLUTO.match(rel), f"deveria ser relativo: {rel}"
    assert "\\" not in rel, "deve usar separador POSIX '/'"
    assert rel == "resultados/experimentos/ghoneim/comparacao.png"


def test_round_trip_relativo_absoluto():
    alvo = Path(RAIZ_PROJETO) / "resultados" / "autoencoder" / "validacao_roc.png"
    rel = to_project_relative_path(alvo)
    de_volta = resolve_project_path(rel)
    assert de_volta.resolve() == alvo.resolve()


def test_resolve_mantem_absoluto_existente():
    # caminho já absoluto (compat. com artefatos antigos) é devolvido absoluto
    p = resolve_project_path(str(Path(RAIZ_PROJETO) / "x.png"))
    assert p.is_absolute()


def test_resultado_json_experimento_sem_caminho_absoluto():
    """Se houver resultado.json de experimento, o campo grafico é relativo."""
    import json

    base = Path(RAIZ_PROJETO) / "resultados" / "experimentos"
    encontrados = list(base.glob("*/resultado.json")) if base.exists() else []
    if not encontrados:
        import pytest

        pytest.skip("nenhum resultado.json de experimento disponível ainda")
    for arq in encontrados:
        d = json.loads(arq.read_text(encoding="utf-8"))
        for campo in ("grafico",):
            val = d.get(campo)
            if val:
                assert not _ABSOLUTO.match(str(val)), f"{arq.name}:{campo} absoluto: {val}"


def test_metadados_pendentes_gravam_caminho_relativo(tmp_path, monkeypatch):
    import json

    import src.conhecimento.processador_pdf as proc
    import src.core.config as config
    import src.core.utils as utils

    monkeypatch.setattr(config, "RAIZ_PROJETO", tmp_path)
    monkeypatch.setattr(utils, "RAIZ_PROJETO", tmp_path, raising=False)
    monkeypatch.setattr(proc, "RAIZ_PROJETO", tmp_path)

    pdf = tmp_path / "novos_pdfs" / "Autor_Titulo_2024.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n")

    proc._registrar_pendencia(pdf, "Autor", "Titulo", "2024")

    d = json.loads((tmp_path / "metadados_pendentes.json").read_text(encoding="utf-8"))
    caminho = d[pdf.name]["arquivo"]
    assert caminho == "novos_pdfs/Autor_Titulo_2024.pdf"
    assert not _ABSOLUTO.match(caminho)
