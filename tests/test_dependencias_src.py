"""Restrições arquiteturais mínimas entre os pacotes de src/."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_ml_nao_importa_conhecimento():
    achados = []
    for arquivo in (RAIZ / "src/ml").glob("*.py"):
        fonte = arquivo.read_text(encoding="utf-8")
        if "src.conhecimento" in fonte:
            achados.append(arquivo.name)
    assert not achados, f"dependência reversa ml -> conhecimento: {achados}"


def test_escritas_chromadb_usam_lock_canonico():
    contratos = {
        "src/conhecimento/indexador.py": ("indexar_pdf_unico", "indexar_sessao"),
        "src/conhecimento/indice_portatil.py": ("importar_colecao",),
        "src/conhecimento/obsidian.py": ("sincronizar_obsidian",),
        "src/conhecimento/consolidar_memoria.py": ("atualizar_chromadb",),
    }
    for relativo, funcoes in contratos.items():
        fonte = (RAIZ / relativo).read_text(encoding="utf-8")
        assert "from src.conhecimento.index_lock import lock_indexacao" in fonte
        for funcao in funcoes:
            assert f"def {funcao}(" in fonte
